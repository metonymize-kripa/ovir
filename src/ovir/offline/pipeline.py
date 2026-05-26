import json
import time
import hashlib
from collections import defaultdict
import numpy as np
import requests
import pysolr
import pickle
from pathlib import Path
from gliner import GLiNER
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from cobweb_language_embedding import CobwebRetriever

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
RETRIEVER_PATH = Path("cobweb_retriever.pkl")

# Solr Configs
SOLR_KEYWORD_URL = "http://localhost:8983/solr/ovir_corpus"
SOLR_VECTOR_URL = "http://localhost:8984/solr/ovir_vectors"

# OVIR Domain Labels
LABELS = [
    "organization", "person", "location", "date", 
    "monetary_amount", "product", "technology"
]

def embed(texts: list[str]) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

def normalize_entity_id(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace(".", "")

class OfflinePipeline:
    def __init__(self):
        self.gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.db = FalkorDB(host="localhost", port=6379)
        self.solr_keyword = pysolr.Solr(SOLR_KEYWORD_URL, always_commit=False, timeout=30)
        self.solr_vector = pysolr.Solr(SOLR_VECTOR_URL, always_commit=False, timeout=30)

    def _wait_for_services(self):
        print("Waiting for Solr services...")
        for url in [SOLR_KEYWORD_URL, SOLR_VECTOR_URL]:
            for _ in range(15):
                try:
                    requests.get(f"{url}/admin/ping")
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError(f"Solr not ready at {url}")
        print("Services ready.")

    def run(self, corpus: list[dict]):
        """
        Runs the full offline preprocessing pipeline over a list of chunks.
        corpus format: [{"id": "...", "source": "...", "text": "..."}]
        """
        self._wait_for_services()

        print("\n=== 1. GLiNER Entity Extraction ===")
        texts = [c["text"] for c in corpus]
        t0 = time.perf_counter()
        batch_results = self.gliner.batch_predict_entities(texts, LABELS, threshold=0.45)
        print(f"Extraction done in {(time.perf_counter() - t0) * 1000:.0f}ms")

        annotated_chunks = []
        for chunk, entities in zip(corpus, batch_results):
            chunk_entities = []
            for e in entities:
                eid = normalize_entity_id(e["text"])
                etype = {"organization": "ORG", "person": "PERSON", "location": "LOCATION",
                         "date": "DATE", "monetary_amount": "AMOUNT", "product": "PRODUCT",
                         "technology": "TECH"}.get(e["label"], e["label"].upper())
                chunk_entities.append({
                    "entity_id": eid, "entity_name": e["text"], "entity_type": etype,
                    "confidence": round(e["score"], 4)
                })
            annotated_chunks.append({
                "chunk_id": chunk["id"], "source": chunk["source"], "text": chunk["text"],
                "entities": chunk_entities
            })

        print("\n=== 2. FalkorDB Graph Ingestion ===")
        try:
            self.db.select_graph("ovir_corpus").delete()
        except ResponseError:
            pass
        g = self.db.select_graph("ovir_corpus")

        for ac in annotated_chunks:
            g.query(
                "MERGE (c:Chunk {id: $id}) SET c.text = $text, c.source = $source, c.hash = $hash",
                {"id": ac["chunk_id"], "text": ac["text"], "source": ac["source"],
                 "hash": hashlib.md5(ac["text"].encode()).hexdigest()[:8]}
            )
            for ent in ac["entities"]:
                g.query(
                    "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type",
                    {"id": ent["entity_id"], "name": ent["entity_name"], "type": ent["entity_type"]}
                )
                g.query("""
                  MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
                  MERGE (c)-[m:MENTIONS]->(e)
                  SET m.confidence = $conf
                """, {"cid": ac["chunk_id"], "eid": ent["entity_id"], "conf": ent["confidence"]})
        print(f"Ingested {len(annotated_chunks)} chunks into FalkorDB.")

        print("\n=== 3. Embedding & COBWEB Clustering ===")
        t0 = time.perf_counter()
        embeddings = embed(texts)
        print(f"Generated embeddings in {(time.perf_counter() - t0) * 1000:.0f}ms")

        retriever = CobwebRetriever(corpus=texts, corpus_embeddings=embeddings)
        with open(RETRIEVER_PATH, "wb") as f:
            pickle.dump({"retriever": retriever, "docs": annotated_chunks}, f)
        print(f"CobwebRetriever saved to {RETRIEVER_PATH}")

        # Determine implicit clusters from Cobweb (simulated for solr ingestion)
        # Using the base of the cobweb tree conceptually
        cluster_ids = ["concept_" + str(i % 5) for i in range(len(corpus))] 

        print("\n=== 4. Solr Ingestion ===")
        self._setup_solr_schemas()

        solr_docs = []
        for ac, emb, cluster_id in zip(annotated_chunks, embeddings, cluster_ids):
            solr_docs.append({
                "id": ac["chunk_id"],
                "chunk_id": ac["chunk_id"],
                "source": ac["source"],
                "chunk_text": ac["text"],
                "entities": [e["entity_name"] for e in ac["entities"]],
                "entity_types": list(set(e["entity_type"] for e in ac["entities"])),
                "confidence": min([e["confidence"] for e in ac["entities"]], default=1.0),
                "cluster_id": cluster_id,
                "chunk_vector": emb.tolist()
            })

        # To Solr Keyword
        solr_kw_docs = [{k: v for k, v in d.items() if k != "chunk_vector"} for d in solr_docs]
        self.solr_keyword.add(solr_kw_docs)
        self.solr_keyword.commit()
        
        # To Solr Vector
        self.solr_vector.add(solr_docs)
        self.solr_vector.commit()

        print(f"Indexed {len(solr_docs)} docs to both Solr cores.")
        print("\nOffline pipeline complete.")

    def _setup_solr_schemas(self):
        def add_field(url, name, field_type, **kwargs):
            payload = {"add-field": {"name": name, "type": field_type, **kwargs}}
            r = requests.post(f"{url}/schema", json=payload)
            if "errors" in r.json() and not any("already exists" in str(e) for e in r.json()["errors"]):
                print(f"Schema warning for {name}: {r.json()['errors']}")

        # Keyword
        add_field(SOLR_KEYWORD_URL, "chunk_id", "string")
        add_field(SOLR_KEYWORD_URL, "source", "string")
        add_field(SOLR_KEYWORD_URL, "chunk_text", "text_general")
        add_field(SOLR_KEYWORD_URL, "entities", "string", multiValued=True)
        add_field(SOLR_KEYWORD_URL, "entity_types", "string", multiValued=True)
        add_field(SOLR_KEYWORD_URL, "confidence", "pfloat")
        add_field(SOLR_KEYWORD_URL, "cluster_id", "string")
        requests.post(f"{SOLR_KEYWORD_URL}/schema", json={"add-copy-field": {"source": "chunk_text", "dest": "_text_"}})

        # Vector
        requests.post(f"{SOLR_VECTOR_URL}/schema", json={"add-field-type": {
            "name": "knn_vector_768", "class": "solr.DenseVectorField", 
            "vectorDimension": 768, "similarityFunction": "cosine"}})
        add_field(SOLR_VECTOR_URL, "chunk_id", "string")
        add_field(SOLR_VECTOR_URL, "source", "string")
        add_field(SOLR_VECTOR_URL, "chunk_text", "text_general")
        add_field(SOLR_VECTOR_URL, "cluster_id", "string")
        add_field(SOLR_VECTOR_URL, "chunk_vector", "knn_vector_768", stored=True, indexed=True)

if __name__ == "__main__":
    sample_corpus = [
        {"id": "c1", "source": "acme_msla_2024.pdf", "text": "ACME Corp agrees to pay Globex $2M annually under this MSA."},
        {"id": "c2", "source": "acme_msla_2024.pdf", "text": "Liability is capped at 12 months of fees paid by ACME Corp under Section 8.2."},
        {"id": "c3", "source": "acme_sow_001.pdf", "text": "Globex will deliver the data pipeline by June 30, 2024."},
        {"id": "c4", "source": "globex_vendor_reg.pdf", "text": "Globex is a wholly-owned subsidiary of Initech Holdings incorporated in Delaware."},
        {"id": "c5", "source": "initech_annual_2023.pdf", "text": "Initech Holdings reported $800M revenue in FY2023. CEO is Bill Lumbergh."},
    ]
    pipeline = OfflinePipeline()
    pipeline.run(sample_corpus)


