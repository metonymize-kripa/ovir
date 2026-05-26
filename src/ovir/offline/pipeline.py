import json
import time
import hashlib
from collections import defaultdict
import numpy as np
import requests
import pysolr
import pickle
from pathlib import Path
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from cobweb_language_embedding import CobwebRetriever
import ray

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


# --- Stateful Actors ---

@ray.remote
class FalkorActor:
    def __init__(self):
        self.db = FalkorDB(host="localhost", port=6379)
        self.g = self.db.select_graph("ovir_corpus")
        self.ingested = 0

    def clear_graph(self):
        try:
            self.db.select_graph("ovir_corpus").delete()
        except ResponseError:
            pass
        self.g = self.db.select_graph("ovir_corpus")

    def ingest(self, chunk: dict, entities: list):
        cid = chunk["id"]
        # Merge chunk
        self.g.query(
            "MERGE (c:Chunk {id: $id}) SET c.text = $text, c.source = $source, c.hash = $hash",
            {"id": cid, "text": chunk["text"], "source": chunk["source"],
             "hash": hashlib.md5(chunk["text"].encode()).hexdigest()[:8]}
        )
        # Merge entities
        for ent in entities:
            self.g.query(
                "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type",
                {"id": ent["entity_id"], "name": ent["entity_name"], "type": ent["entity_type"]}
            )
            self.g.query("""
              MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
              MERGE (c)-[m:MENTIONS]->(e)
              SET m.confidence = $conf
            """, {"cid": cid, "eid": ent["entity_id"], "conf": ent["confidence"]})
        self.ingested += 1
        return True

    def get_stats(self):
        return {"falkor_chunks_ingested": self.ingested}

@ray.remote
class SolrActor:
    def __init__(self):
        self.solr_keyword = pysolr.Solr(SOLR_KEYWORD_URL, always_commit=False, timeout=30)
        self.solr_vector = pysolr.Solr(SOLR_VECTOR_URL, always_commit=False, timeout=30)
        self.ingested = 0
        self._setup_solr_schemas()
        
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

    def ingest(self, chunk: dict, entities: list, emb: list, cluster_id: str):
        solr_doc = {
            "id": chunk["id"],
            "chunk_id": chunk["id"],
            "source": chunk["source"],
            "chunk_text": chunk["text"],
            "entities": [e["entity_name"] for e in entities],
            "entity_types": list(set(e["entity_type"] for e in entities)),
            "confidence": min([e["confidence"] for e in entities], default=1.0),
            "cluster_id": cluster_id,
            "chunk_vector": emb
        }
        
        # Keyword
        solr_kw_doc = {k: v for k, v in solr_doc.items() if k != "chunk_vector"}
        self.solr_keyword.add([solr_kw_doc])
        
        # Vector
        self.solr_vector.add([solr_doc])
        
        self.ingested += 1
        
        if self.ingested % 100 == 0:
            self.solr_keyword.commit()
            self.solr_vector.commit()
            
        return True
        
    def finalize(self):
        self.solr_keyword.commit()
        self.solr_vector.commit()
        return {"solr_docs_ingested": self.ingested}


@ray.remote
class CobwebActor:
    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.docs_metadata = []

    def ingest(self, chunk: dict, entities: list, emb: list):
        self.texts.append(chunk["text"])
        self.embeddings.append(emb)
        self.docs_metadata.append({
            "chunk_id": chunk["id"], 
            "source": chunk["source"], 
            "text": chunk["text"],
            "entities": entities
        })
        # Simulate a cluster id for solr mapping
        cluster_id = f"concept_{len(self.texts) % 5}"
        return cluster_id

    def build_tree(self):
        print("Building CobwebRetriever tree...")
        if not self.texts:
            return {"status": "no data"}
            
        embeddings_arr = np.array(self.embeddings, dtype=np.float32)
        retriever = CobwebRetriever(corpus=self.texts, corpus_embeddings=embeddings_arr)
        
        with open(RETRIEVER_PATH, "wb") as f:
            pickle.dump({"retriever": retriever, "docs": self.docs_metadata}, f)
            
        return {"tree_size": len(self.texts), "path": str(RETRIEVER_PATH)}


# --- Stateless Worker ---

_gliner_model = None

@ray.remote
def process_chunk(chunk: dict, falkor_actor, solr_actor, cobweb_actor):
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        
    # 1. Extraction
    entities = _gliner_model.predict_entities(chunk["text"], LABELS, threshold=0.45)
    
    formatted_entities = []
    for e in entities:
        eid = normalize_entity_id(e["text"])
        etype = {"organization": "ORG", "person": "PERSON", "location": "LOCATION",
                 "date": "DATE", "monetary_amount": "AMOUNT", "product": "PRODUCT",
                 "technology": "TECH"}.get(e["label"], e["label"].upper())
        formatted_entities.append({
            "entity_id": eid, "entity_name": e["text"], "entity_type": etype,
            "confidence": round(e["score"], 4)
        })
        
    # 2. Embedding
    emb = embed([chunk["text"]])[0].tolist()
    
    # 3. Graph Routing
    falkor_actor.ingest.remote(chunk, formatted_entities)
    
    # 4. Cobweb & Solr Routing (sequential to get cluster ID)
    cluster_id = ray.get(cobweb_actor.ingest.remote(chunk, formatted_entities, emb))
    solr_actor.ingest.remote(chunk, formatted_entities, emb, cluster_id)
    
    return chunk["id"]


# --- Main Application ---

def run_pipeline(corpus_file: Path, limit: int = None):
    print("=== Ray-powered OVIR Offline Pipeline ===")
    ray.init(ignore_reinit_error=True)
    
    falkor = FalkorActor.remote()
    solr = SolrActor.remote()
    cobweb = CobwebActor.remote()
    
    # Clear previous graph
    ray.get(falkor.clear_graph.remote())
    
    futures = []
    count = 0
    t0 = time.time()
    
    print(f"Streaming data from {corpus_file}...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunk = json.loads(line)
            
            # Fire and forget to background workers
            futures.append(process_chunk.remote(chunk, falkor, solr, cobweb))
            count += 1
            if limit and count >= limit:
                break
                
    # Wait for all workers to finish their ML tasks
    processed = ray.get(futures)
    print(f"\nDispatched {len(processed)} chunks in {time.time() - t0:.2f} seconds.")
    
    print("\nWaiting for sinks to finalize...")
    solr_stats = ray.get(solr.finalize.remote())
    falkor_stats = ray.get(falkor.get_stats.remote())
    cobweb_stats = ray.get(cobweb.build_tree.remote())
    
    print("\n=== Final Pipeline Statistics ===")
    print(f"FalkorDB: {falkor_stats}")
    print(f"Solr:     {solr_stats}")
    print(f"COBWEB:   {cobweb_stats}")

if __name__ == "__main__":
    # If the user provides the corpus file, use it. Otherwise use a small sample.
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "cfpb_corpus.jsonl"
    if data_path.exists():
        # Limit to 500 chunks for the integration test to save time, but it scales!
        run_pipeline(data_path, limit=500)
    else:
        print(f"Warning: Could not find {data_path}. Please run fetch_cfpb_corpus.py first.")
