"""
01_outer_loop.py — OVIR outer loop on real CFPB corpus.

Runs the full offline pipeline on the first N chunks from
data/cfpb_corpus.jsonl:

  1. GLiNER  — entity extraction with financial labels
  2. Embed   — nomic-embed-text via Ollama (batch)
  3. FalkorDB — CHUNK + ENTITY nodes, MENTIONS edges with confidence
  4. COBWEB  — CobwebRetriever built over embeddings, pickled
  5. Solr    — keyword fields + dense vector field indexed

After this script completes, run 02_query.py to test retrieval.

Prerequisites:
  - FalkorDB running (ports 6379 / 3000)
  - Solr running on :8983 with ovir_corpus collection
      docker compose -f ../solr-playground/docker-compose.yml up -d
  - Ollama running with nomic-embed-text in ollama ls

Run:
  uv run 01_outer_loop.py
  uv run 01_outer_loop.py --chunks 500   # larger run
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import requests
import pysolr
from falkordb import FalkorDB
from gliner import GLiNER
from cobweb_language_embedding import CobwebRetriever
from redis.exceptions import ResponseError

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH  = Path(__file__).parent.parent / "data" / "cfpb_corpus.jsonl"
RETRIEVER_PKL = Path(__file__).parent / "cfpb_retriever.pkl"

OLLAMA_URL   = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text"
SOLR_URL     = "http://localhost:8983/solr/ovir_corpus"
SCHEMA_URL   = f"{SOLR_URL}/schema"
FALKOR_HOST  = "localhost"
FALKOR_PORT  = 6379
GRAPH_NAME   = "cfpb_corpus"

GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_LABELS = [
    "organization",      # Wells Fargo, Citibank, Discover…
    "person",            # named individuals
    "monetary_amount",   # $714.66, $235.95…
    "date",              # 09/30/2023…
    "financial_product", # student loan, mortgage, credit card…
    "complaint_type",    # fraudulent charge, overdraft fee…
]
GLINER_THRESHOLD = 0.45

DIMS = 768

parser = argparse.ArgumentParser()
parser.add_argument("--chunks", type=int, default=100)
args = parser.parse_args()
N_CHUNKS = args.chunks

# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def normalize_entity_id(text: str) -> str:
    return text.lower().strip().replace(" ", "_").replace(".", "").replace(",", "")


# ── 1. Load corpus ────────────────────────────────────────────────────────────

print(f"=== Loading {N_CHUNKS} chunks from CFPB corpus ===")
chunks = []
with open(CORPUS_PATH) as f:
    for i, line in enumerate(f):
        if i >= N_CHUNKS:
            break
        chunks.append(json.loads(line))
print(f"  Loaded {len(chunks)} chunks.")


# ── 2. GLiNER extraction ──────────────────────────────────────────────────────

print(f"\n=== GLiNER extraction ({GLINER_MODEL}) ===")
t0 = time.perf_counter()
gliner = GLiNER.from_pretrained(GLINER_MODEL)
texts = [c["text"] for c in chunks]
batch_results = gliner.batch_predict_entities(texts, GLINER_LABELS, threshold=GLINER_THRESHOLD)
gliner_ms = (time.perf_counter() - t0) * 1000

# Attach entities to chunks
for chunk, entities in zip(chunks, batch_results):
    chunk["entities"] = entities  # list of {text, label, score, start, end}

total_entities = sum(len(c["entities"]) for c in chunks)
print(f"  {total_entities} entities extracted from {N_CHUNKS} chunks in {gliner_ms:.0f}ms")
print(f"  ({gliner_ms/N_CHUNKS:.1f}ms/chunk)")

# Sample
sample = chunks[0]
print(f"\n  Sample chunk [{sample['id']}]: {sample['text'][:80]}...")
for e in sample["entities"]:
    print(f"    [{e['label']:20s}] '{e['text']}'  ({e['score']:.2f})")


# ── 3. nomic-embed-text embeddings ────────────────────────────────────────────

print(f"\n=== Embedding {N_CHUNKS} chunks with nomic-embed-text ===")
t0 = time.perf_counter()

# Batch in groups of 50 to avoid Ollama timeouts on large runs
BATCH_SIZE = 50
all_embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    vecs = embed_batch(batch)
    all_embeddings.append(vecs)
    print(f"  Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}...")

embeddings = np.vstack(all_embeddings)
embed_ms = (time.perf_counter() - t0) * 1000
print(f"  Done: shape={embeddings.shape}  {embed_ms:.0f}ms  ({embed_ms/N_CHUNKS:.1f}ms/chunk)")


# ── 4. FalkorDB graph build ───────────────────────────────────────────────────

print(f"\n=== Building FalkorDB graph '{GRAPH_NAME}' ===")
db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)

# Drop and recreate for a clean run
try:
    db.select_graph(GRAPH_NAME).delete()
    print("  Dropped existing graph.")
except ResponseError:
    pass
g = db.select_graph(GRAPH_NAME)

# Create indexes before bulk insert
g.query("CREATE INDEX FOR (c:Chunk) ON (c.id)")
g.query("CREATE INDEX FOR (e:Entity) ON (e.id)")

t0 = time.perf_counter()
nodes = 0
edges = 0

for chunk in chunks:
    cid = chunk["id"]
    src = chunk["source"]
    product = chunk["metadata"].get("product", "")
    company = chunk["metadata"].get("company", "")
    date    = chunk["metadata"].get("date", "")
    text_escaped = chunk["text"].replace("'", "\\'")[:200]  # truncate for graph storage

    # Chunk node
    g.query(
        f"MERGE (c:Chunk {{id: '{cid}'}}) "
        f"SET c.source='{src}', c.product='{product}', "
        f"    c.company='{company}', c.date='{date}', "
        f"    c.text='{text_escaped}'"
    )
    nodes += 1

    # Entity nodes + MENTIONS edges
    for ent in chunk["entities"]:
        eid = normalize_entity_id(ent["text"])
        label = ent["label"]
        score = round(ent["score"], 3)
        ent_text_escaped = ent["text"].replace("'", "\\'")

        g.query(
            f"MERGE (e:Entity {{id: '{eid}'}}) "
            f"SET e.text='{ent_text_escaped}', e.label='{label}'"
        )
        g.query(
            f"MATCH (c:Chunk {{id: '{cid}'}}), (e:Entity {{id: '{eid}'}}) "
            f"MERGE (c)-[r:MENTIONS {{confidence: {score}}}]->(e)"
        )
        nodes += 1
        edges += 1

falkor_ms = (time.perf_counter() - t0) * 1000

# Stats
result = g.query("MATCH (c:Chunk) RETURN count(c) AS n")
n_chunks_graph = result.result_set[0][0]
result = g.query("MATCH (e:Entity) RETURN count(e) AS n")
n_entities_graph = result.result_set[0][0]

print(f"  Chunk nodes : {n_chunks_graph}")
print(f"  Entity nodes: {n_entities_graph}")
print(f"  MENTIONS    : {edges}")
print(f"  Time        : {falkor_ms:.0f}ms")


# ── 5. COBWEB retriever ───────────────────────────────────────────────────────

print(f"\n=== Building CobwebRetriever ===")
t0 = time.perf_counter()
retriever = CobwebRetriever(corpus=texts, corpus_embeddings=embeddings)
cobweb_ms = (time.perf_counter() - t0) * 1000

with open(RETRIEVER_PKL, "wb") as f:
    pickle.dump({"retriever": retriever, "chunks": chunks}, f)

print(f"  Built in {cobweb_ms:.0f}ms")
print(f"  Saved to {RETRIEVER_PKL}")


# ── 6. Solr indexing ──────────────────────────────────────────────────────────

print(f"\n=== Solr indexing ===")
solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=30)

# Verify Solr is up
try:
    solr.ping()
    print("  Solr is up.")
except Exception:
    print("  ERROR: Solr not reachable at :8983.")
    print("  Run: docker compose -f ../solr-playground/docker-compose.yml up -d")
    raise

# Add vector field type + field if not present
def post_schema(payload):
    r = requests.post(SCHEMA_URL, json=payload)
    data = r.json()
    if "errors" in data:
        if not any("already exists" in str(e) for e in data["errors"]):
            print(f"  Schema warning: {data['errors']}")

post_schema({"add-field-type": {
    "name": f"knn_vector_{DIMS}",
    "class": "solr.DenseVectorField",
    "vectorDimension": DIMS,
    "similarityFunction": "cosine",
}})
post_schema({"add-field": {"name": "chunk_id",    "type": "string",              "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "chunk_text",  "type": "text_general",        "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "source",      "type": "string",              "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "product",     "type": "string",              "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "company",     "type": "string",              "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "doc_date",    "type": "string",              "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "entities",    "type": "string", "multiValued": True, "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "entity_types","type": "string", "multiValued": True, "stored": True, "indexed": True}})
post_schema({"add-field": {"name": "chunk_vector","type": f"knn_vector_{DIMS}",  "stored": True, "indexed": True}})
post_schema({"add-copy-field": {"source": "chunk_text", "dest": "_text_"}})
print("  Schema fields ready.")

# Build and index docs
t0 = time.perf_counter()
solr_docs = []
for chunk, emb in zip(chunks, embeddings):
    entity_texts = [e["text"] for e in chunk["entities"]]
    entity_labels = [e["label"] for e in chunk["entities"]]
    solr_docs.append({
        "id":           chunk["id"],
        "chunk_id":     chunk["id"],
        "chunk_text":   chunk["text"],
        "source":       chunk["source"],
        "product":      chunk["metadata"].get("product", ""),
        "company":      chunk["metadata"].get("company", ""),
        "doc_date":     chunk["metadata"].get("date", ""),
        "entities":     entity_texts,
        "entity_types": entity_labels,
        "chunk_vector": emb.tolist(),
    })

solr.add(solr_docs)
solr.commit()
solr_ms = (time.perf_counter() - t0) * 1000

total_docs = solr.search("*:*", rows=0).hits
print(f"  Indexed {len(solr_docs)} docs in {solr_ms:.0f}ms")
print(f"  Total docs in index: {total_docs}")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"""
=== Outer loop complete ===
  Chunks processed : {N_CHUNKS}
  Entities found   : {total_entities}  ({total_entities/N_CHUNKS:.1f}/chunk avg)
  FalkorDB nodes   : {n_chunks_graph} chunks + {n_entities_graph} entities
  FalkorDB edges   : {edges} MENTIONS
  Solr docs        : {len(solr_docs)}
  COBWEB retriever : {RETRIEVER_PKL}

  Timing (ms):
    GLiNER   : {gliner_ms:>7.0f}ms  ({gliner_ms/N_CHUNKS:.1f}/chunk)
    Embed    : {embed_ms:>7.0f}ms  ({embed_ms/N_CHUNKS:.1f}/chunk)
    FalkorDB : {falkor_ms:>7.0f}ms  ({falkor_ms/N_CHUNKS:.1f}/chunk)
    COBWEB   : {cobweb_ms:>7.0f}ms
    Solr     : {solr_ms:>7.0f}ms  ({solr_ms/N_CHUNKS:.1f}/chunk)

Run 02_query.py to test retrieval against the built index.
""")
