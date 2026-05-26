"""
01_vector_index.py — Solr 9 DenseVectorField with nomic-embed-text via Ollama.

Embeddings come from nomic-embed-text (already in ollama ls, 274MB).
nomic-embed-text produces 768-dimensional embeddings — better quality than
all-MiniLM-L6-v2 (384 dims) and already on your machine, no extra install.

Covers:
  - Embedding via Ollama /api/embed endpoint (batch)
  - DenseVectorField schema (768 dims, cosine similarity)
  - Indexing docs with vectors
  - KNN search: {!knn f=chunk_vector topK=N}
  - Scoped KNN (pre-filtered by cluster/entity from FalkorDB)
  - Hybrid BM25 + KNN scoring

Run: docker compose up -d   (Solr on :8984)
     uv run 01_vector_index.py
"""

import pysolr
import requests
import numpy as np
import time

SOLR_URL  = "http://localhost:8984/solr/ovir_vectors"
SCHEMA_URL = f"{SOLR_URL}/schema"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DIMS = 768  # nomic-embed-text output dimension

solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=30)

# ── Wait for Solr ─────────────────────────────────────────────────────────────
for attempt in range(15):
    try:
        solr.ping()
        print("Solr is up.")
        break
    except Exception:
        print(f"  Waiting... ({attempt+1}/15)")
        time.sleep(2)
else:
    raise RuntimeError("Solr not ready. Run: docker compose up -d")


# ── Embedding via Ollama ──────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using nomic-embed-text via Ollama /api/embed.
    Returns: float32 array of shape (N, 768), L2-normalized.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


# ── Verify Ollama embedding works ─────────────────────────────────────────────
print("\n=== Verifying nomic-embed-text via Ollama ===")
test_vec = embed_one("test")
print(f"  Embedding dims: {len(test_vec)}  (expected {DIMS})")
assert len(test_vec) == DIMS, f"Dimension mismatch: got {len(test_vec)}, expected {DIMS}"
print("  OK.")


# ── Schema ────────────────────────────────────────────────────────────────────

def add_field(name, field_type, **kwargs):
    payload = {"add-field": {"name": name, "type": field_type, **kwargs}}
    r = requests.post(SCHEMA_URL, json=payload)
    data = r.json()
    if "errors" in data:
        if not any("already exists" in str(e) for e in data["errors"]):
            print(f"  Schema warning for {name}: {data['errors']}")

def ensure_vector_field_type():
    payload = {"add-field-type": {
        "name": f"knn_vector_{DIMS}",
        "class": "solr.DenseVectorField",
        "vectorDimension": DIMS,
        "similarityFunction": "cosine",
    }}
    r = requests.post(SCHEMA_URL, json=payload)
    data = r.json()
    if "errors" in data:
        if not any("already exists" in str(e) for e in data["errors"]):
            print(f"  Field type warning: {data['errors']}")

print("\n=== Adding schema fields ===")
ensure_vector_field_type()
add_field("chunk_id",     "string")
add_field("source",       "string")
add_field("chunk_text",   "text_general")
add_field("cluster_id",   "string")
add_field("entity_names", "string",  multiValued=True)
add_field("chunk_vector", f"knn_vector_{DIMS}", stored=True, indexed=True)
print("  Fields added (or already existed).")


# ── Index documents ───────────────────────────────────────────────────────────
corpus = [
    {"id": "c1",  "chunk_id": "c1",  "source": "acme_msla.pdf",      "cluster_id": "financial_obligations", "chunk_text": "ACME Corp agrees to pay Globex two million dollars annually.", "entity_names": ["ACME Corp", "Globex"]},
    {"id": "c2",  "chunk_id": "c2",  "source": "acme_msla.pdf",      "cluster_id": "contract_terms",        "chunk_text": "Liability is capped at twelve months of fees under Section 8.", "entity_names": ["ACME Corp"]},
    {"id": "c3",  "chunk_id": "c3",  "source": "globex_vendor.pdf",  "cluster_id": "org_structure",         "chunk_text": "Globex is a subsidiary of Initech Holdings.", "entity_names": ["Globex", "Initech Holdings"]},
    {"id": "c4",  "chunk_id": "c4",  "source": "initech_annual.pdf", "cluster_id": "financial_obligations", "chunk_text": "Initech Holdings reported eight hundred million in revenue.", "entity_names": ["Initech Holdings"]},
    {"id": "c5",  "chunk_id": "c5",  "source": "initech_annual.pdf", "cluster_id": "org_structure",         "chunk_text": "CEO Bill Lumbergh oversees all subsidiaries.", "entity_names": ["Bill Lumbergh"]},
    {"id": "c6",  "chunk_id": "c6",  "source": "tech_runbook.md",    "cluster_id": "tech",                  "chunk_text": "AuthService handles authentication using JWT tokens.", "entity_names": ["AuthService"]},
    {"id": "c7",  "chunk_id": "c7",  "source": "tech_runbook.md",    "cluster_id": "tech",                  "chunk_text": "TokenService manages token expiry and Redis cache.", "entity_names": ["TokenService", "Redis"]},
    {"id": "c8",  "chunk_id": "c8",  "source": "tech_runbook.md",    "cluster_id": "tech",                  "chunk_text": "Alice owns the authentication and token services.", "entity_names": ["Alice"]},
    {"id": "c9",  "chunk_id": "c9",  "source": "acme_sow.pdf",       "cluster_id": "contract_terms",        "chunk_text": "Globex will deliver the data pipeline by June 30.", "entity_names": ["Globex"]},
    {"id": "c10", "chunk_id": "c10", "source": "acme_msla.pdf",      "cluster_id": "contract_terms",        "chunk_text": "Either party may terminate with thirty days written notice.", "entity_names": []},
]

print("\n=== Generating nomic-embed-text embeddings ===")
texts = [doc["chunk_text"] for doc in corpus]
t0 = time.perf_counter()
embeddings = embed(texts)
print(f"  {len(texts)} embeddings in {(time.perf_counter()-t0)*1000:.0f}ms  (dims={embeddings.shape[1]})")

docs_to_index = [{**doc, "chunk_vector": emb.tolist()} for doc, emb in zip(corpus, embeddings)]
solr.add(docs_to_index)
solr.commit()
print(f"  Indexed {len(docs_to_index)} docs.")


# ── KNN search helper ─────────────────────────────────────────────────────────

def knn_search(query_text: str, top_k: int = 5, fq: str = None):
    query_vec = embed_one(query_text)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
    kwargs = {"fl": "chunk_id,source,cluster_id,chunk_text,score", "rows": top_k}
    if fq:
        kwargs["fq"] = fq
    return solr.search(f"{{!knn f=chunk_vector topK={top_k}}}{vec_str}", **kwargs)


# ── 1. Pure KNN ───────────────────────────────────────────────────────────────
print("\n=== KNN: 'payment obligations' ===")
for doc in knn_search("payment obligations under the contract", top_k=4):
    print(f"  [{doc['chunk_id']}] score={doc['score']:.4f}  {doc['chunk_text'][:65]}...")

print("\n=== KNN: 'who is responsible for the services' ===")
for doc in knn_search("who is responsible for the services", top_k=4):
    print(f"  [{doc['chunk_id']}] score={doc['score']:.4f}  {doc['chunk_text'][:65]}...")


# ── 2. Scoped KNN (FalkorDB scope → Solr fq → KNN within subset) ─────────────
print("\n=== Scoped KNN: cluster_id=tech only ===")
for doc in knn_search("authentication token management", top_k=3, fq="cluster_id:tech"):
    print(f"  [{doc['chunk_id']}] score={doc['score']:.4f}  {doc['chunk_text'][:65]}...")

print("\n=== Scoped KNN: financial_obligations OR contract_terms ===")
for doc in knn_search("fees liability termination", top_k=4,
                       fq="cluster_id:(financial_obligations OR contract_terms)"):
    print(f"  [{doc['chunk_id']}] score={doc['score']:.4f}  {doc['chunk_text'][:65]}...")


# ── 3. Hybrid BM25 + KNN ─────────────────────────────────────────────────────

def hybrid_search(query_text: str, top_k: int = 5, bm25_w=0.3, knn_w=0.7):
    bm25 = solr.search(query_text, fl="chunk_id,chunk_text,score", rows=top_k * 2)
    knn  = knn_search(query_text, top_k=top_k * 2)

    scores: dict[str, dict] = {}
    bm25_max = max((d["score"] for d in bm25), default=1.0)
    for doc in bm25:
        scores[doc["chunk_id"]] = {
            "text": doc["chunk_text"],
            "score": bm25_w * (doc["score"] / bm25_max),
        }
    knn_max = max((d["score"] for d in knn), default=1.0)
    for doc in knn:
        cid = doc["chunk_id"]
        contrib = knn_w * (doc["score"] / knn_max)
        if cid in scores:
            scores[cid]["score"] += contrib
        else:
            scores[cid] = {"text": doc["chunk_text"], "score": contrib}

    return sorted(scores.items(), key=lambda x: -x[1]["score"])[:top_k]

print("\n=== Hybrid (BM25 30% + nomic KNN 70%): 'fees payment liability' ===")
for cid, info in hybrid_search("fees payment liability", top_k=4):
    print(f"  [{cid}] hybrid={info['score']:.4f}  {info['text'][:65]}...")


# ── 4. Embedding quality check ────────────────────────────────────────────────
print("\n=== Similarity spot-check ===")
pairs = [
    ("ACME Corp agrees to pay Globex two million dollars annually.",
     "What are the payment terms?"),
    ("ACME Corp agrees to pay Globex two million dollars annually.",
     "Who owns the authentication service?"),
]
for a, b in pairs:
    va, vb = embed_one(a), embed_one(b)
    sim = float(np.dot(va, vb))
    print(f"  sim={sim:.3f}  '{a[:45]}...' ↔ '{b}'")

print("\nDone.")
