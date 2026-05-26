"""
02_ovir_routing.py — CobwebRetriever as the OVIR query routing layer.

Full offline→online pipeline using nomic-embed-text + CobwebRetriever:

  OFFLINE (outer loop, runs once per corpus version):
    1. Embed all chunks with nomic-embed-text
    2. Build CobwebRetriever over the embeddings
    3. Save retriever to disk

  ONLINE (inner loop, per query):
    1. Embed the user query with nomic-embed-text
    2. CobwebRetriever.query() → top-k candidate chunks
    3. Use chunk IDs as Solr fq scope
    4. Emit retrieval trace

Run: uv run 02_ovir_routing.py
"""

import numpy as np
import requests
import pickle
import time
import json
from pathlib import Path
from cobweb_language_embedding import CobwebRetriever

OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
RETRIEVER_PATH = Path("cobweb_retriever.pkl")


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


# ── 1. OFFLINE: build and persist retriever ───────────────────────────────────

corpus_docs = [
    {"id": "c1",  "source": "acme_msla.pdf",      "text": "ACME Corp agrees to pay Globex two million dollars annually under this MSA."},
    {"id": "c2",  "source": "acme_msla.pdf",       "text": "Liability is capped at twelve months of fees paid by ACME Corp under Section 8."},
    {"id": "c3",  "source": "acme_msla.pdf",       "text": "Either party may terminate with thirty days written notice to the other party."},
    {"id": "c4",  "source": "acme_sow.pdf",        "text": "Globex will deliver the data pipeline milestone by June 30, 2024."},
    {"id": "c5",  "source": "globex_vendor.pdf",   "text": "Globex is a wholly-owned subsidiary of Initech Holdings incorporated in Delaware."},
    {"id": "c6",  "source": "initech_annual.pdf",  "text": "CEO Bill Lumbergh oversees all Initech Holdings subsidiaries and business units."},
    {"id": "c7",  "source": "initech_annual.pdf",  "text": "Initech Holdings reported eight hundred million dollars revenue in FY2023."},
    {"id": "c8",  "source": "tech_runbook.md",     "text": "AuthService handles user authentication using JWT tokens and OAuth2."},
    {"id": "c9",  "source": "tech_runbook.md",     "text": "TokenService manages token expiry and caches sessions in Redis with a 24-hour TTL."},
    {"id": "c10", "source": "tech_runbook.md",     "text": "Alice owns AuthService and TokenService. Bob owns UserDB and the Redis cluster."},
    {"id": "c11", "source": "tech_runbook.md",     "text": "APIGateway routes all inbound traffic through AuthService before downstream processing."},
    {"id": "c12", "source": "initech_annual.pdf",  "text": "Initech Holdings operates in twelve countries including Germany, Japan, and Brazil."},
]

print("=== OFFLINE: Building retriever ===")
texts = [doc["text"] for doc in corpus_docs]
t0 = time.perf_counter()
embeddings = embed(texts)
embed_ms = (time.perf_counter() - t0) * 1000
print(f"  Embedded {len(texts)} chunks in {embed_ms:.0f}ms")

retriever = CobwebRetriever(corpus=texts, corpus_embeddings=embeddings)

# Save retriever + doc metadata for query-time lookup
with open(RETRIEVER_PATH, "wb") as f:
    pickle.dump({"retriever": retriever, "docs": corpus_docs}, f)
print(f"  Saved retriever to {RETRIEVER_PATH}")


# ── 2. ONLINE: query-time routing ─────────────────────────────────────────────

with open(RETRIEVER_PATH, "rb") as f:
    saved = pickle.load(f)

loaded_retriever: CobwebRetriever = saved["retriever"]
docs_by_text = {doc["text"]: doc for doc in saved["docs"]}


def ovir_route(query: str, top_k: int = 4) -> dict:
    """
    OVIR inner loop:
      1. Embed query
      2. CobwebRetriever → top-k candidate chunks
      3. Return chunk IDs (→ pass to Solr fq)
      4. Emit trace
    """
    t0 = time.perf_counter()
    query_emb = embed([query])[0]
    results = loaded_retriever.query(query_emb, k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    chunks_in_scope = []
    for chunk_text in results:  # query() returns list of strings, no scores
        doc = docs_by_text.get(chunk_text, {"id": "?", "source": "?"})
        chunks_in_scope.append({
            "chunk_id": doc["id"],
            "source": doc["source"],
            "text": chunk_text,
        })

    return {
        "query": query,
        "scope": chunks_in_scope,
        "scope_ids": [c["chunk_id"] for c in chunks_in_scope],
        "latency_ms": round(latency_ms, 1),
    }


print("\n=== ONLINE: Query routing ===")
queries = [
    "What is ACME Corp's liability cap under the MSA?",
    "Who is responsible for the authentication service?",
    "What is Initech Holdings' financial performance?",
    "When is the Globex delivery deadline?",
]

for query in queries:
    trace = ovir_route(query, top_k=3)
    print(f"\n  Q: {trace['query']}")
    print(f"  Scope IDs → Solr fq: {trace['scope_ids']}")
    print(f"  Latency: {trace['latency_ms']}ms")
    for c in trace["scope"]:
        print(f"    [{c['chunk_id']}]  {c['text'][:65]}...")


# ── 3. Hop depth analogue: expand scope ───────────────────────────────────────
# CobwebRetriever's k parameter is the OVIR "hop depth" equivalent.
# More k = more scope = higher recall, wider Solr search.

print("\n=== Scope size vs. k (recall/compute tradeoff) ===")
query = "What are ACME Corp's payment and liability obligations?"
for k in [2, 4, 6]:
    trace = ovir_route(query, top_k=k)
    sources = {c["source"] for c in trace["scope"]}
    print(f"  k={k}: {len(trace['scope'])} chunks  sources={sources}")

print("\nKey: k is the COBWEB scope dial — same role as hop depth in FalkorDB traversal.")
print("OVIR starts at k=3, expands to k=6 if confidence is below threshold.")
print("\nDone.")
