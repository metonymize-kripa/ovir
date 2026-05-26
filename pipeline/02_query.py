"""
02_query.py — OVIR inner loop against the CFPB index.

Tests end-to-end retrieval using the artifacts built by 01_outer_loop.py:
  1. Embed query with nomic-embed-text
  2. COBWEB → top-k candidate chunk IDs (semantic scope)
  3. FalkorDB → entity neighborhood chunks (graph scope)
  4. Solr fq → intersect scopes, rank by BM25 or KNN

Run: uv run 02_query.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import requests
import pysolr
from falkordb import FalkorDB

# ── Config ────────────────────────────────────────────────────────────────────

RETRIEVER_PKL = Path(__file__).parent / "cfpb_retriever.pkl"
OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
SOLR_URL      = "http://localhost:8983/solr/ovir_corpus"
FALKOR_HOST   = "localhost"
FALKOR_PORT   = 6379
GRAPH_NAME    = "cfpb_corpus"
DIMS          = 768

# ── Load artifacts ────────────────────────────────────────────────────────────

print("=== Loading artifacts ===")
with open(RETRIEVER_PKL, "rb") as f:
    saved = pickle.load(f)
retriever = saved["retriever"]
chunks_by_id = {c["id"]: c for c in saved["chunks"]}
print(f"  COBWEB retriever: {len(chunks_by_id)} chunks")

db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
g  = db.select_graph(GRAPH_NAME)
solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=10)
print(f"  FalkorDB graph : {GRAPH_NAME}")
print(f"  Solr collection: ovir_corpus")


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_one(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-9)


def cobweb_scope(query_text: str, k: int = 10) -> list[str]:
    """Return top-k chunk IDs from COBWEB semantic routing."""
    qvec = embed_one(query_text)
    results = retriever.query(qvec, k=k)
    # results is list of chunk texts; map back to IDs
    text_to_id = {c["text"]: c["id"] for c in chunks_by_id.values()}
    return [text_to_id[t] for t in results if t in text_to_id]


def falkor_scope(entity_name: str, hops: int = 2) -> list[str]:
    """Return chunk IDs for chunks mentioning entities near the given entity."""
    eid = entity_name.lower().strip().replace(" ", "_").replace(".", "").replace(",", "")
    result = g.query(
        f"MATCH (c:Chunk)-[:MENTIONS*1..{hops}]->(:Entity {{id: '{eid}'}}) "
        f"RETURN c.id AS chunk_id"
    )
    return [row[0] for row in result.result_set]


def solr_search(query: str, scope_ids: list[str], top_k: int = 5) -> list[dict]:
    """BM25 search within a scoped set of chunk IDs."""
    if not scope_ids:
        return []
    fq = "id:(" + " OR ".join(scope_ids) + ")"
    results = solr.search(
        query,
        fq=fq,
        fl="chunk_id,chunk_text,product,company,doc_date,entities,score",
        rows=top_k,
    )
    return list(results)


def solr_knn(query_text: str, scope_ids: list[str], top_k: int = 5) -> list[dict]:
    """KNN search within a scoped set of chunk IDs."""
    if not scope_ids:
        return []
    qvec = embed_one(query_text)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in qvec) + "]"
    fq = "id:(" + " OR ".join(scope_ids) + ")"
    results = solr.search(
        f"{{!knn f=chunk_vector topK={top_k}}}{vec_str}",
        fq=fq,
        fl="chunk_id,chunk_text,product,company,doc_date,score",
        rows=top_k,
    )
    return list(results)


def show(results: list[dict], label: str):
    print(f"\n  [{label}] ({len(results)} results)")
    for doc in results:
        text = doc.get("chunk_text", "")
        if isinstance(text, list):
            text = text[0]
        score = doc.get("score", 0)
        company = doc.get("company", "?")
        if isinstance(company, list):
            company = company[0]
        product = doc.get("product", "?")
        if isinstance(product, list):
            product = product[0]
        print(f"    score={score:.4f}  [{company} / {product}]")
        print(f"      {text[:100]}...")


# ── Queries ───────────────────────────────────────────────────────────────────

queries = [
    {
        "text": "Wells Fargo refused to reverse fraudulent charge",
        "entity": "wells_fargo",
        "description": "Fraudulent charge complaint against Wells Fargo",
    },
    {
        "text": "missed payment reported to credit bureau",
        "entity": "discover",
        "description": "Credit bureau reporting error",
    },
    {
        "text": "overdraft fee student loan",
        "entity": "citibank",
        "description": "Unexpected overdraft fee on student loan",
    },
    {
        "text": "mortgage payment dispute customer service unhelpful",
        "entity": None,
        "description": "Mortgage dispute, no specific entity",
    },
]

print("\n" + "="*60)
print("OVIR inner loop — CFPB retrieval test")
print("="*60)

for q in queries:
    print(f"\nQ: {q['description']}")
    print(f"   '{q['text']}'")

    # 1. COBWEB semantic scope
    cobweb_ids = cobweb_scope(q["text"], k=15)
    print(f"   COBWEB scope : {len(cobweb_ids)} chunks → {cobweb_ids[:5]}...")

    # 2. FalkorDB graph scope (if entity provided)
    if q["entity"]:
        falkor_ids = falkor_scope(q["entity"], hops=1)
        print(f"   FalkorDB scope ({q['entity']}, 1-hop): {len(falkor_ids)} chunks")
        # Intersect: use cobweb scope, further filtered by falkor overlap
        combined_ids = list(set(cobweb_ids) | set(falkor_ids[:20]))
    else:
        combined_ids = cobweb_ids
        print(f"   FalkorDB scope: skipped (no entity)")

    # 3. Solr BM25 within scope
    bm25_results = solr_search(q["text"], combined_ids, top_k=3)
    show(bm25_results, "BM25")

    # 4. Solr KNN within scope
    knn_results = solr_knn(q["text"], combined_ids, top_k=3)
    show(knn_results, "KNN")


# ── FalkorDB inspection ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("FalkorDB graph stats")
print("="*60)

result = g.query("MATCH (e:Entity) RETURN e.label AS label, count(e) AS n ORDER BY n DESC LIMIT 10")
print("\nTop entity types:")
for row in result.result_set:
    print(f"  {row[0]:25s} {row[1]}")

result = g.query(
    "MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) "
    "RETURN e.text AS entity, count(c) AS mentions "
    "ORDER BY mentions DESC LIMIT 10"
)
print("\nMost-mentioned entities:")
for row in result.result_set:
    print(f"  {str(row[0]):30s} {row[1]} mentions")

result = g.query(
    "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
    "WHERE e.label = 'organization' "
    "RETURN c.company AS company, collect(DISTINCT e.text) AS orgs "
    "LIMIT 5"
)
print("\nSample org entities per chunk:")
for row in result.result_set:
    print(f"  company={row[0]}  orgs={row[1]}")

print("\nDone.")
