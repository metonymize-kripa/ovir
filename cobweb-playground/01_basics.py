"""
01_basics.py — CobwebRetriever with nomic-embed-text embeddings.

cobweb-language-embedding (Teachable-AI-Lab) provides CobwebRetriever:
a COBWEB tree built over embedding space for semantic retrieval.
Instead of hand-crafted feature dicts, each chunk is represented by
its nomic-embed-text embedding (768 dims via Ollama).

The retriever does hierarchical probabilistic clustering over the embedding
space — finding the concept node that best matches a query embedding, then
returning the chunks in that concept's neighborhood.

Prereq: nomic-embed-text already in ollama ls
Run:    uv run 01_basics.py  (first run installs cobweb-language-embedding from GitHub)
"""

import numpy as np
import requests
from cobweb_language_embedding import CobwebRetriever

OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


# ── Shared embedding helper ────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """Batch embed via Ollama nomic-embed-text. Returns (N, 768) float32, L2-normalized."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# ── Corpus ────────────────────────────────────────────────────────────────────
corpus = [
    # Contract cluster
    "ACME Corp agrees to pay Globex two million dollars annually under this MSA.",
    "Liability is capped at twelve months of fees paid by ACME Corp under Section 8.",
    "Either party may terminate with thirty days written notice.",
    "Globex will deliver the data pipeline by June 30, 2024.",
    # Org structure cluster
    "Globex is a wholly-owned subsidiary of Initech Holdings incorporated in Delaware.",
    "CEO Bill Lumbergh oversees all Initech Holdings subsidiaries.",
    "Initech Holdings reported eight hundred million dollars in revenue for FY2023.",
    # Tech cluster
    "AuthService handles user authentication using JWT tokens and OAuth2.",
    "TokenService manages token expiry and caches sessions in Redis.",
    "Alice owns AuthService and TokenService; Bob owns UserDB.",
    "APIGateway routes all traffic through AuthService before processing.",
]

print("=== Embedding corpus with nomic-embed-text ===")
corpus_embeddings = embed(corpus)
print(f"  {len(corpus)} chunks embedded  shape={corpus_embeddings.shape}")


# ── 1. Build CobwebRetriever ──────────────────────────────────────────────────
# CobwebRetriever takes the corpus strings and their pre-computed embeddings.
# It builds a COBWEB tree over the embedding space incrementally.

print("\n=== Building CobwebRetriever ===")
retriever = CobwebRetriever(corpus=corpus, corpus_embeddings=corpus_embeddings)
print("  Retriever built.")


# ── 2. Query: embed + retrieve ─────────────────────────────────────────────────
# At query time: embed the query, ask the retriever for top-k matches.
# The retriever walks the COBWEB tree to find the concept node most similar
# to the query embedding, then returns chunks from that neighborhood.

queries = [
    ("What is the liability cap?",             3),
    ("Who owns the authentication service?",    3),
    ("What is Initech Holdings' revenue?",      3),
    ("How is the contract terminated?",         2),
]

print("\n=== Queries ===")
for query_text, k in queries:
    query_emb = embed([query_text])[0]
    results = retriever.query(query_emb, k=k)  # returns list of strings
    print(f"\n  Q: {query_text}")
    for i, chunk in enumerate(results):
        print(f"    [{i+1}]  {chunk[:70]}...")


# ── 3. Retrieval quality: expected cluster cohesion ───────────────────────────
# Good retrieval: results for a contract question should come from contract chunks.
# Check that the top result matches the expected "cluster".

print("\n=== Cluster cohesion check ===")
contract_chunks = set(corpus[:4])
tech_chunks = set(corpus[7:])

for query_text, expected_chunks in [
    ("payment fees annual",         contract_chunks),
    ("JWT authentication service",  tech_chunks),
]:
    query_emb = embed([query_text])[0]
    results = retriever.query(query_emb, k=3)  # returns list of strings
    top_chunks = set(results)
    hits = len(top_chunks & expected_chunks)
    print(f"  '{query_text}': {hits}/3 results from expected cluster")

print("\nDone. Move on to 02_ovir_routing.py")
