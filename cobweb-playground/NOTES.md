# COBWEB Playground

`cobweb-language-embedding` (Teachable-AI-Lab) with nomic-embed-text embeddings via Ollama.

Repo: https://github.com/Teachable-AI-Lab/cobweb-language-embedding
Paper: CobwebTM (arXiv:2604.14489, 2026)

## Setup

```bash
# nomic-embed-text already in ollama ls — no pull needed
# First run installs cobweb-language-embedding from GitHub
uv run 01_basics.py
uv run 02_ovir_routing.py
```

## Scripts

**01_basics.py** — Embed corpus with nomic-embed-text, build `CobwebRetriever`, run semantic queries, cohesion check (results land in the right topic cluster).

**02_ovir_routing.py** — Full offline→online pipeline: embed corpus → build retriever → pickle → load → embed query → `retriever.query(query_emb, k)` → chunk IDs → Solr `fq`. Scope size vs. `k` tradeoff measurement.

## Key concepts

**CobwebRetriever takes embeddings you provide.** No internal embedding model. Pass `corpus_embeddings` from nomic-embed-text (or any model). The retriever builds a COBWEB tree over the embedding space and returns chunks from the concept node most similar to the query embedding.

**`k` is the scope dial.** `retriever.query(query_emb, k=3)` is OVIR's narrow-scope start. `k=6` is wider scope / higher recall. Same role as FalkorDB's hop depth — start small, expand if confidence is low.

**COBWEB vs. FalkorDB scoping.** They're complementary:
- FalkorDB: entity relationship graph → "chunks mentioning entities near X"
- COBWEB: embedding space clusters → "chunks semantically similar to this query"
- Both produce chunk ID sets; Solr ANDs them for the tightest scope

**Persist with pickle.** Build once per corpus version. `pickle.dump` the retriever object. At query time: `pickle.load` → `retriever.query()`. No rebuild needed until the corpus changes materially.

## Embedding model

nomic-embed-text (274MB, already in `ollama ls`). 768 dims. Called via:
```python
requests.post("http://localhost:11434/api/embed",
              json={"model": "nomic-embed-text", "input": [text1, text2, ...]})
```
Returns `{"embeddings": [[...], [...]]}`. L2-normalize before passing to CobwebRetriever.

## cobweb-language-embedding extras

The library also provides:
- `CobwebWrapper` — base tree with visualization support (needs Graphviz)
- BERTopic-compatible topic cluster wrappers (`topic_modeling.py`)
- PCA+ICA whitening for embedding preprocessing (`preprocess_embedding.py`)

Install extras: `pip install "cobweb-language-embedding[retrieval]"` or `".[topic-modeling]"`.
Not needed for OVIR's core routing use case.
