# Solr NN Playground

Solr 9 DenseVectorField (HNSW) + nomic-embed-text via Ollama (768 dims). Runs on :8984 to avoid
conflict with the keyword Solr playground on :8983.

## Setup

```bash
docker compose up -d   # Solr on :8984
uv run 01_vector_index.py
```

## Script

**01_vector_index.py** — Add `DenseVectorField` (768 dims) to schema, embed corpus with nomic-embed-text via Ollama, index vectors, `{!knn}` search, scoped KNN (pre-filter with `fq`), hybrid BM25+KNN scoring.

## Key concepts

**DenseVectorField** stores a fixed-dimension float array and builds an HNSW graph index over it. At query time, `{!knn f=chunk_vector topK=N}[v1,v2,...]` finds the N nearest neighbors by cosine similarity.

**Scoped KNN is the OVIR pattern.** `fq` (from FalkorDB + COBWEB) prunes the candidate set before HNSW search. Smaller candidate set → faster KNN → lower latency. This is the same graph-scoped search as keyword Solr, applied to vectors.

**Hybrid scoring.** Solr doesn't natively combine BM25 and KNN scores. Options: (a) run both in Python and merge, (b) use Solr's `edismax` + `rrf` (Reciprocal Rank Fusion) for a lightweight blend, (c) wait for Solr's experimental hybrid scorer. The Python merge in `01_vector_index.py` is the simplest start.

**Embedding model.** `nomic-embed-text` (768 dims) runs via Ollama — already in your `ollama ls`, no extra Python package. Calls go to `POST http://localhost:11434/api/embed` with batch input. Better quality than all-MiniLM-L6-v2 (384 dims) due to higher dimensionality and nomic's training data. Dimension change requires recreating the DenseVectorField schema.

## OVIR wiring

Dense vectors are most useful for entity descriptions and document summaries — not raw chunks. Entity description embeddings let FalkorDB neighborhood queries be augmented with "entities semantically similar to X." Raw chunk vectors help when keyword search misses paraphrase matches. Both are optional — start with BM25+graph, add vectors if recall gaps appear in eval.
