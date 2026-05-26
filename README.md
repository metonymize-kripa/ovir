# OVIR — Offline Verified Inference Retrieval

Two-loop retrieval architecture. Expensive outer loop preprocesses the corpus once; cheap inner loop serves queries.

**Outer loop** (offline, runs per corpus version): GLiNER entity extraction → FalkorDB graph build → COBWEB embedding index → Solr schema + indexing → DSPy trace generation via `qwen3.6:35b-mlx`.

**Inner loop** (per query): COBWEB scope → FalkorDB graph traversal → Solr `fq` pre-filter → BM25 or KNN search → `gemma4:e4b` reads only scoped candidates → sufficiency check → escalate or return.

All models run locally via Ollama. No cloud dependencies.

## Models

| Role | Model | When |
|---|---|---|
| Inner loop LM | `gemma4:e4b` | Every query |
| Outer loop LM | `qwen3.6:35b-mlx` | Corpus prep, DSPy trace generation |
| Embeddings | `nomic-embed-text` | COBWEB index, Solr KNN |

Both LMs and the embedding model are already in `ollama ls`. No pulls needed.

## Playgrounds

Each subfolder is a self-contained `uv` environment. Run scripts with `uv run <script>.py`.

### `falkordb-playground/`
Entity-chunk graph. Cypher queries, variable-length traversal, OVIR scope pattern.
```bash
# FalkorDB already running (existing container)
uv run 01_basics.py
uv run 02_traversal.py
uv run 03_ovir_pattern.py
uv run 04_index_and_performance.py
```

### `gliner-playground/`
Zero-shot NER with domain-specific label sets. Produces entity annotations and co-occurrence pairs for FalkorDB ingestion.
```bash
uv run 01_basics.py        # extraction, threshold comparison, latency
uv run 02_batch_corpus.py  # batch extraction → annotated_corpus.jsonl + entity_registry.json
```

### `cobweb-playground/`
Semantic routing via `CobwebRetriever` over nomic-embed-text embeddings. Produces chunk ID scope for Solr `fq`.
```bash
uv run 01_basics.py        # build retriever, query, cohesion check
uv run 02_ovir_routing.py  # offline build + pickle, online query → scope IDs
```

### `solr-playground/`
Keyword search, scoped `fq` queries, faceting, atomic updates. Solr 10.0.0.
```bash
docker compose up -d
uv run 01_schema_and_index.py
uv run 02_search.py
uv run 03_updates_and_management.py
```

### `solr-nn-playground/`
Dense vector search via `DenseVectorField` (768 dims, cosine). Scoped KNN and hybrid BM25+KNN. Runs on `:8984` to avoid conflict with `solr-playground`. Solr 10.0.0.
```bash
docker compose up -d
uv run 01_vector_index.py
```

### `dspy-playground/`
DSPy v3 signatures, modules, and BootstrapFewShot optimization. Teacher (`qwen3.6:35b-mlx`) generates traces; student (`gemma4:e4b`) is compiled with them baked in.
```bash
uv run 01_signatures.py  # inline + class-based signatures, ChainOfThought, inspect_history
uv run 02_modules.py     # OVIRQueryModule: extract → route → assess sufficiency
uv run 03_optimizer.py   # teacher/student split, compile → compiled_router.json
```

### `qwen-playground/`
Inner and outer loop via Ollama's OpenAI-compatible endpoint. Entity extraction, thinking mode, JSON output, latency benchmark.
```bash
uv run 01_ollama_basics.py
```

### `gensyn-playground/`
Gensyn REE verifiable inference. Run/verify subprocess wrapper and receipt analysis stub.
```bash
# Requires gensyn-ree Docker image
uv run 01_run_and_receipt.py
uv run 02_receipt_analysis.py
```

## Stack dependencies

| Component | Purpose | Port |
|---|---|---|
| FalkorDB | Entity-chunk graph, scope by traversal | 6379 / 3000 |
| Solr 10 (keyword) | BM25 search over scoped candidates | 8983 |
| Solr 10 (vectors) | KNN search, hybrid scoring | 8984 |
| Ollama | LM inference + embeddings (local) | 11434 |

FalkorDB and Ollama run as persistent containers. Solr instances are started per-playground via `docker compose`.
