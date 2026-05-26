# Pipeline

Full OVIR outer loop on real CFPB complaint data (`data/cfpb_corpus.jsonl`, 5000 chunks).

## Prerequisites

All three services must be running:

```bash
# FalkorDB — already running (persistent container)
# Check: redis-cli -p 6379 ping

# Solr 10 — start if not running
docker compose -f ../solr-playground/docker-compose.yml up -d
# Wait ~10s, verify: curl http://localhost:8983/solr/ovir_corpus/admin/ping

# Ollama — already running
# Check: curl http://localhost:11434/api/tags | grep nomic
```

## Scripts

**01_outer_loop.py** — Outer loop pipeline. Loads N chunks, runs GLiNER, embeds with nomic-embed-text, builds FalkorDB graph + COBWEB retriever + Solr index.

```bash
uv run 01_outer_loop.py              # 100 chunks (default)
uv run 01_outer_loop.py --chunks 500 # larger run
```

**02_query.py** — Inner loop test. Loads COBWEB retriever from pickle, runs 4 test queries using COBWEB scope + FalkorDB traversal + Solr BM25/KNN.

```bash
uv run 02_query.py
```

## What gets built

| Artifact | Location | Description |
|---|---|---|
| FalkorDB graph | `cfpb_corpus` (in-memory) | CHUNK + ENTITY nodes, MENTIONS edges |
| COBWEB retriever | `cfpb_retriever.pkl` | Pickled retriever for query-time scope |
| Solr index | `:8983/solr/ovir_corpus` | Text + vector fields, all chunks |

## CFPB corpus

Synthetic CFPB consumer complaint records. Fields: `id`, `text`, `metadata.product`, `metadata.company`, `metadata.date`.

GLiNER labels used: `organization`, `person`, `monetary_amount`, `date`, `financial_product`, `complaint_type`.

## Timing expectations (100 chunks, M-series MacBook)

| Step | Expected |
|---|---|
| GLiNER | ~4–6s (CPU) |
| Embed (nomic) | ~2–3s |
| FalkorDB | ~5–10s (per-chunk queries) |
| COBWEB build | <1s |
| Solr index | ~1–2s |
| **Total** | **~15–25s** |

For 500+ chunks, use `ray-playground/03_ovir_ray_pipeline.py` as the pattern for parallelising the GLiNER + embed steps.
