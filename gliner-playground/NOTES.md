# GLiNER Playground

GLiNER v0.2.26. Zero-shot NER — no fine-tuning, no domain-specific training.

## Setup

```bash
uv run 01_basics.py       # downloads model on first run (~500MB)
uv run 02_batch_corpus.py
```

## Scripts

**01_basics.py** — `from_pretrained`, `predict_entities`, domain-specific label sets, threshold comparison, latency benchmark.

**02_batch_corpus.py** — `batch_predict_entities`, annotated chunk format for FalkorDB, entity deduplication via normalized IDs, co-occurrence pairs as graph edge seeds, JSONL output.

## Key concepts

**Zero-shot means the labels are the API.** Pass `["party", "liability_cap", "obligation"]` and GLiNER extracts those types from any text, with no training. Change labels per corpus type without re-training.

**Threshold is the recall/precision dial.** 0.4–0.5 for broad entity coverage (more graph nodes, more noise). 0.7+ for high-confidence entities only. OVIR recommendation: run at 0.45, store confidence on the MENTIONS edge, filter at query time.

**`batch_predict_entities` is deprecated — use `GLiNER.inference`.** The `batch_predict_entities` method still works but emits a `FutureWarning`. Replace with `model.inference(texts, entity_types=labels, threshold=t)` for forward compatibility. The scripts use the old API for now; update before the next major GLiNER release.

**Normalize entity IDs.** `"ACME Corp"`, `"Acme Corp"`, `"ACME"` might all refer to the same entity. Normalize to a stable ID (`acme_corp`) before inserting into FalkorDB. The registry in `02_batch_corpus.py` handles this.

**Co-occurrence → graph edges.** Two entities in the same chunk are likely related. Store co-occurrence counts as weak RELATED_TO edges in FalkorDB. Supplement with explicit relationship extraction for stronger signals.

## Model options

| Model | Size | Notes |
|---|---|---|
| `urchade/gliner_small-v2.1` | ~250MB | Fastest, lowest accuracy |
| `urchade/gliner_medium-v2.1` | ~500MB | Good balance — use this |
| `urchade/gliner_large-v2.1` | ~1GB | Best accuracy, 2× slower |

All run on CPU. GPU (MPS/CUDA) gives ~5× speedup for large corpora.
