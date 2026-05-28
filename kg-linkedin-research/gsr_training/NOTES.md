# GSR Training Pipeline — LinkedIn KG

## What this does

Adapts the GSR retriever (EMNLP 2024 — "Less is More") to the LinkedIn+O*NET knowledge graph. GSR trains a small T5 model as a seq2seq subgraph retriever: given a question, predict the sequence of relation tokens that form the gold path through the KG. A separate reader LLM then answers from the retrieved subgraph.

## Files

| File | Purpose |
|------|---------|
| `build_tokenizer.py` | Augments T5 tokenizer with 9 LinkedIn relation tokens + GSR prefix tokens |
| `prepare_data.py` | Converts `data/qa_pairs.jsonl` → stratified HF datasets |
| `train.py` | Fine-tunes T5 (seq2seq) on the LinkedIn dataset |
| `eval.py` | Inference + EM and subgraph-hit evaluation |
| `smoke_test.py` | 1-epoch end-to-end check (~2 min on CPU) |
| `run_pipeline.sh` | Full pipeline: tokenizer → data → train → eval |

## Quick start

```bash
# From repo root (ovir/)

# Smoke test (1 epoch, ~2 min):
uv run kg-linkedin-research/gsr_training/smoke_test.py

# Full training (20 epochs, t5-small):
bash kg-linkedin-research/gsr_training/run_pipeline.sh

# Or step by step:
uv run kg-linkedin-research/gsr_training/build_tokenizer.py
uv run kg-linkedin-research/gsr_training/prepare_data.py
uv run kg-linkedin-research/gsr_training/train.py --model_id t5-small --num_epochs 20
uv run kg-linkedin-research/gsr_training/eval.py \
    --model_path kg-linkedin-research/gsr_training/trained_models/linkedin-gsr-t5-small \
    --kg data/kg --num_beams 5
```

## Data

**KG** (`data/kg/`): 253,794 entities, 9 relation types, 753,513 triples.

| Relation | Count |
|----------|-------|
| requires_skill | 203,167 |
| specialises_in | 169,182 |
| in_industry | 158,355 |
| posted_job | 122,131 |
| uses_technology | 32,570 |
| operates_in | 24,375 |
| has_skill | 18,478 |
| related_to_occupation | 18,460 |
| requires_knowledge | 6,795 |

**QA pairs** (`data/qa_pairs.jsonl`): 1,114 records, split 80/10/10 stratified by type.

| Type | Total | Train | Val | Test |
|------|-------|-------|-----|------|
| career_path | 500 | 400 | 50 | 50 |
| company_target | 500 | 400 | 50 | 50 |
| skill_gap | 114 | 92 | 11 | 11 |
| **Total** | **1,114** | **892** | **111** | **111** |

## Relation encoding

Each LinkedIn relation maps to a single GSR special token:

```
has_skill           → <rel_0_has_skill>
posted_job          → <rel_0_posted_job>
requires_skill      → <rel_0_requires_skill>
uses_technology     → <rel_0_uses_technology>
requires_knowledge  → <rel_0_requires_knowledge>
related_to_occupation → <rel_0_related_to_occupation>
```

Training input/target format (mirrors GSR):
```
input:  "[reasoning] what skills distinguish a data scientist from a data analyst?"
target: "<rel_0_has_skill>"
```

## Key finding: 4 unique output patterns

The model is essentially a **4-class classifier** wrapped in seq2seq:

| Pattern | Question type |
|---------|--------------|
| `<rel_0_has_skill>` | skill_gap |
| `<rel_0_posted_job><rel_0_requires_skill>` | company_target |
| `<rel_0_uses_technology><rel_0_related_to_occupation>` | career_path (no knowledge edges) |
| `<rel_0_uses_technology><rel_0_requires_knowledge><rel_0_related_to_occupation>` | career_path (full) |

T5-small should reach near-100% EM within 10–15 epochs. The interesting signal is whether the model correctly routes question phrasing to the right pattern (skill-gap questions vs. company-search questions).

## Traversal strategies in eval

The KG uses three distinct traversal patterns depending on question type — `eval.py` handles all three via `_walk_chain()`:

- **`skill_gap`**: `occ → has_skill → skill` (1-hop)
- **`career_path`**: `occ → {uses_technology, requires_knowledge, related_to_occupation}` (parallel fan-out from occupation)
- **`company_target`**: `company → posted_job → job → requires_skill → skill` (sequential 2-hop)

Oracle ceiling (gold chain, all question types): **100% subgraph hit rate**.

## PyKEEN RotatE results

The trained RotatE model at `data/kg_ckpt/` produces usable embeddings:

| Metric | Value |
|--------|-------|
| MRR (overall) | 0.186 |
| Hits@10 (overall) | 0.29 |
| MRR (tail prediction) | 0.33 |
| Hits@10 (tail) | 0.49 |

Loss plateaued ~epoch 30; 100 epochs was overkill. MRR > 0.15 confirms real signal — the PLAN.md threshold for proceeding to GSR. These embeddings are ready to use as input features for downstream tasks.

## Can this work?

**Yes.** The signal is there:

1. KG is clean and densely connected (753K triples, 9 meaningful relation types).
2. QA pairs are well-formed and all 1,114 are correctly encodable with the relation token vocabulary.
3. The task is learnable — 4 unique targets, each tied to distinguishable question phrasing.
4. Oracle ceiling is 100% subgraph hit rate, meaning the KG can support the ground truth for every test record.
5. RotatE embeddings confirm real structural signal (MRR 0.33 on tail prediction).

**Realistic trained-model expectations (t5-small, 20 epochs):**
- EM@1: >85% (routing question type is the main challenge; phrasing within type varies)
- Subgraph hit@5: >90%
- Failure mode to watch: `career_path` questions where the occupation has no `requires_knowledge` edges — those produce a 2-token rather than 3-token target

## Next steps (from PLAN.md)

1. Train and verify EM ≥ 85% — if yes, proceed to DSPy GEPA optimization
2. Wire the trained GSR into a `SubgraphRetriever` DSPy module
3. Textualize retrieved subgraphs for the reader LLM
4. Run GEPA with Claude Opus as the reflection LLM
5. Evaluate end-to-end pipeline on the 111-record test split
