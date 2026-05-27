# Repo Notes: KG Completion → LinkedIn Economic Graph

**Goal:** Run a KGC / KGE experiment on LinkedIn Economic Graph data (1.2B members, 69M companies, 41K skills). GLR paper (MDPI 2025) is the anchoring method; KG-FIT is the NeurIPS'24 baseline to beat; PBG is the scale layer.

---

## Paper in scope: GLR (MDPI Applied Sciences, June 2025)

**Title:** GLR: Graph Chain-of-Thought with LoRA Fine-Tuning and Confidence Ranking for Knowledge Graph Completion  
**DOI:** 10.3390/app15137282

**What it does.** Three components bolted together:
1. **Graph-CoT** — feeds local subgraph structure as context into an LLM prompt, forcing step-wise multi-hop reasoning instead of a direct entity-ranking pass.
2. **LoRA fine-tuning** — parameter-efficient adaptation of the LLM to the KGC task; keeps compute tractable.
3. **P(True) confidence ranking** — asks the LLM to self-assess the probability that a candidate answer is correct; uses this as a re-ranking signal.

**Reported numbers (FB15K-237):**  
- GLR MRR: **0.507**  
- Best prior LLM-based baseline (DIFT + CoLE): MRR **0.439** → +6.8%

### Is this SOTA?

**No, not globally.** Caveats:

- Published in *Applied Sciences* (broad multidisciplinary journal, low barrier vs. NeurIPS/ICLR/KDD). This doesn't invalidate the numbers, but the community hasn't stress-tested them.
- The comparison set is scoped to *LLM-based KGC* methods. Pure KGE methods (RotatE, TuckER) and hybrid methods like KG-FIT likely report different MRR figures on FB15K-237. KG-FIT (NeurIPS'24) should be checked directly — it upgrades the same KGE backbones with LLM-guided hierarchy and routinely beats vanilla RotatE/TuckER.
- On WN18RR, GLR's advantage vs. KGE baselines is smaller; WordNet-style hierarchies are exactly what KG-FIT's clustering step was designed for.
- For a LinkedIn-scale graph (billions of edges), neither GLR nor KG-FIT has been validated. Scale is a separate problem.

**Bottom line:** GLR is a credible incremental step for LLM-as-reasoner KGC on standard benchmarks. It's not the global SOTA. The more interesting question is whether Graph-CoT + P(True) transfers to an economic graph where the "facts" are career transitions, skill endorsements, and company-role edges — none of which appear in FB15K-237 or WN18RR.

---

## Repo 1: KG-FIT

**URL:** https://github.com/pat-jj/KG-FIT  
**Paper:** NeurIPS 2024  
**Stars:** 130 | **Forks:** 13  
**Stack:** Python 66%, Jupyter 30%, Shell 3%

### What it does

KG-FIT wraps existing KGE models (RotatE, TransE, DistMult, TuckER, ConvE, HAKE, pRotatE) with two LLM-driven preprocessing steps:

1. **Seed hierarchy** (`cluster.py`) — clusters entity embeddings into a tree structure using entity name/description text.
2. **LLM-Guided Hierarchy Refinement (LHR)** (`llm_refine.py`) — sends GPT-4o calls to audit and correct the cluster assignments.

The resulting hierarchy injects a text-alignment constraint and a parent-child proximity constraint into the KGE training loss. The base KGE model trains on top of this enriched graph.

### Key files

| File | Role |
|---|---|
| `code/precompute/cluster.py` | Builds seed hierarchy; run twice (before and after LHR) |
| `code/precompute/llm_refine/llm_refine.py` | GPT-4o refinement pass; reads `openai_api.key` |
| `code/model_common.py` | KG-FIT for all models except TuckER/ConvE; text embeddings frozen |
| `code/model_flex.py` | Same but text embeddings unfrozen ("on fire") — likely the variant to beat |
| `code/model_tucker_conve.py` | TuckER + ConvE variant |
| `runs_rotate.sh` etc. | Shell scripts to reproduce paper runs per model |

### What to fork / change for LinkedIn Economic Graph

**Data format.** KG-FIT expects TSV triples `(head, relation, tail)` in the standard KGE format. LinkedIn Economic Graph data (if accessed via the Research Institute partnership) will come as CSVs of member-skill edges, member-company edges, company-location edges, etc. You need an ETL step:

```
(member_id, has_skill, skill_name)
(member_id, works_at, company_id)
(company_id, operates_in, industry)
(skill_name, requires, skill_name)   # skill co-occurrence
```

Map these to the triples format and split train/val/test by time (e.g., edges before 2023 = train, 2023 = val, 2024 = test) rather than random split, since career graphs have temporal structure.

**Entity descriptions.** `cluster.py` uses entity names + optional text descriptions to build the seed hierarchy. LinkedIn entities have rich natural-language descriptions (company about-pages, skill definitions, job titles). Feed these in — this is where KG-FIT's LLM enrichment should work well, since the Economic Graph's ontology is domain-specific and the hierarchy (e.g., "Software Engineer → Senior SWE → Staff SWE") is meaningful.

**LHR cost.** `llm_refine.py` calls GPT-4o for every cluster boundary decision. On FB15K-237 (~15K entities) this is manageable. On a LinkedIn-scale subgraph (even a 1M-entity sample) this becomes expensive. Options:
- Use a smaller LLM (GPT-4o-mini, or a locally hosted Llama 3 / Mistral) by swapping the model string in `llm_refine.py`.
- Run LHR only on the top-k highest-degree entities (companies, canonical skills) and accept seed hierarchy for the long tail.

**Loss function.** The hierarchy constraints are added as regularization terms in `model_common.py`. If you're modeling career *transitions* (directed, temporal), the RotatE relational structure (rotation in complex space) is a better fit than TransE. Start with `runs_rotate.sh`.

**Evaluation.** Standard KGC eval is link prediction (MRR, Hits@1/3/10). For Economic Graph, more meaningful eval might be: given a member's skills at time T, predict the company they join at T+1. Rewrite the eval loop in `code/` to frame it this way.

---

## Repo 2: PyTorch-BigGraph (PBG)

**URL:** https://github.com/facebookresearch/PyTorch-BigGraph  
**Paper:** SysML 2019  
**Status:** ⚠️ **ARCHIVED March 2024. Read-only. No new commits.**  
**Stars:** 3.5k | **Forks:** 454  
**Stack:** Python 97%, C++ 3%

### What it does

PBG trains graph embeddings at billion-entity scale via:
- **Graph partitioning** — splits entities into buckets so the full embedding matrix never fits in memory at once.
- **Multi-threaded CPU training** — 40+ cores, >1M edges/sec/machine.
- **Distributed training** — multiple machines via torch.distributed + shared filesystem checkpoint.
- **GPU mode** (experimental) — requires `PBG_INSTALL_CPP=1`; needs larger batch sizes (~100k).

Supported models: TransE, RESCAL, DistMult, ComplEx.

Pre-trained Wikidata embeddings are available: 78M entities, 4,131 relations, dim=200, ~36GiB compressed.

### Limitations for this use case

PBG is the right tool for the **scale problem** — not the accuracy problem. It does not support:
- LLM-derived text embeddings as initialization
- Hierarchical constraints (what KG-FIT adds)
- Confidence-ranked predictions (what GLR adds)

The archived status is a real issue. Dependencies (PyTorch ≥ 1.0, Python ≥ 3.6) are pinned to 2019-era versions. Running on modern hardware will require patching `setup.cfg` and possibly `torchbiggraph/` internals.

### When to use it

Use PBG as the **first pass** on raw Economic Graph data:
1. Train TransE or ComplEx embeddings on the full graph.
2. Export entity embeddings via `torchbiggraph_export_to_tsv`.
3. Use these as warm-start initializations for KG-FIT's `cluster.py` step — instead of random init, you hand it PBG embeddings that already encode structural proximity.

This sidesteps PBG's lack of reasoning features while leveraging its scale capacity.

**Alternative to consider:** [GraphVite](https://github.com/DeepGraphLearning/graphvite) (GPU-accelerated, more actively maintained as of 2023) or [DGL-KE](https://github.com/awslabs/dgl-ke) (supports RotatE at scale, AWS-backed). Both are drop-in replacements for the scale layer.

### Installation note (current)

```bash
# PBG requires Python 3.6–3.9 and PyTorch 1.x
# Create isolated env:
conda create -n pbg python=3.8
conda activate pbg
pip install torch==1.13.1
git clone https://github.com/facebookresearch/PyTorch-BigGraph.git
cd PyTorch-BigGraph
pip install .
```

GPU mode (optional):
```bash
PBG_INSTALL_CPP=1 pip install .
```

---

## Proposed Stack for LinkedIn Economic Graph Experiment

```
LinkedIn EG data (TSV triples)
        ↓
[PBG or DGL-KE] — billion-scale embedding warmup
        ↓
  warm-start embeddings
        ↓
[KG-FIT] — hierarchy clustering + LHR + KGE fine-tuning
        ↓
  fine-tuned entity embeddings + KGE model
        ↓
[GLR-style eval] — Graph-CoT + P(True) re-ranking on a held-out test set
```

**Open questions before starting:**
1. What slice of the Economic Graph is accessible? (Partnership program vs. public datasets)
2. Is the entity set stable? (Members churn; skill taxonomy evolves. This affects whether a static KGE model makes sense.)
3. What's the evaluation target? (Link prediction on skills? Company prediction? Salary inference?)

---

## References

- Guo, Yan (2025). "GLR: Graph Chain-of-Thought with LoRA Fine-Tuning and Confidence Ranking for Knowledge Graph Completion." *Applied Sciences* 15(13):7282. https://doi.org/10.3390/app15137282
- Jiang et al. (2024). "KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge." *NeurIPS 2024*. https://github.com/pat-jj/KG-FIT
- Lerer et al. (2019). "PyTorch-BigGraph: A Large-scale Graph Embedding System." *SysML 2019*. https://github.com/facebookresearch/PyTorch-BigGraph
- LinkedIn Economic Graph. https://economicgraph.linkedin.com/
