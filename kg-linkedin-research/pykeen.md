# PyKEEN

**PyKEEN** (Python Knowledge Embeddings) is an actively maintained library for training and evaluating knowledge graph embedding (KGE) models. It wraps the full training pipeline — data loading, negative sampling, optimization, evaluation, HPO, and checkpointing — behind a single `pipeline()` call, while still exposing every component individually for custom work.

Current version: 1.11.1. License: MIT. Maintained by the PyKEEN Project Team (Fraunhofer / Bonn / Elsevier / Harvard, among others) since 2019.

---

## What it does

**Problem.** A knowledge graph is a set of (head, relation, tail) triples — `(Python, required_for, Software_Engineer)`. Most graphs are incomplete: edges that should exist are missing. KGE learns dense vector representations of entities and relations such that true triples score higher than false ones. The embeddings are then used for link prediction, entity alignment, and as inputs to downstream models (e.g. the retriever in this pipeline).

**How.** Each model defines two things: a *representation* (how entity and relation vectors are structured) and an *interaction function* (how head, relation, and tail vectors are combined into a scalar score). Training minimizes a loss over positive triples vs. corrupted negatives.

PyKEEN ships 40+ interaction functions from the KGE literature, all runnable from the same `pipeline()` interface:

- **Translational** — TransE, TransH, TransR, TransD, TorusE, PairRE, TripleRE
- **Factorization** — DistMult, ComplEx, SimplE, RESCAL, TuckER, CP
- **Rotation** — RotatE (complex space), QuatE (quaternion space), TorusE
- **Neural** — ConvE, ConvKB, ERMLP, NTN, ProjE
- **GNN-based** — R-GCN, CompGCN
- **Compositional / inductive** — NodePiece (anchor-based, works on unseen entities at inference)

**RotatE** (Sun et al., ICLR 2019) is the default recommendation for graphs with directional and compositional structure — which includes career graphs and skill taxonomies. It represents each relation as a rotation in complex space: `tail = head ∘ relation`, where `∘` is element-wise complex multiplication. This captures symmetry, antisymmetry, inversion, and composition in a single model.

---

## Pipeline

The `pipeline()` function is the entry point:

```python
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

tf = TriplesFactory.from_path("train.tsv", delimiter="\t")
train, valid, test = tf.split([0.8, 0.1, 0.1])

result = pipeline(
    training=train,
    validation=valid,
    testing=test,
    model="RotatE",
    model_kwargs=dict(embedding_dim=256),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=1e-3),
    training_kwargs=dict(num_epochs=100, batch_size=1024),
    negative_sampler_kwargs=dict(num_negs_per_pos=200),
    device="mps",   # or "cuda" or "cpu"
)

result.save_to_directory("checkpoints/")
```

`PipelineResult` gives you: the trained model, loss curves, and a full rank-based evaluation (MRR, Hits@1, Hits@3, Hits@10) against the test set. Embeddings are extracted via `result.model.entity_representations[0](indices=None)`.

**HPO** is built in via Optuna:

```python
from pykeen.hpo import hpo_pipeline
hpo_pipeline(model="RotatE", dataset="FB15k237", n_trials=50)
```

**Experiment tracking** integrates with MLflow, W&B, Neptune, and TensorBoard out of the box.

---

## Deployed use cases

PyKEEN's reference list shows where KGE in general (and this library specifically) has been applied in production research:

**Biomedical / drug repurposing.** The most active domain. Santos et al. (2020) built a Clinical Knowledge Graph integrating proteomics data for clinical decision-making (bioRxiv). Himmelstein et al.'s Hetionet (eLife, 2017) used KGE for systematic drug prioritization across 11 biomedical entity types. Walsh et al. (2020) released BioKG for relational learning on biological data. Chandak et al. (2022) built a precision medicine KG. Breit et al.'s OpenBioLink benchmark (Bioinformatics, 2020) standardized evaluation for large-scale biomedical link prediction.

**Pharmaceutical networks.** Königs et al. (2022) released PharMeBINet — a heterogeneous pharmacological network — in Scientific Data. Zheng et al. (2020) introduced PharmKG for biomedical data mining.

**Commonsense and general knowledge.** Speer et al.'s ConceptNet 5.5 (AAAI 2017) is a built-in PyKEEN dataset. Ilievski et al.'s CSKG (2020) combines multiple commonsense sources into a unified graph.

**Visual relation detection.** Zhang et al. (2017) applied translational embeddings to visual scene graphs for relation detection in images.

**Cross-lingual entity alignment.** Shi et al. (2019) used KGE for precise cross-lingual entity alignment across multilingual KGs.

**Inductive link prediction.** Teru et al. (ICML 2020) and Galkin et al. (2021) extended KGE to generalize to unseen entities via subgraph reasoning and NodePiece respectively — directly relevant to the LinkedIn use case where new members and companies appear constantly.

---

## Relevance to this pipeline

PyKEEN is Step 2 in the ovir KG pipeline. It replaces DGL-KE (which has no Apple Silicon wheels) with a maintained, MPS-aware alternative that produces the same artifact: a trained RotatE model and entity embedding matrix.

The output from `result.save_to_directory()` contains `trained_model.pkl`, which you load in Step 3 (GSR) to extract entity vectors for retrieval. The rank-based evaluation metrics (MRR, Hits@10) from this step are the baseline numbers to beat before investing in GEPA optimization.

**Key decision point after training:** if MRR on the test set is below ~0.15, the graph structure is too sparse or the relation design is wrong — fix the ETL before wiring up GSR. If MRR is above ~0.25, the embeddings are capturing real signal and GSR retrieval quality should be measurably better than random subgraph selection.

---

## References

- [PyKEEN documentation](https://pykeen.readthedocs.io/en/stable/)
- Sun et al. (2019). "RotatE: Knowledge Graph Embeddings by relational rotation in complex space." ICLR 2019.
- Ali et al. (2020). "Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models Under a Unified Framework." arXiv:2006.13365.
- Galkin et al. (2021). "NodePiece: Compositional and Parameter-Efficient Representations of Large Knowledge Graphs." arXiv:2106.12144.
- Santos et al. (2020). "Clinical Knowledge Graph Integrates Proteomics Data into Clinical Decision-Making." bioRxiv.
- Chandak et al. (2022). "Building a knowledge graph to enable precision medicine." bioRxiv.
