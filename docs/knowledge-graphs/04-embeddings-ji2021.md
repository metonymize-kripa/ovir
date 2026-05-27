# A Survey on Knowledge Graphs: Representation, Acquisition, and Applications

**Authors:** Shaoxiong Ji (Aalto University), Shirui Pan (Monash University), Erik Cambria (NTU Singapore), Pekka Marttinen (Aalto University), Philip S. Yu (University of Illinois at Chicago)  
**Journal:** *IEEE Transactions on Neural Networks and Learning Systems* (TNNLS)  
**Year:** 2021 (published online April 1, 2021; accepted March 30, 2021)  
**DOI:** [10.1109/TNNLS.2021.3070843](https://doi.org/10.1109/TNNLS.2021.3070843)  
**arXiv:** [2002.00388](https://arxiv.org/abs/2002.00388)

---

## What the paper is

Where Hogan et al. [2021] covers the entire KG stack, Ji et al. [2021] goes deep on the machine learning side: representation learning, knowledge acquisition (completion, entity discovery, relation extraction), temporal KGs, and downstream applications. Its main structural contribution is a **four-view categorization** of KG embedding methods — the most systematic taxonomy of the embedding literature available. Peng et al. [2023] leans on this paper for its embedding coverage; the field cites it alongside Hogan and Wang et al. [2017] as the standard embedding survey. ~4,000+ citations as of 2025.

---

## Abstract (verbatim)

> "Human knowledge provides a formal understanding of the world. Knowledge graphs that represent structural relations between entities have become an increasingly popular research direction towards cognition and human-level intelligence. In this survey, we provide a comprehensive review of knowledge graph covering overall research topics about 1) knowledge graph representation learning, 2) knowledge acquisition and completion, 3) temporal knowledge graph, and 4) knowledge-aware applications, and summarize recent breakthroughs and perspective directions to facilitate future research. We propose a full-view categorization and new taxonomies on these topics. Knowledge graph embedding is organized from four aspects of representation space, scoring function, encoding models, and auxiliary information."

---

## Brief history traced (§II-A)

The paper opens with a concise historical sketch:

- **1956**: Richens proposes semantic networks (graphical knowledge representation)
- **1959**: General Problem Solver (Newell & Simon)
- **~1970s**: Expert systems (MYCIN with ~600 rules); recognition of knowledge acquisition bottleneck
- **1984**: Cyc project begins (Lenat); goal: assemble broad commonsense knowledge; CycL representation language
- **1988**: Stokman and de Vries propose graph-based knowledge structures
- **~1985–1995**: WordNet (Miller et al.); RDF/W3C standards (1999)
- **2007–2008**: DBpedia (Auer et al.), YAGO (Suchanek et al.), Freebase (Bollacker et al.)
- **2011**: IBM Watson wins Jeopardy
- **2012**: **Google Knowledge Graph** — "the term of knowledge graph gained wide popularity" [paper's framing]
- **2013**: TransE (Bordes et al.) — KG embedding as translation; opens the modern embedding research wave

---

## Definition

**Informal (§I):**
> "A knowledge graph is a structured representation of facts, consisting of entities, relationships, and semantic descriptions. Entities can be real-world objects and abstract concepts, relationships represent the relation between entities, and semantic descriptions of entities, and their relationships contain types and properties with a well-defined meaning."

**Formal (§II-B):**
> "We define a knowledge graph as G = {E, R, F}, where E, R and F are sets of entities, relations and facts, respectively. A fact is denoted as a triple (h, r, t) ∈ F."

**On synonymy:**
> "The term of knowledge graph is synonymous with knowledge base with a minor difference. A knowledge graph can be viewed as a graph when considering its graph structure. When it involves formal semantics, it can be taken as a knowledge base for interpretation and inference over facts. For simplicity and following the trend of the research community, this paper uses the terms knowledge graph and knowledge base interchangeably."

---

## Core taxonomy: the four-view KGE framework (§III)

This is the paper's central structural contribution. Every KG embedding model can be characterized along four axes:

### Axis 1: Representation Space

Where entities and relations live:

| Space | Description | Models |
|-------|-------------|--------|
| **Point-wise (real vectors)** | Standard Euclidean ℝ^d | TransE, TransH, TransR, DistMult, RESCAL |
| **Complex vectors** | ℂ^d; handles asymmetric relations | ComplEx, RotatE, QuatE |
| **Gaussian distributions** | Entities as density functions, not points | KG2E, TransG |
| **Manifold / Lie group / Hyperbolic** | Non-Euclidean geometry; better for hierarchies | MuRP (Poincaré ball), RotH, AttH |

### Axis 2: Scoring Function

How to measure plausibility of a triple:

**Distance-based (translational):** `h + r ≈ t`
- TransE: `−||h + r − t||` — simple and effective; fails on N-to-1 / 1-to-N
- TransH: projects h, t onto a relation-specific hyperplane before translating
- TransR: maps entities into relation-specific spaces via projection matrix M_r
- TransD: entity-and-relation-specific projection; more parameters
- TranSparse: sparse projection to handle heterogeneous relation degrees
- RotatE: relation r as rotation in complex space; `h ∘ r = t`

**Semantic matching (bilinear):** inner products between entity and relation representations
- RESCAL: `h^T M_r t` — full M_r per relation
- DistMult: diagonal M_r (symmetric; cannot model asymmetric relations)
- HolE: circular correlation `h ⋆ t · r`
- ComplEx: extends DistMult to complex vectors; handles asymmetry
- SimplE: inverse-aware; two independent embedding vectors per entity
- TuckER: Tucker decomposition; subsumes RESCAL and DistMult
- QuatE: quaternion product; richest rotation group

### Axis 3: Encoding Models

How embeddings are computed (architecture):

| Category | Models |
|----------|--------|
| **Linear/bilinear** | RESCAL, DistMult, ComplEx, AnalyEE |
| **Factorization** | TATEC, IRN |
| **Neural networks (MLP)** | SME, NTN (Neural Tensor Network — Socher et al. 2013) |
| **CNN** | ConvE (2D convolution, Dettmers et al. 2018), ConvKB |
| **RNN** | RSN (Recurrent Skipping Networks — models relation paths) |
| **Transformer / BERT-based** | KG-BERT, KEPLER (pre-trained LM fine-tuned on KG triples) |
| **GNN/GCN** | R-GCN (Schlichtkrull et al. 2018), SACN, KBAT |

### Axis 4: Auxiliary Information

Extra signals beyond raw triples:

| Type | Description | Models |
|------|-------------|--------|
| **Entity types** | Type constraints on valid triples | TKRL, TypedEntRel |
| **Relation paths** | Multi-hop paths between entity pairs | PTransE, GAKE |
| **Visual information** | Entity images encoded alongside triples | IKRL |
| **Textual descriptions** | Entity/relation descriptions from text | DKRL, KG-BERT, KEPLER |
| **Logical rules** | Horn rules constrain embedding geometry | RUGE, IterE, pLogicNet |

---

## Knowledge acquisition (§IV)

### A. Knowledge Graph Completion (KGC)

The task: given an incomplete KG, predict missing triples. Standard formulation: rank candidate entities for `(h, r, ?)` or `(?, r, t)`.

**Embedding-based ranking.** All KGE models above applied to link prediction. Evaluate by filtered MR (mean rank), MRR, Hits@k.

**Relation path inference.** Multi-hop paths carry signal beyond direct edges.
- Path Ranking Algorithm (PRA): random walks to infer relation-path features [Lao et al. 2011]
- Path-RNN: RNN encoder over relation paths [Das et al. 2017]
- DeepPath: REINFORCE policy network selects informative paths [Xiong et al. 2017]
- MINERVA: Go-for-a-walk policy gradient; navigates KG to reach answer entity [Das et al. 2018]
- M-Walk: Monte Carlo tree search over relation paths

**Logical rule learning.** Mine Horn rules from KG data; use them for inference.
- AMIE / AMIE+: Galárraga et al. (2013/2015); partial completeness assumption (PCA); metrics: confidence, support, head coverage
- Neural LP: differentiable rule learning via attention-weighted path matrices [Yang et al. 2017]
- DRUM: differentiable rule mining with binary decision diagrams [Sadeghian et al. 2019]
- pLogicNet: probabilistic logic; soft rules as Markov Logic priors over embeddings
- IterE: alternates between rule learning and embedding learning

**Meta relational learning (few-shot KGC).** Most KG relations have few examples (long-tail). Meta-learning adapts quickly to new relations.
- GMatching: match entity neighbor patterns for one-shot link prediction
- MetaR: relation meta-learning; gradient-based adaptation (MAML-style) per new relation
- FSRL: few-shot relation learning with memory networks
- FAAN, MTransH: other few-shot variants

**Triple classification.** Binary task: is a given triple true or false? SVM on embedding scores; NTN classifier.

---

### B. Entity discovery

**Named Entity Recognition (NER).** Identify entity mentions in text.
- Sequence labeling: BiLSTM-CRF (standard architecture), BERT-based (BERT-NER)
- Cross-lingual NER: transfer via multilingual embeddings
- Nested NER: entities can be nested (e.g., "New York City" inside "New York City Police Department")

**Entity typing.** Assign semantic types to entities.
- Hierarchical typing: AFET (attentive fine-grained entity typing), NFETC
- Ultra-fine entity typing: thousands of fine-grained types

**Entity disambiguation / entity linking.** Map an entity mention to a KG node.
- BERT-based EL (global coherence), attention mechanisms, BLINK
- Challenge: short-text disambiguation where context is minimal (twitter, titles)

**Entity alignment (cross-KG).** Given two KGs, find entity pairs referring to the same real-world entity.
- Translation-based: MTransE [Chen et al. 2017], IPTransE, JAPE [Sun et al. 2017]
- GCN-based: GCN-Align [Wang et al. 2018], MuGNN, AliNet
- Iterative semi-supervised: BootEA [Sun et al. 2018] — alternates between alignment and KGE training
- Attention-based: RDGCN, HGCN

**Key benchmark**: DBP15k (cross-lingual DBpedia: ZH-EN, JA-EN, FR-EN; 15K entity pairs each). Standard metrics: Hits@1, Hits@10, MRR.

---

### C. Relation extraction

Extract relation triples from raw text. Backbone: distant supervision — align text with existing KG to generate noisy labeled data.

- **CNN-based**: Zeng et al. (2014) — sentence-level RE using CNN; PCNN (Piecewise CNN) [Zeng et al. 2015] — piecewise max-pooling
- **Selective attention**: Lin et al. (2016) — entity-pair level attention over sentences; reduces noise from distant supervision
- **GCN-based**: dependency-tree GCN for within-sentence RE; inter-sentence / document-level RE (DocRED dataset; ATLOP model)
- **Adversarial training**: generate adversarial noise to improve robustness against distant supervision noise
- **Reinforcement learning**: instance selection (select informative sentences); multi-hop reasoning combined with RE
- **Document-level RE**: cross-sentence aggregation models — ATLOP [Zhou et al. 2021], SSAN

**Key benchmarks**: NYT10 (10 relations, distant supervision), SemEval-2010 Task 8 (9 types), TACRED (42 types, challenging), DocRED (96 types, document-level).

---

## Temporal knowledge graphs (§V)

Standard KGs treat facts as timeless. Temporal KGs attach time stamps or intervals to triples: `(h, r, t, τ)` where τ is a time point or interval.

**Temporal information embedding.**
- t-TransE [Leblay & Chekol 2018]: time-specific relation vectors
- HyTE [Dasgupta et al. 2018]: hyperplane projection parameterized by time
- TNTComplEx [Lacroix et al. 2020]: 4th-order tensor decomposition; best performance on standard temporal benchmarks

**Entity dynamics.** Entities themselves change over time (a person's employer, a country's president). Know-Evolve [Trivedi et al. 2017]: continuous-time dynamic KG using temporal point processes.

**Temporal relational reasoning.** Predict future facts: given `(h, r, ?, t+1)`, what entity will be reached?
- RE-NET [Jin et al. 2020]: autoregressive relational event networks; models temporal KG as a sequence of events

**Temporal KGC.** Standard link prediction but evaluated at specific time points.

**Key benchmarks**: ICEWS (political events, fine-grained time stamps), GDELT (global events, large-scale), YAGO2 (Wikipedia-sourced temporal facts), Wikidata temporal.

---

## Knowledge-aware applications (§VI)

### Natural language understanding

**KGQA (Question answering over KGs).**
- Simple QA (one hop): entity linking + relation classification
- Multi-hop QA: path traversal or embedding-based approaches
- Key systems: EmbedKGQA [Saxena et al. 2020], PullNet [Sun et al. 2019], MHGRN [Feng et al. 2020]
- Open-domain: hybrid text + KG retrieval (KEPLER, REALM, RAG)

**Dialogue systems.** Knowledge-grounded generation: integrate KG entity/relation information into seq2seq generation; commonsense-grounded response generation using ConceptNet or ATOMIC.

### Recommendation systems

KGs provide structured item attributes and user-item-entity graphs that collaborative filtering lacks.
- KGCN [Wang et al. 2019]: graph convolutional propagation over KG for user preference
- RippleNet [Wang et al. 2018]: propagate user preferences through KG like ripples
- KGAT [Wang et al. 2019]: attentive KG embedding for recommendation; separately models user-item and KG graphs then jointly

### World knowledge / commonsense

- ConceptNet [Speer et al. 2017]: 34M+ commonsense assertions in 301 languages
- ATOMIC [Sap et al. 2019]: event-centered if-then commonsense (causes, effects, intents); 880K entries
- COMET [Bosselut et al. 2019]: train a transformer (GPT) on ATOMIC triples → generates novel commonsense inferences; treats commonsense generation as KGC

### Other applications

Drug-drug interaction prediction (KGNN [Lin et al. 2020]), biomedical KG for drug discovery, event detection, zero-shot learning via KG class hierarchies, entity-enhanced language models (ERNIE [Zhang et al. 2019], KEPLER [Wang et al. 2021]).

---

## Named datasets and benchmarks

### KG completion

| Dataset | Source | Entities | Relations | Train triples | Notes |
|---------|--------|----------|-----------|--------------|-------|
| **FB15k** | Freebase | 14,951 | 1,345 | 483,142 | Inverse-relation leakage |
| **FB15k-237** | Freebase | 14,541 | 237 | 272,115 | Standard; no leakage |
| **WN18** | WordNet | 40,943 | 18 | 141,442 | Inverse-relation leakage |
| **WN18RR** | WordNet | 40,943 | 11 | 86,835 | Standard; no leakage |
| **YAGO3-10** | YAGO | 123,182 | 37 | 1,079,040 | Larger scale |
| **Nations** | — | 14 | 55 | 1,592 | Small-scale |
| **UMLS** | Medical | 135 | 46 | 5,216 | Biomedical |
| **Nell-995** | NELL | 75,492 | 200 | 149,678 | Used for RL path reasoning |
| **DB100K** | DBpedia | 99,604 | 470 | 597,572 | |

### Temporal KG

| Dataset | Description |
|---------|-------------|
| **ICEWS** | Political events; time-stamped to the day; ~400K events |
| **GDELT** | Global event database; very large-scale |
| **YAGO2** | YAGO with temporal annotations |

### Open-source libraries

| Library | Coverage |
|---------|---------|
| **OpenKE** | TransE, TransH, TransR, TransD, RESCAL, DistMult, ComplEx, RotatE |
| **PyKEEN** | Wide range of KGE models; evaluation pipelines |
| **DGL-KE** | Distributed, large-scale KGE |
| **AmpliGraph** | Sklearn-style API for KGE |
| **LibKGE** | Research-oriented; reproducible baselines |

---

## Open problems and future directions (§VII)

Eight research directions identified:

1. **Complex reasoning.** Multi-hop, compositional, and logical reasoning. Current models handle simple patterns; integrating symbolic rules with neural models for interpretable chaining is open.

2. **Unified framework.** KGC, entity alignment, and relation extraction are separate pipelines. A joint model handling multiple tasks in a shared representation would reduce annotation cost and improve transfer.

3. **Interpretability.** Embedding models are black boxes. Path-based reasoning (MINERVA, DeepPath) is more interpretable; combining with explanations for rule-based systems is an active area.

4. **Scalability.** Freebase / Wikidata have billions of facts. Most KGE training runs on sub-million-triple subsets. Distributed and approximate methods (DGL-KE, RotatE on GPUs) partially address this.

5. **Dynamic KGs.** Facts change over time; entities evolve. Temporal KG models (TNTComplEx, Know-Evolve) exist but continuous updating as new facts stream in is unsolved.

6. **Few-shot and zero-shot learning.** Long-tail relations with few training triples dominate real KGs. Meta-relational learning (MetaR, FSRL) is promising; zero-shot relation learning via KG-encoded class descriptions.

7. **Knowledge aggregation and fusion.** Cross-lingual, cross-modal, and cross-domain KG merging requires entity alignment + ontology alignment + multimodal feature matching simultaneously. No single system handles all three.

8. **Commonsense KGs.** ConceptNet and ATOMIC have different structural properties than encyclopedic KGs. COMET shows that generative models can extend commonsense KGs, but reliable common-sense reasoning in downstream NLP tasks remains unsolved.

---

## What this paper is good for

Ji et al. [2021] is the right starting point for any ML engineer working on KG embeddings or knowledge-augmented NLP. The four-view taxonomy (representation space / scoring function / encoding model / auxiliary information) is the most systematic framework available for understanding and comparing KGE models. The knowledge acquisition sections (NER, entity linking, relation extraction, entity alignment) are detailed and current through ~2020. For formal semantics and query languages, go to Hogan et al. [2021]; for history and philosophical grounding, go to Chaudhri et al. [2022].

---

## Selected references

[1] Ji S, Pan S, Cambria E, Marttinen P, Yu PS (2021). "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications." *IEEE Trans Neural Netw Learn Syst.* DOI: 10.1109/TNNLS.2021.3070843

[2] Bordes A et al. (2013). "Translating Embeddings for Modeling Multi-Relational Data." [TransE] *NeurIPS.*

[3] Nickel M et al. (2011). "A Three-Way Model for Collective Learning on Multi-Relational Data." [RESCAL] *ICML.*

[4] Yang B et al. (2015). "Embedding Entities and Relations for Learning and Inference in Knowledge Bases." [DistMult] *ICLR.*

[5] Trouillon T et al. (2016). "Complex Embeddings for Simple Link Prediction." [ComplEx] *ICML.*

[6] Sun Z et al. (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." *ICLR.*

[7] Zhang S et al. (2019). "Quaternion Knowledge Graph Embeddings." [QuatE] *NeurIPS.*

[8] Balazevic I et al. (2019). "TuckER: Tensor Factorization for Knowledge Graph Completion." *EMNLP.*

[9] Schlichtkrull M et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." [R-GCN] *ESWC.*

[10] Dettmers T et al. (2018). "Convolutional 2D Knowledge Graph Embeddings." [ConvE] *AAAI.*

[11] Galárraga L et al. (2013). "AMIE: Association Rule Mining under Incomplete Evidence." *WWW.*

[12] Sadeghian A et al. (2019). "DRUM: End-to-End Differentiable Rule Mining on Knowledge Graphs." *NeurIPS.*

[13] Xiong W et al. (2017). "DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning." *EMNLP.*

[14] Das R et al. (2018). "Go for a Walk and Arrive at the Answer." [MINERVA] *ICLR.*

[15] Saxena A et al. (2020). "Improving Multi-Hop QA over Knowledge Graphs." [EmbedKGQA] *ACL.*

[16] Bosselut A et al. (2019). "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction." *ACL.*

[17] Lacroix T et al. (2020). "Tensor Decompositions for Temporal Knowledge Base Completion." [TNTComplEx] *ICLR.*

[18] Wang H et al. (2018). "RippleNet: Propagating User Preferences on the Knowledge Graph." *CIKM.*

[19] Wang X et al. (2019). "KGAT: Knowledge Graph Attention Network for Recommendation." *KDD.*

[20] Speer R et al. (2017). "ConceptNet 5.5." *AAAI.*
