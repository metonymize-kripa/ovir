# Knowledge Graphs (Hogan et al.)

**Authors:** Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d'Amato, Gerard de Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, Antoine Zimmermann  
**Venue:** *ACM Computing Surveys* 54(4), Article 71, pp. 71:1–71:37 (2021)  
**Book version:** *Knowledge Graphs*, Synthesis Lectures on Data, Semantics, and Knowledge No. 22, Springer 2021  
**DOI (paper):** [10.1145/3447772](https://doi.org/10.1145/3447772)  
**DOI (book):** [10.2200/S01125ED1V01Y202109DSK022](https://doi.org/10.2200/S01125ED1V01Y202109DSK022)  
**arXiv:** [2003.02320](https://arxiv.org/abs/2003.02320)  
**Full text (book):** [kgbook.org](https://kgbook.org/) (open access)

---

## What the paper is

This is the field's **canonical technical reference** — 18 authors, ~4,000+ citations, the definition source most other KG papers cite. It is structured as a graduate-level textbook that covers the entire KG stack: from graph data models and query languages through schema, reasoning, embeddings, and GNNs, to creation, quality assessment, refinement, publication, and real-world systems. Peng et al. [2023] cite it for definitions; Ji et al. [2021] cite it as a comprehensive reference. If you want to understand any one technical component of KGs — how SPARQL works, what SHACL does, how R-GCN operates — this is the place to go.

---

## Primary definition (verbatim)

> "Herein we adopt an inclusive definition, where we view a **knowledge graph** as *a graph of data intended to accumulate and convey knowledge of the real world, whose nodes represent entities of interest and whose edges represent relations between these entities*."

A second, more precise definition distinguishes levels:

> "Henceforth, we refer to a *data graph* as a collection of data represented as nodes and edges using one of the models discussed in Chapter 2. We refer to a *knowledge graph* as a data graph potentially enhanced with representations of schema, identity, context, ontologies and/or rules."

This makes an important structural point: "knowledge graph" is not a fixed data format — it is a *data graph* (which has a well-defined format) plus layers of enrichment. You can have a KG that is just a data graph, or one enriched with ontology, rules, embeddings, and provenance metadata.

---

## Graph data models taxonomy (Chapter 2)

The paper's most precise technical contribution is its formal taxonomy of graph data models:

### Directed edge-labelled graph (DEL graph)
The minimal model. Formally: `G = (V, E, L)` where V = nodes, L = edge labels, E ⊆ V × L × V. This is the basis of **RDF** (W3C standard): triples using IRIs, literals, and blank nodes. Queried by **SPARQL 1.1** (W3C 2013).

### Heterogeneous graph
A DEL graph plus a type label per node and per edge: `G = (V, E, L, l)`. Each node and edge carries exactly one type. Used in machine learning (heterogeneous information networks). Not queried by SPARQL — typically traversed by GNN or meta-path methods.

### Property graph (labeled property graph, LPG)
Formally: `G = (V, E, L, P, U, e, l, p)` where nodes and edges each carry a label set and a property–value set. This is the model used by Neo4j / Cypher, FalkorDB, etc. More flexible than RDF for operational data; less interoperable. Queried by **Cypher** (Neo4j), **Gremlin** (Apache TinkerPop), **G-CORE** (proposed standard).

### Graph dataset
A default graph plus a set of named graphs: `D = (G_D, N)`. Enables provenance tracking — you can ask "which named graph does this triple come from?" RDF datasets (W3C) are a standardized case. Used in situations requiring multi-source data management.

### Other models
Hypernodes (nested graphs), hypergraphs. Used in specialized settings; not mainstream for KGs.

---

## Query languages

| Language | Model | Distinguishing features |
|----------|-------|------------------------|
| **SPARQL 1.1** | RDF (DEL graph) | Basic graph patterns, complex patterns (UNION, OPTIONAL, FILTER), property paths (regular expressions over edges), aggregation, CONSTRUCT/INSERT/DELETE |
| **Cypher** | Property graph | Pattern matching using ASCII-art syntax; isomorphism semantics (no repeated nodes by default) |
| **Gremlin** | Property graph | Traversal-based, imperative; used in Apache TinkerPop, Amazon Neptune |
| **G-CORE** | Property graph | Academic proposal for a unified language; paths as first-class results |

Query types: **basic graph patterns** (BGPs — substitute variables for constants, find homomorphisms); **complex graph patterns** (BGPs + projection, selection, union, difference, join, anti-join); **navigational graph patterns** (BGPs + regular path queries using Kleene star, concatenation, disjunction, inverse).

Query interfaces discussed: faceted browsing (VisiNav, GraFa), graphical query builders (QueryVOWL, Sparklis, RDF Explorer), query-by-example (GQBE), natural language QA (WDAqua-core1, TemplateQA).

---

## Schema, identity, context (Chapter 3)

### Three types of schema

**Semantic schema** (RDFS, OWL): defines class hierarchies, property domains/ranges, disjointness, cardinality, equivalence. Enables inference — from the schema you can derive new triples. OWL 2 DL is based on the SROIQ description logic.

**Validating schema** (SHACL, ShEx): prescribes structural constraints the data must satisfy, without inference. SHACL targets are graphs matched against shapes; ShEx uses a grammar-like syntax. These are the right tool for data quality enforcement.

**Emergent schema**: induced from data patterns via graph summarisation or quotient graphs. No explicit prescription required — the structure reveals itself from the data.

### Identity mechanisms
- **Persistent identifiers**: IRIs in RDF; HTTP URIs enable dereferencing
- **External identity links**: `owl:sameAs`, `schema:sameAs` link nodes across KGs
- **Datatypes**: `xsd:date`, `xsd:integer` — typed literals
- **Lexicalisation**: `rdfs:label`, `skos:prefLabel` — human-readable names for entities
- **Existential nodes**: RDF blank nodes for unknown/anonymous entities

### Context mechanisms
- **Direct representation**: add context facts as additional triples in the main graph
- **Reification**: make a triple itself a node, then annotate that node — standard RDF reification, n-ary relations
- **Higher-arity representation**: property graph edge attributes, Wikidata qualifiers
- **Annotations**: attach metadata (timestamps, confidence, provenance) to triples — temporal logic, fuzzy logic, probabilistic weights
- **Other**: named graphs, RDF-star, singleton properties, N3 logic

Wikidata uses qualifiers (a form of higher-arity representation) to encode context — e.g., "Barack Obama (head of state) (start: 2009) (end: 2017)."

---

## Deductive knowledge (Chapter 4)

**Ontologies** encode knowledge as class definitions, property restrictions, and axioms. The semantics are Tarski-style model theory: an interpretation assigns extensions to all terms, and a model is an interpretation where all axioms hold.

Key ontology features: class/property hierarchies (subClassOf, subPropertyOf), equivalence (owl:equivalentClass), inverses (owl:inverseOf), transitivity (owl:TransitiveProperty), symmetry, cardinality constraints (owl:maxCardinality), disjointness, nominals (owl:oneOf), existential/universal restrictions.

**Open World Assumption (OWA)** — standard in ontologies: absence of a fact doesn't mean it's false; you simply don't know. This is the correct stance for incomplete KGs. Contrast with the **Closed World Assumption (CWA)** in relational databases: absence means false.

**Reasoning** with rules (Datalog, SWRL, RIF, ASP) and description logics. OWL 2 profiles trade expressivity for tractability:
- **OWL 2 EL**: polynomial-time reasoning (used in large biomedical ontologies like SNOMED CT)
- **OWL 2 QL**: conjunctive query rewriting, compatible with relational databases
- **OWL 2 RL**: rule-based reasoning, polynomial in data size
- **OWL 2 DL**: based on SROIQ; decidable but worst-case exponential

Scalable reasoners: ELK (OWL 2 EL), Pellet, HermiT, RDFox (datalog + OWL 2 RL, scales to large KGs).

---

## Inductive knowledge (Chapter 5)

### Graph analytics
Centrality (PageRank, betweenness), community detection (label propagation, Louvain), graph similarity, motif discovery. Frameworks: Pregel (Malewicz et al. 2010), GraphX (Apache Spark). These can operate on the raw data graph, on SPARQL query results, or on the entailed graph.

### Knowledge graph embeddings (§5.2)

Three families:

**Tensor-based models.** The KG is a 3-D tensor (entities × entities × relations); factorize it.

| Model | Scoring function | Notes |
|-------|-----------------|-------|
| TransE | −‖h + r − t‖ | h + r ≈ t; simple, effective |
| DistMult | h · diag(M_r) · t | Symmetric; misses asymmetric relations |
| ComplEx | Re(h · r · t̄) | Complex vectors; handles asymmetry |
| RotatE | h ∘ r = t (complex rotation) | Models all relation patterns |
| RESCAL | h^T M_r t | Full bilinear; M_r is a full matrix per relation |
| TuckER | Tucker decomposition | Subsumes RESCAL and DistMult |

**Language models.** RDF2Vec, KGloVe — apply Word2Vec/GloVe objectives to random-walk sequences over the KG. Bridge graph structure and text.

**Entailment-aware models.** Incorporate ontological constraints into the embedding objective — KALE, TransC, ComplEx-NNE. Force the embedding space to respect known class/property axioms.

### Graph neural networks (§5.3)

**Recursive GNNs** aggregate neighbor information iteratively:
- **R-GCN** (Schlichtkrull et al. 2018): relation-specific linear transformations; message passing. Used for entity classification and link prediction on relational graphs.
- CompGCN: jointly embeds nodes and relations.

**Non-recursive (convolutional)**:
- **ConvE** (Dettmers et al. 2018): 2D convolution over reshaped entity/relation matrices; more parameters, better at large-scale link prediction.
- InteractE: extends ConvE with feature interactions.

**Expressivity note**: GNNs with standard 1-Weisfeiler-Leman (1-WL) aggregation cannot distinguish some non-isomorphic graphs — an active theoretical research area.

### Symbolic learning (§5.4)

**Rule mining**: learn Horn rules (e.g., `bornIn(x,y) ∧ locatedIn(y,z) → nationality(x,z)`) from observed KG triples.
- AMIE (Galárraga et al. 2013/2015): mines rules under partial completeness assumption (PCA); scores by confidence, support, head coverage.
- AnyBURL: any-time bottom-up rule learning; efficient on large KGs.

**Axiom mining**: induce ontology axioms from data patterns.

**Hypothesis mining**: discover more complex relational patterns; used for open-world knowledge completion.

---

## Creation and enrichment (Chapter 6)

**Human collaboration**: Wikidata, Freebase, OpenStreetMap — crowd-sourced; games-with-a-purpose (JeuxDeMots); human-in-the-loop for verification.

**Text pipeline** (NLP):
1. Pre-processing: tokenisation, PoS tagging, coreference resolution, dependency parsing
2. Named Entity Recognition (NER): identify entity mentions
3. Entity Linking (EL): map mentions to KG nodes (tools: TAGME, AIDA, BLINK)
4. Relation Extraction (RE): classify relationships between entity pairs (distant supervision, OpenIE, neural)
5. Joint end-to-end methods: combine NER + EL + RE

**Markup sources**: HTML wrappers for structured web extraction; web table extraction; Deep Web crawling behind forms/APIs.

**Structured sources**: R2RML (relational DB → RDF mapping language), D2RQ, XML/JSON → RDF via XSLT/JSONiq, cross-KG mapping using entity alignment systems (PARIS, AgreementMakerLight, LIMES, SILK).

**Schema creation**: ontology engineering (Protégé), ontology learning from data and text (taxonomy induction, concept extraction).

---

## Quality assessment (Chapter 7)

Four primary dimensions:

| Dimension | Sub-dimensions |
|-----------|---------------|
| **Accuracy** | Syntactic (valid syntax, datatypes), Semantic (factual correctness), Timeliness (up-to-date) |
| **Coverage** | Completeness (are all relevant facts present?), Representativeness (is coverage biased?) |
| **Coherency** | Consistency (no logical contradictions with schema), Validity (conforms to validating schema) |
| **Succinctness** | Conciseness (no redundant facts), Representational conciseness, Understandability (labels, docs) |

Tools: TripleCheckMate, RDFUnit, Luzzu. Completeness is notoriously hard — measuring it requires a gold standard that typically doesn't exist.

---

## Refinement (Chapter 8)

**Completion**: general link prediction (embeddings, rule mining), type-link prediction, identity-link prediction (entity resolution using string similarity, embedding matching, or logic).

**Correction**: fact validation (embedding-based, text corpus corroboration, crowdsourcing), inconsistency repair (removing triples to restore ontology consistency).

---

## KGs in practice (Chapter 10)

### Open KGs

| KG | Origin | Scale | Notes |
|----|--------|-------|-------|
| **DBpedia** | Wikipedia infoboxes | ~4.58M entities, ~583M triples (2016) | RDF/OWL; SPARQL endpoint; community ontology |
| **YAGO** | Wikipedia + WordNet + GeoNames | ~10M entities, ~120M facts | High precision (~95%); YAGO 1 (2007), 2, 3 |
| **Freebase** | Community contributions | ~58M entities, ~3B triples (peak) | Shut down 2016; data donated to Wikidata |
| **Wikidata** | Community (Wikimedia Foundation) | ~90M+ items (2021) | CC0; JSON/RDF; custom data model with qualifiers |
| **BabelNet** | WordNet + Wikipedia (multilingual) | 15M+ synsets, 271 languages | [Navigli & Ponzetto 2012] |
| **ConceptNet** | Crowd-sourced + databases | 34M+ assertions | Commonsense; [Speer et al. 2017] |
| **NELL** | CMU web-crawling | ~2.8M beliefs | Never-Ending Language Learning |

Domain-specific open KGs: MusicBrainz, GovTrack/data.gov, LinkedGeoData, SIDER, DrugBank, UniProt, Gene Ontology.

### Enterprise KGs

| Company | KG / Notes |
|---------|-----------|
| **Google** | Knowledge Graph (2012, Singhal); powers Search entity cards and QA |
| **Microsoft** | Bing KG (Shrivastava 2017); Microsoft Academic Graph |
| **Facebook** | Entity graph (Noy et al. 2019) |
| **LinkedIn** | Knowledge graph (He et al. 2016) for recommendations and business analytics |
| **Amazon** | Product graph (Krishnan 2018; Dong 2019) |
| **eBay** | KG for personal shopping agents (Pittman et al. 2017) |
| **Airbnb** | Listing/recommendation KG (Chang 2018) |
| **Uber** | Geospatial recommendation KG (Hamad et al. 2018) |
| **Bloomberg** | Finance KG (Meij 2019) |
| **Thomson Reuters** | News/legal KG (Tobin 2017) |
| **IBM** | Watson KG (Devarajan 2017) |

---

## Open problems and future directions (Chapter 11)

1. **Definition consensus.** No universally accepted definition of "knowledge graph" exists (Ehrlinger & Wöß 2016; Bonatti et al. 2018). The field still argues about this.
2. **Property graph standardisation.** ISO GQL (a standard for property graph queries analogous to SQL) was under development; G-CORE is the academic proposal.
3. **Scalable deductive reasoning.** Efficient reasoning over large KGs with expressive ontologies (OWL 2 DL) remains hard. Tractable profiles (OWL 2 EL/RL) are more feasible but sacrifice expressivity.
4. **KG embedding robustness.** Most models see only surface triples and ignore type, temporal, and textual signals. Interpretability of embedding-based predictions is poor.
5. **GNN expressivity.** 1-WL GNNs cannot distinguish all non-isomorphic graphs; higher-order GNNs scale poorly.
6. **Temporal KGs.** Representing and reasoning over facts that change over time; versioning and streaming updates.
7. **Multilingual KGs.** Cross-lingual entity linking and RE; building KGs in low-resource languages.
8. **Open-world completeness.** Estimating what is missing from a KG; partial completeness axioms.
9. **Privacy and access control.** GDPR-compliant KG publishing; machine-readable usage policies (ODRL); differential privacy.
10. **Federated querying.** Efficient SPARQL federation across heterogeneous, distributed endpoints.
11. **Explainability and trust.** Provenance tracking; explaining KG-derived answers; uncertainty quantification.
12. **Neuro-symbolic integration.** Combining symbolic KG reasoning with LLM-based neural approaches — the major open frontier at time of publication.

---

## Named benchmarks

| Dataset | Source | Entities | Relations | Training triples | Notes |
|---------|--------|----------|-----------|-----------------|-------|
| **FB15k** | Freebase | 14,951 | 1,345 | 483,142 | Has test leakage via inverse relations |
| **FB15k-237** | Freebase (cleaned) | 14,541 | 237 | 272,115 | Standard; removes inverse leakage |
| **WN18** | WordNet | 40,943 | 18 | 141,442 | Has test leakage |
| **WN18RR** | WordNet (cleaned) | 40,943 | 11 | 86,835 | Standard |
| **YAGO3-10** | YAGO | 123,182 | 37 | 1,079,040 | Large-scale eval |
| **OGBL-WikiKG2** | Wikidata | 2.5M | 535 | 17M | OGB; realistic scale |

---

## What this paper is good for

This is the right reference for any claim about KG fundamentals: data models, query semantics, schema types, ontology reasoning, embedding families, quality dimensions. The book version (kgbook.org) includes formal definitions, running examples, and full proofs. Use it when you need precise technical grounding rather than a historical narrative (for history, use Chaudhri et al. [2022]) or an applications survey (for applications, use Peng et al. [2023]).

---

## Selected references

[1] Hogan A et al. (2021). "Knowledge Graphs." *ACM Comput Surv* 54(4):71. DOI: 10.1145/3447772

[2] Bordes A et al. (2013). "Translating Embeddings for Modeling Multi-Relational Data." [TransE] *NeurIPS.*

[3] Nickel M et al. (2011). "A Three-Way Model for Collective Learning on Multi-Relational Data." [RESCAL] *ICML.*

[4] Schlichtkrull M et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." [R-GCN] *ESWC.*

[5] Galárraga L et al. (2013). "AMIE: Association Rule Mining under Incomplete Evidence." *WWW.*

[6] Vrandecic D, Krötzsch M (2014). "Wikidata: A Free Collaborative Knowledgebase." *Commun ACM.*

[7] Harris S, Seaborne A (2013). "SPARQL 1.1 Query Language." W3C Recommendation.

[8] Francis N et al. (2018). "Cypher: An Evolving Query Language for Property Graphs." *SIGMOD.*

[9] Singhal A (2012). "Introducing the Knowledge Graph: Things, not strings." Google Blog.

[10] Ehrlinger L, Wöß W (2016). "Towards a Definition of Knowledge Graphs." *SEMANTiCS.*

[11] Navigli R, Ponzetto S (2012). "BabelNet." *Artif Intell* 193:217–250.

[12] Speer R et al. (2017). "ConceptNet 5.5." *AAAI.*

[13] Wilkinson MD et al. (2016). "The FAIR Guiding Principles." *Sci Data.*

[14] Noy N et al. (2019). "Industry-scale Knowledge Graphs." *ACM Queue.*

[15] Dettmers T et al. (2018). "Convolutional 2D Knowledge Graph Embeddings." [ConvE] *AAAI.*

[16] Balazevic I et al. (2019). "TuckER: Tensor Factorization for KG Completion." *EMNLP.*
