# Knowledge Graphs: Survey Overview

This folder traces the history and structure of knowledge graph (KG) research. Each file covers a landmark paper or period. This is the first entry.

---

## Paper: "Knowledge Graphs: Opportunities and Challenges"

**Authors:** Ciyuan Peng, Feng Xia, Mehdi Naseriparsa, Francesco Osborne  
**Journal:** *Artificial Intelligence Review*, Springer Nature (Open Access, CC BY 4.0)  
**Published:** 2023 (accepted March 9, 2023)  
**DOI:** [10.1007/s10462-023-10465-9](https://doi.org/10.1007/s10462-023-10465-9)  
**PMC:** [PMC10068207](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068207/)

---

## What the paper is

A systematic survey of knowledge graph research organized around two axes: **opportunities** (AI systems and application domains that KGs unlock) and **challenges** (the five hard technical problems that remain unsolved). It is not a history paper per se — for history, it defers to Chaudhri et al. [2022] and Hogan et al. [2021] — but its §2 "Overview" gives the canonical set of definitions and the short historical provenance that every subsequent KG paper references.

---

## Core definitions

The paper gives three nested definitions that build on each other.

**Triple / fact.** The atomic unit is the triple `(subject, predicate, object)` — equivalently `(head, relation, tail)`. Example: `(Bill Gates, founderOf, Microsoft)`. Relations are directed; `founderOf` ≠ `foundedBy`.

**Knowledge base.** A dataset of triples encoding real-world facts and semantic relations [Bordes et al. 2011]. A schema (ontology) constrains the property types and their domains.

**Knowledge graph.** When triples are laid out as a directed graph — nodes = entities, edges = relations — the result is a knowledge graph [Hogan et al. 2021; Cheng et al. 2022]. The paper treats "knowledge graph" and "knowledge base" as interchangeable in practice (a common convention in the field).

Two storage formats dominate: **RDF** (Resource Description Framework, W3C standard) and **Labeled Property Graphs** (LPGs, used by Neo4j, FalkorDB, etc.) [Färber et al. 2018; Baken 2020].

---

## Historical provenance (§2.1)

The paper traces seven named systems before the Google KG coin the term:

| Year | System | Key fact |
|------|--------|----------|
| 2004 | **WordNet** | Lexical KG; measures semantic similarity between concepts in a hierarchical graph [Pedersen et al. 2004] |
| 2007 | **DBpedia** | Converts Wikipedia infoboxes into an ontological knowledge base; "nucleus for a web of open data" [Auer et al. 2007] |
| 2008 | **Freebase** | Collaboratively created, multi-source structured graph database [Bollacker et al. 2008] |
| 2011 | **Facebook entity graph** | Converts unstructured user-profile content into structured entity data [Ugander et al. 2011] |
| **2012** | **Google Knowledge Graph** | First mainstream use of the term "knowledge graph"; catalyzed the field [Ehrlinger & Wöß 2016] |
| 2014 | **Wikidata** | Cross-lingual, document-oriented KG powering Wikipedia and many downstream services [Vrandečić & Krötzsch 2014] |
| 2016 | **Yago** | High-quality KG from Wikipedia + WordNet; emphasizes accuracy and multilingual coverage [Rebele et al. 2016] |

**Takeaway.** Google's 2012 announcement is the inflection point the community treats as year zero. Everything before it is precursor; everything after is the modern KG research wave.

---

## Research landscape (§2.2)

The paper maps KG research into seven tracks, illustrated in their Figure 2:

1. **KG Embedding** — represent entities and relations as low-dimensional vectors
2. **Knowledge Acquisition** — extract entities and relations from raw text, structured sources, and multimodal data
3. **KG Completion** — predict missing facts (link prediction, entity prediction)
4. **Knowledge Fusion** — align and merge KGs from different sources or languages
5. **Knowledge Reasoning** — infer new facts from existing triples (single-hop and multi-hop)
6. **AI Systems** — downstream systems powered by KGs (recommenders, QA, IR)
7. **Application Fields** — domain deployments (education, science, social, health)

The paper's scope is tracks 1–5 as challenges and 6–7 as opportunities.

---

## Opportunities: AI systems powered by KGs (§3)

### Recommender systems

Traditional content-based and collaborative filtering (CF) systems hit two structural limits: **data sparsity** (most user–item pairs are unobserved) and **cold start** (no history for new users or items). KG-based systems thread item attributes, user preferences, and external knowledge into a single graph, addressing both problems and adding explainability.

Representative systems built on this pattern: RippleNet [Wang et al. 2018], KPRN [Wang et al. 2019], MKR [Wang et al. 2019], MKGAT [Sun et al. 2020].

### Question-answering systems

KG-based QA parses a natural-language question, maps it to entities and relations in the graph, and retrieves or traverses to the answer — either via similarity search (KEQA [Huang et al. 2019]) or SPARQL-like traversal (PCQA [Shin et al. 2019]). Multi-hop QA (EmbedKGQA [Saxena et al. 2020]) extends this by chaining multiple relation steps, which pure text retrieval handles poorly.

### Information retrieval

Classic IR matches query keywords against an inverted index; it has no semantic model of what entities mean or how they relate. KG-based IR uses entity representations to interpret query intent and rank documents by semantic relevance rather than term overlap. Systems include EDRM [Liu et al. 2018] and the COVID-19 Knowledge Graph [Wise et al. 2020].

---

## Opportunities: Application fields (§4)

The paper identifies four high-value verticals:

**Education.** KGs model curriculum structure, concept prerequisites, and learner state. KnowEdu [Chen et al. 2018] automates educational KG construction; Aliyu et al. [2020] uses KGs for course allocation. Demand accelerated post-COVID as online learning scaled.

**Scientific research.** Academic KGs link publications, authors, concepts, institutions, and funding. Scale: Microsoft Academic Graph (MAG) holds 248M+ publications [Wang et al. 2020]; AMiner Graph holds 200M+ [Zhang et al. 2018]; AI-KG [Dessì et al. 2020] covers 800K entities from 330K AI papers; AIDA [Angioni et al. 2021] spans 21M publications and 8M patents. Applications: paper recommendation, reviewer assignment, trend detection.

**Social networks.** Social KGs encode user–user and user–item relations from Facebook, Twitter, and similar platforms. Applications: social recommendation (GraphRec [Fan et al. 2019]), social relationship extraction from photos [Wang et al. 2018], fake news detection (DEAP-FAKED [Mayank et al. 2021]).

**Health and medical care.** Medical KGs integrate clinical knowledge for intelligent systems. Applications: drug–drug interaction prediction (KGNN [Lin et al. 2020]), medicine recommendation (SMR [Gong et al. 2021]), health misinformation detection (DETERRENT [Cui et al. 2020]), COVID-19 drug repurposing (COVID-KG [Wang et al. 2020]).

---

## Technical challenges (§5)

### 1. Knowledge graph embeddings (§5.1)

Embedding maps entities and relations into a continuous vector space so that structural relationships are preserved arithmetically. Three families exist:

**Tensor factorization-based** — treat the KG as a 3D tensor (entities × entities × relations) and factorize it. Key models: RESCAL [Nickel et al. 2011], HolE [Nickel et al. 2016], ComplEx [Trouillon et al. 2016], SimplE [Kazemi & Poole 2018], RotatE [Sun et al. 2019], QuatE [Zhang et al. 2019].

**Translation-based** — model relations as vector translations: `head + relation ≈ tail`. Key models: TransE [Bordes et al. 2013] (the foundational model), TransH [Wang et al. 2014], TransR [Lin et al. 2015], TransD [Ji et al. 2015], KG2E [He et al. 2015], TransG [Xiao et al. 2015].

**Neural network-based** — learn embeddings via deep architectures. Key models: NTN/SLM [Socher et al. 2013], RMNN [Liu et al. 2016], R-GCN [Schlichtkrull et al. 2018], ConvKB [Nguyen et al. 2017], KBGAN [Cai & Wang 2017].

**Benchmark results (Table 3, Hits@10):** QuatE 90% on FB15K, RMNN 89.9% on WN11, KBGAN 89.2% on WN18.

**Shared limitation.** Most embedding methods see only surface-level triples. They ignore entity type, relation paths, temporal metadata, and textual descriptions — all of which carry signal the models discard.

### 2. Knowledge acquisition (§5.2)

Building a KG from raw data requires extracting three kinds of things: entities, relations, and attributes [Fu et al. 2019]. The field struggles with: (a) low extraction accuracy producing noisy or incomplete graphs; (b) domain-specific KG construction requiring labeled corpora that don't exist; (c) multilingual extraction — non-English datasets are sparse; cross-lingual models exist [Bekoulis et al. 2018] but accuracy drops significantly; (d) multimodal construction — integrating images, tables, and text remains an open problem [Zhu et al. 2022].

### 3. Knowledge graph completion (§5.3)

KGs are structurally incomplete — Freebase, for example, is missing birthplace data for more than half of its person entities. Completion predicts missing links (link prediction [Wang et al. 2020; Akrami et al. 2020]) or missing entities (entity prediction [Ji et al. 2021]).

The hard problem is the **closed-world assumption**: standard completion methods can only fill in blanks for entities already in the graph. Open-world completion (ConMask [Shi & Weninger 2018]) extends this to unseen entities but with low accuracy. Temporal KG completion — adding a timestamp dimension to each triple — introduces further computational cost [Shao et al. 2022].

### 4. Knowledge fusion (§5.4)

Multiple KGs built from different sources describe the same real-world entities with different identifiers. Fusion aligns them via entity alignment or ontology alignment [Ren et al. 2021; Zhao et al. 2020]. Open problems: cross-language alignment accuracy remains low [Xu et al. 2019]; entity disambiguation in short texts (e.g., tweets) is hard because context is thin (SCSNED [Zhu & Iglesias 2018]); multimodal fusion — aligning across text, image, and structured data — requires mapping between incompatible feature spaces (HMEA [Guo et al. 2021] uses hyperbolic space).

### 5. Knowledge reasoning (§5.5)

Reasoning infers new triples from existing ones. Three approaches: logic rule-based [De Meester et al. 2021], distributed representation-based [Chen et al. 2020], neural network-based [Xiong et al. 2017; Xian et al. 2019].

**The scale problem.** Most published multi-hop reasoning experiments use KGs with ~63K entities and ~592K relations. Real KGs have millions of entities; current models fail at that scale because the computation cost grows exponentially with hop count [Zhang et al. 2021; Zhu et al. 2022]. Verifying inferred facts (rather than just generating them) is an additional unsolved problem.

---

## What this paper is good for / where it points

Peng et al. (2023) is a clean current-state survey — good for understanding the taxonomy of KG research and the dominant embedding approaches. It is not the place to look for deep history; for that, the paper itself cites two better sources:

- **Chaudhri et al. (2022).** "Knowledge graphs: introduction, history and perspectives." *AI Magazine* 43(1):17–29. — The dedicated history paper.
- **Hogan et al. (2021).** "Knowledge graphs." *ACM Computing Surveys.* — The authoritative technical reference; the paper the field treats as the canonical definition source.

Both should be next in this survey.

---

## Selected references from this paper

[1] Peng C, Xia F, Naseriparsa M, Osborne F (2023). "Knowledge Graphs: Opportunities and Challenges." *Artif Intell Rev.* DOI: 10.1007/s10462-023-10465-9

[2] Hogan A et al. (2021). "Knowledge graphs." *ACM Comput Surv.*

[3] Chaudhri V et al. (2022). "Knowledge graphs: introduction, history and perspectives." *AI Mag* 43(1):17–29.

[4] Ehrlinger L, Wöß W (2016). "Towards a definition of knowledge graphs." *SEMANTiCS.*

[5] Auer S et al. (2007). "DBpedia: a nucleus for a web of open data." In: *The Semantic Web.* Springer.

[6] Bollacker K et al. (2008). "Freebase: a collaboratively created graph database for structuring human knowledge." *ACM SIGMOD.*

[7] Vrandečić D, Krötzsch M (2014). "Wikidata: a free collaborative knowledgebase." *Commun ACM.*

[8] Rebele T et al. (2016). "YAGO: a multilingual knowledge base from Wikipedia, Wordnet, and Wikidata." *ISWC.*

[9] Bordes A et al. (2013). "Translating embeddings for modeling multi-relational data." [TransE] *NeurIPS.*

[10] Ji S et al. (2021). "A survey on knowledge graphs: representation, acquisition, and applications." *IEEE Trans Neural Netw Learn Syst.*

[11] Nickel M, Tresp V, Gabrilovich M (2011). "A three-way model for collective learning on multi-relational data." [RESCAL] *ICML.*

[12] Schlichtkrull M et al. (2018). "Modeling relational data with graph convolutional networks." [R-GCN] *ESWC.*

[13] Saxena A, Tripathi A, Talukdar P (2020). "Improving multi-hop question answering over knowledge graphs using knowledge base embeddings." [EmbedKGQA] *ACL.*

[14] Dessì D, Osborne F et al. (2020). "AI-KG: an automatically generated knowledge graph of artificial intelligence." *ISWC.* Springer.

[15] Färber M et al. (2018). "Linked data quality of DBpedia, Freebase, OpenCyc, Wikidata, and Yago." *Semantic Web* 9(1):77–129.
