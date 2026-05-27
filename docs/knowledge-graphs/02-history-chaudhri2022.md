# Knowledge Graphs: Introduction, History and Perspectives

**Authors:** Vinay K. Chaudhri (Stanford), Chaitanya Baru (UCSD), Naren Chittar (JPMorgan Chase), Xin Luna Dong (Meta), Michael Genesereth (Stanford), James Hendler (RPI), Aditya Kalyanpur (Elemental Cognition), Douglas B. Lenat (Cycorp), Juan Sequeda (data.world), Denny Vrandecic (Wikimedia Foundation), Kuansan Wang (Microsoft)  
**Journal:** *AI Magazine*, 43(1):17–29  
**Year:** 2022 (published 2022-03-31)  
**DOI:** [10.1609/aimag.v43i1.19119](https://doi.org/10.1609/aimag.v43i1.19119) (open access via AAAI OJS)

---

## What the paper is

This is the dedicated KG **history paper** — the one Peng et al. [2023] and Ji et al. [2021] defer to for historical context. It grew out of a Stanford graduate seminar on knowledge graphs that featured 50+ speakers; all eleven co-authors were speakers. The team was assembled to represent competing traditions: Lenat (Cyc/commonsense logic), Hendler (Semantic Web/linked data), Vrandecic (Wikidata/crowdsourcing), Wang (Microsoft Academic Graph), Dong (large-scale web KG construction), Genesereth (logical foundations), Kalyanpur (IBM Watson), Sequeda (enterprise data integration). The paper explicitly presents "contrasting perspectives" from these traditions rather than a single canonical view. It also emerged from the NSF Convergence Accelerator Track A program on the **Open Knowledge Network (OKN)**.

The paper's stated goals: (a) introduce KGs and discuss high-prominence application areas; (b) situate KGs in prior AI work; (c) present contrasting perspectives that clarify KGs in relation to related technologies.

---

## Definition

A knowledge graph is a **directed labeled graph** in which nodes represent entities (real-world objects or abstract concepts) and edges represent relations between them. The fundamental unit is the triple `(subject, predicate, object)` — also written `(head, relation, tail)`. Two main storage formats:

- **RDF** (W3C standard): triples with IRIs, supports OWL ontologies and SPARQL queries
- **Property graphs (LPGs)**: nodes and edges with associated key-value property sets; used in Neo4j, FalkorDB, etc.

The paper also covers JSON/key-value representations and mappings of KG data into relational databases. "A directed labeled graph containing data and taxonomy is often referred to as an ontology." CycL (Lenat & Guha, 1991) is noted as combining first-order logic (FOL) ideas with semantic-network pragmatics — an early bridge between logical and graph-based representations.

---

## History: the timeline this paper traces

This section is the paper's primary contribution to the survey literature.

**1956–1970s: Semantic networks and expert systems.** Richens proposed semantic networks in 1956. Ross Quillian (1966/1968) formalized them as networks of nodes and labeled arcs for representing conceptual knowledge, used in early NLP and IR. Minsky's frame theory (1975) introduced structured object-oriented knowledge representation. Expert systems DENDRAL (~1965) and MYCIN (~1970s) at Stanford demonstrated rule-based inference over explicit domain knowledge.

**1984–1995: Cyc and the commonsense bottleneck.** Douglas Lenat starts the Cyc project in 1984 at MCC (later Cycorp) with the goal of building the first large-scale commonsense knowledge base — explicitly aiming to build the background knowledge needed for AI agents to reason like humans. CycL (Lenat & Guha, 1991) combines FOL with semantic-network pragmatics. The project recognized what became known as the knowledge acquisition bottleneck: manually encoding commonsense knowledge at scale is intractable. Miller et al. develop WordNet (~1985–1995) as a more tractable lexical resource. Lenat publishes "A Large-Scale Investment in Knowledge Infrastructure" in CACM (1995) making the case for explicit, large-scale knowledge engineering.

**Late 1990s–2006: Semantic Web.** W3C standardizes RDF (1999) and OWL (2004). Berners-Lee, Hendler, and Lassila publish "The Semantic Web" in Scientific American (2001), envisioning a web of machine-readable linked data. The linked open data (LOD) movement follows, producing interconnected open datasets. The practical results fall short of the vision — adoption outside academia is thin.

**2007–2011: Modern precursor KGs.**

| Year | System | Key detail |
|------|--------|-----------|
| 2007 | **DBpedia** | Auer et al. extract structured data from Wikipedia infoboxes into RDF; becomes a hub of the LOD cloud |
| 2007 | **YAGO 1** | Suchanek et al. fuse Wikipedia and WordNet into a high-precision KG |
| 2008 | **Freebase** | Bollacker et al. build a collaboratively created multi-source graph database; acquired by Google in 2010 |
| 2011 | **IBM Watson** | Wins Jeopardy using knowledge-intensive QA; KGs are one component |
| 2011 | **Facebook entity graph** | Converts unstructured user profiles into structured entity data |

**2012: The naming event.** Google launches its public **Knowledge Graph** (Singhal, 2012). The term "knowledge graph" is coined for general use. The KG powers search result cards, entity disambiguation, and related-search suggestions. This is the inflection point: the term proliferates and the modern research wave begins.

**2012–2018: Industrial and academic scaling.**

| Year | System | Key detail |
|------|--------|-----------|
| 2012 | **Wikidata** | Vrandecic & Krötzsch; cross-lingual, crowd-sourced; CC0 license; replaces Freebase after Google shuts it down (2016) |
| 2012–| **Microsoft Academic Graph (MAG)** | Wang et al.; heterogeneous scholarly KG; 248M+ publications |
| 2013 | **TransE** | Bordes et al.; KG embedding as translation; opens the embedding research wave |
| 2014–| **Data Commons** (Google) | Integrates statistical datasets into a public KG |
| 2016 | **Freebase shutdown** | Data donated to Wikidata; research community migrates |
| 2017–| **Open Knowledge Network (OKN)** | NSF Convergence Accelerator Track A; goal: an open, federated, queryable infrastructure KG for the US |

---

## Contrasting perspectives (the paper's key structural feature)

The eleven co-authors represent genuinely different traditions. The "contrasting perspectives" section brings these into explicit dialogue — unusual for a survey paper.

**KGs vs. knowledge bases.** KGs emphasize graph structure and scalability; traditional KBs (Cyc) emphasized expressivity and reasoning depth. The paper treats these as complementary, not competitive. A KG can be the storage layer for a KB; whether you call it a KB depends on whether you attach formal semantics.

**KGs vs. ontologies.** An ontology is a schema/taxonomy — it defines classes, properties, and their relationships. A KG is data conforming to (or accompanied by) an ontology. They are not equivalent; conflating them is a common source of confusion.

**KGs and first-order logic.** Genesereth and Lenat's tradition argues full FOL expressivity is necessary for general AI agents. KG practitioners (commercial and academic) trade expressivity for scale and query performance. CycL shows this was recognized early — it hedges between FOL rigor and practical graph navigability.

**Manually curated vs. automatically constructed KGs.** Cyc was hand-coded at enormous cost. DBpedia and YAGO are automatically extracted. Wikidata is crowd-sourced. Each trades quality for coverage in a different way: manual = deep, narrow; extraction = broad, noisy; crowd-sourced = broad, variable quality with editorial governance.

**Data integration perspective (Sequeda).** KGs as the new paradigm for enterprise data integration — sitting atop relational and other databases via virtual KG / ontology-based data access (OBDA). The KG provides a semantic layer; the underlying data doesn't move.

**Web/linked data perspective (Hendler).** KGs as the practical realization of the Semantic Web vision: open URIs, RDF triples, SPARQL endpoints, owl:sameAs links across datasets.

---

## Applications

The paper covers four application areas where KGs have gained recent prominence:

**Question answering.** Structured KGs enable precise, multi-hop QA that text retrieval cannot support. IBM Watson demonstrated this; current systems (KGQA, EmbedKGQA) scale it.

**Information retrieval / search.** Google's KG powers entity cards and contextual results. The structured representation of entities and their relations enables semantic search beyond keyword matching.

**Recommender systems.** KGs provide structured item/user attributes that collaborative filtering lacks, enabling explainable recommendations with reduced cold-start sensitivity.

**Entity resolution and NER.** KG construction requires entity linking; entity resolution tools use KGs as reference. A feedback loop: KGs are built by NER/EL tools and used to improve NER/EL tools.

---

## Open problems

The paper identifies these as unresolved:

- **Commonsense knowledge gaps.** Large-scale encyclopedic KGs (Wikidata, DBpedia) don't capture everyday commonsense reasoning. Cyc addressed this but at unscalable cost. ConceptNet and ATOMIC are partial solutions. No system has achieved both scale and commonsense coverage.
- **Expressivity vs. scalability.** RDF/OWL and property graphs sacrifice logical expressivity for query performance. Bridging full FOL reasoning with billion-edge KGs is unresolved.
- **Automatic construction quality.** Extraction is noisy; crowd-sourcing requires editorial governance. There is no fully automated, high-quality KG construction pipeline.
- **KG completion.** Most KGs are structurally incomplete. Link prediction addresses this but with closed-world limitations.
- **Neuro-symbolic integration.** Combining symbolic KG reasoning with neural/LLM approaches is the frontier the 2022 paper identifies and the 2025 follow-up (Chaudhri et al. 2025) frames as the central open challenge.
- **Knowledge provenance and trust.** Where did a fact come from; how reliable is it; how do you propagate uncertainty through KG-based inference.
- **Multimodal knowledge.** KGs are text/structured-data focused. Incorporating images, video, sensor data remains open.

---

## What this paper is good for

This is the reference for anyone who wants to understand **why the KG field exists** and **where it came from**. It is the best single source for: (a) tracing the lineage from semantic networks → expert systems → Cyc → Semantic Web → modern KGs; (b) understanding the field's internal debates (FOL vs. graph, manual vs. automatic, ontology vs. data); (c) understanding the industrial context — why Google, Microsoft, Amazon, and Facebook all built KGs in the 2012–2018 window. For technical depth on embeddings or querying, defer to Ji et al. [2021] or Hogan et al. [2021].

---

## Selected references

[1] Chaudhri VK et al. (2022). "Knowledge Graphs: Introduction, History and, Perspectives." *AI Mag* 43(1):17–29. DOI: 10.1609/aimag.v43i1.19119

[2] Lenat DB, Guha RV (1991). *Building Large Knowledge-Based Systems.* Addison-Wesley.

[3] Lenat DB (1995). "A Large-Scale Investment in Knowledge Infrastructure." *Commun ACM* 38(11).

[4] Singhal A (2012). "Introducing the Knowledge Graph: Things, not strings." Google Blog.

[5] Berners-Lee T, Hendler J, Lassila O (2001). "The Semantic Web." *Scientific American.*

[6] Bollacker K et al. (2008). "Freebase: A Collaboratively Created Graph Database." *ACM SIGMOD.*

[7] Auer S et al. (2007). "DBpedia: A Nucleus for a Web of Open Data." *ISWC.* Springer.

[8] Suchanek FM et al. (2007). "YAGO: A Core of Semantic Knowledge." *WWW.*

[9] Vrandecic D, Krötzsch M (2014). "Wikidata: A Free Collaborative Knowledgebase." *Commun ACM.*

[10] Quillian MR (1968). "Semantic memory." In: *Semantic Information Processing.* MIT Press.

[11] Minsky M (1975). "A Framework for Representing Knowledge." In: *The Psychology of Computer Vision.*

[12] Hogan A et al. (2021). "Knowledge Graphs." *ACM Comput Surv* 54(4):71.
