# O*NET × FalkorDB: Why Graph Beats Flat Lookup for Labor Market Reasoning

## Problem

Standard labor market platforms have a structural ceiling on what they can reason about. The O\*NET public API and web app use an RDBMS/document model optimized for isolated lookups: "what skills does occupation X need?" That query terminates at one hop. Any cross-occupational inference requires downloading whole tables, parsing rows in external memory, and running set logic yourself.

LinkedIn and similar talent intelligence platforms add a semantic layer — typically two-tower BERT or bi-encoders — to map text to flat skill entities or coarse taxonomies (Parent/Child/Sibling tags). This handles keyword-to-entity matching but not structural inference. "Engineered financial balance sheets" maps to the entity *Accounting*; it doesn't expose that *Accounting* decomposes into sub-abilities at different required levels across different occupation families.

## Why

Both limitations share a root cause: the data lives in a model that doesn't natively represent the relationships.

An RDBMS stores the O\*NET tables as join targets. Multi-hop traversal means multiple sequential queries or expensive cross-joins. Latency kills the use case before it starts.

A bi-encoder produces a dense vector per job title or skill phrase. Similarity is cosine distance in embedding space. This recovers semantic overlap but loses the explicit structural weights — the Importance (IM) and Level (LV) scores O\*NET already provides. Two occupations that use identical corporate jargon in their descriptions will score high similarity even if their actual competency profiles are structurally distant.

## Pattern

FalkorDB stores the O\*NET taxonomy as a native labeled property graph backed by GraphBLAS sparse matrix operations. The graph models what the data actually *is*: a weighted directed ontology.

```
(Occupation)-[:HAS_SKILL {im: 4.12, lv: 4.62, score: 19.03}]->(Skill)
(Occupation)-[:HAS_ABILITY {im: 4.62, lv: 4.88, score: 22.55}]->(Ability)
(Task)-[:MAPS_TO]->(DWA)-[:PART_OF]->(IWA)-[:PART_OF]->(WorkActivity)
```

The `score` edge property is `im × lv` — the product of how important a competency is and at what level it must be performed. This is the mathematical weight that makes structural cosine similarity meaningful. Two occupations are similar not because their descriptions read similarly but because their competency weight vectors overlap.

An LLM paired with this graph doesn't need to reason about similarity in its context window. It issues one Cypher query, gets back a structured delta, and emits deterministic output.

## Traditional System vs. GraphBLAS

```
             [Traditional — Flat Tables]
 ┌──────────────────┐             ┌──────────────────┐
 │  Occupation A    │             │   Occupation B   │
 │  - Task 1        │             │   - Task 4       │
 │  - Task 2        │             │   - Task 5       │
 └──────────────────┘             └──────────────────┘
          ▲                                ▲
          └──────────[ No Link ]───────────┘
          (requires full table download + external join)

             [FalkorDB — GraphBLAS Traversal]
 ┌──────────────────┐             ┌──────────────────┐
 │  Occupation A    │             │   Occupation B   │
 └────────┬─────────┘             └────────┬─────────┘
          │ [:PERFORMS]                     │ [:PERFORMS]
          ▼                                 ▼
 ┌──────────────┐                 ┌──────────────┐
 │   Task 1     │                 │   Task 4     │
 └──────┬───────┘                 └──────┬───────┘
        │ [:MAPS_TO]                     │ [:MAPS_TO]
        ▼                                ▼
        └──────────────┬─────────────────┘
               ┌───────┴──────────────────┐
               │  Detailed Work Activity  │
               │  (structural intersection│
               │   computed in-engine)    │
               └──────────────────────────┘
```

## Failed Approaches

*Full-table download + pandas merge.* Works once. Doesn't compose. Rebuilding this for each new query type is the data engineering tax that makes these projects stall.

*Embedding similarity on job descriptions.* Fast to prototype. Fails when two roles share jargon but differ structurally (e.g., "Communication Skills" appears in 900+ O\*NET occupations at wildly different required levels).

*LLM reasoning over raw TSV context.* Putting the full O\*NET competency table in context works at small scale. 288k rows of Abilities + Skills + Knowledge + Work Activities exceeds any practical context window and introduces hallucination on numerical comparisons.

## Use Cases This Playground Demonstrates

### Use Case A: Career Transition Engine (implemented)

Extract the exact competency delta between two O\*NET occupations. No generic advice ("learn Python") — specific missing nodes, deficient edge weights, and tech tool gaps.

| Category | Item | Source Score | Target Score | Delta |
|---|---|---|---|---|
| Missing node | Tableau | — | 3.2 | +3.2 |
| Deficient edge | Mathematics (IM×LV) | 8.2 | 16.5 | +8.3 |
| Transferable | Active Listening | 19.0 | 18.5 | −0.5 |

The cosine similarity between the two competency vectors is computed via GraphBLAS matrix operations. The LLM receives structured data, not a description.

### Use Case B: Behavioral JD Generation (future)

Traverse `Task → DWA → IWA → WorkActivity` from a free-text input. Map a non-standard task ("Build custom WebGPU canvas pipelines") to its standardized federal taxonomy counterpart. Output a JD grounded in O\*NET's verified behavioral framework rather than generated from statistical text patterns.

### Use Case C: Automation Vulnerability Analysis (future)

Map Automation Occupation (AO) scores on Work Activities to individual tasks via the `Tasks → DWAs → WorkActivities` chain. Factor in FT (task frequency) scale weights. Compute personalized automation risk for a specific task distribution rather than title-wide averages. Find lateral pivots that share the user's toolset but carry lower baseline AO exposure.

## Strategic Point

Standard tools tell an LLM *what* is inside a single profile. This graph architecture lets the LLM compute *how* the entire labor market fits together structurally. The LLM stops guessing from statistical text probabilities and starts executing deterministic queries over an exact mathematical model of human labor.

---

**References**

[1] O\*NET Resource Center. "O\*NET Database." U.S. Department of Labor, August 2025. https://www.onetcenter.org/database.html  
[2] FalkorDB. "GraphBLAS-based Graph Database." https://falkordb.com  
[3] Kepner, J. et al. (2016). "Mathematical Foundations of the GraphBLAS." IEEE High Performance Extreme Computing Conference.
