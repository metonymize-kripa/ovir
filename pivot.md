Here's the complete edit list, ordered by section, ready to hand to the front-end team.

---

**1. `<title>` and meta tags**

- `<title>`: change from `OVIR.net — Opportunistic Verifiable Inference Runtime Network` to `OVIR.net — Offline Verified Inference Retrieval Network`
- `meta-description`: change to `OVIR is an offline verified inference retrieval network: corpus preprocessing with GLiNER entity extraction, COBWEB topic hierarchy, knowledge graph, and BM25 — composed at query time by a small DSPy-orchestrated LM.`
- `meta-og:description`: change to `A workload-first offline preprocessing and retrieval runtime for repeated queries over known corpora.`
- `meta-og:title`: change to `OVIR.net — Offline Verified Inference Retrieval Network`
- `meta-theme-color`: keep `#0d0f12`

---

**2. Nav bar**

- Logo text: change `OOVIR.net` (note: there is a typo in the current nav — double O) to `OVIR.net`
- Nav items: keep `Workloads`, `Pattern`, `Runtime`, `Stack`, `Build`, `Simulation`, `GitHub`
- Add one new nav item between `Stack` and `Build`: `Corpus` — anchors to the new Corpus section (see section 6 below)

---

**3. Hero section**

- Full name display: change `Opportunistic Verifiable Inference Runtime Network` to `Offline Verified Inference Retrieval Network`
- H1: change `A runtime for inference that repeats.` to `A corpus goes in. Verified inferences come out. Queries run in milliseconds.`
- Body copy: replace current paragraph (`OVIR.net targets the workload shape where many short questions repeatedly hit the same long corpus. It fragments the corpus, routes each query to likely evidence, reuses cached fragment work, escalates uncertain cases, and emits receipts for what ran.`) with:

> OVIR decomposes a known corpus offline into a verified inference network — entity links, topic hierarchies, and search indices — so that query-time reasoning is fast composition over pre-verified facts, not generation over raw text. The hard work happens once, offline. Every subsequent query is a fast, auditable lookup.

- CTA buttons: keep `View GitHub Repo` and `Open runtime simulation` — no changes

- Five callout pills — replace all five:

| Pill label | Current value | New value |
|---|---|---|
| Wedge | Repeated short queries | Offline preprocessing |
| Context | Stable long corpus | Verified inference network |
| Runtime | Elixir / OTP + Nx | DSPy + small LM (3–7B) |
| Compute | Sparse + cached | BM25 + graph + COBWEB |
| Trust | Receipts + recompute | Auditable retrieval chains |

---

**4. Workloads section**

- Section header `Workloads first` and intro copy: keep as-is
- Opening paragraph (`OVIR is not a general transformer replacement...`): update to:

> OVIR is not a general-purpose LLM wrapper. It is an offline preprocessing and retrieval runtime for domains where the corpus is known in advance and can be decomposed into verified, composable inference units before any query runs.

- **Data rooms and diligence** card: keep title and description. Replace tags `contract review`, `evidence packs`, `audit trail` with `entity linking`, `graph traversal`, `verified citations`

- **Stable archives** card: keep title and description. Replace tags `cached answers`, `routing memory`, `fallback reuse` with `BM25 scoping`, `COBWEB routing`, `offline precomputation`

- **Large repositories** card: keep title and description. Replace tags `fragment locality`, `call graph hints`, `spot checks` with `call graph indexing`, `entity extraction`, `BM25 + graph hybrid`

- **Policy corpora** card: keep title and description. Replace tags `citations`, `receipts`, `recompute` with `offline verified fragments`, `auditable retrieval chain`, `receipts`

- **Literature sets** card: keep title and description. Replace tags `stable corpus`, `fragment summaries`, `conflict checks` with `entity-linked papers`, `COBWEB concept clusters`, `conflict detection`

- **Bad fit** card: keep the card. Replace all content with:

> **Not the right tool if** the corpus is not known in advance, changes faster than the preprocessing cadence allows, or queries require global reasoning across the full document rather than composable local inferences. Unique question over unique context: use the full model path.

---

**5. Pattern section**

- Section header `The repeated-inference pattern`: change to `The offline-first pattern`
- Section intro copy: replace current paragraph with:

> The preprocessing phase does the expensive work once. The runtime composes pre-verified outputs fast. Uncertainty triggers escalation; escalation results are stored and reused.

- Remove all three current panels (BigBird analogy, PEER analogy, Runtime memory)

- Add three new panels:

**Panel 1 — Offline verified decomposition**
> The corpus is parsed once, offline. GLiNER tags every chunk with entity IDs, types, and confidence scores. A knowledge graph connects entity relationships — explicit links from source material plus inferred transitive paths. COBWEB clustering builds a probabilistic topic hierarchy over the chunk space. BM25 indices and graph edges are materialized and verified against source before the runtime sees any query.

**Panel 2 — Query-time composition**
> A small LM (3–7B), orchestrated by DSPy, extracts entities and routing intent from the user query. The COBWEB hierarchy returns probable concept clusters given the extracted entities. BM25 and graph search run scoped to those clusters. Results assemble from pre-verified fragments. The LM orchestrates lookups — it does not generate reasoning over raw documents.

**Panel 3 — Approximate query processing**
> The same retrieval network scales with available compute. Minimal resources: traverse the hierarchy shallowly, return top results fast. Abundant resources: explore multiple branches in parallel, run consistency checks across answer paths, aggregate by confidence. More compute → higher recall. Graceful degradation is structural, not an afterthought.

---

**6. New section: Corpus (insert between Pattern and Runtime)**

This is a new section, anchor id `#corpus`, added to nav as described above.

- Section header: `Corpus preprocessing`
- Body:

> OVIR's performance advantage is entirely upstream of the query. A known corpus — contracts, runbooks, papers, code, policy documents — is passed through a preprocessing pipeline that produces a verified inference network ready for fast runtime composition.

- Five pipeline steps, displayed as a numbered vertical list or timeline:

1. **Parse and chunk.** Structural chunking — section boundaries for documents, records for tables, AST nodes for code. Not sliding windows.
2. **Entity extraction.** GLiNER runs over every chunk. Zero-shot, no domain-specific training required. Outputs: entity spans, types, confidence scores. Stored alongside chunks.
3. **Knowledge graph construction.** Entity relationships from co-occurrence, explicit source links, and inferred transitive paths. Neo4j or FalkorDB. Pre-indexed for O(1) neighborhood lookup at query time.
4. **COBWEB topic hierarchy.** Hierarchical probabilistic clustering over chunk representations. Produces `P(chunk | cluster)` for every chunk. Incrementally updatable. Fast tree traversal at query time prunes low-probability branches before BM25 runs.
5. **Index materialization.** BM25 index (turbopuffer or Postgres with GiST), dense embeddings for entity descriptions and document summaries, columnar metadata index (dates, entity types, confidence scores). All stored, versioned, and verifiable against source hashes.

- Footer note for this section:

> Preprocessing cost is amortized. You pay once per corpus version. Every subsequent query draws from the same verified network.

---

**7. Runtime loop section**

- Section header `Runtime loop`: keep
- Intro copy (`The runtime is a graph of small verifiable jobs...`): replace with:

> At query time, the runtime composes pre-verified outputs. The unit of work is a lookup, not an inference. The unit of trust is the retrieval trace.

- Replace all six current steps with:

1. **Extract query intent.** DSPy module extracts entities and routing signal from the user query. Typed output: `[entity, entity_type, confidence]`. Fast, deterministic.
2. **Traverse COBWEB hierarchy.** Given extracted entities, compute `P(cluster | entities)`. Descend the hierarchy. Return ranked concept clusters. Low-probability branches pruned before search runs.
3. **Execute scoped retrieval.** BM25 and/or graph traversal, scoped to high-probability clusters. Search space is pre-narrowed. Results ranked by relevance within scope.
4. **Merge and decide.** Accept result when confidence, citation coverage, and conflict checks pass. If below threshold: expand cluster scope, increase parallel branch count, or trigger fallback.
5. **Fallback when needed.** Escalate to a larger model, broader search, or human review. Fallback result is stored — it becomes reusable network memory for similar future queries.
6. **Emit retrieval trace.** Record: entities extracted, clusters traversed, fragments retrieved, ranking rationale, latency, cache hits. Every answer reproducible from its trace.

---

**8. Technical spine section**

- Section header `Technical spine`: keep
- Remove all four current panels (Attention / BigBird masks, Experts / PEER routing, Execution / ONNX + Ortex, Distribution / Elixir / OTP)

- Add five new panels:

**GLiNER — Entity extraction**
> Zero-shot named entity recognition. Runs offline over the full corpus once. No domain-specific training. Outputs entity spans, types, and confidence scores stored alongside every chunk.

**Knowledge graph — Neo4j / FalkorDB**
> Entity relationships from co-occurrence, explicit source links, and inferred transitive paths. Pre-indexed for fast neighborhood traversal. Handles relationship queries the BM25 index cannot.

**COBWEB — Topic hierarchy**
> Hierarchical probabilistic clustering over chunk representations. Gives `P(chunk | cluster)` for every chunk. Incrementally updatable when corpus changes. O(log n) traversal at query time.

**BM25 + metadata — turbopuffer / Postgres**
> Keyword search on well-formed, pre-parsed chunks. Native metadata filtering on entity types, dates, confidence scores. Scoped to COBWEB clusters — search space pre-narrowed before BM25 executes.

**DSPy — Orchestration and optimization**
> Small LM (3–7B instruct) with typed DSPy signatures for entity extraction, routing, and result assembly. MIPROv2 or custom RL optimizer trained on labeled `(query, expected_entities, expected_results)` examples. 500–2k labeled examples sufficient to tune routing decisions.

---

**9. Build target section**

- Section header `Build target`: keep
- Replace current week-5 integration test with three milestones:

**Milestone 1 — Offline pipeline**
> Single domain corpus (e.g., 10k documents). GLiNER entity extraction, knowledge graph construction, COBWEB clustering, BM25 index — all materialized and hashed against source. Passing conditions: entity coverage > 90%, cluster coherence measurable, pipeline fully reproducible from source hash.

**Milestone 2 — Query runtime**
> DSPy modules wired to preprocessed corpus. Entity extraction → hierarchy traversal → scoped BM25 → result assembly. Target: < 500ms P95 on a single query. Passing conditions: end-to-end retrieval trace emitted, every retrieved fragment traceable to source chunk and cluster.

**Milestone 3 — Parallel approximate processing**
> Same pipeline, N parallel branches. Measure recall vs. compute budget. Passing condition: doubling parallel branches improves recall on held-out eval set. Publish recall@k curve vs. compute budget as a public benchmark artifact.

---

**10. References section**

- Keep: Gensyn REE (receipts remain relevant)
- Remove: BigBird, PEER, Nx, Ortex, ONNX, Erlang distribution — move to a footnote or separate architecture history page if preservation is needed
- Add the following entries:

| Label | Title | Link |
|---|---|---|
| GLiNER | Zero-shot named entity recognition | arxiv.org/abs/2311.08526 |
| DSPy | Declarative self-improving language programs | dspy.ai |
| COBWEB | Incremental concept formation | reference Fisher 1987 |
| turbopuffer | Fast search over large namespaces | turbopuffer.com |
| Neo4j | Graph database | neo4j.com |
| FalkorDB | Graph database for low-latency queries | falkordb.com |

---

**11. Footer**

- Change `OVIR.net — Opportunistic Verifiable Inference Runtime Network` to `OVIR.net — Offline Verified Inference Retrieval Network`
- Change scope line `Scope: repeated short-query / shared-long-context inference.` to `Scope: offline preprocessing of known corpora for fast, verified, composable retrieval.`

---

**What does not change.**

The color scheme, layout structure, font choices, GitHub and simulation links, and the five workload categories all stay. The trust model (receipts, verifiability, auditability) stays — it's now a structural property of the retrieval trace rather than a separate mechanism. The "bad fit" card stays, with updated copy. The overall page length and section count stays approximately the same, with the Corpus section as the one net addition.