# OVIR.net: Scaling Assessment — Opportunities & Risks

This document provides a critical assessment of the current OVIR (Offline Verified Inference Retrieval) codebase as it scales from toy examples to the demanding production use cases defined in the project vision (Data rooms, Stable archives, Large code repositories, Policy corpora, Literature sets).

## 1. Opportunities for Scaling

### 1.1 Deep Specialization via the Outer Loop
The separation of the "outer" preprocessing loop and the "inner" query loop is the codebase's strongest architectural asset. For demanding use cases like **Data rooms** or **Policy corpora**, you can invest massive asynchronous compute (e.g., using larger models like Qwen 72B or even GPT-4o offline) to build the graph and synthesize evaluation sets without impacting the sub-500ms latency requirement of the runtime. 

### 1.2 Graceful Degradation & Approximate Computing
The implementation of COBWEB (topic hierarchy) and FalkorDB (graph traversals) introduces explicit "hop depth" and "scope" dials (`top_k` and `k`). As the corpus scales to millions of chunks (e.g., **Large repositories**), the system can natively limit search space to high-probability clusters. This avoids the O(N) scaling costs of brute-force vector search and allows the runtime to gracefully trade compute for recall.

### 1.3 Auditability and Receipt Chains
For **Compliance** and **Scientific Review**, the ability to emit a deterministic retrieval trace (entities extracted &rarr; clusters explored &rarr; chunks retrieved &rarr; confidence scores) provides a significant competitive advantage over black-box RAG pipelines.

---

## 2. Technical Risks & Bottlenecks

While the architectural pattern is sound, the current Python implementation (`ovir/offline/pipeline.py` and `ovir/runtime/query.py`) contains several bottlenecks that will break under load.

### 2.1 In-Memory Preprocessing Limits
**Risk**: The current `pipeline.py` loads the entire corpus into memory (`texts = [c["text"] for c in corpus]`), passes it synchronously to GLiNER, and holds all embeddings in memory to build the `CobwebRetriever`.
**Impact**: Processing a 10,000-page data room will cause out-of-memory (OOM) crashes. 
**Mitigation**: The offline pipeline must be refactored into a streaming, chunked architecture using generators. GLiNER extraction and Ollama embedding should happen in batched streams, persisting intermediate results to disk (e.g., JSONL or Parquet) before graph/Solr ingestion.

### 2.2 COBWEB Clustering Scalability
**Risk**: The `cobweb-language-embedding` library relies on iterative tree building. While COBWEB is incrementally updatable, Python-based tree construction over millions of 768-dimensional dense vectors will become a severe CPU and memory bottleneck.
**Impact**: The outer loop execution time will scale non-linearly, making frequent corpus updates impossible for fast-moving **Stable archives**.
**Mitigation**: Evaluate if the COBWEB implementation can be parallelized, backed by a vector database like Faiss for nearest-neighbor approximations during tree insertion, or replaced by a more scalable hierarchical clustering algorithm (e.g., HDBSCAN) if incremental updates aren't strictly required.

### 2.3 Single-Node FalkorDB Constraints
**Risk**: FalkorDB is an exceptionally fast, in-memory graph database. However, memory is finite. A massive corpus with dense entity co-occurrence edges (e.g., **Literature sets**) will balloon the graph size.
**Impact**: RAM exhaustion on the database node.
**Mitigation**: Implement edge pruning during the outer loop (only keeping high-confidence `MENTIONS` and explicitly defined relationships). Ensure the FalkorDB instance is properly tuned for persistence (AOF/RDB) to prevent data loss on restarts.

### 2.4 DSPy Prompt Brittleness at Scale
**Risk**: The `OVIRQueryModule` currently uses simple string interpolation for retrieved context. In dense corpora, the Solr Hybrid search might return highly fragmented context chunks.
**Impact**: The inner-loop LM (`gemma4:e4b` or `qwen` 3B) will suffer from "lost in the middle" syndrome or hallucinate connections between disjointed chunks.
**Mitigation**: Use DSPy to rigorously optimize the prompt structure for the specific inner-loop model, potentially utilizing strict JSON outputs or XML tagging to clearly demarcate distinct context chunks.

## 3. Summary Recommendation

To survive the transition to demanding use cases, the immediate engineering priority must be **streaming the offline pipeline**. The runtime (`query.py`) is structurally sound for scale (thanks to Solr and scoped graph traversals), but the pipeline (`pipeline.py`) will fail on the first realistic dataset unless chunk processing, embedding, and indexing are decoupled into batched, asynchronous worker queues.
