# Ray Playground

Ray 2.x for parallelising OVIR's outer loop: GLiNER extraction and Ollama embedding are CPU/network-bound per-chunk operations that can run across all cores simultaneously.

## Setup

```bash
uv run 01_ray_basics.py
uv run 02_ray_actors.py
uv run 03_ovir_ray_pipeline.py
```

No Docker needed. Ray spins up a local cluster automatically.

## Scripts

**01_ray_basics.py** — `@ray.remote` function vs sequential loop. 8 tasks × 1s each: sequential = 8s, Ray parallel ≈ 1s. Demonstrates the core speedup primitive.

**02_ray_actors.py** — Stateful Actor holding a COBWEB tree. 20 parallel workers embed and ingest chunks; one Actor serializes tree insertions. Shows how to properly await actor futures instead of using `time.sleep()` drain hacks.

**03_ovir_ray_pipeline.py** — Full outer loop simulation: parallel workers run GLiNER + embed per chunk, fan out to three stateful sinks (FalkorDB, Solr, COBWEB). Assertions confirm all 50 chunks land in every sink.

## Key concepts

**`runtime_env={"excludes": [".venv/", "pyproject.toml", "uv.lock"]}`** — required on `ray.init()`. Without it Ray auto-packages the working directory, reads `pyproject.toml`, and reinstalls deps in every worker process. Excluding `pyproject.toml` and `uv.lock` prevents the per-worker venv setup entirely. Excluding `.venv/` keeps the bundle small.

**`@ray.remote` function** — stateless, parallelisable. Right primitive for per-chunk GLiNER extraction and Ollama embedding calls.

**`@ray.remote` class (Actor)** — stateful, serialised. Right primitive for COBWEB tree (accumulates state), FalkorDB writer (idempotent MERGE), Solr indexer (batched commits). One Actor per sink; Ray queues calls automatically.

**Future collection pattern** — actors called inside workers return futures. Workers return those futures to the caller; the caller does `ray.get()` on all of them. Never use `time.sleep()` to drain an actor queue.

```python
# Correct pattern
worker_futures = [process_chunk.remote(doc, actor) for doc in corpus]
sink_futures = ray.get(worker_futures)      # resolve workers
ray.get(sink_futures)                       # resolve actor writes
stats = ray.get(actor.get_stats.remote())   # now safe to read
```

## OVIR wiring

Ray is an outer loop accelerator only. It doesn't change the query-time architecture.

At corpus prep time: `ray.get([process_chunk.remote(doc) for doc in corpus])` replaces a sequential for-loop. Wall time drops from O(N × per-chunk latency) to O(per-chunk latency) across available cores.

At query time Ray adds no value — the inner loop is already fast (sub-100ms) and single-threaded.
