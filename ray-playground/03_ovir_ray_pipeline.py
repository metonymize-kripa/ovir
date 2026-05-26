"""
03_ovir_ray_pipeline.py — Ray-parallelised OVIR outer loop simulation.

The OVIR outer loop has three sequential stages per chunk:
  1. GLiNER entity extraction  (~40ms/chunk, CPU-bound)
  2. nomic-embed-text embedding (~15ms/chunk, calls Ollama)
  3. Fan-out to FalkorDB, Solr, COBWEB (I/O-bound, can overlap)

Ray maps naturally onto this: stateless workers handle stages 1–2
in parallel; stateful Actors represent the three sinks and serialize
their own writes.

This script simulates the pipeline with sleep-based timing.
Replace the sleep blocks with real GLiNER + Ollama calls once
you're ready to run against the actual corpus.

Key fix: workers return actor futures; all futures are collected and
awaited before reading sink stats. No time.sleep() drain hacks.

Run: uv run 03_ovir_ray_pipeline.py
"""

import ray
import time
import random
from typing import List

ray.init(runtime_env={"excludes": [".venv/"]})


# ── Stateful sinks ────────────────────────────────────────────────────────────

@ray.remote
class FalkorDBActor:
    def __init__(self):
        self.nodes = 0
        self.edges = 0

    def merge_chunk(self, chunk_id: str, entities: list) -> str:
        self.nodes += 1 + len(entities)   # chunk node + entity nodes
        self.edges += len(entities)        # MENTIONS edges
        return chunk_id

    def get_stats(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}


@ray.remote
class SolrActor:
    def __init__(self):
        self.indexed = 0

    def index(self, chunk_id: str, text: str, vector: list) -> str:
        self.indexed += 1
        return chunk_id

    def get_stats(self) -> dict:
        return {"indexed": self.indexed}


@ray.remote
class CobwebActor:
    def __init__(self):
        self.leaves = 0

    def insert(self, chunk_id: str, embedding: list) -> str:
        time.sleep(0.02)  # Simulate COBWEB tree insertion cost
        self.leaves += 1
        return chunk_id

    def get_stats(self) -> dict:
        return {"leaves": self.leaves}


# ── Stateless extractor/embedder worker ───────────────────────────────────────

@ray.remote
def process_chunk(
    chunk: dict,
    falkor: "FalkorDBActor",
    solr: "SolrActor",
    cobweb: "CobwebActor",
) -> List:
    """
    Per-chunk outer loop work:
      1. Simulate GLiNER extraction
      2. Simulate Ollama embedding
      3. Fan out to all three sinks; return their futures
    """
    # 1. GLiNER — CPU-bound NER
    time.sleep(random.uniform(0.05, 0.15))
    entities = [f"ent_{random.randint(1, 20)}" for _ in range(random.randint(1, 3))]

    # 2. Ollama embed — network I/O
    time.sleep(random.uniform(0.01, 0.05))
    embedding = [random.random() for _ in range(768)]

    # 3. Fan out — collect futures for upstream awaiting
    f1 = falkor.merge_chunk.remote(chunk["id"], entities)
    f2 = solr.index.remote(chunk["id"], chunk["text"], embedding)
    f3 = cobweb.insert.remote(chunk["id"], embedding)
    return [f1, f2, f3]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    falkor = FalkorDBActor.remote()
    solr   = SolrActor.remote()
    cobweb = CobwebActor.remote()

    corpus = [{"id": f"c{i}", "text": f"Chunk text {i}"} for i in range(50)]

    print(f"=== OVIR outer loop: {len(corpus)} chunks ===")
    t0 = time.time()

    # Fan out all chunks in parallel; each returns a list of 3 sink futures
    worker_futures = [process_chunk.remote(doc, falkor, solr, cobweb) for doc in corpus]
    sink_future_lists = ray.get(worker_futures)   # resolve workers

    # Flatten and await all sink futures — guarantees all writes complete
    all_sink_futures = [f for futures in sink_future_lists for f in futures]
    ray.get(all_sink_futures)

    elapsed = time.time() - t0
    print(f"  Processed in {elapsed:.2f}s  ({len(corpus)/elapsed:.1f} chunks/s)")

    falkor_stats = ray.get(falkor.get_stats.remote())
    solr_stats   = ray.get(solr.get_stats.remote())
    cobweb_stats = ray.get(cobweb.get_stats.remote())

    print(f"\n=== Sink stats ===")
    print(f"  FalkorDB : {falkor_stats}")
    print(f"  Solr     : {solr_stats}")
    print(f"  COBWEB   : {cobweb_stats}")

    assert solr_stats["indexed"]  == 50, f"Solr: expected 50, got {solr_stats['indexed']}"
    assert cobweb_stats["leaves"] == 50, f"COBWEB: expected 50, got {cobweb_stats['leaves']}"
    print("\n  All assertions passed.")

    ray.shutdown()
