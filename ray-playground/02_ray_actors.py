"""
02_ray_actors.py — Ray Actors: stateful, long-lived workers.

Actors are the right primitive for OVIR's stateful outer-loop sinks:
the COBWEB tree accumulates state across all ingested chunks, so it
can't be a stateless remote function. One Actor instance holds the tree;
many parallel workers feed it.

Key fix over naive implementation: collect the futures returned by
actor.method.remote() and await them with ray.get() instead of
time.sleep() hacks. This guarantees the actor has processed all
messages before reading stats.

Run: uv run 02_ray_actors.py
"""

import ray
import time
import random

ray.init(runtime_env={"excludes": [".venv/", "pyproject.toml", "uv.lock"]})


# ── Stateful Actor — one instance, holds shared tree state ───────────────────

@ray.remote
class CobwebActor:
    def __init__(self):
        self.processed_chunks = 0
        self.tree_state: dict = {}

    def ingest(self, chunk_id: str, embedding: list) -> str:
        time.sleep(0.05)  # Simulate tree insertion cost
        self.processed_chunks += 1
        self.tree_state[chunk_id] = "inserted"
        return chunk_id

    def get_stats(self) -> dict:
        return {"processed": self.processed_chunks}


# ── Stateless worker — many instances, parallelised across corpus ─────────────

@ray.remote
def process_chunk(chunk_id: str, actor_handle) -> str:
    """Simulate embedding a chunk and inserting it into the central Actor."""
    time.sleep(random.uniform(0.1, 0.3))  # Simulate ML extraction + embed
    embedding = [random.random() for _ in range(768)]
    # Collect the actor future so the caller can await completion
    ingest_future = actor_handle.ingest.remote(chunk_id, embedding)
    return ingest_future  # return the actor future, not just chunk_id


print("=== Ray Actor: parallel ingestion into shared COBWEB tree ===")
cobweb = CobwebActor.remote()

t0 = time.time()
# Each worker returns an actor future; collect all of them
worker_futures = [process_chunk.remote(f"chunk_{i}", cobweb) for i in range(20)]
actor_futures = ray.get(worker_futures)   # resolve worker tasks first
ray.get(actor_futures)                    # then resolve actor ingestion calls
print(f"  20 chunks ingested in {time.time() - t0:.2f}s")

stats = ray.get(cobweb.get_stats.remote())
print(f"  Actor stats: {stats}")
assert stats["processed"] == 20, f"Expected 20, got {stats['processed']}"
print("  All chunks confirmed in actor state.")

ray.shutdown()
