"""
01_ray_basics.py — Ray remote functions: sequential vs parallel.

Demonstrates the core Ray primitive: @ray.remote turns a function into
a distributed task. ray.get() blocks until all futures resolve.

Key fix: runtime_env={"excludes": [".venv/"]} prevents Ray from packaging
the local virtualenv into the worker bundle — without this, Ray tries to
reinstall all deps in each worker, which is slow and can fail.

Run: uv run 01_ray_basics.py
"""

import ray
import time

ray.init(runtime_env={"excludes": [".venv/", "pyproject.toml", "uv.lock"]})


def slow_task(item_id: int) -> str:
    time.sleep(1)  # Simulate slow work (embedding, extraction)
    return f"Processed item {item_id}"


@ray.remote
def ray_slow_task(item_id: int) -> str:
    time.sleep(1)
    return f"Processed item {item_id}"


print("=== Sequential (Slow) ===")
t0 = time.time()
results_seq = [slow_task(i) for i in range(8)]
print(f"  Sequential time: {time.time() - t0:.2f}s")

print("\n=== Parallel with Ray (Fast) ===")
t0 = time.time()
futures = [ray_slow_task.remote(i) for i in range(8)]
results_ray = ray.get(futures)
print(f"  Ray parallel time: {time.time() - t0:.2f}s")
print(f"  Speedup: {8 / (time.time() - t0 + 1e-9):.1f}x  (8 tasks, ~1s each, run in parallel)")

ray.shutdown()
