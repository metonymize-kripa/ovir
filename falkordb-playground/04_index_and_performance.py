"""
04_index_and_performance.py — Indexes, query plans, and scale behavior.

Covers:
  - Creating node property indexes
  - EXPLAIN / PROFILE (query plan inspection)
  - Scale: inserting 1000 entities and measuring traversal latency
  - The index payoff: unindexed vs. indexed lookup

Run: python 04_index_and_performance.py
"""

from falkordb import FalkorDB
from redis.exceptions import ResponseError
import time
import random
import string

def reset_graph(db, name):
    try:
        db.select_graph(name).delete()
    except ResponseError:
        pass
    return db.select_graph(name)

db = FalkorDB(host="localhost", port=6379)
g = reset_graph(db, "ovir_perf")


# ── 1. Create indexes before bulk insert ──────────────────────────────────────
# Rule: create indexes BEFORE inserting data. FalkorDB builds the index over
# existing nodes, but pre-creation means every insert is indexed as it lands.

print("=== Creating indexes ===")
g.query("CREATE INDEX FOR (e:Entity) ON (e.id)")
g.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
g.query("CREATE INDEX FOR (c:Chunk) ON (c.id)")
g.query("CREATE INDEX FOR (c:Chunk) ON (c.source)")
print("  Indexes created: Entity(id), Entity(name), Chunk(id), Chunk(source)")


# ── 2. Bulk insert ─────────────────────────────────────────────────────────────
# Insert N entities and chunks, connect randomly, measure time.

N_ENTITIES = 500
N_CHUNKS   = 500
EDGE_FANOUT = 3  # each chunk mentions ~3 entities

print(f"\n=== Inserting {N_ENTITIES} entities + {N_CHUNKS} chunks ===")

entity_ids = [f"ent_{i:04d}" for i in range(N_ENTITIES)]
chunk_ids  = [f"chk_{i:04d}" for i in range(N_CHUNKS)]

def rand_name():
    return "".join(random.choices(string.ascii_uppercase, k=1)) + \
           "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))

# Batch insert entities (one CREATE per batch for speed)
BATCH = 100
t0 = time.perf_counter()
for batch_start in range(0, N_ENTITIES, BATCH):
    batch = entity_ids[batch_start:batch_start + BATCH]
    params = {f"id_{i}": eid for i, eid in enumerate(batch)}
    params.update({f"name_{i}": rand_name() for i in range(len(batch))})
    params.update({f"type_{i}": random.choice(["ORG", "PERSON", "CONCEPT"]) for i in range(len(batch))})
    clauses = ", ".join(
        f"(:Entity {{id: $id_{i}, name: $name_{i}, type: $type_{i}}})"
        for i in range(len(batch))
    )
    g.query(f"CREATE {clauses}", params)

entity_insert_ms = (time.perf_counter() - t0) * 1000
print(f"  {N_ENTITIES} entities inserted in {entity_insert_ms:.0f}ms")

# Batch insert chunks
t0 = time.perf_counter()
for batch_start in range(0, N_CHUNKS, BATCH):
    batch = chunk_ids[batch_start:batch_start + BATCH]
    params = {f"id_{i}": cid for i, cid in enumerate(batch)}
    params.update({f"src_{i}": f"doc_{random.randint(0, 50):03d}.pdf" for i in range(len(batch))})
    params.update({f"text_{i}": f"Sample text {rand_name()} {rand_name()}" for i in range(len(batch))})
    clauses = ", ".join(
        f"(:Chunk {{id: $id_{i}, source: $src_{i}, text: $text_{i}}})"
        for i in range(len(batch))
    )
    g.query(f"CREATE {clauses}", params)

chunk_insert_ms = (time.perf_counter() - t0) * 1000
print(f"  {N_CHUNKS} chunks inserted in {chunk_insert_ms:.0f}ms")

# Random MENTIONS edges
t0 = time.perf_counter()
random.seed(42)
for cid in chunk_ids:
    mentioned = random.sample(entity_ids, EDGE_FANOUT)
    for eid in mentioned:
        g.query("""
          MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
          CREATE (c)-[:MENTIONS {confidence: $conf}]->(e)
        """, {"cid": cid, "eid": eid, "conf": round(random.uniform(0.8, 1.0), 2)})

edge_insert_ms = (time.perf_counter() - t0) * 1000
total_edges = N_CHUNKS * EDGE_FANOUT
print(f"  {total_edges} MENTIONS edges inserted in {edge_insert_ms:.0f}ms")

# Add a small chain of DEPENDS_ON between random entity pairs (for traversal tests)
chain_ids = random.sample(entity_ids, 20)
for i in range(len(chain_ids) - 1):
    g.query("""
      MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
      CREATE (a)-[:DEPENDS_ON]->(b)
    """, {"a": chain_ids[i], "b": chain_ids[i+1]})


# ── 3. Measure lookup latency (indexed) ───────────────────────────────────────

print("\n=== Indexed lookup latency (warm) ===")
sample_ids = random.sample(entity_ids, 10)
times = []
for eid in sample_ids:
    t0 = time.perf_counter()
    g.query("MATCH (e:Entity {id: $id}) RETURN e.name, e.type", {"id": eid})
    times.append((time.perf_counter() - t0) * 1000)

avg_ms = sum(times) / len(times)
print(f"  10 indexed MATCH lookups: avg={avg_ms:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")


# ── 4. Traversal latency at scale ─────────────────────────────────────────────

print("\n=== Traversal latency: 1-hop chunk scope ===")
times = []
for eid in random.sample(entity_ids, 10):
    t0 = time.perf_counter()
    result = g.query("""
      MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $eid})
      RETURN c.id
    """, {"eid": eid})
    times.append((time.perf_counter() - t0) * 1000)

avg_ms = sum(times) / len(times)
print(f"  10 chunk-neighborhood queries: avg={avg_ms:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")


print("\n=== Traversal latency: 2-hop entity neighborhood + chunk scope ===")
times = []
scope_sizes = []
for eid in random.sample(entity_ids, 10):
    t0 = time.perf_counter()
    result = g.query("""
      MATCH (root:Entity {id: $eid})-[:DEPENDS_ON*1..2]-(neighbor:Entity)
      WITH collect(DISTINCT neighbor.id) + [$eid] AS scope_ids
      MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
      WHERE e.id IN scope_ids
      RETURN DISTINCT c.id
    """, {"eid": eid})
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    scope_sizes.append(len(result.result_set))

avg_ms = sum(times) / len(times)
avg_scope = sum(scope_sizes) / len(scope_sizes)
print(f"  10 2-hop queries: avg={avg_ms:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")
print(f"  avg scope size: {avg_scope:.1f} chunks")


# ── 5. EXPLAIN: see the query plan ────────────────────────────────────────────
# FalkorDB's EXPLAIN shows what the planner will do before executing.

print("\n=== EXPLAIN: indexed lookup ===")
plan = g.explain("MATCH (e:Entity {id: 'ent_0001'}) RETURN e.name")
print(plan)


# ── 6. Node count check ───────────────────────────────────────────────────────

print("\n=== Graph stats ===")
r = g.query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
for row in r.result_set:
    print(f"  {row[0]}: {row[1]} nodes")
r = g.query("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC")
for row in r.result_set:
    print(f"  {row[0]}: {row[1]} edges")

print("\nDone. You now have enough FalkorDB intuition to implement the OVIR corpus prep layer.")
print("Key takeaways:")
print("  1. Index before bulk insert.")
print("  2. MERGE for idempotent ingestion; CREATE for bulk speed when you control uniqueness.")
print("  3. Variable-length paths (*1..N) are cheap at small N; benchmark your hop depth.")
print("  4. Chunk scope size grows fast with hop depth — that's the recall/compute tradeoff.")
