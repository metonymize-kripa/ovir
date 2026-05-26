"""
01_basics.py — Create nodes, relationships, and run simple lookups.

Covers:
  - Connecting to FalkorDB
  - CREATE / MERGE / MATCH / DELETE
  - Node labels and relationship types
  - Basic property filtering

Run: python 01_basics.py
"""

from falkordb import FalkorDB
from redis.exceptions import ResponseError

def reset_graph(db, name):
    """Delete graph if it exists, then return a fresh handle."""
    try:
        db.select_graph(name).delete()
    except ResponseError:
        pass  # graph didn't exist yet — fine
    return db.select_graph(name)

db = FalkorDB(host="localhost", port=6379)

# ── Clean slate ───────────────────────────────────────────────────────────────
g = reset_graph(db, "ovir_basics")

# ── 1. Create nodes ───────────────────────────────────────────────────────────
# OVIR mental model: chunks are nodes; entities are nodes; relationships are edges.
# Start simple: a few document chunks and the entities GLiNER would extract.

g.query("""
  CREATE
    (c1:Chunk {id: 'c1', text: 'Apple acquired Beats Electronics in 2014.', source: 'doc_001'}),
    (c2:Chunk {id: 'c2', text: 'Beats was founded by Dr. Dre and Jimmy Iovine.', source: 'doc_001'}),
    (c3:Chunk {id: 'c3', text: 'Apple reported $383B revenue in FY2023.', source: 'doc_002'}),
    (e1:Entity {id: 'e_apple',   name: 'Apple',           type: 'ORG'}),
    (e2:Entity {id: 'e_beats',   name: 'Beats Electronics', type: 'ORG'}),
    (e3:Entity {id: 'e_dre',     name: 'Dr. Dre',          type: 'PERSON'}),
    (e4:Entity {id: 'e_iovine',  name: 'Jimmy Iovine',     type: 'PERSON'})
""")

# ── 2. Create relationships ───────────────────────────────────────────────────
# MENTIONS: chunk → entity (GLiNER output)
# ACQUIRED: entity → entity (inferred or extracted)
# FOUNDED_BY: entity → entity

g.query("""
  MATCH (c1:Chunk {id: 'c1'}), (c2:Chunk {id: 'c2'}), (c3:Chunk {id: 'c3'})
  MATCH (apple:Entity {id: 'e_apple'}), (beats:Entity {id: 'e_beats'})
  MATCH (dre:Entity {id: 'e_dre'}), (iovine:Entity {id: 'e_iovine'})
  CREATE
    (c1)-[:MENTIONS {confidence: 0.97}]->(apple),
    (c1)-[:MENTIONS {confidence: 0.95}]->(beats),
    (c2)-[:MENTIONS {confidence: 0.98}]->(beats),
    (c2)-[:MENTIONS {confidence: 0.96}]->(dre),
    (c2)-[:MENTIONS {confidence: 0.91}]->(iovine),
    (c3)-[:MENTIONS {confidence: 0.99}]->(apple),
    (apple)-[:ACQUIRED {year: 2014}]->(beats),
    (beats)-[:FOUNDED_BY]->(dre),
    (beats)-[:FOUNDED_BY]->(iovine)
""")

print("Graph created.\n")

# ── 3. Basic lookups ──────────────────────────────────────────────────────────

print("=== All entities ===")
result = g.query("MATCH (e:Entity) RETURN e.name, e.type ORDER BY e.type, e.name")
for row in result.result_set:
    print(f"  {row[1]}: {row[0]}")

print("\n=== Chunks mentioning Apple ===")
result = g.query("""
  MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: 'Apple'})
  RETURN c.id, c.text
""")
for row in result.result_set:
    print(f"  [{row[0]}] {row[1]}")

print("\n=== High-confidence mentions (>0.95) ===")
result = g.query("""
  MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
  WHERE m.confidence > 0.95
  RETURN c.id, e.name, m.confidence
  ORDER BY m.confidence DESC
""")
for row in result.result_set:
    print(f"  {row[0]} → {row[1]}  (conf={row[2]:.2f})")

# ── 4. Relationship traversal ─────────────────────────────────────────────────
print("\n=== Who founded Beats? (1-hop traversal) ===")
result = g.query("""
  MATCH (beats:Entity {name: 'Beats Electronics'})<-[:FOUNDED_BY]-(person:Entity)
  RETURN person.name
""")
# Note: direction is BEATS -[:FOUNDED_BY]-> PERSON in our schema
# Let's correct: we created (beats)-[:FOUNDED_BY]->(dre), so:
result = g.query("""
  MATCH (org:Entity {name: 'Beats Electronics'})-[:FOUNDED_BY]->(person:Entity)
  RETURN person.name
""")
for row in result.result_set:
    print(f"  {row[0]}")

print("\n=== What did Apple acquire? (entity → entity) ===")
result = g.query("""
  MATCH (apple:Entity {name: 'Apple'})-[r:ACQUIRED]->(target:Entity)
  RETURN target.name, r.year
""")
for row in result.result_set:
    print(f"  {row[0]} (year={row[1]})")

# ── 5. Key concept: MERGE (idempotent upsert) ─────────────────────────────────
# MERGE is the workhorse for corpus ingestion — run the pipeline twice, no duplicates.
print("\n=== MERGE demo (re-adding Apple is a no-op) ===")
before = g.query("MATCH (e:Entity {name: 'Apple'}) RETURN count(e)").result_set[0][0]
g.query("MERGE (e:Entity {id: 'e_apple', name: 'Apple', type: 'ORG'})")
after  = g.query("MATCH (e:Entity {name: 'Apple'}) RETURN count(e)").result_set[0][0]
print(f"  Apple nodes before: {before}, after MERGE: {after}  (idempotent ✓)")

print("\nDone. Move on to 02_traversal.py")
