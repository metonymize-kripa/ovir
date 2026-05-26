"""
02_traversal.py — Multi-hop traversal and neighborhood queries.

This is the core OVIR capability: given an entity extracted from a user query,
find the neighborhood of chunks and related entities to scope the Solr search.

Covers:
  - Variable-length paths
  - Neighborhood expansion (1-hop, 2-hop)
  - Collecting adjacent nodes (the "scope" OVIR passes to Solr)
  - Path inspection
  - Shortest path

Run: python 02_traversal.py
"""

from falkordb import FalkorDB
from redis.exceptions import ResponseError

def reset_graph(db, name):
    try:
        db.select_graph(name).delete()
    except ResponseError:
        pass
    return db.select_graph(name)

db = FalkorDB(host="localhost", port=6379)
g = reset_graph(db, "ovir_traversal")

# ── Seed: a small knowledge graph about a fictional codebase ──────────────────
# Entities: services, engineers, concepts
# Relationships: DEPENDS_ON, OWNS, IMPLEMENTS, CO_OCCURS_IN

g.query("""
  CREATE
    (auth:Entity   {id: 'auth',    name: 'AuthService',     type: 'SERVICE'}),
    (token:Entity  {id: 'token',   name: 'TokenService',    type: 'SERVICE'}),
    (db:Entity     {id: 'db',      name: 'UserDB',          type: 'SERVICE'}),
    (cache:Entity  {id: 'cache',   name: 'RedisCache',      type: 'SERVICE'}),
    (api:Entity    {id: 'api',     name: 'APIGateway',      type: 'SERVICE'}),
    (alice:Entity  {id: 'alice',   name: 'Alice',           type: 'PERSON'}),
    (bob:Entity    {id: 'bob',     name: 'Bob',             type: 'PERSON'}),
    (jwt:Entity    {id: 'jwt',     name: 'JWT',             type: 'CONCEPT'}),
    (oauth:Entity  {id: 'oauth',   name: 'OAuth2',          type: 'CONCEPT'})
""")

g.query("""
  MATCH
    (auth:Entity {id:'auth'}), (token:Entity {id:'token'}),
    (db:Entity {id:'db'}),     (cache:Entity {id:'cache'}),
    (api:Entity {id:'api'}),   (alice:Entity {id:'alice'}),
    (bob:Entity {id:'bob'}),   (jwt:Entity {id:'jwt'}),
    (oauth:Entity {id:'oauth'})
  CREATE
    (api)-[:DEPENDS_ON]->(auth),
    (auth)-[:DEPENDS_ON]->(token),
    (auth)-[:DEPENDS_ON]->(db),
    (token)-[:DEPENDS_ON]->(cache),
    (alice)-[:OWNS]->(auth),
    (alice)-[:OWNS]->(token),
    (bob)-[:OWNS]->(db),
    (auth)-[:IMPLEMENTS]->(jwt),
    (auth)-[:IMPLEMENTS]->(oauth),
    (token)-[:IMPLEMENTS]->(jwt)
""")

# Add some chunks so we know what OVIR would retrieve
g.query("""
  CREATE
    (c1:Chunk {id:'c1', text:'AuthService validates tokens via TokenService', source:'runbook_001'}),
    (c2:Chunk {id:'c2', text:'AuthService owns JWT signing keys', source:'runbook_001'}),
    (c3:Chunk {id:'c3', text:'UserDB schema: users, sessions, roles', source:'schema_002'}),
    (c4:Chunk {id:'c4', text:'RedisCache TTL policy for token storage', source:'runbook_003'}),
    (c5:Chunk {id:'c5', text:'APIGateway routes all traffic through AuthService', source:'arch_004'})
""")

g.query("""
  MATCH
    (auth:Entity {id:'auth'}), (token:Entity {id:'token'}),
    (db:Entity {id:'db'}),     (cache:Entity {id:'cache'}),
    (api:Entity {id:'api'})
  MATCH
    (c1:Chunk{id:'c1'}),(c2:Chunk{id:'c2'}),(c3:Chunk{id:'c3'}),
    (c4:Chunk{id:'c4'}),(c5:Chunk{id:'c5'})
  CREATE
    (c1)-[:MENTIONS]->(auth), (c1)-[:MENTIONS]->(token),
    (c2)-[:MENTIONS]->(auth), (c2)-[:MENTIONS]->(token),
    (c3)-[:MENTIONS]->(db),
    (c4)-[:MENTIONS]->(cache), (c4)-[:MENTIONS]->(token),
    (c5)-[:MENTIONS]->(api), (c5)-[:MENTIONS]->(auth)
""")

print("Graph seeded.\n")

# ── 1. Direct chunk neighborhood ──────────────────────────────────────────────
# Query: "How does AuthService work?"
# Step 1: find the entity node for AuthService.
# Step 2: find all chunks that mention it.
# This is the 1-hop scope OVIR sends to Solr.

print("=== Chunks mentioning AuthService (1-hop, OVIR's Solr scope) ===")
result = g.query("""
  MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: 'AuthService'})
  RETURN c.id, c.source, c.text
  ORDER BY c.id
""")
for row in result.result_set:
    print(f"  [{row[0]}] ({row[1]}) {row[2]}")

# ── 2. Expand scope: entity neighborhood ──────────────────────────────────────
# Query: "What does AuthService depend on?"
# 2-hop: AuthService → DEPENDS_ON → * → also get chunks for those entities.

print("\n=== 2-hop: entities reachable from AuthService via DEPENDS_ON ===")
result = g.query("""
  MATCH (auth:Entity {name: 'AuthService'})-[:DEPENDS_ON*1..2]->(dep:Entity)
  RETURN DISTINCT dep.name, dep.type
  ORDER BY dep.name
""")
for row in result.result_set:
    print(f"  {row[1]}: {row[0]}")

# ── 3. Collect the full chunk scope for a 2-hop query ─────────────────────────
# This is what OVIR would pass to Solr as a filter: chunk IDs or source docs.

print("\n=== Chunks in 2-hop neighborhood of AuthService (Solr scope) ===")
result = g.query("""
  MATCH (auth:Entity {name: 'AuthService'})-[:DEPENDS_ON*1..2]->(dep:Entity)
  MATCH (c:Chunk)-[:MENTIONS]->(dep)
  RETURN DISTINCT c.id, c.source, c.text
  ORDER BY c.id
""")
for row in result.result_set:
    print(f"  [{row[0]}] ({row[1]}) {row[2]}")

# ── 4. Ownership traversal ─────────────────────────────────────────────────────
# Query: "Who do I contact about JWT issues?"
# Path: JWT ← IMPLEMENTS ← * ← OWNS ← Person

print("\n=== Who owns something that implements JWT? ===")
result = g.query("""
  MATCH (person:Entity {type: 'PERSON'})-[:OWNS]->(svc:Entity)-[:IMPLEMENTS]->(jwt:Entity {name: 'JWT'})
  RETURN person.name, svc.name
""")
for row in result.result_set:
    print(f"  {row[0]} owns {row[1]}")

# ── 5. Any-direction traversal ────────────────────────────────────────────────
# Sometimes you want the full entity neighborhood regardless of edge direction.

print("\n=== All entities within 1 hop of TokenService (any direction) ===")
result = g.query("""
  MATCH (token:Entity {name: 'TokenService'})-[r]-(neighbor:Entity)
  RETURN DISTINCT neighbor.name, neighbor.type, type(r) AS rel
  ORDER BY neighbor.name
""")
for row in result.result_set:
    print(f"  {row[1]}: {row[0]}  (via {row[2]})")

# ── 6. Shortest path ──────────────────────────────────────────────────────────
# How is APIGateway connected to RedisCache?

print("\n=== Shortest path: APIGateway → RedisCache ===")
result = g.query("""
  MATCH p = shortestPath(
    (src:Entity {name: 'APIGateway'})-[*]->(dst:Entity {name: 'RedisCache'})
  )
  RETURN [n IN nodes(p) | n.name] AS path
""")
for row in result.result_set:
    print("  " + " → ".join(row[0]))

print("\nDone. Move on to 03_ovir_pattern.py")
