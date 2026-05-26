# FalkorDB Playground

Four scripts, meant to be run in order. Each builds on the last.

## Setup

```bash
# FalkorDB is already running (ports 6379 + 3000 are live)
# Browser UI: http://localhost:3000

# Run any script — uv resolves dependencies from pyproject.toml automatically
uv run 01_basics.py
```

## Scripts

**01_basics.py** — CRUD, MERGE, property filtering. Mental model: nodes are chunks and entities, edges are MENTIONS and semantic relationships.

**02_traversal.py** — Multi-hop traversal, neighborhood expansion, shortest path. This is the core FalkorDB capability OVIR depends on.

**03_ovir_pattern.py** — Full OVIR inner loop simulation. Offline corpus ingestion via `ingest_chunk()`, then query-time traversal that collects a chunk scope to pass to Solr. Watch how scope size changes with hop depth.

**04_index_and_performance.py** — Indexes, EXPLAIN, scale to 500 entities / 500 chunks. Measures real latency numbers for lookup and 1–2 hop traversal.

## Key concepts to internalize

**MERGE vs CREATE.** `MERGE` is idempotent — run the ingestion pipeline twice, get one copy of each node. Use it for entities. `CREATE` is faster for bulk inserts when you're sure you're not duplicating.

**Hop depth is the recall/compute dial.** 1-hop: only chunks directly mentioning the query entity. 2-hop: chunks mentioning the entity's neighbors too. Scope size grows fast. OVIR starts shallow and expands only when confidence is low.

**Chunk scope → Solr filter.** FalkorDB's job in OVIR is scope narrowing, not full-text search. The output of a traversal query is a set of chunk IDs or source document names. That set becomes a Solr filter. Solr handles the keyword matching; FalkorDB handles the relationship-aware scoping.

**Indexes must exist before bulk insert.** FalkorDB indexes entities on demand but pre-creating indexes before data load means every insert is indexed as it lands.

**EXPLAIN before you optimize.** `g.explain(query)` returns the query plan. Check it before assuming an index is being used.

## Cypher quick reference

```cypher
-- Create
CREATE (n:Label {prop: value})

-- Idempotent upsert
MERGE (n:Label {id: $id}) SET n.prop = $val

-- Simple lookup
MATCH (n:Label {id: $id}) RETURN n

-- Relationship traversal
MATCH (a)-[:REL]->(b) RETURN a, b

-- Variable-length path
MATCH (a)-[:REL*1..3]->(b) RETURN b

-- Any-direction
MATCH (a)-[:REL]-(b) RETURN b

-- Shortest path
MATCH p = shortestPath((a)-[*]->(b)) RETURN nodes(p)

-- Collect into list
MATCH (n:Entity) RETURN collect(n.id) AS ids

-- Count
MATCH (n:Chunk) RETURN count(n)

-- Delete graph
CALL db.dropGraph()   -- or via Python: g.delete()
```

## FalkorDB vs Neo4j

FalkorDB is a Redis module — graph lives in memory, persisted to RDB/AOF. Much lower latency than Neo4j for the in-memory case. Same Cypher dialect for most queries. Tradeoff: no enterprise features (auth, clustering, RBAC). Fine for OVIR's single-node corpus server.
