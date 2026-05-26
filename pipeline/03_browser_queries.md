# FalkorDB Browser Queries — CFPB Corpus

Open **http://localhost:3000**, connect to `localhost:6379`, select the `cfpb_corpus` graph.
Paste any query into the editor and hit Run. Results render as an interactive graph.

---

## Orientation

```cypher
// How many nodes and edges?
MATCH (n) RETURN labels(n) AS type, count(n) AS count

MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count
```

---

## Company graphs

```cypher
// Everything connected to Wells Fargo (1 hop)
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {id: 'wells_fargo'})
RETURN c, r, e
```

```cypher
// All chunks mentioning any of the top banks
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
WHERE e.id IN ['wells_fargo', 'citibank', 'discover', 'bank_of_america', 'chase']
RETURN c, r, e
LIMIT 80
```

```cypher
// Two-hop: chunks → entities → other chunks sharing those entities
MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
WHERE c1.company = 'Wells Fargo' AND c1 <> c2
RETURN c1, e, c2
LIMIT 50
```

---

## Entity type views

```cypher
// All monetary amounts and the chunks they appear in
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {label: 'monetary_amount'})
RETURN c, r, e
LIMIT 60
```

```cypher
// All organizations extracted by GLiNER
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {label: 'organization'})
RETURN c, r, e
LIMIT 80
```

```cypher
// Financial products mentioned across chunks
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {label: 'financial_product'})
RETURN c, r, e
LIMIT 60
```

---

## Most connected entities (hub view)

```cypher
// Top 15 entities by mention count — good starting nodes to explore
MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
WITH e, count(c) AS mentions
ORDER BY mentions DESC
LIMIT 15
MATCH (c:Chunk)-[r:MENTIONS]->(e)
RETURN c, r, e
LIMIT 100
```

---

## Entity co-occurrence (implicit relationships)

```cypher
// Entities that appear in the same chunk as Wells Fargo
MATCH (c:Chunk)-[:MENTIONS]->(hub:Entity {id: 'wells_fargo'})
MATCH (c)-[r:MENTIONS]->(neighbor:Entity)
WHERE neighbor <> hub
RETURN c, r, neighbor
LIMIT 60
```

```cypher
// All co-occurring entity pairs (dense — use with LIMIT)
MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
WHERE id(e1) < id(e2)
RETURN e1, c, e2
LIMIT 80
```

---

## Product-based views

```cypher
// All student loan complaints and their entities
MATCH (c:Chunk {product: 'Student loan'})-[r:MENTIONS]->(e:Entity)
RETURN c, r, e
LIMIT 60
```

```cypher
// Mortgage chunks only
MATCH (c:Chunk {product: 'Mortgage'})-[r:MENTIONS]->(e:Entity)
RETURN c, r, e
LIMIT 60
```

---

## Traversal depth comparison

```cypher
// 1-hop from a specific chunk
MATCH (c:Chunk {id: 'cfpb_100000'})-[r:MENTIONS]->(e:Entity)
RETURN c, r, e
```

```cypher
// 2-hop: chunks that share entities with cfpb_100000
MATCH (c1:Chunk {id: 'cfpb_100000'})-[:MENTIONS]->(e:Entity)<-[r:MENTIONS]-(c2:Chunk)
RETURN c1, e, r, c2
LIMIT 40
```

---

## Tips for the browser UI

- **Node colours** are assigned by label automatically — Chunk nodes and Entity nodes will render in different colours.
- **Hover** over a node to see all its properties.
- **Double-click** a node to expand its neighbours.
- If the graph is a hairball, add `LIMIT 50` and increase gradually.
- The browser saves query history — use the up arrow to recall previous queries.
