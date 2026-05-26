"""
03_ovir_pattern.py — The actual OVIR query pattern end-to-end.

Simulates exactly what happens at query time in the OVIR inner loop:
  1. Entity extracted from user query (simulating DSPy output)
  2. Graph traversal to find entity neighborhood
  3. Collect chunk IDs in scope
  4. (Would pass chunk IDs to Solr — here we just print them)
  5. Emit retrieval trace

Also shows the offline corpus prep step: how you'd ingest a new document.

Run: python 03_ovir_pattern.py
"""

from falkordb import FalkorDB
from redis.exceptions import ResponseError
import time
import hashlib
import json

def reset_graph(db, name):
    try:
        db.select_graph(name).delete()
    except ResponseError:
        pass
    return db.select_graph(name)

db = FalkorDB(host="localhost", port=6379)
g = reset_graph(db, "ovir_corpus")


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE: corpus prep (runs once per corpus version)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_chunk(chunk_id, text, source, entities):
    """
    Ingest one chunk into the graph.
    entities: list of {name, type, confidence}

    In production: GLiNER produces this entity list.
    Here we pass it manually to keep the script self-contained.
    """
    # Upsert chunk
    g.query(
        "MERGE (c:Chunk {id: $id}) SET c.text = $text, c.source = $source, c.hash = $hash",
        {"id": chunk_id, "text": text, "source": source,
         "hash": hashlib.md5(text.encode()).hexdigest()[:8]}
    )
    for ent in entities:
        ent_id = ent["name"].lower().replace(" ", "_")
        # Upsert entity
        g.query(
            "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type",
            {"id": ent_id, "name": ent["name"], "type": ent["type"]}
        )
        # Upsert MENTIONS edge with confidence
        g.query("""
          MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
          MERGE (c)-[m:MENTIONS]->(e)
          SET m.confidence = $conf
        """, {"cid": chunk_id, "eid": ent_id, "conf": ent["confidence"]})


def add_entity_relationship(from_name, rel_type, to_name, props=None):
    """Add a relationship between two entities (co-occurrence or explicit)."""
    from_id = from_name.lower().replace(" ", "_")
    to_id   = to_name.lower().replace(" ", "_")
    prop_str = ""
    if props:
        prop_str = " SET r." + ", r.".join(f"{k} = {repr(v)}" for k, v in props.items())
    g.query(f"""
      MATCH (a:Entity {{id: $from_id}}), (b:Entity {{id: $to_id}})
      MERGE (a)-[r:{rel_type}]->(b){prop_str}
    """, {"from_id": from_id, "to_id": to_id})


print("=== OFFLINE: Ingesting corpus ===")

# Simulated GLiNER output for a small contract corpus
corpus = [
    {
        "id": "c1", "source": "acme_msla_2024.pdf",
        "text": "ACME Corp agrees to pay Globex $2M annually under this MSA.",
        "entities": [
            {"name": "ACME Corp",  "type": "ORG",    "confidence": 0.97},
            {"name": "Globex",     "type": "ORG",    "confidence": 0.95},
        ]
    },
    {
        "id": "c2", "source": "acme_msla_2024.pdf",
        "text": "Liability is capped at 12 months of fees paid by ACME Corp.",
        "entities": [
            {"name": "ACME Corp",  "type": "ORG",    "confidence": 0.98},
        ]
    },
    {
        "id": "c3", "source": "acme_sow_001.pdf",
        "text": "Globex will deliver the data pipeline by 2024-06-30.",
        "entities": [
            {"name": "Globex",     "type": "ORG",    "confidence": 0.96},
        ]
    },
    {
        "id": "c4", "source": "globex_vendor_reg.pdf",
        "text": "Globex is a subsidiary of Initech Holdings.",
        "entities": [
            {"name": "Globex",           "type": "ORG", "confidence": 0.99},
            {"name": "Initech Holdings", "type": "ORG", "confidence": 0.94},
        ]
    },
    {
        "id": "c5", "source": "initech_annual_2023.pdf",
        "text": "Initech Holdings reported $800M revenue in FY2023. CEO: Bill Lumbergh.",
        "entities": [
            {"name": "Initech Holdings", "type": "ORG",    "confidence": 0.97},
            {"name": "Bill Lumbergh",    "type": "PERSON", "confidence": 0.93},
        ]
    },
]

for chunk in corpus:
    ingest_chunk(chunk["id"], chunk["text"], chunk["source"], chunk["entities"])
    print(f"  ingested {chunk['id']} ({chunk['source']})")

# Add inferred relationships (these would come from NLP or explicit source links)
add_entity_relationship("Globex", "SUBSIDIARY_OF", "Initech Holdings")
add_entity_relationship("ACME Corp", "CONTRACTED_WITH", "Globex", {"contract": "acme_msla_2024"})

print("Corpus ingested.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE: query-time inner loop
# ─────────────────────────────────────────────────────────────────────────────

def ovir_query(query_text, extracted_entity, hop_depth=1):
    """
    Simulate the OVIR inner loop for one query.

    In production:
      - extracted_entity comes from DSPy entity extraction module
      - chunk_ids would be passed to Solr for keyword search
      - confidence threshold gates result acceptance

    Returns a retrieval trace.
    """
    t0 = time.perf_counter()
    trace = {
        "query": query_text,
        "extracted_entity": extracted_entity,
        "hop_depth": hop_depth,
    }

    # Step 1: find entity node
    ent_id = extracted_entity.lower().replace(" ", "_")
    match_result = g.query(
        "MATCH (e:Entity {id: $id}) RETURN e.name, e.type",
        {"id": ent_id}
    )
    if not match_result.result_set:
        trace["status"] = "entity_not_found"
        return trace
    trace["entity_type"] = match_result.result_set[0][1]

    # Step 2: expand entity neighborhood
    neighbor_result = g.query(f"""
      MATCH (root:Entity {{id: $eid}})-[*1..{hop_depth}]-(neighbor:Entity)
      RETURN DISTINCT neighbor.id, neighbor.name, neighbor.type
    """, {"eid": ent_id})
    neighbor_ids = [r[0] for r in neighbor_result.result_set]
    all_entity_ids = [ent_id] + neighbor_ids
    trace["entity_neighborhood"] = [r[1] for r in neighbor_result.result_set]

    # Step 3: collect chunk scope
    # In production: these chunk IDs go to Solr as a filter.
    scope_result = g.query("""
      MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
      WHERE e.id IN $entity_ids AND m.confidence >= $min_conf
      RETURN DISTINCT c.id, c.source, c.text, m.confidence
      ORDER BY m.confidence DESC
    """, {"entity_ids": all_entity_ids, "min_conf": 0.90})

    chunks_in_scope = [
        {"chunk_id": r[0], "source": r[1], "text": r[2], "confidence": r[3]}
        for r in scope_result.result_set
    ]
    trace["chunks_in_scope"] = chunks_in_scope
    trace["scope_size"] = len(chunks_in_scope)
    trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    trace["status"] = "ok"

    return trace


# Run three queries that represent real OVIR workload shapes

queries = [
    {
        "text": "What are ACME Corp's payment obligations?",
        "entity": "ACME Corp",
        "hops": 1,
    },
    {
        "text": "Who controls Globex and what are its contracts?",
        "entity": "Globex",
        "hops": 2,  # needs to reach Initech via SUBSIDIARY_OF
    },
    {
        "text": "What is Initech Holdings' financial exposure through subsidiaries?",
        "entity": "Initech Holdings",
        "hops": 2,
    },
]

for q in queries:
    print(f"=== Query: {q['text']} ===")
    trace = ovir_query(q["text"], q["entity"], hop_depth=q["hops"])
    print(f"  entity       : {trace['extracted_entity']} ({trace.get('entity_type', '?')})")
    print(f"  neighborhood : {trace.get('entity_neighborhood', [])}")
    print(f"  scope_size   : {trace['scope_size']} chunks")
    print(f"  latency      : {trace['latency_ms']}ms")
    print(f"  chunks:")
    for c in trace["chunks_in_scope"]:
        print(f"    [{c['chunk_id']}] (conf={c['confidence']:.2f}) {c['text'][:60]}...")
    print()

# ── Observation: 2-hop expands scope significantly ────────────────────────────
print("=== Scope size vs. hop depth for 'Globex' ===")
for depth in [1, 2, 3]:
    t = ovir_query("", "Globex", hop_depth=depth)
    print(f"  depth={depth}: {t['scope_size']} chunks in scope")

print("\nKey insight: hop depth is a tunable compute/recall tradeoff.")
print("OVIR's approximate query processing = start shallow, expand if confidence is low.")
print("\nDone. Move on to 04_index_and_performance.py")
