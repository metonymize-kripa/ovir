"""
02_batch_corpus.py — Batch corpus processing and FalkorDB output format.

This is the OVIR offline pipeline step: run GLiNER over every chunk in
the corpus, then format the output for FalkorDB ingestion.

Covers:
  - Batch prediction (GLiNER's native batch API)
  - Producing the entity annotation format FalkorDB expects
  - Deduplicating entities across chunks
  - Writing output to JSONL for inspection

Run: uv run 02_batch_corpus.py
"""

from gliner import GLiNER
import json
import time
from collections import defaultdict

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# ── OVIR domain labels ────────────────────────────────────────────────────────
LABELS = [
    "organization",
    "person",
    "location",
    "date",
    "monetary_amount",
    "product",
    "technology",
]

# ── Simulated corpus (would be loaded from parsed PDFs/docs in production) ───
corpus = [
    {"id": "c1",  "source": "acme_msla_2024.pdf",     "text": "ACME Corp agrees to pay Globex $2M annually under this MSA."},
    {"id": "c2",  "source": "acme_msla_2024.pdf",     "text": "Liability is capped at 12 months of fees paid by ACME Corp under Section 8.2."},
    {"id": "c3",  "source": "acme_sow_001.pdf",       "text": "Globex will deliver the data pipeline by June 30, 2024."},
    {"id": "c4",  "source": "globex_vendor_reg.pdf",  "text": "Globex is a wholly-owned subsidiary of Initech Holdings incorporated in Delaware."},
    {"id": "c5",  "source": "initech_annual_2023.pdf","text": "Initech Holdings reported $800M revenue in FY2023. CEO is Bill Lumbergh."},
    {"id": "c6",  "source": "initech_annual_2023.pdf","text": "Initech Holdings operates in 12 countries including Germany, Japan, and Brazil."},
    {"id": "c7",  "source": "acme_msla_2024.pdf",     "text": "Either party may terminate upon 30 days written notice to the other party."},
    {"id": "c8",  "source": "tech_runbook.md",        "text": "AuthService depends on TokenService and UserDB for session management."},
    {"id": "c9",  "source": "tech_runbook.md",        "text": "TokenService uses Redis for caching JWT tokens with a 24-hour TTL."},
    {"id": "c10", "source": "tech_runbook.md",        "text": "Alice owns AuthService and TokenService. Bob owns UserDB and the Redis cluster."},
]


# ── 1. Batch prediction ───────────────────────────────────────────────────────
# GLiNER's batch API processes multiple texts in one forward pass.
# Much faster than calling predict_entities() in a loop.

print("=== Batch GLiNER extraction ===")
texts = [c["text"] for c in corpus]

t0 = time.perf_counter()
batch_results = model.batch_predict_entities(texts, LABELS, threshold=0.45)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"  {len(texts)} chunks processed in {elapsed_ms:.0f}ms ({elapsed_ms/len(texts):.1f}ms/chunk)")


# ── 2. Format for FalkorDB ─────────────────────────────────────────────────────
# Each annotated chunk becomes:
#   - A Chunk node (id, text, source)
#   - N Entity nodes (name, type, normalized_id)
#   - N MENTIONS edges (chunk → entity, with confidence)

annotated_chunks = []
entity_registry = {}  # entity_id → {name, type}

def normalize_entity_id(name: str) -> str:
    """Stable ID for deduplication across chunks."""
    return name.lower().strip().replace(" ", "_").replace(".", "")

for chunk, entities in zip(corpus, batch_results):
    chunk_entities = []
    for e in entities:
        eid = normalize_entity_id(e["text"])
        # Normalize GLiNER type label to uppercase short form
        etype = {
            "organization": "ORG",
            "person": "PERSON",
            "location": "LOCATION",
            "date": "DATE",
            "monetary_amount": "AMOUNT",
            "product": "PRODUCT",
            "technology": "TECH",
        }.get(e["label"], e["label"].upper())

        entity_registry[eid] = {"id": eid, "name": e["text"], "type": etype}
        chunk_entities.append({
            "entity_id": eid,
            "entity_name": e["text"],
            "entity_type": etype,
            "confidence": round(e["score"], 4),
            "span_start": e["start"],
            "span_end": e["end"],
        })

    annotated_chunks.append({
        "chunk_id": chunk["id"],
        "source": chunk["source"],
        "text": chunk["text"],
        "entities": chunk_entities,
    })


# ── 3. Print summary ──────────────────────────────────────────────────────────
print(f"\n  Unique entities found: {len(entity_registry)}")
print(f"  Entity registry:")
for eid, ent in sorted(entity_registry.items()):
    print(f"    [{ent['type']:10s}] {ent['name']}")

print(f"\n  Per-chunk annotations:")
for ac in annotated_chunks:
    ent_summary = ", ".join(f"{e['entity_name']}({e['confidence']:.2f})" for e in ac["entities"])
    print(f"  [{ac['chunk_id']}] {ent_summary or '(none)'}")


# ── 4. Write JSONL output ─────────────────────────────────────────────────────
# In the real pipeline: this feeds directly into FalkorDB ingestion.

output_path = "annotated_corpus.jsonl"
with open(output_path, "w") as f:
    for ac in annotated_chunks:
        f.write(json.dumps(ac) + "\n")
print(f"\n  Written to {output_path}")

entity_path = "entity_registry.json"
with open(entity_path, "w") as f:
    json.dump(entity_registry, f, indent=2)
print(f"  Written to {entity_path}")


# ── 5. Entity co-occurrence (seed for FalkorDB relationship inference) ────────
# Two entities that appear in the same chunk have a co-occurrence relationship.
# This is one of the heuristics OVIR uses to infer graph edges.

print("\n=== Co-occurrence pairs (→ FalkorDB edges) ===")
cooccurrence = defaultdict(int)
for ac in annotated_chunks:
    eids = [e["entity_id"] for e in ac["entities"]]
    for i in range(len(eids)):
        for j in range(i+1, len(eids)):
            pair = tuple(sorted([eids[i], eids[j]]))
            cooccurrence[pair] += 1

for (a, b), count in sorted(cooccurrence.items(), key=lambda x: -x[1]):
    na = entity_registry[a]["name"]
    nb = entity_registry[b]["name"]
    print(f"  {na} ↔ {nb}: {count} co-occurrence(s)")

print("\nDone.")
