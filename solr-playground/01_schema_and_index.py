"""
01_schema_and_index.py — Schema definition and document indexing.

Solr 9 uses a managed schema. We add fields via the Schema API (REST),
then index documents via pysolr.

Covers:
  - Adding custom fields to the schema (text, string, int, float)
  - Indexing documents with pysolr
  - Commit vs softCommit
  - Verifying the index

Run: docker compose up -d   (waits ~10s for Solr to start)
     uv run 01_schema_and_index.py
"""

import pysolr
import requests
import time

SOLR_URL = "http://localhost:8983/solr/ovir_corpus"
SCHEMA_URL = f"{SOLR_URL}/schema"

solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=10)

# ── Wait for Solr to be ready ─────────────────────────────────────────────────
for attempt in range(15):
    try:
        solr.ping()
        print("Solr is up.")
        break
    except Exception:
        print(f"  Waiting for Solr... ({attempt+1}/15)")
        time.sleep(2)
else:
    raise RuntimeError("Solr did not start in time. Run: docker compose up -d")


# ── 1. Add schema fields ──────────────────────────────────────────────────────
# Solr ships with a managed schema. We extend it with OVIR-specific fields.
# Text fields use the built-in text_general analyzer (tokenized, lowercased).
# String fields are exact-match (good for IDs, entity names, facets).

def add_field(name, field_type, stored=True, indexed=True, multiValued=False):
    payload = {"add-field": {
        "name": name, "type": field_type,
        "stored": stored, "indexed": indexed, "multiValued": multiValued
    }}
    r = requests.post(SCHEMA_URL, json=payload)
    data = r.json()
    if "errors" in data:
        # Field already exists — fine
        if any("already exists" in str(e) for e in data["errors"]):
            return
        print(f"  Schema warning for {name}: {data['errors']}")

print("=== Adding schema fields ===")
add_field("chunk_id",     "string")       # unique doc ID
add_field("source",       "string")       # source filename
add_field("chunk_text",   "text_general") # full-text searchable
add_field("entities",     "string",  multiValued=True)  # entity names (filter)
add_field("entity_types", "string",  multiValued=True)  # ORG, PERSON, etc.
add_field("confidence",   "pfloat")       # GLiNER min confidence for this chunk
add_field("cluster_id",   "string")       # COBWEB cluster assignment
add_field("doc_date",     "pdate",  stored=True, indexed=True)
print("  Fields added (or already existed).")

# Route chunk_text into Solr's default _text_ field so bare keyword queries work.
r = requests.post(SCHEMA_URL, json={"add-copy-field": {"source": "chunk_text", "dest": "_text_"}})
data = r.json()
if "errors" in data and not any("already exists" in str(e) for e in data["errors"]):
    print(f"  copyField warning: {data['errors']}")


# ── 2. Index documents ────────────────────────────────────────────────────────
# Each doc is a dict. 'id' is Solr's required unique key.

print("\n=== Indexing documents ===")
docs = [
    {
        "id": "c1",
        "chunk_id": "c1",
        "source": "acme_msla_2024.pdf",
        "chunk_text": "ACME Corp agrees to pay Globex $2M annually under this MSA.",
        "entities": ["ACME Corp", "Globex"],
        "entity_types": ["ORG", "ORG"],
        "confidence": 0.97,
        "cluster_id": "financial_obligations",
        "doc_date": "2024-01-15T00:00:00Z",
    },
    {
        "id": "c2",
        "chunk_id": "c2",
        "source": "acme_msla_2024.pdf",
        "chunk_text": "Liability is capped at 12 months of fees paid by ACME Corp under Section 8.2.",
        "entities": ["ACME Corp"],
        "entity_types": ["ORG"],
        "confidence": 0.98,
        "cluster_id": "contract_terms",
        "doc_date": "2024-01-15T00:00:00Z",
    },
    {
        "id": "c3",
        "chunk_id": "c3",
        "source": "globex_vendor_reg.pdf",
        "chunk_text": "Globex is a wholly-owned subsidiary of Initech Holdings incorporated in Delaware.",
        "entities": ["Globex", "Initech Holdings"],
        "entity_types": ["ORG", "ORG"],
        "confidence": 0.99,
        "cluster_id": "org_structure",
        "doc_date": "2023-06-01T00:00:00Z",
    },
    {
        "id": "c4",
        "chunk_id": "c4",
        "source": "initech_annual_2023.pdf",
        "chunk_text": "Initech Holdings reported $800M revenue in FY2023. CEO is Bill Lumbergh.",
        "entities": ["Initech Holdings", "Bill Lumbergh"],
        "entity_types": ["ORG", "PERSON"],
        "confidence": 0.95,
        "cluster_id": "financial_obligations",
        "doc_date": "2023-12-31T00:00:00Z",
    },
    {
        "id": "c5",
        "chunk_id": "c5",
        "source": "acme_sow_001.pdf",
        "chunk_text": "Globex will deliver the data pipeline by 2024-06-30 per SOW-001 terms.",
        "entities": ["Globex"],
        "entity_types": ["ORG"],
        "confidence": 0.96,
        "cluster_id": "contract_terms",
        "doc_date": "2024-02-01T00:00:00Z",
    },
]

solr.add(docs)
solr.commit()
print(f"  Indexed {len(docs)} documents and committed.")

# Verify
count = solr.search("*:*", rows=0).hits
print(f"  Total docs in index: {count}")

print("\nDone. Move on to 02_search.py")
