"""
02_search.py — Keyword search, filtering, and faceting.

This covers OVIR's core Solr usage: scoped search where FalkorDB
has already provided a list of cluster IDs or entity names to filter on.

Covers:
  - Full-text keyword search
  - Field filtering (fq — the OVIR scope filter)
  - Multi-value field filtering (entities)
  - Faceting (count by cluster, entity type)
  - Highlighting
  - Relevance boosting
  - Pagination with cursor

Run: uv run 02_search.py   (requires 01_schema_and_index.py to have run first)
"""

import pysolr

SOLR_URL = "http://localhost:8983/solr/ovir_corpus"
solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=10)


def show(results, label):
    print(f"\n=== {label} ({results.hits} hits) ===")
    for doc in results:
        ents = ", ".join(doc.get("entities", []))
        text = doc.get("chunk_text", "(not in fl)")
        print(f"  [{doc['chunk_id']}] cluster={doc.get('cluster_id','?')}  entities=[{ents}]")
        print(f"          {text[:80]}...")


# ── 1. Unscoped full-text search ──────────────────────────────────────────────
results = solr.search("liability", fl="chunk_id,cluster_id,chunk_text,entities")
show(results, "Full-text: 'liability'")

# ── 2. Scoped search — the core OVIR pattern ─────────────────────────────────
# FalkorDB provides cluster IDs; Solr filters to them before ranking.
# fq (filter query) does NOT affect relevance score — it just prunes candidates.
# This is equivalent to OVIR's "search space pre-narrowed before Solr executes."

results = solr.search(
    "fees payment",
    fq="cluster_id:contract_terms OR cluster_id:financial_obligations",
    fl="chunk_id,cluster_id,chunk_text,entities",
)
show(results, "Scoped to contract_terms + financial_obligations")

# ── 3. Entity filter — narrow to chunks mentioning a specific entity ──────────
# FalkorDB gives us entity names from traversal; Solr filters to those chunks.

results = solr.search(
    "*:*",
    fq='entities:"ACME Corp"',
    fl="chunk_id,cluster_id,chunk_text",
)
show(results, "Chunks mentioning ACME Corp")

# Multiple entities (OR): chunks mentioning any of these
results = solr.search(
    "*:*",
    fq='entities:("Globex" OR "Initech Holdings")',
    fl="chunk_id,cluster_id,chunk_text,entities",
)
show(results, "Chunks mentioning Globex OR Initech Holdings")

# ── 4. Combined: keyword + scope + entity ─────────────────────────────────────
# Full OVIR query pattern in one call.

results = solr.search(
    "revenue subsidiary",                                          # keyword
    fq=['cluster_id:org_structure', 'entity_types:ORG'],          # scope filters
    fl="chunk_id,source,chunk_text,entities,confidence",
    sort="confidence desc",
)
show(results, "keyword='revenue subsidiary' + scope=org_structure + entity_types=ORG")

# ── 5. Faceting — understand the corpus distribution ─────────────────────────
results = solr.search(
    "*:*",
    facet="on",
    **{"facet.field": ["cluster_id", "entity_types"], "facet.mincount": 1},
    rows=0,  # don't need docs, just facets
)
print("\n=== Facets ===")
for field, counts in results.facets.get("facet_fields", {}).items():
    pairs = [(counts[i], counts[i+1]) for i in range(0, len(counts), 2)]
    print(f"  {field}: " + "  ".join(f"{v}({n})" for v, n in pairs))

# ── 6. Highlighting — know which term matched ─────────────────────────────────
results = solr.search(
    "liability cap",
    hl="true",
    **{"hl.fl": "chunk_text", "hl.snippets": 1, "hl.fragsize": 100},
    fl="chunk_id,chunk_text",
)
print("\n=== Highlighting ===")
for doc in results:
    cid = doc["chunk_id"]
    snippets = results.highlighting.get(cid, {}).get("chunk_text", [])
    print(f"  [{cid}] {snippets[0] if snippets else '(no highlight)'}")

# ── 7. Date range filter ──────────────────────────────────────────────────────
results = solr.search(
    "*:*",
    fq="doc_date:[2024-01-01T00:00:00Z TO *]",
    fl="chunk_id,source,doc_date",
)
show(results, "Documents dated 2024 or later")

print("\nDone. Move on to 03_updates_and_management.py")
