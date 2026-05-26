"""
03_updates_and_management.py — Atomic updates, deletes, and commit policy.

In OVIR, corpus updates happen when the outer loop re-runs.
Understanding Solr's commit model is important for keeping index state clean.

Covers:
  - Atomic field updates (update one field without re-indexing the full doc)
  - Soft delete vs hard delete
  - Commit policy (when does a search see the new data?)
  - Querying index stats

Run: uv run 03_updates_and_management.py
"""

import pysolr
import requests
import time

SOLR_URL = "http://localhost:8983/solr/ovir_corpus"
solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=10)

# ── 1. Atomic update — change one field without re-sending the whole doc ──────
# Use case: outer loop updates cluster assignment without re-extracting entities.
# Syntax: {"field": {"set": new_value}} or {"field": {"add": value}} for multiValued

print("=== Atomic update: reassign cluster_id ===")
before = solr.search("id:c4", fl="chunk_id,cluster_id").docs[0]
print(f"  Before: cluster_id={before['cluster_id']}")

solr.add([{"id": "c4", "cluster_id": {"set": "org_structure"}}])
solr.commit()

after = solr.search("id:c4", fl="chunk_id,cluster_id").docs[0]
print(f"  After:  cluster_id={after['cluster_id']}")

# ── 2. Add a new entity to a multi-valued field ───────────────────────────────
print("\n=== Atomic update: add entity to c3 ===")
before = solr.search("id:c3", fl="chunk_id,entities").docs[0]
print(f"  Before: entities={before['entities']}")

solr.add([{"id": "c3", "entities": {"add": "Delaware"}}])
solr.commit()

after = solr.search("id:c3", fl="chunk_id,entities").docs[0]
print(f"  After:  entities={after['entities']}")

# ── 3. Delete ─────────────────────────────────────────────────────────────────
print("\n=== Delete by ID ===")
solr.add([{"id": "c_temp", "chunk_text": "Temporary doc for delete demo.", "source": "test"}])
solr.commit()
count_before = solr.search("*:*", rows=0).hits
print(f"  Count before delete: {count_before}")

solr.delete(id="c_temp")
solr.commit()
count_after = solr.search("*:*", rows=0).hits
print(f"  Count after delete:  {count_after}")

# Delete by query — remove all docs from a source
solr.add([
    {"id": "stale_1", "source": "old_version.pdf", "chunk_text": "Old content A."},
    {"id": "stale_2", "source": "old_version.pdf", "chunk_text": "Old content B."},
])
solr.commit()
print(f"\n=== Delete by query (source=old_version.pdf) ===")
print(f"  Before: {solr.search('*:*', rows=0).hits} docs")
solr.delete(q='source:"old_version.pdf"')
solr.commit()
print(f"  After:  {solr.search('*:*', rows=0).hits} docs")

# ── 4. Commit policy ──────────────────────────────────────────────────────────
# hard commit: writes to disk, opens a new searcher — expensive but durable
# soft commit: makes docs visible to search, no disk flush — cheap, fast
# OVIR outer loop: hard commit once at end of each corpus prep run
# OVIR inner loop: never writes, so no commit needed

print("\n=== Commit timing ===")
# Add without committing — doc not visible yet
solr.add([{"id": "uncommitted", "chunk_text": "This doc is not committed yet.", "source": "test"}])
before = solr.search("id:uncommitted", rows=0).hits
print(f"  Hits before commit: {before}")

# Soft commit — visible but not durable
solr.commit(softCommit=True)
after_soft = solr.search("id:uncommitted", rows=0).hits
print(f"  Hits after softCommit: {after_soft}")

# Hard commit — durable
solr.commit()
after_hard = solr.search("id:uncommitted", rows=0).hits
print(f"  Hits after hard commit: {after_hard}")

solr.delete(id="uncommitted")
solr.commit()

# ── 5. Index stats ────────────────────────────────────────────────────────────
print("\n=== Index stats ===")
resp = requests.get(f"{SOLR_URL}/admin/luke?wt=json&numTerms=0")
luke = resp.json()
index = luke.get("index", {})
print(f"  numDocs    : {index.get('numDocs')}")
print(f"  maxDoc     : {index.get('maxDoc')}")
print(f"  segmentCount: {index.get('segmentCount')}")
print(f"  lastModified: {index.get('lastModified')}")

print("\nDone.")
