# Solr Playground

pysolr 3.11.0 against Solr 10.0.0.

## Setup

```bash
docker compose up -d   # starts Solr on :8983, creates ovir_corpus collection
# Wait ~10s for Solr to initialize, then:
uv run 01_schema_and_index.py
uv run 02_search.py
uv run 03_updates_and_management.py

# Admin UI: http://localhost:8983/solr
```

## Scripts

**01_schema_and_index.py** — Schema API (add fields), `solr.add()`, `solr.commit()`. Defines the OVIR field set: chunk_id, source, chunk_text, entities (multi), entity_types (multi), confidence, cluster_id.

**02_search.py** — Full-text search, `fq` scope filters, entity filtering, faceting, highlighting, date range. The `fq` pattern (FalkorDB scope → Solr filter) is the core OVIR search call.

**03_updates_and_management.py** — Atomic updates (change one field), delete by ID/query, soft vs hard commit, index stats.

## Known issues / gotchas

**copyField required for bare keyword queries.** Solr's default search field is `_text_`, not `chunk_text`. Custom fields don't get copyField'd automatically. `01_schema_and_index.py` adds `copyField chunk_text → _text_` so bare keyword queries work. Without it, full-text searches return 0 hits.

**Solr 10 upgrade.** Both Solr playgrounds moved from `solr:9` to `solr:10.0.0`. No API changes for the features used here — schema API, `fq`, `DenseVectorField`, and pysolr are unchanged.

## Key concepts

**fq (filter query)** is the most important Solr primitive for OVIR. It prunes the candidate set before relevance scoring — no score impact, just fast boolean filtering. This is how FalkorDB's cluster and entity scope becomes a Solr constraint.

**text_general vs string fields.** `chunk_text` is text_general: tokenized, lowercased, stemmed — good for keyword search. `cluster_id` and `entities` are string: exact-match only — good for filter queries and facets. Never full-text search on a string field; never facet on a text field.

**Commit policy.** Hard commit is expensive (opens a new Lucene searcher). In OVIR: commit once at the end of each outer-loop run. Don't commit per document. `softCommit=True` is useful for making docs visible immediately during development.

**Atomic updates.** You don't need to re-index a full document to update one field. Use `{"field": {"set": value}}`. Critical for OVIR's outer loop which may update cluster assignments without re-extracting entities.

## OVIR query pattern

```python
results = solr.search(
    query_keywords,                          # from user query
    fq=[
        f"cluster_id:({' OR '.join(cluster_ids)})",   # from FalkorDB traversal
        f"entities:({' OR '.join(entity_names)})",    # from DSPy entity extraction
    ],
    fl="chunk_id,source,chunk_text,confidence",
    sort="confidence desc",
    rows=20,
)
```
