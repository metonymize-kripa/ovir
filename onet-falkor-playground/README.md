# onet-falkor-playground

Career transition engine backed by FalkorDB. Computes structural competency deltas between O*NET occupations via GraphBLAS matrix traversal.

See `notes.md` for the full motivation and use case architecture.

---

## Prerequisites

- Docker (FalkorDB runs in the root `docker-compose.yml`)
- Python ≥ 3.11 + [uv](https://github.com/astral-sh/uv)
- Node.js ≥ 18

---

## 1. Start FalkorDB

From the repo root:

```bash
docker compose up falkordb -d
```

FalkorDB browser UI: http://localhost:3000  
Redis port: 6379

---

## 2. Load O*NET data

```bash
cd onet-falkor-playground/loader
uv run load_onet.py --fresh
```

Pass `--data-dir <path>` if your O*NET files are elsewhere. Default: `../../data/db_30_0_text`.

Takes ~2–4 minutes. Output ends with a sanity-check node/edge count.

To re-run without dropping the graph, omit `--fresh` (loader uses MERGE — idempotent).

---

## 3. Start the backend

```bash
cd onet-falkor-playground/backend
uv run uvicorn main:app --reload --port 8000
```

Health check: http://localhost:8000/health  
API docs: http://localhost:8000/docs

---

## 4. Start the frontend

```bash
cd onet-falkor-playground/frontend
npm install
npm run dev
```

Open http://localhost:3001

---

## Project layout

```
onet-falkor-playground/
├── notes.md                    # Architecture + motivation
├── README.md
├── loader/
│   ├── pyproject.toml
│   └── load_onet.py            # O*NET TSV → FalkorDB graph
├── backend/
│   ├── pyproject.toml
│   └── main.py                 # FastAPI: search + transition endpoints
└── frontend/
    ├── package.json
    ├── next.config.js           # Proxies /api/* to localhost:8000
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx             # Career transition explorer
    │   └── globals.css
    └── components/
        ├── OccupationSearch.tsx # Debounced typeahead
        └── TransitionView.tsx   # Gap table + bar chart
```

---

## Graph schema

```
(Occupation)-[:HAS_SKILL        {im, lv, score}]->(Skill)
(Occupation)-[:HAS_ABILITY      {im, lv, score}]->(Ability)
(Occupation)-[:HAS_KNOWLEDGE    {im, lv, score}]->(Knowledge)
(Occupation)-[:HAS_WORK_ACTIVITY{im, lv, score}]->(WorkActivity)
(Occupation)-[:USES_TECH        {hot_tech, in_demand}]->(Technology)
(Occupation)-[:RELATED_TO       {tier, rank}]->(Occupation)
```

`score = im × lv` — the combined importance × level weight. Cosine similarity over this vector space gives structural occupation distance with no embedding models involved.

---

## API

| Endpoint | Description |
|---|---|
| `GET /api/search?q=<text>` | Occupation typeahead (full-text index) |
| `GET /api/occupation/<code>` | Occupation details + top 10 competencies |
| `POST /api/transition` `{source_code, target_code}` | Full gap analysis |
| `GET /health` | DB connectivity + occupation count |

---

## Sample FalkorDB queries

Connect via redis-cli or the browser UI at http://localhost:3000:

```cypher
-- Occupation competency profile
MATCH (o:Occupation {code: "15-2051.00"})-[r:HAS_SKILL]->(s:Skill)
RETURN s.name, r.im, r.lv, r.score
ORDER BY r.score DESC LIMIT 10

-- Structural intersection between two occupations
MATCH (a:Occupation {code: "13-2011.00"})-[:HAS_SKILL]->(s:Skill)
MATCH (b:Occupation {code: "15-2051.00"})-[:HAS_SKILL]->(s)
RETURN s.name

-- Occupations sharing a specific skill at high level
MATCH (o:Occupation)-[r:HAS_SKILL]->(s:Skill {name: "Programming"})
WHERE r.lv > 5
RETURN o.code, o.title, r.im, r.lv
ORDER BY r.lv DESC

-- Find occupations most similar by shared technology
MATCH (a:Occupation {code: "15-2051.00"})-[:USES_TECH]->(t:Technology)
MATCH (b:Occupation)-[:USES_TECH]->(t)
WHERE b.code <> "15-2051.00"
RETURN b.code, b.title, count(t) AS shared_tech
ORDER BY shared_tech DESC LIMIT 10
```
