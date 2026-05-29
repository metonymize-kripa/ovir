"""
main.py — Career transition API backed by FalkorDB.

Endpoints:
  GET  /api/search?q=<text>           Occupation search (full-text index)
  GET  /api/occupation/<code>         Single occupation + top competencies
  POST /api/transition                Competency delta between two occupations

Run:
    cd onet-falkor-playground/backend
    uv run uvicorn main:app --reload --port 8000
"""

import os
from typing import Optional

import numpy as np
from falkordb import FalkorDB
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="O*NET Transition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FALKOR_HOST = os.getenv("FALKORDB_HOST", "127.0.0.1")
FALKOR_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("ONET_GRAPH", "onet")

db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
g = db.select_graph(GRAPH_NAME)


# ─── Models ──────────────────────────────────────────────────────────────────

class Occupation(BaseModel):
    code: str
    title: str
    description: str
    job_zone: Optional[int] = None


class CompetencyItem(BaseModel):
    element_id: str
    name: str
    type: str  # Skill | Ability | Knowledge | WorkActivity
    im: float
    lv: float
    score: float  # im × lv


class CompetencyGap(BaseModel):
    element_id: str
    name: str
    type: str
    target_im: float
    target_lv: float
    target_score: float
    source_im: Optional[float] = None
    source_lv: Optional[float] = None
    source_score: Optional[float] = None
    delta: float  # target_score - source_score (= target_score for missing items)


class TechGap(BaseModel):
    commodity_code: str
    title: str
    hot_tech: bool
    in_demand: bool


class TransitionRequest(BaseModel):
    source_code: str
    target_code: str


class TransitionResult(BaseModel):
    source: Occupation
    target: Occupation
    similarity: float  # 0–1 cosine similarity over im×lv vectors
    missing: list[CompetencyGap]      # in target, absent in source
    deficient: list[CompetencyGap]    # in both, target score > source + threshold
    transferable: list[CompetencyGap] # in both, close scores
    tech_gaps: list[TechGap]          # tech tools target uses, source doesn't
    summary: dict


# ─── Graph helpers ───────────────────────────────────────────────────────────

DEFICIENCY_THRESHOLD = 3.0  # im×lv delta to classify as deficient vs. transferable

COMPETENCY_QUERIES = [
    ("Skill",        "HAS_SKILL"),
    ("Ability",      "HAS_ABILITY"),
    ("Knowledge",    "HAS_KNOWLEDGE"),
    ("WorkActivity", "HAS_WORK_ACTIVITY"),
]


def fetch_occupation(code: str) -> Optional[Occupation]:
    res = g.query(
        "MATCH (o:Occupation {code: $code}) RETURN o.code, o.title, o.description, o.job_zone",
        {"code": code},
    )
    if not res.result_set:
        return None
    r = res.result_set[0]
    return Occupation(
        code=r[0],
        title=r[1],
        description=r[2] or "",
        job_zone=r[3],
    )


def fetch_profile(code: str) -> list[CompetencyItem]:
    """
    Fetch all competency nodes reachable from an occupation via the four
    weighted relationship types. Returns one CompetencyItem per (type, element_id).
    """
    items = []
    for node_label, rel_type in COMPETENCY_QUERIES:
        res = g.query(
            f"""
            MATCH (o:Occupation {{code: $code}})-[r:{rel_type}]->(n:{node_label})
            RETURN n.element_id, n.name, r.im, r.lv, r.score
            """,
            {"code": code},
        )
        for row in res.result_set:
            eid, name, im, lv, score = row
            if im is None or lv is None:
                continue
            items.append(
                CompetencyItem(
                    element_id=eid,
                    name=name,
                    type=node_label,
                    im=float(im),
                    lv=float(lv),
                    score=float(score) if score is not None else float(im) * float(lv),
                )
            )

    # Fallback to sister occupations if no competencies found
    if not items and len(code) >= 7:
        prefix = code[:7]
        # Count total sister occupations under this prefix
        sisters_count_res = g.query(
            "MATCH (o:Occupation) WHERE o.code STARTS WITH $prefix AND o.code <> $code RETURN count(o)",
            {"prefix": prefix, "code": code}
        )
        total_sisters = sisters_count_res.result_set[0][0] if sisters_count_res.result_set else 1
        if total_sisters == 0:
            total_sisters = 1

        fallback_query = """
            MATCH (o:Occupation)-[r]->(n)
            WHERE o.code STARTS WITH $prefix AND o.code <> $code AND type(r) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
            RETURN labels(n)[0] AS type, n.element_id, n.name, sum(r.im), sum(r.lv), sum(r.score)
        """
        res = g.query(fallback_query, {"prefix": prefix, "code": code})
        for row in res.result_set:
            comp_type, eid, name, sum_im, sum_lv, sum_score = row
            if sum_im is None or sum_lv is None:
                continue

            # Zero-filled average over all sister occupations under this prefix
            avg_im = float(sum_im) / total_sisters
            avg_lv = float(sum_lv) / total_sisters
            avg_score = float(sum_score) / total_sisters

            items.append(
                CompetencyItem(
                    element_id=eid,
                    name=name,
                    type=comp_type,
                    im=round(avg_im, 4),
                    lv=round(avg_lv, 4),
                    score=round(avg_score, 4),
                )
            )

    return items


def fetch_tech(code: str) -> dict[str, TechGap]:
    """Return tech tools for an occupation keyed by commodity_code."""
    res = g.query(
        """
        MATCH (o:Occupation {code: $code})-[r:USES_TECH]->(t:Technology)
        RETURN t.commodity_code, t.title, t.hot_tech, t.in_demand
        """,
        {"code": code},
    )
    out = {}
    for row in res.result_set:
        cc, title, hot, demand = row
        out[cc] = TechGap(
            commodity_code=cc,
            title=title or "",
            hot_tech=bool(hot),
            in_demand=bool(demand),
        )

    # Fallback to sister occupations if no tech tools found
    if not out and len(code) >= 7:
        prefix = code[:7]
        res_fallback = g.query(
            """
            MATCH (o:Occupation)-[r:USES_TECH]->(t:Technology)
            WHERE o.code STARTS WITH $prefix AND o.code <> $code
            RETURN t.commodity_code, t.title, max(toInteger(r.hot_tech)), max(toInteger(r.in_demand))
            """,
            {"prefix": prefix, "code": code},
        )
        for row in res_fallback.result_set:
            cc, title, hot, demand = row
            out[cc] = TechGap(
                commodity_code=cc,
                title=title or "",
                hot_tech=bool(hot),
                in_demand=bool(demand),
            )

    return out


def cosine_similarity(a_items: list[CompetencyItem], b_items: list[CompetencyItem]) -> float:
    """
    Pure structural cosine similarity over im×lv score vectors.
    No embeddings — just the weighted competency overlap.
    """
    # Key: (node_label, element_id)
    a_map = {(x.type, x.element_id): x.score for x in a_items}
    b_map = {(x.type, x.element_id): x.score for x in b_items}
    all_keys = list(set(a_map) | set(b_map))

    if not all_keys:
        return 0.0

    a_vec = np.array([a_map.get(k, 0.0) for k in all_keys])
    b_vec = np.array([b_map.get(k, 0.0) for k in all_keys])

    norm_a = np.linalg.norm(a_vec)
    norm_b = np.linalg.norm(b_vec)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_vec, b_vec) / (norm_a * norm_b))


def compute_gap(
    source_items: list[CompetencyItem],
    target_items: list[CompetencyItem],
) -> tuple[list[CompetencyGap], list[CompetencyGap], list[CompetencyGap]]:
    """
    Partition target competencies into missing / deficient / transferable
    relative to source.
    """
    source_map = {(x.type, x.element_id): x for x in source_items}

    missing: list[CompetencyGap] = []
    deficient: list[CompetencyGap] = []
    transferable: list[CompetencyGap] = []

    for item in target_items:
        key = (item.type, item.element_id)
        src = source_map.get(key)

        # Competency is a negligible requirement in target role - automatically transferable
        if item.score < 8.0:
            transferable.append(
                CompetencyGap(
                    element_id=item.element_id,
                    name=item.name,
                    type=item.type,
                    target_im=item.im,
                    target_lv=item.lv,
                    target_score=item.score,
                    source_im=src.im if src else None,
                    source_lv=src.lv if src else None,
                    source_score=src.score if src else None,
                    delta=round(item.score - src.score, 4) if src else item.score,
                )
            )
            continue

        if src is None:
            missing.append(
                CompetencyGap(
                    element_id=item.element_id,
                    name=item.name,
                    type=item.type,
                    target_im=item.im,
                    target_lv=item.lv,
                    target_score=item.score,
                    delta=item.score,
                )
            )
        else:
            delta = item.score - src.score
            gap = CompetencyGap(
                element_id=item.element_id,
                name=item.name,
                type=item.type,
                target_im=item.im,
                target_lv=item.lv,
                target_score=item.score,
                source_im=src.im,
                source_lv=src.lv,
                source_score=src.score,
                delta=round(delta, 4),
            )
            if delta >= DEFICIENCY_THRESHOLD:
                deficient.append(gap)
            else:
                transferable.append(gap)

    # Sort descending by delta (largest gaps first)
    missing.sort(key=lambda x: x.target_score, reverse=True)
    deficient.sort(key=lambda x: x.delta, reverse=True)
    transferable.sort(key=lambda x: x.target_score, reverse=True)

    return missing, deficient, transferable


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/search")
async def search(q: str = "") -> list[Occupation]:
    """
    Search occupations by title. Uses FalkorDB full-text index when available,
    falls back to toLower CONTAINS scan (fine for 1k nodes).
    """
    if not q.strip():
        return []

    # Full-text search
    try:
        res = g.query(
            """
            CALL db.idx.fulltext.queryNodes('Occupation', $q) YIELD node
            RETURN node.code, node.title, node.description, node.job_zone
            LIMIT 20
            """,
            {"q": q + "*"},
        )
    except Exception:
        # Fallback: CONTAINS scan
        res = g.query(
            """
            MATCH (o:Occupation)
            WHERE toLower(o.title) CONTAINS toLower($q) OR o.code CONTAINS $q
            RETURN o.code, o.title, o.description, o.job_zone
            LIMIT 20
            """,
            {"q": q.lower()},
        )

    return [
        Occupation(code=r[0], title=r[1], description=r[2] or "", job_zone=r[3])
        for r in res.result_set
    ]


@app.get("/api/occupation/{code}")
async def get_occupation(code: str) -> dict:
    occ = fetch_occupation(code)
    if not occ:
        raise HTTPException(status_code=404, detail=f"Occupation {code} not found")

    profile = fetch_profile(code)
    top = sorted(profile, key=lambda x: x.score, reverse=True)[:10]

    return {
        "occupation": occ.model_dump(),
        "top_competencies": [c.model_dump() for c in top],
        "competency_count": len(profile),
    }


@app.post("/api/transition")
async def transition(req: TransitionRequest) -> TransitionResult:
    source_occ = fetch_occupation(req.source_code)
    target_occ = fetch_occupation(req.target_code)

    if not source_occ:
        raise HTTPException(status_code=404, detail=f"Source occupation {req.source_code} not found")
    if not target_occ:
        raise HTTPException(status_code=404, detail=f"Target occupation {req.target_code} not found")

    source_profile = fetch_profile(req.source_code)
    target_profile = fetch_profile(req.target_code)

    sim = cosine_similarity(source_profile, target_profile)
    missing, deficient, transferable = compute_gap(source_profile, target_profile)

    # Technology gap: tools in target not in source
    source_tech = fetch_tech(req.source_code)
    target_tech = fetch_tech(req.target_code)
    tech_gaps = [
        t for cc, t in target_tech.items()
        if cc not in source_tech
    ]
    # Sort: hot + in-demand first
    tech_gaps.sort(key=lambda x: (x.hot_tech, x.in_demand), reverse=True)

    return TransitionResult(
        source=source_occ,
        target=target_occ,
        similarity=round(sim, 4),
        missing=missing,
        deficient=deficient,
        transferable=transferable,
        tech_gaps=tech_gaps,
        summary={
            "missing_count": len(missing),
            "deficient_count": len(deficient),
            "transferable_count": len(transferable),
            "tech_gap_count": len(tech_gaps),
            "source_competency_count": len(source_profile),
            "target_competency_count": len(target_profile),
        },
    )


@app.get("/health")
async def health():
    try:
        res = g.query("MATCH (n:Occupation) RETURN count(n) AS c")
        occ_count = res.result_set[0][0]
        return {"status": "ok", "occupation_count": occ_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
