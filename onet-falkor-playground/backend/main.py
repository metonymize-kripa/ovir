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
import re
import math
import csv
import io
import time
from collections import Counter
from typing import Optional

import numpy as np
from falkordb import FalkorDB
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

def text_cosine_similarity(desc1: str, desc2: str) -> float:
    if not desc1 or not desc2:
        return 0.0
    words1 = re.findall(r'\w+', desc1.lower())
    words2 = re.findall(r'\w+', desc2.lower())
    
    c1 = Counter(words1)
    c2 = Counter(words2)
    
    all_words = set(c1.keys()) | set(c2.keys())
    
    dot_product = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
    norm1 = math.sqrt(sum(val ** 2 for val in c1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in c2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

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
    text_similarity: float # 0–1 description similarity
    relatedness_tier: Optional[str] = None
    missing: list[CompetencyGap]      # in target, absent in source
    deficient: list[CompetencyGap]    # in both, target score > source + threshold
    transferable: list[CompetencyGap] # in both, close scores
    tech_gaps: list[TechGap]          # tech tools target uses, source doesn't
    summary: dict


class JdRequest(BaseModel):
    jd_text: str


class AutomationRiskRequest(BaseModel):
    occupation_code: str
    active_task_ids: list[str]


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


def fetch_relatedness_tier(src_code: str, tgt_code: str) -> Optional[str]:
    res = g.query(
        "MATCH (a:Occupation {code: $src})-[r:RELATED_TO]->(b:Occupation {code: $tgt}) RETURN r.tier",
        {"src": src_code, "tgt": tgt_code}
    )
    if res.result_set and len(res.result_set) > 0:
        return res.result_set[0][0]
    return None


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
    text_sim = text_cosine_similarity(source_occ.description, target_occ.description)
    tier = fetch_relatedness_tier(req.source_code, req.target_code)
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
        text_similarity=round(text_sim, 4),
        relatedness_tier=tier,
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


# Automation vulnerability scores mapped to O*NET Work Activities
AUTOMATION_VULNERABILITY = {
    "4.A.1.a.1": 0.40,  # Getting Information
    "4.A.1.a.2": 0.65,  # Monitoring Processes, Materials, or Surroundings
    "4.A.1.b.1": 0.50,  # Identifying Objects, Actions, and Events
    "4.A.2.a.1": 0.55,  # Estimating the Quantifiable Characteristics
    "4.A.2.a.2": 0.60,  # Evaluating Information to Determine Compliance
    "4.A.2.a.3": 0.50,  # Analyzing Data or Information
    "4.A.2.a.4": 0.85,  # Processing Information
    "4.A.2.b.1": 0.15,  # Thinking Creatively
    "4.A.2.b.2": 0.30,  # Making Decisions and Solving Problems
    "4.A.2.b.3": 0.45,  # Updating and Using Relevant Knowledge
    "4.A.2.b.4": 0.20,  # Developing Objectives and Strategies
    "4.A.2.b.5": 0.70,  # Scheduling Work and Activities
    "4.A.2.b.6": 0.25,  # Organizing, Planning, and Prioritizing Work
    "4.A.3.a.1": 0.80,  # Performing General Physical Activities
    "4.A.3.a.2": 0.85,  # Handling and Moving Objects
    "4.A.3.a.3": 0.75,  # Controlling Machines and Processes
    "4.A.3.a.4": 0.85,  # Operating Vehicles, Mechanized Devices, or Equipment
    "4.A.3.b.1": 0.35,  # Working with the Public
    "4.A.3.b.2": 0.90,  # Documenting/Recording Information
    "4.A.3.b.4": 0.40,  # Interpreting the Meaning of Information for Others
    "4.A.3.b.5": 0.75,  # Coding/Translating Information
    "4.A.3.b.6": 0.65,  # Drafting, Laying Out, and Specifying Technical Devices
    "4.A.4.a.1": 0.30,  # Performing for or Working Directly with the Public
    "4.A.4.a.2": 0.15,  # Establishing and Maintaining Interpersonal Relationships
    "4.A.4.a.3": 0.20,  # Assisting and Caring for Others
    "4.A.4.a.4": 0.25,  # Selling or Influencing Others
    "4.A.4.a.5": 0.15,  # Resolving Conflicts and Negotiating with Others
    "4.A.4.a.6": 0.80,  # Performing Administrative Activities
    "4.A.4.a.7": 0.25,  # Staffing Organizational Units
    "4.A.4.a.8": 0.60,  # Monitoring and Controlling Resources
    "4.A.4.b.1": 0.15,  # Guiding, Directing, and Motivating Subordinates
    "4.A.4.b.2": 0.15,  # Coaching and Developing Others
    "4.A.4.b.3": 0.15,  # Providing Consultation and Advice to Others
    "4.A.4.b.4": 0.20,  # Coordinating the Work and Activities of Others
    "4.A.4.b.5": 0.10,  # Developing and Building Teams
    "4.A.4.b.6": 0.15,  # Training and Teaching Others
}


@app.get("/api/occupation/{code}/closest")
async def get_closest_occupations(code: str):
    t_start = time.perf_counter()
    src = fetch_occupation(code)
    if not src:
        raise HTTPException(status_code=404, detail=f"Occupation {code} not found")

    # Check direct competencies count
    check_res = g.query(
        "MATCH (o:Occupation {code: $code})-[r]->() WHERE type(r) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY'] RETURN count(r)",
        {"code": code}
    )
    has_comps = check_res.result_set[0][0] > 0 if check_res.result_set else False

    if has_comps:
        cypher = """
            MATCH (src:Occupation {code: $code})-[r]->(c)
            WHERE type(r) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
            WITH src, sum(r.score * r.score) AS src_norm_sq

            MATCH (src)-[r1]->(c)<-[r2]-(other:Occupation)
            WHERE type(r1) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
              AND type(r2) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
              AND other.code <> $code
            WITH src, src_norm_sq, other, sum(r1.score * r2.score) AS dot_product

            MATCH (other)-[r3]->(c2)
            WHERE type(r3) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
            WITH other, dot_product, src_norm_sq, sum(r3.score * r3.score) AS other_norm_sq

            WITH other, (dot_product / (sqrt(src_norm_sq) * sqrt(other_norm_sq))) AS similarity
            RETURN other.code, other.title, similarity
            ORDER BY similarity DESC
            LIMIT 10
        """
        res = g.query(cypher, {"code": code})
    else:
        prefix = code[:7] if len(code) >= 7 else code
        cypher = """
            MATCH (s:Occupation)
            WHERE s.code STARTS WITH $prefix AND s.code <> $code
            MATCH (s)-[r1]->(c)
            WHERE type(r1) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
            WITH c, avg(r1.score) AS avg_src_score
            WITH sum(avg_src_score * avg_src_score) AS src_norm_sq

            MATCH (s:Occupation)
            WHERE s.code STARTS WITH $prefix AND s.code <> $code
            MATCH (s)-[r1]->(c)<-[r2]-(other:Occupation)
            WHERE type(r1) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
              AND type(r2) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
              AND other.code <> $code
            WITH src_norm_sq, other, c, avg(r1.score) AS avg_src_score, r2.score AS other_score
            WITH src_norm_sq, other, sum(avg_src_score * other_score) AS dot_product

            MATCH (other)-[r3]->(c2)
            WHERE type(r3) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
            WITH other, dot_product, src_norm_sq, sum(r3.score * r3.score) AS other_norm_sq

            WITH other, (dot_product / (sqrt(src_norm_sq) * sqrt(other_norm_sq))) AS similarity
            RETURN other.code, other.title, similarity
            ORDER BY similarity DESC
            LIMIT 10
        """
        res = g.query(cypher, {"prefix": prefix, "code": code})

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    closest = []
    for row in res.result_set:
        closest.append({
            "code": row[0],
            "title": row[1],
            "similarity": round(float(row[2]), 4) if row[2] is not None else 0.0
        })

    return {
        "source": src.model_dump(),
        "closest": closest,
        "execution_time_ms": round(elapsed_ms, 2),
        "cypher_query": cypher.strip()
    }


@app.get("/api/transition/{src}/{tgt}/export")
async def export_transition_gap(src: str, tgt: str):
    source_occ = fetch_occupation(src)
    target_occ = fetch_occupation(tgt)

    if not source_occ or not target_occ:
        raise HTTPException(status_code=404, detail="Source or target occupation not found")

    source_profile = fetch_profile(src)
    target_profile = fetch_profile(tgt)

    missing, deficient, transferable = compute_gap(source_profile, target_profile)

    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([f"# Source: {source_occ.title} ({source_occ.code})"])
    writer.writerow([f"# Target: {target_occ.title} ({target_occ.code})"])
    writer.writerow([])
    writer.writerow(["competency_id", "competency_name", "competency_type", "source_score", "target_score", "delta", "classification"])

    for x in missing:
        writer.writerow([
            x.element_id,
            x.name,
            x.type,
            "",
            round(x.target_score, 1),
            round(x.delta, 1),
            "Missing"
        ])

    for x in deficient:
        writer.writerow([
            x.element_id,
            x.name,
            x.type,
            round(x.source_score, 1) if x.source_score is not None else "",
            round(x.target_score, 1),
            round(x.delta, 1),
            "Deficient"
        ])

    for x in transferable:
        writer.writerow([
            x.element_id,
            x.name,
            x.type,
            round(x.source_score, 1) if x.source_score is not None else "",
            round(x.target_score, 1),
            round(x.delta, 1),
            "Transferable"
        ])

    output.seek(0)
    filename = f"transition_gap_{src}_to_{tgt}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/occupation/{code}/tasks")
async def get_occupation_tasks(code: str):
    res = g.query(
        """
        MATCH (o:Occupation {code: $code})-[r:PERFORMS]->(t:Task)
        OPTIONAL MATCH (t)-[:MAPS_TO]->(d:DWA)-[:PART_OF]->(wa:WorkActivity)
        RETURN t.task_id, t.statement, t.task_type, r.importance, r.relevance, 
               collect(DISTINCT {element_id: wa.element_id, name: wa.name}) AS activities
        """,
        {"code": code}
    )
    
    tasks_list = []
    for row in res.result_set:
        task_id, statement, task_type, imp, rel, activities = row
        
        wa_list = []
        for act in activities:
            if act and act.get("element_id"):
                wa_list.append(act)
                
        wa_scores = [AUTOMATION_VULNERABILITY.get(wa["element_id"], 0.5) for wa in wa_list]
        task_risk = sum(wa_scores) / len(wa_scores) if wa_scores else 0.5
        
        tasks_list.append({
            "task_id": task_id,
            "statement": statement,
            "task_type": task_type or "Core",
            "importance": float(imp) if imp is not None else 3.0,
            "relevance": float(rel) if rel is not None else 50.0,
            "work_activities": wa_list,
            "automation_risk": round(task_risk, 4)
        })
        
    tasks_list.sort(key=lambda x: (x["task_type"] == "Core", x["importance"] * x["relevance"]), reverse=True)
    return tasks_list


@app.post("/api/automation-risk")
async def calculate_automation_risk(req: AutomationRiskRequest):
    tasks = await get_occupation_tasks(req.occupation_code)
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found for occupation")
        
    active_set = set(req.active_task_ids)
    
    total_weight_all = 0.0
    weighted_risk_all = 0.0
    
    total_weight_active = 0.0
    weighted_risk_active = 0.0
    
    for t in tasks:
        w = t["importance"] * t["relevance"]
        r = t["automation_risk"]
        
        total_weight_all += w
        weighted_risk_all += r * w
        
        if t["task_id"] in active_set:
            total_weight_active += w
            weighted_risk_active += r * w
            
    baseline_risk = (weighted_risk_all / total_weight_all) if total_weight_all > 0 else 0.5
    personalized_risk = (weighted_risk_active / total_weight_active) if total_weight_active > 0 else baseline_risk
    
    return {
        "baseline_risk": round(baseline_risk, 4),
        "personalized_risk": round(personalized_risk, 4),
        "active_count": len(active_set),
        "total_count": len(tasks)
    }


@app.post("/api/match-jd")
async def match_jd(req: JdRequest):
    words = [w.lower().strip() for w in re.findall(r'\w+', req.jd_text) if len(w) >= 4]
    stopwords = {'with', 'that', 'this', 'from', 'their', 'they', 'have', 'were', 'about', 'would', 'could'}
    keywords = [w for w in words if w not in stopwords]
    
    if not keywords:
        return []
        
    keywords = keywords[:10]
    
    cypher = """
        UNWIND $keywords AS kw
        MATCH (o:Occupation)-[:PERFORMS]->(t:Task)
        WHERE toLower(t.statement) CONTAINS kw
        RETURN o.code, o.title, count(t) AS match_count, collect(t.statement)[0..3] AS sample_tasks
        ORDER BY match_count DESC
        LIMIT 10
    """
    res = g.query(cypher, {"keywords": keywords})
    
    results = []
    for row in res.result_set:
        results.append({
            "code": row[0],
            "title": row[1],
            "match_count": int(row[2]),
            "sample_tasks": row[3]
        })
        
    return results


@app.get("/health")
async def health():
    try:
        res = g.query("MATCH (n:Occupation) RETURN count(n) AS c")
        occ_count = res.result_set[0][0]
        return {"status": "ok", "occupation_count": occ_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
