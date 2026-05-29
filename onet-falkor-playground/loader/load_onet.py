"""
load_onet.py — Load O*NET 30.0 TSV files into FalkorDB.

Graph schema:
  Nodes:   Occupation, Skill, Ability, Knowledge, WorkActivity, Technology
  Edges:   HAS_SKILL, HAS_ABILITY, HAS_KNOWLEDGE, HAS_WORK_ACTIVITY (all with im, lv, score)
           USES_TECH (with hot_tech, in_demand flags)
           RELATED_TO (with tier, rank)

Edge property `score` = im × lv — the combined importance+level weight used for
cosine similarity in the transition engine.

Run:
    cd onet-falkor-playground/loader
    uv run load_onet.py [--data-dir ../../data/db_30_0_text] [--host localhost] [--port 6379] [--fresh]

Takes ~2–4 minutes for the full dataset on local hardware.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

from falkordb import FalkorDB
from redis.exceptions import ResponseError
from tqdm import tqdm

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Load O*NET into FalkorDB")
parser.add_argument("--data-dir", default="../../data/db_30_0_text", type=Path)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=6379, type=int)
parser.add_argument("--fresh", action="store_true", help="Drop and recreate the graph")
parser.add_argument("--graph", default="onet", help="Graph name in FalkorDB")
args = parser.parse_args()

DATA = args.data_dir
BATCH = 400  # rows per UNWIND call; tune based on row width


# ─── Connection ──────────────────────────────────────────────────────────────

print(f"Connecting to FalkorDB at {args.host}:{args.port} ...")
db = FalkorDB(host=args.host, port=args.port)

if args.fresh:
    try:
        db.select_graph(args.graph).delete()
        print(f"Dropped existing graph '{args.graph}'")
    except ResponseError:
        pass

g = db.select_graph(args.graph)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def tsv(filename: str):
    """Open a TSV file and yield DictReader rows."""
    path = DATA / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found at {path}", file=sys.stderr)
        return
    with open(path, encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t")


def run_batch(query: str, rows: list[dict], desc: str = ""):
    """Batch UNWIND a list of row dicts. Shows progress bar per 1000 rows."""
    if not rows:
        return
    for i in tqdm(range(0, len(rows), BATCH), desc=desc or query[:40], unit="batch"):
        chunk = rows[i : i + BATCH]
        g.query(query, {"rows": chunk})


def create_indexes():
    index_defs = [
        ("Occupation", "code"),
        ("Occupation", "title"),
        ("Skill", "element_id"),
        ("Ability", "element_id"),
        ("Knowledge", "element_id"),
        ("WorkActivity", "element_id"),
        ("Technology", "commodity_code"),
        ("Task", "task_id"),
        ("DWA", "dwa_id"),
    ]
    for label, prop in index_defs:
        try:
            g.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
        except ResponseError:
            pass  # already exists

    # Full-text index on occupation title for search
    try:
        g.query("CALL db.idx.fulltext.createNodeIndex('Occupation', 'title', 'code')")
    except ResponseError:
        pass

    print("  Indexes created.")


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_occupations():
    print("\n[1/7] Loading occupations ...")

    # Primary occupation data
    occs = {}
    for row in tsv("Occupation Data.txt"):
        code = row["O*NET-SOC Code"].strip()
        occs[code] = {
            "code": code,
            "title": row["Title"].strip(),
            "description": row["Description"].strip(),
            "job_zone": None,
        }

    # Attach job zones
    for row in tsv("Job Zones.txt"):
        code = row["O*NET-SOC Code"].strip()
        if code in occs:
            try:
                occs[code]["job_zone"] = int(row["Job Zone"])
            except (ValueError, KeyError):
                pass

    rows = list(occs.values())
    run_batch(
        """
        UNWIND $rows AS row
        MERGE (o:Occupation {code: row.code})
        SET o.title = row.title,
            o.description = row.description,
            o.job_zone = row.job_zone
        """,
        rows,
        desc="  occupations",
    )
    print(f"  {len(rows)} occupations loaded.")


def load_competency(
    filename: str,
    node_label: str,
    rel_type: str,
    step_num: int,
    total_steps: int,
):
    """
    Generic loader for Abilities, Skills, Knowledge, Work Activities.
    All share the same TSV schema:
      O*NET-SOC Code | Element ID | Element Name | Scale ID | Data Value | ... | Recommend Suppress | Not Relevant
    """
    print(f"\n[{step_num}/{total_steps}] Loading {node_label} ({filename}) ...")

    # Group rows by (occ_code, element_id) to merge IM and LV
    groups: dict[tuple, dict] = defaultdict(dict)
    suppress_set: set[tuple] = set()
    not_relevant_set: set[tuple] = set()

    for row in tsv(filename):
        code = row["O*NET-SOC Code"].strip()
        eid = row["Element ID"].strip()
        scale = row.get("Scale ID", "").strip()
        suppress = row.get("Recommend Suppress", "N").strip()
        not_rel = row.get("Not Relevant", "N").strip()
        key = (code, eid)

        if suppress == "Y":
            suppress_set.add(key)
            continue

        groups[key]["element_id"] = eid
        groups[key]["name"] = row["Element Name"].strip()
        groups[key]["occ_code"] = code

        if not_rel == "Y":
            not_relevant_set.add(key)

        try:
            val = float(row["Data Value"])
        except (ValueError, TypeError):
            continue

        if scale == "IM":
            groups[key]["im"] = val
        elif scale == "LV":
            groups[key]["lv"] = val

    # Build edge rows — only include pairs with both IM and LV
    edge_rows = []
    for key, props in groups.items():
        if key in suppress_set or key in not_relevant_set:
            continue
        if "im" not in props or "lv" not in props:
            continue
        edge_rows.append(
            {
                "occ_code": props["occ_code"],
                "element_id": props["element_id"],
                "name": props["name"],
                "im": props["im"],
                "lv": props["lv"],
                "score": round(props["im"] * props["lv"], 4),
            }
        )

    if not edge_rows:
        print(f"  No valid rows for {node_label}.")
        return

    # First pass: upsert element nodes
    unique_elements = {
        r["element_id"]: {"element_id": r["element_id"], "name": r["name"]}
        for r in edge_rows
    }
    run_batch(
        f"""
        UNWIND $rows AS row
        MERGE (n:{node_label} {{element_id: row.element_id}})
        SET n.name = row.name
        """,
        list(unique_elements.values()),
        desc=f"  {node_label} nodes",
    )

    # Second pass: upsert edges with weights
    run_batch(
        f"""
        UNWIND $rows AS row
        MATCH (o:Occupation {{code: row.occ_code}})
        MATCH (n:{node_label} {{element_id: row.element_id}})
        MERGE (o)-[r:{rel_type}]->(n)
        SET r.im = row.im, r.lv = row.lv, r.score = row.score
        """,
        edge_rows,
        desc=f"  {rel_type} edges",
    )
    print(f"  {len(unique_elements)} {node_label} nodes, {len(edge_rows)} edges.")


def load_technologies():
    print("\n[6/8] Loading technologies ...")

    tech_nodes: dict[str, dict] = {}
    edge_rows: list[dict] = []

    for row in tsv("Technology Skills.txt"):
        code = row["O*NET-SOC Code"].strip()
        commodity = (row.get("Commodity Code") or row.get("Example Commodity Code") or "").strip()
        title = row.get("Commodity Title", "").strip()
        hot = row.get("Hot Technology", "N").strip().upper() == "Y"
        demand = row.get("In Demand", "N").strip().upper() == "Y"

        if not commodity:
            continue

        tech_nodes[commodity] = {
            "commodity_code": commodity,
            "title": title,
            "hot_tech": hot,
            "in_demand": demand,
        }
        edge_rows.append(
            {
                "occ_code": code,
                "commodity_code": commodity,
                "hot_tech": hot,
                "in_demand": demand,
            }
        )

    run_batch(
        """
        UNWIND $rows AS row
        MERGE (t:Technology {commodity_code: row.commodity_code})
        SET t.title = row.title, t.hot_tech = row.hot_tech, t.in_demand = row.in_demand
        """,
        list(tech_nodes.values()),
        desc="  Technology nodes",
    )

    run_batch(
        """
        UNWIND $rows AS row
        MATCH (o:Occupation {code: row.occ_code})
        MATCH (t:Technology {commodity_code: row.commodity_code})
        MERGE (o)-[r:USES_TECH]->(t)
        SET r.hot_tech = row.hot_tech, r.in_demand = row.in_demand
        """,
        edge_rows,
        desc="  USES_TECH edges",
    )
    print(f"  {len(tech_nodes)} technology nodes, {len(edge_rows)} edges.")


def load_related_occupations():
    print("\n[7/8] Loading related occupations ...")

    rows = []
    for row in tsv("Related Occupations.txt"):
        a = row["O*NET-SOC Code"].strip()
        b = row["Related O*NET-SOC Code"].strip()
        tier = row.get("Relatedness Tier", "").strip()
        try:
            rank = int(row.get("Index", 0))
        except ValueError:
            rank = 0

        rows.append({"a": a, "b": b, "tier": tier, "rank": rank})

    run_batch(
        """
        UNWIND $rows AS row
        MATCH (a:Occupation {code: row.a})
        MATCH (b:Occupation {code: row.b})
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.tier = row.tier, r.rank = row.rank
        """,
        rows,
        desc="  RELATED_TO edges",
    )
    print(f"  {len(rows)} related occupation edges.")


def load_tasks():
    print("\n[8/8] Loading Tasks and DWAs ...")

    # 1. Load DWAs from DWA Reference.txt
    dwas = {}
    for row in tsv("DWA Reference.txt"):
        dwa_id = row["DWA ID"].strip()
        dwas[dwa_id] = {
            "dwa_id": dwa_id,
            "title": row["DWA Title"].strip(),
            "element_id": row["Element ID"].strip(),
        }

    # Upsert DWA nodes
    run_batch(
        """
        UNWIND $rows AS row
        MERGE (d:DWA {dwa_id: row.dwa_id})
        SET d.title = row.title
        """,
        list(dwas.values()),
        desc="  DWA nodes",
    )

    # Relate DWA -> WorkActivity (element_id)
    run_batch(
        """
        UNWIND $rows AS row
        MATCH (d:DWA {dwa_id: row.dwa_id})
        MATCH (w:WorkActivity {element_id: row.element_id})
        MERGE (d)-[:PART_OF]->(w)
        """,
        list(dwas.values()),
        desc="  DWA -> WorkActivity edges",
    )

    # 2. Load Task Statements.txt
    tasks = {}
    occ_task_edges = []
    for row in tsv("Task Statements.txt"):
        task_id = row["Task ID"].strip()
        occ_code = row["O*NET-SOC Code"].strip()
        statement = row["Task"].strip()
        task_type = row.get("Task Type", "").strip()
        
        tasks[task_id] = {
            "task_id": task_id,
            "statement": statement,
            "task_type": task_type
        }
        occ_task_edges.append({
            "occ_code": occ_code,
            "task_id": task_id,
            "task_type": task_type,
            "importance": None,
            "relevance": None,
        })

    # Upsert Task nodes
    run_batch(
        """
        UNWIND $rows AS row
        MERGE (t:Task {task_id: row.task_id})
        SET t.statement = row.statement, t.task_type = row.task_type
        """,
        list(tasks.values()),
        desc="  Task nodes",
    )

    # Load ratings from Task Ratings.txt
    ratings = defaultdict(dict)
    for row in tsv("Task Ratings.txt"):
        occ_code = row["O*NET-SOC Code"].strip()
        task_id = row["Task ID"].strip()
        scale = row["Scale ID"].strip()
        try:
            val = float(row["Data Value"])
        except (ValueError, TypeError):
            continue
        ratings[(occ_code, task_id)][scale] = val

    # Enrich edge items with ratings
    for edge in occ_task_edges:
        key = (edge["occ_code"], edge["task_id"])
        if key in ratings:
            edge["importance"] = ratings[key].get("IM")
            edge["relevance"] = ratings[key].get("RT")

    # Relate Occupation -> Task
    run_batch(
        """
        UNWIND $rows AS row
        MATCH (o:Occupation {code: row.occ_code})
        MATCH (t:Task {task_id: row.task_id})
        MERGE (o)-[r:PERFORMS]->(t)
        SET r.task_type = row.task_type,
            r.importance = row.importance,
            r.relevance = row.relevance
        """,
        occ_task_edges,
        desc="  Occupation -> Task edges",
    )

    # 3. Load Tasks to DWAs.txt
    task_dwa_edges = []
    for row in tsv("Tasks to DWAs.txt"):
        task_id = row["Task ID"].strip()
        dwa_id = row["DWA ID"].strip()
        task_dwa_edges.append({
            "task_id": task_id,
            "dwa_id": dwa_id
        })

    # Relate Task -> DWA
    run_batch(
        """
        UNWIND $rows AS row
        MATCH (t:Task {task_id: row.task_id})
        MATCH (d:DWA {dwa_id: row.dwa_id})
        MERGE (t)-[:MAPS_TO]->(d)
        """,
        task_dwa_edges,
        desc="  Task -> DWA edges",
    )
    print(f"  Tasks, DWAs and mappings loaded.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print("Creating indexes ...")
    create_indexes()

    load_occupations()

    load_competency("Abilities.txt",       "Ability",      "HAS_ABILITY",      2, 8)
    load_competency("Skills.txt",          "Skill",        "HAS_SKILL",        3, 8)
    load_competency("Knowledge.txt",       "Knowledge",    "HAS_KNOWLEDGE",    4, 8)
    load_competency("Work Activities.txt", "WorkActivity", "HAS_WORK_ACTIVITY",5, 8)

    load_technologies()
    load_related_occupations()
    load_tasks()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Quick sanity check
    print("\n--- Sanity check ---")
    for label in ["Occupation", "Skill", "Ability", "Knowledge", "WorkActivity", "Technology", "Task", "DWA"]:
        result = g.query(f"MATCH (n:{label}) RETURN count(n) AS c")
        print(f"  {label}: {result.result_set[0][0]}")

    result = g.query("MATCH ()-[r]->() RETURN count(r) AS c")
    print(f"  Total edges: {result.result_set[0][0]}")


if __name__ == "__main__":
    main()
