"""
generate_qa.py
Generates synthetic (question, gold_relation_chain) training pairs for GSR
from the LinkedIn + O*NET knowledge graph, using a local Qwen model via Ollama.

Three question types:
  1. skill_gap       — What skills does role A need that role B doesn't?
  2. company_target  — Which companies hire for skill X in industry Y?
  3. career_path     — What technologies and knowledge does role X require?

Usage (from repo root):
    uv run kg-linkedin-research/generate_qa.py \\
        --kg data/kg/ \\
        --linkedin data/LinkedIn-JobPostings-2023-2024/ \\
        --out data/qa_pairs.jsonl \\
        --n 500 \\
        --model qwen3.6:35b-mlx

Output: JSONL, one record per line:
    {
      "question": "What skills does a data scientist need that a data analyst doesn't?",
      "relation_chain": ["has_skill"],
      "question_type": "skill_gap",
      "subgraph_entities": ["occ:data_scientist", "occ:data_analyst"],
      "subgraph_triples": [["occ:data_scientist", "has_skill", "skill:python"], ...]
    }
"""

import os
import re
import csv
import json
import random
import argparse
import requests
from collections import defaultdict

random.seed(42)

OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(s: str) -> str:
    """Convert normalised entity name to readable label. occ:chief_executives → Chief Executives"""
    # Strip prefix
    s = re.sub(r"^[a-z]+:", "", s)
    # Underscores to spaces, title case
    return s.replace("_", " ").title()


def ask_qwen(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.8},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

def load_graph(kg_dir: str):
    """Return entity list, relation list, and adjacency structures."""

    def read_dict(fname):
        out = {}
        with open(os.path.join(kg_dir, fname)) as f:
            for line in f:
                idx, name = line.rstrip("\n").split("\t", 1)
                out[int(idx)] = name
        return out

    entities  = read_dict("entities.dict")
    relations = read_dict("relations.dict")
    rel_index = {v: k for k, v in relations.items()}

    # Build adjacency: head_name -> [(relation_name, tail_name)]
    adj = defaultdict(list)
    # And reverse: tail_name -> [(relation_name, head_name)]
    radj = defaultdict(list)

    for fname in ("train.tsv", "valid.tsv", "test.tsv"):
        path = os.path.join(kg_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                h_name = entities[int(h)]
                r_name = relations[int(r)]
                t_name = entities[int(t)]
                adj[h_name].append((r_name, t_name))
                radj[t_name].append((r_name, h_name))

    return entities, relations, adj, radj


def load_lookups(linkedin_dir: str):
    """Return company_id→name and industry_id→name dicts."""

    def read_csv(path):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    companies = {
        str(r["company_id"]): r["name"]
        for r in read_csv(os.path.join(linkedin_dir, "companies", "companies.csv"))
        if r.get("company_id") and r.get("name")
    }
    industries = {
        str(r["industry_id"]): r["industry_name"]
        for r in read_csv(os.path.join(linkedin_dir, "mappings", "industries.csv"))
        if r.get("industry_id") and r.get("industry_name")
    }
    return companies, industries


def resolve_company(entity_name: str, companies: dict) -> str:
    """company:7096.0 → 'IBM' (or fallback to normalised id)"""
    raw_id = entity_name.replace("company:", "").rstrip(".0")
    return companies.get(raw_id, companies.get(entity_name.replace("company:", ""), denorm(entity_name)))


def resolve_industry(entity_name: str, industries: dict) -> str:
    raw_id = entity_name.replace("industry:", "")
    return industries.get(raw_id, denorm(entity_name))


# ---------------------------------------------------------------------------
# Question type 1: Skill gap
# Relation chain: has_skill  (occ → skill)
# Pattern: find two occupations sharing some skills; highlight unique ones
# ---------------------------------------------------------------------------

def sample_skill_gap(adj, n: int):
    occs = [e for e in adj if e.startswith("occ:")]
    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        occ_a, occ_b = random.sample(occs, 2)
        skills_a = {t for r, t in adj[occ_a] if r == "has_skill"}
        skills_b = {t for r, t in adj[occ_b] if r == "has_skill"}
        unique_to_a = skills_a - skills_b
        if len(unique_to_a) < 2:
            continue
        sample_skills = random.sample(sorted(unique_to_a), min(4, len(unique_to_a)))
        triples = (
            [(occ_a, "has_skill", s) for s in skills_a] +
            [(occ_b, "has_skill", s) for s in skills_b]
        )
        samples.append({
            "question_type": "skill_gap",
            "relation_chain": ["has_skill"],
            "subgraph_entities": [occ_a, occ_b],
            "subgraph_triples": triples,
            "prompt_context": {
                "occ_a": denorm(occ_a),
                "occ_b": denorm(occ_b),
                "unique_skills": [denorm(s) for s in sample_skills],
            },
        })
    return samples


def prompt_skill_gap(ctx: dict) -> str:
    return f"""You are generating training data for a knowledge graph QA system about careers and skills.

Context:
- Occupation A: {ctx['occ_a']}
- Occupation B: {ctx['occ_b']}
- Skills that {ctx['occ_a']} requires but {ctx['occ_b']} typically does not: {', '.join(ctx['unique_skills'])}

Write ONE natural question that a job seeker or career advisor might ask to understand what distinguishes {ctx['occ_a']} from {ctx['occ_b']} in terms of required skills.

Rules:
- Sound like a real person asking, not a database query
- Do not mention specific skill names from the list above — ask generally
- Return only the question, no explanation

Question:"""


# ---------------------------------------------------------------------------
# Question type 2: Company targeting
# Relation chain: posted_job → requires_skill  (company → job → skill)
# Pattern: find a skill, find companies hiring for it, optionally filter by industry
# ---------------------------------------------------------------------------

def sample_company_target(adj, radj, industries: dict, n: int):
    skills = [e for e in radj if e.startswith("skill:") and len(radj[e]) >= 3]
    if not skills:
        return []
    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        skill = random.choice(skills)
        # Jobs requiring this skill
        jobs_with_skill = [h for r, h in radj[skill] if r == "requires_skill" and h.startswith("job:")]
        if len(jobs_with_skill) < 2:
            continue
        # Companies that posted those jobs
        companies_for_job = {}
        for job in jobs_with_skill[:50]:
            for r, h in radj[job]:
                if r == "posted_job" and h.startswith("company:"):
                    companies_for_job[h] = job
        if len(companies_for_job) < 2:
            continue
        # Optional: industry filter
        industry = None
        sample_companies = random.sample(sorted(companies_for_job.keys()), min(3, len(companies_for_job)))
        job_sample = companies_for_job[sample_companies[0]]
        industry_edges = [(r, t) for r, t in adj[job_sample] if r == "in_industry"]
        if industry_edges:
            industry = resolve_industry(industry_edges[0][1], industries)

        triples = (
            [(c, "posted_job", companies_for_job[c]) for c in sample_companies] +
            [(companies_for_job[c], "requires_skill", skill) for c in sample_companies]
        )
        samples.append({
            "question_type": "company_target",
            "relation_chain": ["posted_job", "requires_skill"],
            "subgraph_entities": [skill] + sample_companies,
            "subgraph_triples": triples,
            "prompt_context": {
                "skill": denorm(skill),
                "industry": industry,
            },
        })
    return samples


def prompt_company_target(ctx: dict) -> str:
    industry_clause = f" in the {ctx['industry']} industry" if ctx.get("industry") else ""
    return f"""You are generating training data for a knowledge graph QA system about job market data.

Context:
- Skill: {ctx['skill']}
- Industry: {ctx.get('industry', 'various industries')}

Write ONE natural question that a job seeker might ask to find companies{industry_clause} that are actively hiring for {ctx['skill']} skills.

Rules:
- Sound like a real person, not a database query
- Return only the question, no explanation

Question:"""


# ---------------------------------------------------------------------------
# Question type 3: Career path / role requirements
# Relation chain: uses_technology + requires_knowledge  (occ → tech, occ → knowledge)
# Pattern: given an occupation, what tools and knowledge does it need?
# ---------------------------------------------------------------------------

def sample_career_path(adj, n: int):
    occs = [e for e in adj if e.startswith("occ:")]
    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        occ = random.choice(occs)
        techs     = [t for r, t in adj[occ] if r == "uses_technology"]
        knowledge = [t for r, t in adj[occ] if r == "requires_knowledge"]
        related   = [t for r, t in adj[occ] if r == "related_to_occupation"]
        if not techs and not knowledge:
            continue
        sample_tech = random.sample(techs, min(3, len(techs))) if techs else []
        sample_know = random.sample(knowledge, min(2, len(knowledge))) if knowledge else []
        sample_rel  = random.sample(related, min(2, len(related))) if related else []
        chain = []
        if techs:
            chain.append("uses_technology")
        if knowledge:
            chain.append("requires_knowledge")
        if related:
            chain.append("related_to_occupation")
        triples = (
            [(occ, "uses_technology", t) for t in sample_tech] +
            [(occ, "requires_knowledge", k) for k in sample_know] +
            [(occ, "related_to_occupation", r) for r in sample_rel]
        )
        samples.append({
            "question_type": "career_path",
            "relation_chain": chain,
            "subgraph_entities": [occ] + sample_tech + sample_know + sample_rel,
            "subgraph_triples": triples,
            "prompt_context": {
                "occ": denorm(occ),
                "techs": [denorm(t) for t in sample_tech],
                "knowledge": [denorm(k) for k in sample_know],
                "related": [denorm(r) for r in sample_rel],
            },
        })
    return samples


def prompt_career_path(ctx: dict) -> str:
    parts = []
    if ctx["techs"]:
        parts.append(f"Technologies used: {', '.join(ctx['techs'])}")
    if ctx["knowledge"]:
        parts.append(f"Knowledge domains required: {', '.join(ctx['knowledge'])}")
    if ctx["related"]:
        parts.append(f"Related roles: {', '.join(ctx['related'])}")

    return f"""You are generating training data for a knowledge graph QA system about occupational requirements.

Context:
- Occupation: {ctx['occ']}
- {chr(10).join(f'- {p}' for p in parts)}

Write ONE natural question that someone planning their career or exploring {ctx['occ']} roles might ask about the tools, technologies, or knowledge required.

Rules:
- Sound like a real person exploring career options
- Do not list the specific technologies or domains — ask generally
- Return only the question, no explanation

Question:"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROMPT_FNS = {
    "skill_gap":      prompt_skill_gap,
    "company_target": prompt_company_target,
    "career_path":    prompt_career_path,
}

SAMPLE_FNS = {
    "skill_gap":      lambda adj, radj, ind, n: sample_skill_gap(adj, n),
    "company_target": lambda adj, radj, ind, n: sample_company_target(adj, radj, ind, n),
    "career_path":    lambda adj, radj, ind, n: sample_career_path(adj, n),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg",       default="data/kg",                          help="KG directory")
    parser.add_argument("--linkedin", default="data/LinkedIn-JobPostings-2023-2024", help="LinkedIn data dir")
    parser.add_argument("--out",      default="data/qa_pairs.jsonl",              help="Output JSONL")
    parser.add_argument("--n",        type=int, default=500,                      help="Pairs per question type")
    parser.add_argument("--model",    default="qwen3.6:35b-mlx",                  help="Ollama model name")
    parser.add_argument("--types",    nargs="+",
                        default=["skill_gap", "company_target", "career_path"],
                        help="Question types to generate")
    args = parser.parse_args()

    print("Loading graph...")
    entities, relations, adj, radj = load_graph(args.kg)
    print(f"  {len(entities):,} entities, {len(relations)} relations")

    print("Loading lookups...")
    companies, industries = load_lookups(args.linkedin)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    total = 0

    with open(args.out, "w") as out_f:
        for qtype in args.types:
            print(f"\n--- Generating {args.n} '{qtype}' pairs ---")
            candidates = SAMPLE_FNS[qtype](adj, radj, industries, args.n)
            print(f"  Sampled {len(candidates)} graph paths")

            for i, item in enumerate(candidates[:args.n]):
                try:
                    prompt  = PROMPT_FNS[qtype](item["prompt_context"])
                    question = ask_qwen(args.model, prompt)
                    # Strip any thinking tags qwen3 might emit
                    question = re.sub(r"<think>.*?</think>", "", question, flags=re.DOTALL).strip()
                    record = {
                        "question":         question,
                        "relation_chain":   item["relation_chain"],
                        "question_type":    item["question_type"],
                        "subgraph_entities": item["subgraph_entities"],
                        "subgraph_triples": item["subgraph_triples"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total += 1
                    if (i + 1) % 10 == 0:
                        print(f"  [{qtype}] {i+1}/{min(args.n, len(candidates))} — last: {question[:80]}")
                except Exception as e:
                    print(f"  [skip] {e}")
                    continue

    print(f"\nDone. Wrote {total} QA pairs to {args.out}")
