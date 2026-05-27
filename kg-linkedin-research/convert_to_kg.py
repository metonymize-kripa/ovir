"""
convert_to_kg.py
Converts O*NET and arshkon LinkedIn Job Postings into TSV triple format
ready for DGL-KE training.

Usage:
    # Step 1: Download O*NET (free, no account)
    curl -O https://www.onetcenter.org/dl_files/database/db_30_0_text.zip
    unzip db_30_0_text.zip -d data/onet/

    # Step 2: Download arshkon (requires kaggle CLI + account)
    kaggle datasets download -d arshkon/linkedin-job-postings
    unzip linkedin-job-postings.zip -d data/linkedin_postings/

    # Step 3: Run this script
    python convert_to_kg.py --onet data/onet/ --linkedin data/linkedin_postings/ --out data/kg/

Output files (in --out dir):
    entities.dict      entity_id <TAB> entity_name
    relations.dict     relation_id <TAB> relation_name
    train.tsv          head_id <TAB> relation_id <TAB> tail_id
    valid.tsv          (10% of edges, sampled by time if available)
    test.tsv           (10% of edges)
    stats.txt          entity counts, edge counts per relation type
"""

import os
import re
import csv
import argparse
import random
from collections import defaultdict

random.seed(42)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


# ---------------------------------------------------------------------------
# O*NET converter
# ---------------------------------------------------------------------------

def load_onet(onet_dir: str):
    """
    Load O*NET text files and return lists of (head, relation, tail) string triples.

    Key files used:
      Occupation Data.txt       -> occupation nodes
      Skills.txt                -> occupation -[has_skill]-> skill (importance > 3.0)
      Knowledge.txt             -> occupation -[has_knowledge]-> knowledge_domain
      Technology Skills.txt     -> occupation -[uses_technology]-> tool
      Related Occupations.txt   -> occupation -[related_to]-> occupation
    """
    triples = []

    def read_tab(fname):
        path = os.path.join(onet_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            return []
        rows = []
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
        return rows

    # Occupation nodes (just to ensure they exist as entities)
    occ_rows = read_tab("Occupation Data.txt")
    occ_ids = set()
    for r in occ_rows:
        occ_id = "occ:" + norm(r.get("Title", r.get("O*NET-SOC Code", "")))
        occ_ids.add(occ_id)

    # Skills
    for r in read_tab("Skills.txt"):
        try:
            importance = float(r.get("Data Value", 0))
        except ValueError:
            continue
        if r.get("Scale ID", "") != "IM" or importance < 2.5:
            continue
        occ = "occ:" + norm(r.get("Title", ""))
        skill = "skill:" + norm(r.get("Element Name", ""))
        triples.append((occ, "has_skill", skill))

    # Knowledge domains
    for r in read_tab("Knowledge.txt"):
        try:
            importance = float(r.get("Data Value", 0))
        except ValueError:
            continue
        if r.get("Scale ID", "") != "IM" or importance < 3.0:
            continue
        occ = "occ:" + norm(r.get("Title", ""))
        domain = "knowledge:" + norm(r.get("Element Name", ""))
        triples.append((occ, "requires_knowledge", domain))

    # Technology skills
    for r in read_tab("Technology Skills.txt"):
        occ = "occ:" + norm(r.get("Title", ""))
        tech = "tech:" + norm(r.get("Example", r.get("Commodity Title", "")))
        if tech != "tech:":
            triples.append((occ, "uses_technology", tech))

    # Related occupations
    for r in read_tab("Related Occupations.txt"):
        occ_a = "occ:" + norm(r.get("Title", ""))
        occ_b = "occ:" + norm(r.get("Related O*NET-SOC 2019 Title", ""))
        if occ_a and occ_b and occ_a != occ_b:
            triples.append((occ_a, "related_to_occupation", occ_b))

    print(f"O*NET: loaded {len(triples):,} triples from {onet_dir}")
    return triples


# ---------------------------------------------------------------------------
# arshkon LinkedIn converter
# ---------------------------------------------------------------------------

def load_linkedin(linkedin_dir: str):
    """
    Load arshkon LinkedIn Job Postings CSVs and return triples.

    Files used:
      job_postings.csv                       -> job nodes
      companies/companies.csv                -> company nodes
      mappings/job_skills.csv                -> job -[requires_skill]-> skill
      mappings/job_industries.csv            -> job -[in_industry]-> industry
      mappings/company_industries.csv        -> company -[operates_in]-> industry
      mappings/company_specialities.csv      -> company -[specialises_in]-> specialty
    """
    triples = []

    def read_csv(path):
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            return []
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # Job postings -> company edges
    postings = read_csv(os.path.join(linkedin_dir, "job_postings.csv"))
    job_company = {}
    for r in postings:
        job_id = r.get("job_id", "")
        company_id = r.get("company_id", "")
        title = norm(r.get("title", ""))
        if job_id and company_id and title:
            job_node = f"job:{job_id}_{title[:30]}"
            company_node = f"company:{company_id}"
            job_company[job_id] = (job_node, company_node)
            triples.append((company_node, "posted_job", job_node))

    # Job skills
    for r in read_csv(os.path.join(linkedin_dir, "mappings", "job_skills.csv")):
        job_id = r.get("job_id", "")
        skill = norm(r.get("skill_abr", r.get("skill_name", "")))
        if job_id in job_company and skill:
            job_node = job_company[job_id][0]
            triples.append((job_node, "requires_skill", f"skill:{skill}"))

    # Job industries
    for r in read_csv(os.path.join(linkedin_dir, "mappings", "job_industries.csv")):
        job_id = r.get("job_id", "")
        industry = norm(r.get("industry_id", r.get("industry", "")))
        if job_id in job_company and industry:
            job_node = job_company[job_id][0]
            triples.append((job_node, "in_industry", f"industry:{industry}"))

    # Company industries
    for r in read_csv(os.path.join(linkedin_dir, "mappings", "company_industries.csv")):
        company_id = r.get("company_id", "")
        industry = norm(r.get("industry", ""))
        if company_id and industry:
            triples.append((f"company:{company_id}", "operates_in", f"industry:{industry}"))

    # Company specialities
    for r in read_csv(os.path.join(linkedin_dir, "mappings", "company_specialities.csv")):
        company_id = r.get("company_id", "")
        spec = norm(r.get("speciality", ""))
        if company_id and spec:
            triples.append((f"company:{company_id}", "specialises_in", f"speciality:{spec}"))

    print(f"LinkedIn: loaded {len(triples):,} triples from {linkedin_dir}")
    return triples


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_kg(triples, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Build entity and relation dictionaries
    entities = {}
    relations = {}
    int_triples = []

    for h, r, t in triples:
        if h not in entities:
            entities[h] = len(entities)
        if t not in entities:
            entities[t] = len(entities)
        if r not in relations:
            relations[r] = len(relations)
        int_triples.append((entities[h], relations[r], entities[t]))

    # Shuffle and split 80/10/10
    random.shuffle(int_triples)
    n = len(int_triples)
    train = int_triples[: int(n * 0.8)]
    valid = int_triples[int(n * 0.8) : int(n * 0.9)]
    test  = int_triples[int(n * 0.9) :]

    def write_tsv(path, rows):
        with open(path, "w") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")

    write_tsv(os.path.join(out_dir, "train.tsv"), train)
    write_tsv(os.path.join(out_dir, "valid.tsv"), valid)
    write_tsv(os.path.join(out_dir, "test.tsv"),  test)

    with open(os.path.join(out_dir, "entities.dict"), "w") as f:
        for name, idx in sorted(entities.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")

    with open(os.path.join(out_dir, "relations.dict"), "w") as f:
        for name, idx in sorted(relations.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")

    # Stats
    rel_counts = defaultdict(int)
    for _, r, _ in triples:
        rel_counts[r] += 1

    stats_path = os.path.join(out_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Entities:  {len(entities):>8,}\n")
        f.write(f"Relations: {len(relations):>8,}\n")
        f.write(f"Triples:   {len(triples):>8,}\n")
        f.write(f"  Train:   {len(train):>8,}\n")
        f.write(f"  Valid:   {len(valid):>8,}\n")
        f.write(f"  Test:    {len(test):>8,}\n\n")
        f.write("Edges per relation type:\n")
        for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {rel:<35} {count:>8,}\n")

    print(f"\nWrote KG to {out_dir}")
    print(open(stats_path).read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onet",     default=None, help="Path to O*NET text files dir")
    parser.add_argument("--linkedin", default=None, help="Path to arshkon LinkedIn dir")
    parser.add_argument("--out",      default="data/kg", help="Output dir for KG files")
    args = parser.parse_args()

    all_triples = []

    if args.onet:
        all_triples += load_onet(args.onet)

    if args.linkedin:
        all_triples += load_linkedin(args.linkedin)

    if not all_triples:
        print("No data sources specified. Pass --onet and/or --linkedin.")
        print("Example: python convert_to_kg.py --onet data/onet/ --linkedin data/linkedin_postings/")
        exit(1)

    # Deduplicate
    all_triples = list(set(all_triples))
    write_kg(all_triples, args.out)
