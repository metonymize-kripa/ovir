# Data Sources for LinkedIn Economic Graph Experimentation

**Goal:** Find data to derisk the KG pipeline (DGL-KE → GSR → DSPy GEPA) before accessing real LinkedIn Economic Graph data. Datasets are ranked by structural fit to the pipeline, not by size.

---

## What "structural fit" means

The pipeline needs a graph with at least three entity types and at least two distinct relation types — ideally: people/roles, companies/organizations, and skills, connected by directed edges. The closer the data is to that shape, the less ETL work before training.

---

## Tier 1: Start here (downloadable now, graph-ready)

### 1. arshkon LinkedIn Job Postings 2023–2024

**URL:** https://www.kaggle.com/datasets/arshkon/linkedin-job-postings  
**Size:** ~500 MB, ~124K job postings  
**License:** CC0 (scraped public data, use carefully)  
**Requires:** Kaggle account + `kaggle` CLI

This is the closest publicly available proxy for the LinkedIn Economic Graph structure. The schema is already relational and maps directly to KG edge tables:

| File | KG interpretation |
|---|---|
| `job_postings.csv` | Job nodes (job_id, title, description, company_id) |
| `companies/companies.csv` | Company nodes (company_id, name, size, industry) |
| `job_skills.csv` | Edge: (job_id) –[requires_skill]→ (skill_name) |
| `job_industries.csv` | Edge: (job_id) –[in_industry]→ (industry_name) |
| `company_industries.csv` | Edge: (company_id) –[operates_in]→ (industry_name) |
| `company_specialities.csv` | Edge: (company_id) –[specialises_in]→ (specialty) |
| `benefits.csv` | Edge: (job_id) –[offers_benefit]→ (benefit_type) |

Resulting graph: ~50K entity nodes, ~300–500K edges across 7 relation types. Fits on a single GPU for DGL-KE.

**Download:**
```bash
pip install kaggle
# Put kaggle.json in ~/.kaggle/
kaggle datasets download -d arshkon/linkedin-job-postings
unzip linkedin-job-postings.zip -d data/linkedin_postings/
```

---

### 2. ESCO v1.2.1 (EU Skills & Occupations Ontology)

**URL:** https://esco.ec.europa.eu/en/use-esco/download  
**Size:** ~6.5M RDF triples, CSV download ~30–50 MB  
**License:** CC BY 4.0  
**Requires:** Nothing — direct download from EU portal

ESCO is the EU's official taxonomy of 3,000+ occupations and 13,890 skills, with typed relations between them. It's a real knowledge graph, not a job posting dataset. It complements arshkon by providing the skills taxonomy layer.

Key relation types in ESCO:
- `occupation –[hasEssentialSkill]→ skill` (~80K edges)
- `occupation –[hasOptionalSkill]→ skill` (~50K edges)
- `skill –[broaderSkill]→ skill_group` (hierarchy)
- `skill –[relatedSkill]→ skill` (lateral links)
- `occupation –[isInGroup]→ occupation_group` (ISCO hierarchy)

**Why it matters for the pipeline:** ESCO gives you the skill hierarchy that KG-FIT's clustering step would use. Instead of learning skill clusters from scratch, you start from a human-curated taxonomy. The ESCO skill IDs also align with many job posting datasets.

**Download (CSV):**
```bash
# Direct download from EU ESCO portal
curl -L "https://ec.europa.eu/esco/portal/download?fullDownload=true&format=csv&language=en" \
  -o data/esco_v1_2.zip
unzip data/esco_v1_2.zip -d data/esco/
```

---

### 3. O*NET 30.0 (US Occupational Network)

**URL:** https://www.onetcenter.org/database.html  
**Size:** ~15 MB compressed, ~50 MB uncompressed CSV  
**License:** Public domain (US government)  
**Requires:** Nothing — direct download

O*NET is the US government's occupational taxonomy: 923 occupations, each with structured skill requirements, knowledge areas, work activities, and technology skills. The relation tables are already in a clean edge format.

Key files:
- `Skills.txt` — (O*NET-SOC code, skill, importance score, level score)
- `Knowledge.txt` — (O*NET-SOC code, knowledge domain, importance, level)
- `Abilities.txt` — (O*NET-SOC code, ability, importance, level)
- `Technology Skills.txt` — (O*NET-SOC code, technology/tool name)
- `Occupation Data.txt` — occupations with titles and descriptions
- `Related Occupations.txt` — (occupation_A) –[relatedTo]→ (occupation_B)

This is the cleanest, most immediately usable graph dataset in this list. No Kaggle account needed, completely public domain.

**Direct download:**
```bash
curl -O https://www.onetcenter.org/dl_files/database/db_30_0_text.zip
unzip db_30_0_text.zip -d data/onet/
```

---

## Tier 2: Good supporting data

### 4. 1.3M LinkedIn Jobs & Skills (2024)

**URL:** https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024  
**Size:** ~2 GB  
**Note:** Larger than arshkon but less structured — fewer separate edge tables. Good for scale testing DGL-KE once the pipeline is validated on arshkon.

### 5. skill2vec dataset

**URL:** https://github.com/duyet/skill2vec-dataset  
**Size:** Small (~10K job descriptions + skills)  
**Note:** Already processed as (job_description, [skills]) pairs. Useful as training data for the GSR step.

### 6. Job-SDF (Job Skill Demand Forecasting)

**URL:** https://arxiv.org/abs/2406.11920  
**Note:** Academic dataset for temporal skill demand. Good for the "entity stability" problem — shows which skills grow/decay over time, which matters for the LinkedIn EG's dynamic nature.

---

## What the LinkedIn Economic Graph Research Program actually offers

The EGRP is an academic research partnership, not a data download:
- Apply at: `EconomicGraphResearchProposal@LinkedIn.com`
- Access is de-identified or aggregate data, not raw member graphs
- Timeline: 3–6 months from application to data access
- Typically grants access to workforce reports, skill demand trends, and migration patterns — not raw member–company–skill triples

**Bottom line:** The EGRP is for validating conclusions, not for building a pipeline. Build and validate the pipeline on Tier 1 data first.

---

## Recommended starting combination

**arshkon + O*NET** — download both today, no accounts needed for O*NET.

- arshkon provides the job market graph (company → job → skill)
- O*NET provides the occupational ontology (occupation → skill, with importance weights)
- The two connect via skill names (normalize to lowercase, strip punctuation)

Once the pipeline runs on this combination, add ESCO for the multilingual / European taxonomy layer, and use 1.3M dataset for scale testing.

Combined entity count: ~55K nodes. Combined edge count: ~600K–800K. Reasonable for a single 4-GPU training run in DGL-KE.

---

## Conversion script

See `convert_to_kg.py` in this folder — downloads O*NET and converts to TSV triple format ready for DGL-KE.

---

## References

- [arshkon LinkedIn Job Postings 2023–2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- [ESCO v1.2.1 download](https://esco.ec.europa.eu/en/use-esco/download)
- [O*NET 30.0 database](https://www.onetcenter.org/database.html)
- [1.3M LinkedIn Jobs & Skills 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)
- [skill2vec dataset](https://github.com/duyet/skill2vec-dataset)
- [LinkedIn EGRP](https://engineering.linkedin.com/teams/data/projects/economic-graph-research)
