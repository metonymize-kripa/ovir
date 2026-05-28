"""
build_tokenizer.py
Build an augmented T5 tokenizer with LinkedIn relation special tokens.

Each relation becomes a single special token <rel_0_RELATION_NAME>,
mirroring GSR's encoding for Freebase relations (which used multi-part
tokens like <rel_0_people><rel_1_person><rel_2_nationality>). LinkedIn
relations are single-word, so each maps to exactly one <rel_0_xxx> token.

Saves to:
    gsr_training/tokenizer/t5-linkedin-rel/
    gsr_training/tokenizer/t5-linkedin-rel/relation2semid.json

Usage:
    uv run kg-linkedin-research/gsr_training/build_tokenizer.py
"""

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

# 9 relation types in the LinkedIn+O*NET graph (from data/kg/relations.dict)
RELATIONS = [
    "requires_skill",
    "in_industry",
    "posted_job",
    "related_to_occupation",
    "has_skill",
    "specialises_in",
    "operates_in",
    "uses_technology",
    "requires_knowledge",
]

HERE = Path(__file__).parent
OUT_DIR = HERE / "tokenizer" / "t5-linkedin-rel"


def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # One special token per relation: <rel_0_RELATION_NAME>
    relation2semid = {}
    new_tokens = []
    for rel in RELATIONS:
        token = f"<rel_0_{rel}>"
        relation2semid[rel] = token
        new_tokens.append(token)

    tokenizer.add_tokens(new_tokens)

    # Prefix / separator tokens from GSR (preserve compatibility)
    tokenizer.add_tokens(["[reasoning]", "[query to id]", "[relation to id]", "<SEP>"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(OUT_DIR))

    with open(OUT_DIR / "relation2semid.json", "w") as f:
        json.dump(relation2semid, f, indent=2)

    print(f"Saved tokenizer to {OUT_DIR}")
    print(f"Vocab size: {len(tokenizer):,} (base T5-small: 32,100)")
    print("\nRelation → token mapping:")
    for rel, tok in relation2semid.items():
        print(f"  {rel:<30} → {tok}")


if __name__ == "__main__":
    main()
