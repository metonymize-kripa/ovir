"""
prepare_data.py
Convert data/qa_pairs.jsonl → HuggingFace datasets ready for train.py.

Input format (qa_pairs.jsonl):
    {
      "question":          "What skills does a data scientist need...",
      "relation_chain":    ["has_skill"],
      "question_type":     "skill_gap",
      "subgraph_entities": ["occ:data_scientist", "occ:data_analyst"],
      "subgraph_triples":  [["occ:data_scientist", "has_skill", "skill:python"], ...]
    }

Output:
    processed_data/linkedin_train/   HF dataset (input_ids, attention_mask, labels)
    processed_data/linkedin_val/     HF dataset
    processed_data/linkedin_test/    HF dataset
    data_splits/train.jsonl          raw records (for eval graph traversal)
    data_splits/val.jsonl
    data_splits/test.jsonl

Encoding:
    input  = "[reasoning] {question}?"         (lowercased)
    target = "<rel_0_r1><rel_0_r2>..."         (gold relation chain as special tokens)

Split: 80/10/10, stratified by question_type.

Usage:
    uv run kg-linkedin-research/gsr_training/prepare_data.py
"""

import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

random.seed(42)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
QA_PATH = REPO_ROOT / "data" / "qa_pairs.jsonl"
TOKENIZER_DIR = HERE / "tokenizer" / "t5-linkedin-rel"
PROCESSED_DIR = HERE / "processed_data"
SPLITS_DIR = HERE / "data_splits"

# Percentile caps for sequence lengths
SRC_PERCENTILE = 99.9
TGT_PERCENTILE = 100.0


def load_records():
    records = []
    with open(QA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_split(records, val_frac=0.1, test_frac=0.1):
    """Split by question_type to keep type distribution consistent."""
    by_type = {}
    for r in records:
        by_type.setdefault(r["question_type"], []).append(r)

    train, val, test = [], [], []
    for qt, items in sorted(by_type.items()):
        random.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        test += items[:n_test]
        val += items[n_test : n_test + n_val]
        train += items[n_test + n_val :]
        print(f"  {qt}: {len(items)} total → {len(items[n_test + n_val:])} train / "
              f"{n_val} val / {n_test} test")
    return train, val, test


def encode_chain(chain: list[str], relation2semid: dict) -> str:
    """['has_skill', 'requires_skill'] → '<rel_0_has_skill><rel_0_requires_skill>'"""
    tokens = []
    for rel in chain:
        if rel not in relation2semid:
            return None  # unknown relation — skip record
        tokens.append(relation2semid[rel])
    return "".join(tokens)


def to_model_inputs(records: list[dict], relation2semid: dict) -> list[dict]:
    """Convert raw records to {input, target} pairs, dropping unknown relations."""
    out = []
    skipped = 0
    for r in records:
        target = encode_chain(r["relation_chain"], relation2semid)
        if target is None:
            skipped += 1
            continue
        question = r["question"].strip()
        if not question.endswith("?"):
            question += "?"
        out.append({
            "input": "[reasoning] " + question.lower(),
            "target": target,
        })
    if skipped:
        print(f"  [warn] skipped {skipped} records with unknown relations")
    return out


def compute_max_lengths(model_inputs: list[dict], tokenizer) -> tuple[int, int]:
    src_lens = [len(tokenizer(d["input"])["input_ids"]) for d in model_inputs]
    tgt_lens = [len(tokenizer(d["target"])["input_ids"]) for d in model_inputs]
    max_src = int(np.percentile(src_lens, SRC_PERCENTILE))
    max_tgt = int(np.percentile(tgt_lens, TGT_PERCENTILE))
    return max(max_src, 32), max(max_tgt, 8)


def tokenize_split(
    model_inputs: list[dict],
    tokenizer,
    max_src: int,
    max_tgt: int,
) -> list[dict]:
    inputs = [d["input"] for d in model_inputs]
    targets = [d["target"] for d in model_inputs]

    tok_in = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=max_src
    )
    tok_tgt = tokenizer(
        text_target=targets,
        truncation=True,
        padding="max_length",
        max_length=max_tgt,
    )
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lab]
        for lab in tok_tgt["input_ids"]
    ]
    return [
        {"input_ids": ii, "attention_mask": am, "labels": li}
        for ii, am, li in zip(
            tok_in["input_ids"], tok_in["attention_mask"], label_ids
        )
    ]


def save_hf_dataset(tokenized: list[dict], path: Path):
    ds = Dataset.from_list(tokenized)
    ds.save_to_disk(str(path))
    print(f"  Saved {len(ds)} records → {path}")


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    with open(TOKENIZER_DIR / "relation2semid.json") as f:
        relation2semid = json.load(f)

    print(f"Loading QA pairs from {QA_PATH}...")
    records = load_records()
    print(f"  {len(records)} records loaded")

    print("\nStratified split (80/10/10):")
    train_raw, val_raw, test_raw = stratified_split(records)
    print(f"  Total: {len(train_raw)} train / {len(val_raw)} val / {len(test_raw)} test")

    # Save raw splits for eval graph traversal
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_raw, SPLITS_DIR / "train.jsonl")
    save_jsonl(val_raw,   SPLITS_DIR / "val.jsonl")
    save_jsonl(test_raw,  SPLITS_DIR / "test.jsonl")

    print("\nConverting to model inputs...")
    train_inputs = to_model_inputs(train_raw, relation2semid)
    val_inputs   = to_model_inputs(val_raw,   relation2semid)
    test_inputs  = to_model_inputs(test_raw,  relation2semid)

    # Fit max lengths on training set only
    max_src, max_tgt = compute_max_lengths(train_inputs, tokenizer)
    print(f"  max_src_len={max_src}, max_tgt_len={max_tgt}")

    print("\nTokenizing...")
    train_tok = tokenize_split(train_inputs, tokenizer, max_src, max_tgt)
    val_tok   = tokenize_split(val_inputs,   tokenizer, max_src, max_tgt)
    test_tok  = tokenize_split(test_inputs,  tokenizer, max_src, max_tgt)

    print("\nSaving HF datasets...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_hf_dataset(train_tok, PROCESSED_DIR / "linkedin_train")
    save_hf_dataset(val_tok,   PROCESSED_DIR / "linkedin_val")
    save_hf_dataset(test_tok,  PROCESSED_DIR / "linkedin_test")

    # Save metadata for train.py / eval.py
    meta = {
        "max_src_len": max_src,
        "max_tgt_len": max_tgt,
        "n_train": len(train_tok),
        "n_val":   len(val_tok),
        "n_test":  len(test_tok),
        "relations": list(relation2semid.keys()),
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta}")
    print("\nDone.")


if __name__ == "__main__":
    main()
