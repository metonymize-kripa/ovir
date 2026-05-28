"""
eval.py
Evaluate a trained GSR model on the LinkedIn KG test split.

Two metrics:
  1. Exact Match (EM) — predicted relation chain string == gold chain string
  2. Subgraph Hit@k — starting from the first entity in subgraph_entities,
     walk the predicted relation chain in the KG; hit if we reach any entity
     in the gold subgraph_triples tails.

Usage (from repo root):
    uv run kg-linkedin-research/gsr_training/eval.py \\
        --model_path kg-linkedin-research/gsr_training/trained_models/linkedin-gsr-t5-small \\
        --kg data/kg \\
        --num_beams 5

For eval-only on already-generated predictions:
    uv run kg-linkedin-research/gsr_training/eval.py \\
        --predictions_path kg-linkedin-research/gsr_training/predictions.json \\
        --kg data/kg \\
        --eval_only
"""

import json
import re
import string
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
SPLITS_DIR = HERE / "data_splits"
TOKENIZER_DIR = HERE / "tokenizer" / "t5-linkedin-rel"


# ---------------------------------------------------------------------------
# KG loading  (mirrors generate_qa.py's load_graph)
# ---------------------------------------------------------------------------

def load_graph(kg_dir: str):
    """Return adjacency dict: entity_name → [(relation_name, tail_name)]"""

    def read_dict(fname):
        out = {}
        with open(Path(kg_dir) / fname) as f:
            for line in f:
                idx, name = line.rstrip("\n").split("\t", 1)
                out[int(idx)] = name
        return out

    entities  = read_dict("entities.dict")
    relations = read_dict("relations.dict")
    adj = defaultdict(list)

    for fname in ("train.tsv", "valid.tsv", "test.tsv"):
        path = Path(kg_dir) / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                adj[entities[int(h)]].append((relations[int(r)], entities[int(t)]))

    return adj


# ---------------------------------------------------------------------------
# Prediction decoding  (mirrors GSR's decode_predictions)
# ---------------------------------------------------------------------------

def decode_chain(pred: str) -> list[str]:
    """
    '<rel_0_has_skill><rel_0_requires_skill>' → ['has_skill', 'requires_skill']
    """
    rels = pred.split("<rel_0")[1:]
    path = []
    for rel_str in rels:
        tokens = ("<rel_0" + rel_str).split(">")[:-1]
        parts = [t[7:] for t in tokens]     # strip <rel_0_ or <rel_N_ prefix (7 chars)
        path.append(".".join(parts))
    return path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, records: list[dict], num_beams: int, device: str) -> list[list[str]]:
    """Return list of beam predictions per record (list of decoded strings)."""
    all_preds = []
    for record in tqdm(records, desc="Inference"):
        question = record["question"].strip()
        if not question.endswith("?"):
            question += "?"
        input_text = "[reasoning] " + question.lower()

        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=64,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        preds = [
            tokenizer.decode(o.detach().cpu().numpy(), skip_special_tokens=True)
            for o in outputs
        ]
        all_preds.append(preds)
    return all_preds


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def gold_chain_str(record: dict, relation2semid: dict) -> str:
    tokens = [relation2semid.get(r, "") for r in record["relation_chain"]]
    return "".join(tokens)


def exact_match_score(predictions: list[list[str]], records: list[dict], relation2semid: dict) -> dict:
    hits_at_1 = []
    hits_at_k = []
    for preds, record in zip(predictions, records):
        gold = "".join(relation2semid.get(r, "") for r in record["relation_chain"])
        top1_match = int(preds[0] == gold) if preds else 0
        topk_match = int(any(p == gold for p in preds))
        hits_at_1.append(top1_match)
        hits_at_k.append(topk_match)
    return {
        "em_at_1": float(np.mean(hits_at_1)),
        "em_at_k": float(np.mean(hits_at_k)),
        "n": len(records),
    }


# ---------------------------------------------------------------------------
# Graph traversal hit rate
# ---------------------------------------------------------------------------

def _walk_chain(chain: list[str], start_entities: list[str], adj: dict) -> set:
    """
    Walk a relation chain from one or more start entities and return all reached nodes.

    Two strategies are tried per start entity:
      Sequential: start → r0 → ... → rN  (multi-hop path, e.g. company→job→skill)
      Parallel:   start → r0, start → r1, ... simultaneously (fan-out, e.g. occ→{tech,knowledge,related})

    The union of both strategies is returned so the caller doesn't need to know
    which pattern applies to a given question type.
    """
    reached = set()
    if not chain:
        return reached

    for start in start_entities:
        # Sequential walk: treat chain as A→B→C path
        frontier = [start]
        for rel in chain:
            frontier = [t for ent in frontier for r, t in adj.get(ent, []) if r == rel]
            frontier = list(set(frontier))
        reached.update(frontier)

        # Parallel walk: each relation independently from start
        for rel in chain:
            reached.update(t for r, t in adj.get(start, []) if r == rel)

    return reached


def subgraph_hit(
    predictions: list[list[str]],
    records: list[dict],
    adj: dict,
    top_k: int = 3,
) -> dict:
    """
    For each record, decode the top-k predicted chains, walk from ALL entities
    in subgraph_entities using both sequential and parallel strategies, then
    check if any reached entity appears in the gold subgraph_triples.

    Three question-type traversal patterns this handles correctly:
      skill_gap      occ → has_skill (1-hop, trivially both strategies agree)
      career_path    occ → {uses_technology, requires_knowledge, related_to_occupation}
                     (parallel fan-out from occ node)
      company_target company → posted_job → job → requires_skill → skill
                     (sequential 2-hop; subgraph_entities includes the company nodes)
    """
    hits = []
    empty_preds = 0

    for preds, record in zip(predictions, records):
        start_entities = record.get("subgraph_entities", [])
        if not start_entities:
            hits.append(0)
            continue

        # Gold target entities: all entities appearing as tails in gold triples
        gold_tails = set(t for _, _, t in record.get("subgraph_triples", []))

        reached = set()
        for pred in preds[:top_k]:
            chain = decode_chain(pred)
            if not chain:
                continue
            reached |= _walk_chain(chain, start_entities, adj)

        if not reached:
            empty_preds += 1

        hits.append(int(bool(reached & gold_tails)))

    return {
        "subgraph_hit_at_k": float(np.mean(hits)),
        "empty_pred_rate": empty_preds / max(len(predictions), 1),
        "n": len(records),
    }


# ---------------------------------------------------------------------------
# Per-type breakdown
# ---------------------------------------------------------------------------

def per_type_breakdown(predictions, records, relation2semid, adj, top_k=3):
    by_type = defaultdict(lambda: {"preds": [], "records": []})
    for pred, record in zip(predictions, records):
        qt = record.get("question_type", "unknown")
        by_type[qt]["preds"].append(pred)
        by_type[qt]["records"].append(record)

    results = {}
    for qt, data in sorted(by_type.items()):
        em = exact_match_score(data["preds"], data["records"], relation2semid)
        sg = subgraph_hit(data["preds"], data["records"], adj, top_k)
        results[qt] = {**em, **{f"sg_{k}": v for k, v in sg.items()}}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Path to trained GSR model directory")
    parser.add_argument("--tokenizer_path", default=str(TOKENIZER_DIR),
                        help="Path to augmented tokenizer (default: gsr_training/tokenizer/t5-linkedin-rel)")
    parser.add_argument("--kg", default=str(REPO_ROOT / "data" / "kg"),
                        help="Path to KG directory with entities.dict, train/valid/test.tsv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--predictions_path", default=None,
                        help="Skip inference and load pre-generated predictions JSON")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_dir", default=str(HERE))
    args = parser.parse_args()

    # Load raw records for this split
    split_path = SPLITS_DIR / f"{args.split}.jsonl"
    records = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} {args.split} records from {split_path}")

    # Load relation2semid
    tokenizer_path = args.model_path or args.tokenizer_path
    with open(Path(args.tokenizer_path) / "relation2semid.json") as f:
        relation2semid = json.load(f)

    # Load KG for graph traversal
    print(f"Loading KG from {args.kg}...")
    adj = load_graph(args.kg)
    print(f"  {len(adj):,} head entities in adjacency index")

    # Inference or load pre-generated
    if args.eval_only and args.predictions_path:
        with open(args.predictions_path) as f:
            predictions = json.load(f)
        print(f"Loaded predictions from {args.predictions_path}")
    else:
        if not args.model_path:
            raise ValueError("--model_path required unless --eval_only with --predictions_path")

        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"Device: {device}")

        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()

        predictions = run_inference(model, tokenizer, records, args.num_beams, device)

        preds_path = Path(args.output_dir) / f"predictions_{args.split}.json"
        with open(preds_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {preds_path}")

    # Evaluate
    print("\n=== Exact Match ===")
    em = exact_match_score(predictions, records, relation2semid)
    print(f"  EM@1:  {em['em_at_1']:.4f}")
    print(f"  EM@{args.num_beams}: {em['em_at_k']:.4f}")
    print(f"  N:     {em['n']}")

    print("\n=== Subgraph Hit (graph traversal) ===")
    sg = subgraph_hit(predictions, records, adj, top_k=args.num_beams)
    print(f"  Hit@{args.num_beams}:         {sg['subgraph_hit_at_k']:.4f}")
    print(f"  Empty pred rate: {sg['empty_pred_rate']:.4f}")

    print("\n=== Per-type breakdown ===")
    breakdown = per_type_breakdown(predictions, records, relation2semid, adj, top_k=args.num_beams)
    for qt, scores in breakdown.items():
        print(f"  {qt}:")
        print(f"    EM@1={scores['em_at_1']:.4f}  EM@k={scores['em_at_k']:.4f}  "
              f"SG_hit@k={scores['sg_subgraph_hit_at_k']:.4f}  n={scores['n']}")

    # Save summary
    summary = {
        "split": args.split,
        "num_beams": args.num_beams,
        "overall": {**em, **{f"sg_{k}": v for k, v in sg.items()}},
        "per_type": breakdown,
    }
    summary_path = Path(args.output_dir) / f"eval_results_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
