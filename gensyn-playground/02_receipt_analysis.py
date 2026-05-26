"""
02_receipt_analysis.py — Analyze and compare REE receipts.

Once you have receipts from 01_run_and_receipt.py, use this to:
  - Parse and display the receipt structure
  - Compare receipts from different runs (same model + prompt = same output)
  - Understand what the receipt proves
  - Generate a stub receipt for development/testing (no Docker required)

Run: uv run 02_receipt_analysis.py
"""

import json
import hashlib
import datetime
from pathlib import Path


# ── 1. Stub receipt (for development without Docker) ─────────────────────────
# This mirrors the structure of a real REE receipt.
# Use this to build OVIR receipt handling without running REE.

def make_stub_receipt(model: str, prompt: str, output: str) -> dict:
    """Generate a stub receipt in REE's expected format."""
    prompt_bytes = prompt.encode("utf-8")
    output_bytes = output.encode("utf-8")
    return {
        "version": "1.0",
        "model": model,
        "prompt": prompt,
        "output": output,
        "prompt_hash": hashlib.sha256(prompt_bytes).hexdigest(),
        "output_hash": hashlib.sha256(output_bytes).hexdigest(),
        "model_hash": hashlib.sha256(model.encode()).hexdigest(),  # stub: real hash is weights hash
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "ree_version": "stub-0.1",
        "reproducible": True,
        "hardware": "stub",
    }


# ── 2. Compare two receipts ───────────────────────────────────────────────────
# If two receipts have the same model_hash + prompt_hash → output_hash must match.
# That's the REE guarantee: deterministic execution.

def receipts_match(r1: dict, r2: dict) -> dict:
    """
    Compare two receipts. Returns a dict of:
      - same_model: do they reference the same model weights?
      - same_prompt: same input?
      - same_output: deterministic result confirmed?
      - verdict: "verified" | "mismatch" | "different_input"
    """
    same_model  = r1.get("model_hash")  == r2.get("model_hash")
    same_prompt = r1.get("prompt_hash") == r2.get("prompt_hash")
    same_output = r1.get("output_hash") == r2.get("output_hash")

    if not same_model or not same_prompt:
        verdict = "different_input"
    elif same_output:
        verdict = "verified"  # same model + prompt → same output ✓
    else:
        verdict = "mismatch"  # same model + prompt → DIFFERENT output ✗

    return {
        "same_model": same_model,
        "same_prompt": same_prompt,
        "same_output": same_output,
        "verdict": verdict,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-8B"
PROMPT = "Extract entities from: 'ACME Corp owes Globex $2M by January 2025.'"
OUTPUT = "ACME Corp (ORG), Globex (ORG), $2M (AMOUNT), January 2025 (DATE)"

print("=== Stub receipts (no Docker required) ===")
r1 = make_stub_receipt(MODEL, PROMPT, OUTPUT)
r2 = make_stub_receipt(MODEL, PROMPT, OUTPUT)   # identical run
r3 = make_stub_receipt(MODEL, PROMPT + " Extra text.", OUTPUT)  # different prompt

print(f"\nReceipt r1:")
for k, v in r1.items():
    val = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
    print(f"  {k:15s}: {val}")

print(f"\nCompare r1 vs r2 (same model + prompt):")
cmp = receipts_match(r1, r2)
for k, v in cmp.items():
    print(f"  {k}: {v}")

print(f"\nCompare r1 vs r3 (different prompt):")
cmp = receipts_match(r1, r3)
for k, v in cmp.items():
    print(f"  {k}: {v}")


# ── 3. Load real receipts if they exist ──────────────────────────────────────
RECEIPT_DIR = Path("ree_outputs")
real_receipts = list(RECEIPT_DIR.glob("*.json")) if RECEIPT_DIR.exists() else []

if real_receipts:
    print(f"\n=== Real receipts found in {RECEIPT_DIR} ===")
    for path in real_receipts:
        with open(path) as f:
            receipt = json.load(f)
        print(f"\n  {path.name}:")
        for k in ["model", "output_hash", "timestamp", "ree_version"]:
            if k in receipt:
                print(f"    {k}: {str(receipt[k])[:80]}")
else:
    print(f"\n(No real receipts found in {RECEIPT_DIR} — run 01_run_and_receipt.py first)")


# ── 4. OVIR receipt integration sketch ────────────────────────────────────────
# In OVIR, every inference result is emitted with a receipt.
# The retrieval trace already records chunks, clusters, and latency.
# Attach the REE receipt to make the LM call portion also verifiable.

print("\n=== OVIR retrieval trace + REE receipt ===")
ovir_trace = {
    "query": PROMPT,
    "extracted_entities": ["ACME Corp", "Globex"],
    "clusters_traversed": ["financial_obligations"],
    "chunks_retrieved": ["c1", "c2"],
    "latency_ms": 42,
    "answer": OUTPUT,
    "receipt": r1,  # attach the REE receipt for the LM call
}
print(json.dumps(ovir_trace, indent=2)[:800] + "\n  ...")
print("\nThe receipt proves: this answer came from Qwen3-8B on this exact prompt.")
print("Anyone can re-run ree.py verify --receipt <file> to confirm.")
print("\nDone.")
