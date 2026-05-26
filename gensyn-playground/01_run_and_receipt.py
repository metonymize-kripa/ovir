"""
01_run_and_receipt.py — Gensyn REE: run inference, get a receipt, verify it.

REE (Reproducible Execution Environment) runs LLM inference inside a
Docker container using deterministic operators (RepOps). Every run
produces a receipt — a JSON artifact with the model, prompt, output,
and cryptographic metadata needed for anyone to reproduce the exact output.

For OVIR: receipts are the trust layer. Every inference answer is
accompanied by a receipt that proves the output came from a specific
model run and can be re-verified.

Setup:
  git clone https://github.com/gensyn-ai/ree.git /tmp/gensyn-ree
  docker pull falkordb/falkordb:latest   # REE pulls its own image on first use
  python3 /tmp/gensyn-ree/ree.py        # TUI — run this interactively first

This script wraps ree.py's subcommands programmatically.

Run: uv run 01_run_and_receipt.py
"""

import subprocess
import json
import os
import sys
import hashlib
from pathlib import Path

REE_REPO = Path.home() / "gensyn-ree"   # adjust if you cloned elsewhere
REE_SCRIPT = REE_REPO / "ree.py"


def check_ree_available():
    if not REE_SCRIPT.exists():
        print(f"REE not found at {REE_SCRIPT}")
        print("Clone it first:")
        print("  git clone https://github.com/gensyn-ai/ree.git ~/gensyn-ree")
        return False
    return True


def run_inference(model: str, prompt: str, output_dir: Path) -> Path:
    """
    Run reproducible inference via ree.py run subcommand.
    Returns the path to the generated receipt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "prompt.jsonl"
    receipt_file = output_dir / "receipt.json"

    # REE accepts prompts as JSONL — one prompt per line
    with open(prompt_file, "w") as f:
        f.write(json.dumps({"prompt": prompt}) + "\n")

    cmd = [
        sys.executable, str(REE_SCRIPT),
        "run",
        "--model", model,
        "--prompt-file", str(prompt_file),
        "--output-dir", str(output_dir),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  REE error:\n{result.stderr}")
        return None

    # REE writes receipt.json to output_dir
    if receipt_file.exists():
        return receipt_file
    # Find any .json in output_dir
    jsons = list(output_dir.glob("*.json"))
    return jsons[0] if jsons else None


def verify_receipt(receipt_path: Path) -> bool:
    """Run ree.py verify against a receipt."""
    cmd = [sys.executable, str(REE_SCRIPT), "verify", "--receipt", str(receipt_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  stderr: {result.stderr.strip()}")
    return result.returncode == 0


def inspect_receipt(receipt_path: Path) -> dict:
    """Parse and summarize a receipt JSON."""
    with open(receipt_path) as f:
        receipt = json.load(f)
    return receipt


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if not check_ree_available():
    sys.exit(1)

OUTPUT_DIR = Path("ree_outputs")
MODEL = "Qwen/Qwen3-8B"   # any HuggingFace model REE supports

# ── 1. Run inference ──────────────────────────────────────────────────────────
PROMPT = (
    "Extract all named entities from the following text. "
    "Format: NAME (TYPE)\n\n"
    "Text: ACME Corp agrees to pay Globex $2M annually. "
    "The agreement was signed by Bill Lumbergh on January 15, 2024."
)

print(f"=== Running REE inference ===")
print(f"  Model : {MODEL}")
print(f"  Prompt: {PROMPT[:80]}...")

receipt_path = run_inference(MODEL, PROMPT, OUTPUT_DIR)

if receipt_path and receipt_path.exists():
    print(f"\n  Receipt written to: {receipt_path}")

    # ── 2. Inspect the receipt ────────────────────────────────────────────────
    receipt = inspect_receipt(receipt_path)
    print("\n=== Receipt contents ===")
    print(f"  model       : {receipt.get('model', '?')}")
    print(f"  output      : {str(receipt.get('output', '?'))[:120]}...")
    # The receipt contains hashes that allow verification
    for key in ["prompt_hash", "output_hash", "model_hash", "timestamp"]:
        if key in receipt:
            print(f"  {key:15s}: {receipt[key]}")

    print(f"\n  Full receipt keys: {list(receipt.keys())}")

    # ── 3. Verify the receipt ─────────────────────────────────────────────────
    print("\n=== Verifying receipt ===")
    verified = verify_receipt(receipt_path)
    print(f"  Verification: {'PASSED ✓' if verified else 'FAILED ✗'}")

else:
    print("\n  No receipt generated. Check that Docker is running and REE is set up.")
    print("  Interactive setup: python3 ~/gensyn-ree/ree.py")

print("\nDone.")
