"""
smoke_test.py
End-to-end pipeline check: build tokenizer → prep data → 1-epoch train → eval.

Runs in ~2 minutes on CPU. If all steps pass without errors, the full pipeline
is wired up correctly. EM will be near-zero at 1 epoch — that's expected.

Usage:
    uv run kg-linkedin-research/gsr_training/smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent


def run(cmd: list[str], desc: str):
    print(f"\n{'=' * 60}")
    print(f"[smoke_test] {desc}")
    print(f"{'=' * 60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n[FAIL] Step failed: {desc}")
        sys.exit(1)
    print(f"\n[OK] {desc}")


def main():
    uv = ["uv", "run"]

    run(
        uv + [str(HERE / "build_tokenizer.py")],
        "Step 1: Build augmented tokenizer",
    )

    run(
        uv + [str(HERE / "prepare_data.py")],
        "Step 2: Prepare HF datasets from qa_pairs.jsonl",
    )

    run(
        uv + [
            str(HERE / "train.py"),
            "--model_id", "t5-small",
            "--num_epochs", "1",
            "--bsz", "8",
            "--smoke_test",
        ],
        "Step 3: 1-epoch smoke train (t5-small, 50 samples)",
    )

    # Find the most recently created model directory
    trained_dir = HERE / "trained_models"
    model_dirs = sorted(trained_dir.glob("linkedin-gsr-t5-small-smoke"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_dirs:
        model_dirs = sorted(trained_dir.glob("*smoke*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_dirs:
        # Try any directory
        model_dirs = sorted(trained_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not model_dirs:
        print("\n[FAIL] No trained model directory found — did train.py succeed?")
        sys.exit(1)

    model_path = model_dirs[0]
    print(f"\n[smoke_test] Using model at {model_path}")

    run(
        uv + [
            str(HERE / "eval.py"),
            "--model_path", str(model_path),
            "--kg", str(REPO_ROOT / "data" / "kg"),
            "--split", "test",
            "--num_beams", "3",
        ],
        "Step 4: Eval on test split (beam=3)",
    )

    print("\n" + "=" * 60)
    print("[smoke_test] ALL STEPS PASSED")
    print("=" * 60)
    print("\nNext: run a full training pass:")
    print("  uv run kg-linkedin-research/gsr_training/train.py \\")
    print("      --model_id t5-small --num_epochs 20 --bsz 32")
    print("\nThen evaluate:")
    print("  uv run kg-linkedin-research/gsr_training/eval.py \\")
    print("      --model_path kg-linkedin-research/gsr_training/trained_models/linkedin-gsr-t5-small \\")
    print("      --kg data/kg --num_beams 5")


if __name__ == "__main__":
    main()
