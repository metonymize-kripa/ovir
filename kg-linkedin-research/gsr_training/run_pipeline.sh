#!/usr/bin/env bash
# run_pipeline.sh
# Full GSR training pipeline for the LinkedIn+O*NET knowledge graph.
#
# Steps:
#   1. Build augmented T5 tokenizer with LinkedIn relation tokens
#   2. Prepare HF datasets from qa_pairs.jsonl
#   3. Train T5-small for 20 epochs
#   4. Evaluate on test split
#
# Usage (from repo root):
#   bash kg-linkedin-research/gsr_training/run_pipeline.sh
#
# Optional flags:
#   MODEL=t5-base   bash kg-linkedin-research/gsr_training/run_pipeline.sh
#   EPOCHS=50       bash kg-linkedin-research/gsr_training/run_pipeline.sh
#   BSZ=64          bash kg-linkedin-research/gsr_training/run_pipeline.sh

set -euo pipefail

MODEL="${MODEL:-t5-small}"
EPOCHS="${EPOCHS:-20}"
BSZ="${BSZ:-32}"
LR="${LR:-5e-4}"
NUM_BEAMS="${NUM_BEAMS:-5}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GSR_DIR="$REPO_ROOT/kg-linkedin-research/gsr_training"

echo "============================================================"
echo "LinkedIn GSR Training Pipeline"
echo "  Model:   $MODEL"
echo "  Epochs:  $EPOCHS"
echo "  Batch:   $BSZ"
echo "  LR:      $LR"
echo "  Beams:   $NUM_BEAMS"
echo "  Repo:    $REPO_ROOT"
echo "============================================================"

# Step 1: Tokenizer
echo ""
echo "[1/4] Building augmented tokenizer..."
uv run "$GSR_DIR/build_tokenizer.py"

# Step 2: Data prep
echo ""
echo "[2/4] Preparing HF datasets..."
uv run "$GSR_DIR/prepare_data.py"

# Step 3: Train
echo ""
echo "[3/4] Training ($MODEL, $EPOCHS epochs)..."
uv run "$GSR_DIR/train.py" \
    --model_id "$MODEL" \
    --num_epochs "$EPOCHS" \
    --bsz "$BSZ" \
    --lr "$LR"

# Step 4: Eval — find the best model checkpoint
MODEL_DIR="$GSR_DIR/trained_models/linkedin-gsr-$(echo $MODEL | tr '/' '-')"
echo ""
echo "[4/4] Evaluating on test split (beam=$NUM_BEAMS)..."
uv run "$GSR_DIR/eval.py" \
    --model_path "$MODEL_DIR" \
    --kg "$REPO_ROOT/data/kg" \
    --split test \
    --num_beams "$NUM_BEAMS"

echo ""
echo "============================================================"
echo "Pipeline complete."
echo "Results: $GSR_DIR/eval_results_test.json"
echo "Model:   $MODEL_DIR"
echo "============================================================"
