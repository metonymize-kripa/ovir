#!/usr/bin/env bash
# train_kg.sh — RotatE training on the LinkedIn+O*NET knowledge graph via PyKEEN
#
# Run from repo root (ovir/):
#   bash kg-linkedin-research/train_kg.sh
#
# Prerequisites:
#   uv add pykeen
#
# Hardware notes (M3 Max, 96 GB):
#   - PyKEEN supports MPS (Apple Silicon GPU) — faster than CPU-only DGL-KE
#   - 96 GB unified memory is well above what this graph needs (~300 MB at dim=256)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

uv run python - <<EOF
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pathlib import Path

data = Path("$REPO_ROOT/data/kg")
out  = Path("$REPO_ROOT/data/kg_ckpt")
out.mkdir(parents=True, exist_ok=True)

tf = TriplesFactory.from_path(
    data / "train.tsv",
    delimiter="\\t",
)
train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=42)

result = pipeline(
    training=train,
    validation=valid,
    testing=test,
    model="RotatE",
    model_kwargs=dict(embedding_dim=256),
    loss="NSSALoss",
    optimizer="Adam",
    optimizer_kwargs=dict(lr=1e-3),
    training_kwargs=dict(num_epochs=100, batch_size=1024),
    negative_sampler_kwargs=dict(num_negs_per_pos=200),
    device="mps",        # Apple Silicon GPU — change to "cpu" if MPS causes issues
    random_seed=42,
)

result.save_to_directory(out)
print(f"\\nCheckpoints saved to {out}")
print(result.metric_results.to_df().to_string())
EOF
