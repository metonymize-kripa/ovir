# Gensyn REE Playground

REE = Reproducible Execution Environment. Docker-based; SDK is MIT licensed.
Repo: https://github.com/gensyn-ai/ree (updated April 2026, 2 commits, production-early)

## Setup

```bash
# Clone REE (separate from this repo)
git clone https://github.com/gensyn-ai/ree.git ~/gensyn-ree

# Run interactively first — TUI walks you through model + prompt
python3 ~/gensyn-ree/ree.py

# Then run the playground scripts
uv run 02_receipt_analysis.py   # no Docker needed (uses stub receipts)
uv run 01_run_and_receipt.py    # needs Docker + REE cloned
```

## Scripts

**01_run_and_receipt.py** — Wraps `ree.py run` and `ree.py verify` as Python subprocesses. Runs inference on a specified HuggingFace model, retrieves the receipt JSON, verifies it. Requires Docker running.

**02_receipt_analysis.py** — Receipt structure, stub receipt generation (no Docker needed), two-receipt comparison (same model+prompt → same output hash), OVIR trace + receipt integration sketch.

## REE subcommands

```bash
python3 ~/gensyn-ree/ree.py                          # TUI
python3 ~/gensyn-ree/ree.py run --model Qwen/Qwen3-8B --prompt-file prompts.jsonl --output-dir ./out
python3 ~/gensyn-ree/ree.py verify --receipt ./out/receipt.json
python3 ~/gensyn-ree/ree.py validate --receipt ./out/receipt.json   # structure check only
```

## Receipt structure

```json
{
  "model": "Qwen/Qwen3-8B",
  "prompt": "...",
  "output": "...",
  "prompt_hash": "<sha256>",
  "output_hash": "<sha256>",
  "model_hash": "<sha256 of weights>",
  "timestamp": "2026-05-26T...",
  "ree_version": "..."
}
```

The guarantee: `model_hash + prompt_hash → output_hash` is deterministic. Anyone with REE can rerun `verify` and confirm the output is unchanged.

## OVIR integration

REE solves OVIR's trust layer for LM calls. OVIR already records a retrieval trace (chunks, clusters, latency). Attach the REE receipt to the trace and the entire answer pipeline is auditable end-to-end:

- **Retrieval trace**: which chunks were fetched and how (Solr + FalkorDB)
- **REE receipt**: which model ran on which prompt and produced which output

The combination is what OVIR means by "verified inference" — not just where the data came from, but proof that the LM call itself is reproducible.

## Maturity note

REE is early (17 stars, 2 commits as of May 2026). The binary is proprietary (REE Binary License). The SDK wrapper (`ree.py`) is MIT. Treat it as an aspirational trust layer for OVIR v1 — build the receipt data structure now, wire up real verification when REE matures.
