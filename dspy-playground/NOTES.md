# DSPy Playground

DSPy v3.2.1. Replaces hand-crafted prompts with typed signatures the optimizer rewrites.

## Models

| Role | Model | When |
|---|---|---|
| Inner loop (student) | `gemma4:e4b` | All inference — fast, per-query |
| Outer loop (teacher) | `qwen3.6:35b-mlx` | Compilation only — generates training traces |

## Setup

```bash
# Both models already in ollama ls — no pull needed
uv run 01_signatures.py   # uses gemma4:e4b
uv run 02_modules.py      # uses gemma4:e4b
uv run 03_optimizer.py    # qwen3.6:35b-mlx generates traces → compiled into gemma4:e4b
```

## Scripts

**01_signatures.py** — Inline vs class-based signatures, `Predict` vs `ChainOfThought`, typed `Literal` outputs, `inspect_history`.

**02_modules.py** — `dspy.Module` subclass, composing multiple predictors in `forward()`, the OVIR inner loop as a module, save/load.

**03_optimizer.py** — `dspy.Example` dataset, metric function, `BootstrapFewShot` compilation, pre/post accuracy comparison. MIPROv2 commented out for when you have a real eval set.

## Key concepts

**Signature** = typed interface. The optimizer can rewrite its instructions and inject few-shot examples. Never hand-tune the docstring for production — let the optimizer do it.

**Module** = composable unit. Chain multiple signatures. The optimizer treats the whole module as one optimization target.

**Metric** = the objective. Everything the optimizer does is to maximize this function on your labeled set. Writing a sharp metric is the most important step.

**BootstrapFewShot** = fast optimizer. O(trainset) LM calls. Collects traces from successful runs and injects them as few-shot examples. Good for ≤50 labeled examples.

**MIPROv2** = production optimizer. Generates and tests candidate instructions across many trials. Needs ~200+ LM calls. Use when you have 50–500 labeled examples and a real eval set.

**save/load** = compile once offline, load at inference. The compiled JSON stores learned instructions and few-shot examples, not model weights.

**Teacher/student split** = BootstrapFewShot runs `qwen3.6:35b-mlx` (outer loop) to generate high-quality reasoning traces, then injects them as few-shot examples into `gemma4:e4b` (inner loop) prompts. The small model ends up routing as well as the large one — on this domain, with this prompt.

## OVIR wiring

- Inner loop module: `OVIRQueryModule` in `02_modules.py`
- Outer loop: compile `OVIRQueryModule` with `MIPROv2` against your synthetic eval dataset
- Deploy: `loaded.load("compiled_ovir.json")` at startup — zero re-optimization at query time
