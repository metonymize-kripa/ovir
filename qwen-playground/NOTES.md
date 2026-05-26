# Qwen Playground

Qwen3 series (Alibaba Cloud). All model access via Ollama — no HuggingFace downloads needed.

## Models

| Role | Model | Notes |
|---|---|---|
| Inner loop | `gemma4:e4b` | Per-query, fast. Already in `ollama ls`. |
| Outer loop | `qwen3.6:35b-mlx` | Offline only — corpus prep, DSPy trace generation. Already in `ollama ls`. |

## Setup

```bash
# Both models already available — no pull needed
uv run 01_ollama_basics.py
```

## Scripts

**01_ollama_basics.py** — OpenAI-compatible API via Ollama. Basic completion, entity extraction, thinking mode, JSON output, latency benchmark. Use this for DSPy integration and interactive testing.

## Model selection for OVIR

| Model | Role | Notes |
|---|---|---|
| `gemma4:e4b` (9.6GB) | Inner loop | Serves every query. Fast on Apple Silicon MLX. |
| `qwen3.6:35b-mlx` (21GB) | Outer loop | Runs offline for corpus prep and DSPy compilation. MLX format optimized for M-series. |

DSPy optimization bridges the gap: `qwen3.6:35b-mlx` generates high-quality training traces once; those traces are injected as few-shot examples into `gemma4:e4b`'s prompts. At query time only the small model runs.

## Thinking mode

Qwen3 has a built-in reasoning mode (`"think": True` in Ollama extra_body). Use for:
- Routing decisions where multiple clusters are plausible
- Sufficiency assessment ("do I need to escalate?")
- NOT for entity extraction — overkill, adds latency

## DSPy integration

```python
import dspy
# Inner loop (query time)
inner_lm = dspy.LM("ollama_chat/gemma4:e4b", api_base="http://localhost:11434", temperature=0.0)
# Outer loop (compilation / trace generation)
outer_lm = dspy.LM("ollama_chat/qwen3.6:35b-mlx", api_base="http://localhost:11434", temperature=0.3)
dspy.configure(lm=inner_lm)
```
DSPy wraps Ollama via LiteLLM. Model string format: `ollama_chat/<name>`. Pass `teacher_settings={"lm": outer_lm}` to BootstrapFewShot/MIPROv2.
