"""
01_ollama_basics.py — Qwen3 via Ollama (OpenAI-compatible API).

Ollama is the fastest path to local Qwen inference. It handles model
download, quantization, and serving. The API is OpenAI-compatible so
the same client works for both.

Models (already in ollama ls — no pull needed):
  INNER_MODEL = "gemma4:e4b"         — fast, per-query inference
  OUTER_MODEL = "qwen3.6:35b-mlx"   — offline reasoning, DSPy compilation

Run: uv run 01_ollama_basics.py
"""

from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client, ignored by Ollama
)

# Architecture split:
#   INNER_MODEL = per-query inference (fast, cheap, domain-capable after DSPy optimization)
#   OUTER_MODEL = offline corpus prep, DSPy trace generation, synthetic dataset creation
INNER_MODEL = "gemma4:e4b"
OUTER_MODEL = "qwen3.6:35b-mlx"

MODEL = INNER_MODEL   # default for most examples below


# ── 1. Basic completion ───────────────────────────────────────────────────────

print("=== Basic completion ===")
t0 = time.perf_counter()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a precise assistant. Be concise."},
        {"role": "user",   "content": "What is the capital of France? One word."},
    ],
    temperature=0.0,
)
elapsed = time.perf_counter() - t0
print(f"  response : {resp.choices[0].message.content}")
print(f"  latency  : {elapsed*1000:.0f}ms")
print(f"  tokens   : {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out")


# ── 2. Entity extraction (OVIR inner loop task) ───────────────────────────────
# This is the first step in OVIR's query-time pipeline.
# In production this is wrapped in a DSPy Predict — here we call raw.

ENTITY_SYSTEM = """Extract named entities from the user's query.
Output format (strict):
ENTITIES: <comma-separated entity names>
TYPES: <corresponding types: ORG | PERSON | CONCEPT | LOCATION>"""

queries = [
    "What is ACME Corp's liability cap under the Globex MSA?",
    "Who owns TokenService and what does it depend on?",
    "What did Tim Cook say about the Beats acquisition in 2014?",
]

print("\n=== Entity extraction ===")
for q in queries:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ENTITY_SYSTEM},
            {"role": "user",   "content": q},
        ],
        temperature=0.0,
    )
    print(f"  Q: {q}")
    print(f"  A: {resp.choices[0].message.content.strip()}")
    print()


# ── 3. Thinking mode (Qwen3 extended reasoning) ───────────────────────────────
# Qwen3 supports a "thinking" mode where it reasons before answering.
# Useful for the routing/assessment step in OVIR, not entity extraction.
# Enable by passing "think" in extra_body (Ollama-specific).

print("=== Thinking mode: query routing decision ===")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a document routing assistant."},
        {"role": "user",   "content": (
            "Query: 'What is the total liability exposure across all Globex subsidiaries?'\n"
            "Clusters: contract_terms | financial_obligations | org_structure | compliance\n"
            "Which cluster should be searched first? Explain briefly."
        )},
    ],
    temperature=0.6,  # slightly higher for reasoning tasks
    extra_body={"think": True},
)
msg = resp.choices[0].message
# Ollama exposes the thinking trace in model_extra when think=True
thinking = getattr(msg, "thinking", None) or "(not exposed by this Ollama version)"
print(f"  thinking : {str(thinking)[:200]}...")
print(f"  answer   : {msg.content.strip()}")


# ── 4. Structured output (JSON mode) ─────────────────────────────────────────
# Forces the model to output valid JSON. Use this for typed DSPy outputs.

import json

print("\n=== Structured JSON output ===")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Output valid JSON only. No markdown."},
        {"role": "user",   "content": (
            "Extract entities from: 'Globex, a subsidiary of Initech Holdings, "
            "agreed to pay ACME Corp $2M by 2024-06-30.'\n"
            "Return: {\"entities\": [{\"name\": ..., \"type\": ...}]}"
        )},
    ],
    temperature=0.0,
    response_format={"type": "json_object"},
)
raw = resp.choices[0].message.content
try:
    parsed = json.loads(raw)
    for e in parsed.get("entities", []):
        print(f"  {e['type']:10s} {e['name']}")
except json.JSONDecodeError:
    print(f"  (parse failed) raw: {raw}")


# ── 5. Latency benchmark ──────────────────────────────────────────────────────
print("\n=== Latency benchmark (10 entity extraction calls) ===")
times = []
for q in queries * 4:  # repeat to get 10+ samples
    t0 = time.perf_counter()
    client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Extract entities: {q}"}],
        temperature=0.0,
        max_tokens=64,
    )
    times.append((time.perf_counter() - t0) * 1000)
    if len(times) >= 10:
        break

print(f"  avg={sum(times)/len(times):.0f}ms  min={min(times):.0f}ms  max={max(times):.0f}ms")
print("  Note: first call is slower (KV cache cold). Subsequent calls amortize.")

# ── 6. Outer loop: qwen3.6:35b-mlx for offline reasoning ─────────────────────
# Use the large model for tasks that run once offline:
#   - generating the synthetic eval dataset
#   - DSPy BootstrapFewShot trace generation (teacher_settings={"lm": outer_lm})
#   - producing entity relationship inferences during corpus prep
# Never call this at query time.

print("\n=== Outer loop: qwen3.6:35b-mlx (offline reasoning example) ===")
resp = client.chat.completions.create(
    model=OUTER_MODEL,
    messages=[
        {"role": "system", "content": "You are a corpus analyst. Be thorough."},
        {"role": "user", "content": (
            "Given this contract clause: 'ACME Corp agrees to pay Globex $2M annually.'\n"
            "Generate 3 test questions a user might ask about this clause, "
            "with the expected answer for each. Format as Q: / A: pairs."
        )},
    ],
    temperature=0.7,
)
print(resp.choices[0].message.content.strip())
print("\n  (This is synthetic eval dataset generation — runs once offline)")

print("\nDone. Move on to 02_transformers.py")
