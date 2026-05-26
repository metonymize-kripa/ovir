"""
01_signatures.py — DSPy signatures and basic predictors.

A Signature is DSPy's typed interface between a task description and
an LM call. It replaces hand-crafted prompt strings with a declared
input/output schema the optimizer can rewrite.

Covers:
  - Inline and class-based signatures
  - dspy.Predict (the base caller)
  - dspy.ChainOfThought (adds reasoning field)
  - Typed outputs with Literal and int
  - Inspecting what the LM actually saw (history)

Inner loop model: gemma4:e4b  (fast, cheap, runs per-query)
Outer loop model: qwen3.6:35b-mlx  (used in 03_optimizer.py for compilation)

Run: uv run 01_signatures.py
"""

import dspy

# ── Configure the inner loop LM ──────────────────────────────────────────────
# gemma4:e4b is the inference-time model — small, fast, domain-capable
# after DSPy optimization bakes good prompts into it.
INNER_LM = "ollama_chat/gemma4:e4b"
lm = dspy.LM(INNER_LM, api_base="http://localhost:11434", temperature=0.0)
dspy.configure(lm=lm)

# ── 1. Inline signature (one-liner) ───────────────────────────────────────────
# Format: "input_field -> output_field"
# Good for throwaway calls. Bad for anything you'll optimize.

classify = dspy.Predict("text -> sentiment: str")
result = classify(text="The contract renewal was delayed again.")
print("=== Inline signature ===")
print(f"  sentiment: {result.sentiment}")

# ── 2. Class-based signature ──────────────────────────────────────────────────
# This is the form you use in production. The docstring becomes the task
# instruction; field descriptions tune what the LM fills in.

class ExtractEntities(dspy.Signature):
    """Extract named entities from a document chunk.
    Return as a comma-separated list. If none found, return 'NONE'."""

    chunk: str = dspy.InputField(desc="A passage of text from a document")
    entities: str = dspy.OutputField(desc="Comma-separated list of named entities (ORG, PERSON, CONCEPT)")
    entity_types: str = dspy.OutputField(desc="Corresponding entity types, comma-separated")

extractor = dspy.Predict(ExtractEntities)
result = extractor(chunk="Apple acquired Beats Electronics in 2014 for $3B. Tim Cook announced the deal.")
print("\n=== Class-based signature ===")
print(f"  entities    : {result.entities}")
print(f"  entity_types: {result.entity_types}")

# ── 3. ChainOfThought — adds a reasoning field before the output ──────────────
# The LM is forced to reason first. Improves accuracy on harder tasks.
# In OVIR: use this for query routing decisions, not entity extraction.

class RouteQuery(dspy.Signature):
    """Given a user query and a list of available topic clusters,
    select the most relevant cluster ID to search first."""

    query: str = dspy.InputField()
    clusters: str = dspy.InputField(desc="JSON list of {id, label} cluster objects")
    best_cluster_id: str = dspy.OutputField(desc="The single cluster ID most relevant to the query")
    confidence: str = dspy.OutputField(desc="High / Medium / Low")

router = dspy.ChainOfThought(RouteQuery)
clusters = '[{"id":"c1","label":"contract terms"},{"id":"c2","label":"financial obligations"},{"id":"c3","label":"personnel and org structure"}]'
result = router(
    query="What is ACME Corp's liability cap?",
    clusters=clusters
)
print("\n=== ChainOfThought routing ===")
print(f"  reasoning        : {result.reasoning[:120]}...")
print(f"  best_cluster_id  : {result.best_cluster_id}")
print(f"  confidence       : {result.confidence}")

# ── 4. Typed output with Literal ──────────────────────────────────────────────
# DSPy 3.x supports Python type annotations on output fields.
# The LM's response is parsed and validated against the declared type.

from typing import Literal

class ClassifyQueryType(dspy.Signature):
    """Classify whether a user query is a lookup (single fact retrieval)
    or a reasoning query (requires combining multiple facts)."""

    query: str = dspy.InputField()
    query_type: Literal["lookup", "reasoning"] = dspy.OutputField()

classifier = dspy.Predict(ClassifyQueryType)
for q in [
    "Who founded Beats Electronics?",
    "What is the total liability exposure across all subsidiaries?",
]:
    result = classifier(query=q)
    print(f"\n  Q: {q}")
    print(f"  type: {result.query_type}")

# ── 5. Inspect what was sent to the LM ────────────────────────────────────────
# dspy.inspect_history() shows the last N prompt/response pairs.
# Critical for debugging: see exactly what the optimizer will rewrite.

print("\n=== Last LM call (inspect_history) ===")
dspy.inspect_history(n=1)

print("\nDone. Move on to 02_modules.py")
