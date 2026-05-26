"""
03_optimizer.py — DSPy optimization (the outer loop).

The optimizer rewrites prompt instructions and few-shot examples to maximize
a metric on a labeled dataset. This IS the OVIR outer loop: you run it once
offline per corpus version; the result is a saved module you load at inference.

Covers:
  - Building a labeled dataset (dspy.Example)
  - Writing an evaluation metric
  - BootstrapFewShot (fast, label-efficient optimizer)
  - MIPROv2 (the production optimizer — commented out, needs more LM calls)
  - Comparing pre/post optimization accuracy

Outer loop model: qwen3.6:35b-mlx — generates the training traces (teacher)
Inner loop model: gemma4:e4b      — the compiled module that runs at query time (student)

BootstrapFewShot runs qwen3.6:35b-mlx on the trainset to collect high-quality
reasoning traces, then injects them as few-shot examples into gemma4:e4b's prompts.
The result: a small fast model that routes as well as the large one on this domain.

Run: uv run 03_optimizer.py
     Expect ~20 LM calls (outer) + ~4 eval calls (inner) for BootstrapFewShot.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot

# Outer loop: large model generates training traces (the "teacher")
OUTER_LM = "ollama_chat/qwen3.6:35b-mlx"
# Inner loop: small fast model that will be compiled and deployed (the "student")
INNER_LM = "ollama_chat/gemma4:e4b"

outer_lm = dspy.LM(OUTER_LM, api_base="http://localhost:11434", temperature=0.3)
inner_lm = dspy.LM(INNER_LM, api_base="http://localhost:11434", temperature=0.0)

# Set the inner loop model as the default — the compiled module will run on it
dspy.configure(lm=inner_lm)


# ── 1. The task: query routing ─────────────────────────────────────────────────
# Given a query, route it to one of four cluster IDs.
# In OVIR: this replaces ad-hoc prompt engineering for routing.

CLUSTERS = "contract_terms | financial_obligations | org_structure | compliance"

class RouteQuery(dspy.Signature):
    """Route a user query to the most relevant document cluster.
    Return exactly one cluster label from the provided list."""

    query: str = dspy.InputField()
    clusters: str = dspy.InputField(desc="Pipe-separated cluster labels")
    cluster: str = dspy.OutputField(desc="One cluster label from the list, exactly as written")

class QueryRouter(dspy.Module):
    def __init__(self):
        self.route = dspy.Predict(RouteQuery)

    def forward(self, query: str):
        return self.route(query=query, clusters=CLUSTERS)


# ── 2. Labeled dataset ────────────────────────────────────────────────────────
# dspy.Example is a lightweight dict-like object.
# inputs= marks which fields are inputs (the rest are labels).

trainset = [
    dspy.Example(query="What is ACME's liability cap?",                        cluster="contract_terms").with_inputs("query"),
    dspy.Example(query="How much does Globex charge annually?",                cluster="financial_obligations").with_inputs("query"),
    dspy.Example(query="Who is the CEO of Initech Holdings?",                  cluster="org_structure").with_inputs("query"),
    dspy.Example(query="Does the MSA comply with GDPR?",                       cluster="compliance").with_inputs("query"),
    dspy.Example(query="What happens if ACME terminates early?",               cluster="contract_terms").with_inputs("query"),
    dspy.Example(query="What are the payment terms in the SOW?",               cluster="financial_obligations").with_inputs("query"),
]

devset = [
    dspy.Example(query="Who reports to the Globex board?",                     cluster="org_structure").with_inputs("query"),
    dspy.Example(query="Is there an indemnification clause?",                  cluster="contract_terms").with_inputs("query"),
]


# ── 3. Metric function ────────────────────────────────────────────────────────
# The metric receives (example, prediction, trace=None).
# Return True/False or a float 0–1.
# This is the function the optimizer maximizes.

def routing_accuracy(example, pred, trace=None):
    return pred.cluster.strip().lower() == example.cluster.strip().lower()


# ── 4. Baseline: unoptimized accuracy ─────────────────────────────────────────
router = QueryRouter()
baseline_correct = sum(
    routing_accuracy(ex, router(query=ex.query))
    for ex in devset
)
print(f"Baseline accuracy on devset: {baseline_correct}/{len(devset)}  (gemma4:e4b, no optimization)")


# ── 5. BootstrapFewShot optimization ─────────────────────────────────────────
# teacher_settings injects qwen3.6:35b-mlx as the trace generator.
# The optimizer runs the teacher on trainset, collects successful traces,
# and bakes them as few-shot examples into the student (gemma4:e4b) prompts.
# Result: gemma4:e4b routes with the quality of the 35B model on this domain.

teleprompter = BootstrapFewShot(
    metric=routing_accuracy,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
    teacher_settings={"lm": outer_lm},  # outer loop: qwen3.6:35b-mlx generates traces
)
compiled_router = teleprompter.compile(
    QueryRouter(),          # student module (will run on inner_lm / gemma4:e4b)
    trainset=trainset,
)

# Eval runs on the student (gemma4:e4b, the default lm)
optimized_correct = sum(
    routing_accuracy(ex, compiled_router(query=ex.query))
    for ex in devset
)
print(f"Baseline accuracy  : {baseline_correct}/{len(devset)}  (gemma4:e4b, no optimization)")
print(f"Optimized accuracy : {optimized_correct}/{len(devset)}  (gemma4:e4b + qwen3.6:35b-mlx traces)")


# ── 6. Inspect what changed ───────────────────────────────────────────────────
print("\n=== Compiled module prompt (last call) ===")
dspy.inspect_history(n=1)


# ── 7. Save the compiled module ───────────────────────────────────────────────
compiled_router.save("compiled_router.json")
print("\nSaved to compiled_router.json")
print("At inference time: load this file instead of re-optimizing.")
print("Re-optimize only when corpus or query distribution changes.")

# ── MIPROv2 (production optimizer) — commented out ───────────────────────────
# MIPROv2 generates and tests candidate instructions across many trials.
# Needs ~200+ LM calls. Use when you have a real eval set (50+ examples).
# Teacher = qwen3.6:35b-mlx; student = gemma4:e4b — same pattern as above.
#
# from dspy.teleprompt import MIPROv2
# tp = MIPROv2(metric=routing_accuracy, auto="medium",
#              teacher_settings={"lm": outer_lm})
# compiled = tp.compile(QueryRouter(), trainset=trainset, valset=devset,
#                        num_trials=20, minibatch=True)
# compiled.save("compiled_router_mipro.json")

print("\nDone.")
