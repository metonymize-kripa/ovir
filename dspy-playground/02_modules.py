"""
02_modules.py — Building composable DSPy modules.

A Module wraps one or more Predict/ChainOfThought calls into a reusable,
optimizable unit. This is what you wire together to build the OVIR inner loop.

Covers:
  - dspy.Module subclass with forward()
  - Composing multiple signatures in one module
  - dspy.Retrieve (retrieval integration placeholder)
  - The OVIR inner loop as a DSPy module

Run: uv run 02_modules.py
"""

import dspy
from typing import Literal

# Inner loop: gemma4:e4b — the model that serves queries after optimization
lm = dspy.LM("ollama_chat/gemma4:e4b", api_base="http://localhost:11434", temperature=0.0)
dspy.configure(lm=lm)


# ── 1. Simple module ──────────────────────────────────────────────────────────

class EntityExtractor(dspy.Module):
    """Extract entities from a chunk. Single Predict call, but wrapped
    in a Module so it can be composed and optimized."""

    def __init__(self):
        self.extract = dspy.Predict(
            "chunk -> entities: str, entity_types: str"
        )

    def forward(self, chunk: str):
        return self.extract(chunk=chunk)


# ── 2. Multi-step module ──────────────────────────────────────────────────────
# This is the shape of OVIR's inner loop:
#   1. Extract entities from query
#   2. Route to a cluster
#   3. Decide whether the retrieved context is sufficient

class OVIRQueryModule(dspy.Module):
    """
    Simulates the OVIR inner loop as a DSPy module.

    In production:
      - self.retrieve would call FalkorDB + Solr
      - Each step's signature is an optimization target

    Here: retrieval is mocked; focus is on the module structure.
    """

    def __init__(self):
        self.extract_entities = dspy.Predict(
            "query -> entities: str"
        )
        self.route = dspy.ChainOfThought(
            "query, entities, clusters -> best_cluster: str, confidence: Literal['high','medium','low']"
        )
        self.assess = dspy.Predict(
            "query, retrieved_context -> answer: str, is_sufficient: Literal['yes','no','escalate']"
        )

    def forward(self, query: str, clusters: str, retrieved_context: str):
        # Step 1: entity extraction
        ent_result = self.extract_entities(query=query)

        # Step 2: cluster routing
        route_result = self.route(
            query=query,
            entities=ent_result.entities,
            clusters=clusters
        )

        # Step 3: assess sufficiency of retrieved context
        assess_result = self.assess(
            query=query,
            retrieved_context=retrieved_context
        )

        return dspy.Prediction(
            entities=ent_result.entities,
            best_cluster=route_result.best_cluster,
            routing_confidence=route_result.confidence,
            answer=assess_result.answer,
            is_sufficient=assess_result.is_sufficient,
        )


# ── Run it ────────────────────────────────────────────────────────────────────

ovir = OVIRQueryModule()

clusters = "contract_terms | financial_obligations | org_structure | compliance"

# Simulate a query where context is sufficient
result = ovir(
    query="What is ACME Corp's liability cap under the MSA?",
    clusters=clusters,
    retrieved_context="Section 8.2: Liability is capped at 12 months of fees paid by ACME Corp in the preceding year.",
)

print("=== OVIRQueryModule — sufficient context ===")
print(f"  entities           : {result.entities}")
print(f"  best_cluster       : {result.best_cluster}")
print(f"  routing_confidence : {result.routing_confidence}")
print(f"  answer             : {result.answer}")
print(f"  is_sufficient      : {result.is_sufficient}")

# Simulate a query where context is insufficient → should escalate
result2 = ovir(
    query="What is the total combined liability across all Globex subsidiaries?",
    clusters=clusters,
    retrieved_context="Globex is a subsidiary of Initech Holdings. Revenue: $800M FY2023.",
)

print("\n=== OVIRQueryModule — insufficient context ===")
print(f"  entities           : {result2.entities}")
print(f"  answer             : {result2.answer}")
print(f"  is_sufficient      : {result2.is_sufficient}")
print("  → OVIR would expand hop depth or escalate here")


# ── 3. Saving and loading a module ────────────────────────────────────────────
# After optimization, you save the module's learned prompt parameters.
# Load it at inference time — no re-optimization needed.

import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    save_path = os.path.join(tmp, "ovir_module.json")
    ovir.save(save_path)
    print(f"\nModule saved to {save_path}")

    loaded = OVIRQueryModule()
    loaded.load(save_path)
    print("Module loaded back successfully.")

print("\nDone. Move on to 03_optimizer.py")
