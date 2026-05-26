"""
01_basics.py — GLiNER zero-shot NER basics.

GLiNER is a small BERT-like model that extracts entities with any label
you provide at inference time — no retraining required. This is what
replaces hand-crafted regex or expensive LLM entity extraction in OVIR.

Model: urchade/gliner_medium-v2.1 (~0.5GB, runs on CPU)
       urchade/gliner_large-v2.1  (~1GB, better accuracy)

First run downloads the model from HuggingFace.

Run: uv run 01_basics.py
"""

from gliner import GLiNER
import time

print("Loading GLiNER model (first run downloads ~500MB)...")
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print("Model loaded.\n")

# ── 1. Basic entity extraction ────────────────────────────────────────────────
# labels = the entity types YOU define at inference time.
# No fine-tuning needed. Change labels per domain.

LABELS = ["organization", "person", "location", "date", "monetary_amount", "legal_concept"]

text = "Apple acquired Beats Electronics for $3 billion in 2014. Tim Cook announced the deal in Cupertino."

entities = model.predict_entities(text, LABELS, threshold=0.5)

print("=== Basic extraction ===")
print(f"  Text: {text}")
for e in entities:
    print(f"  [{e['label']:20s}] '{e['text']}'  score={e['score']:.3f}  span=({e['start']},{e['end']})")


# ── 2. Domain-specific labels (contract corpus) ───────────────────────────────
# The power of GLiNER: define labels that match your domain.
# OVIR contract corpus example.

CONTRACT_LABELS = [
    "party",           # named party to a contract
    "date",            # effective dates, deadlines
    "monetary_amount", # fees, caps, damages
    "obligation",      # what a party must do
    "liability_cap",   # max liability terms
]

contract_chunks = [
    "ACME Corp agrees to pay Globex $2,000,000 annually on the first of each January.",
    "Liability under this Agreement shall not exceed twelve (12) months of fees paid by ACME Corp.",
    "Globex will deliver the data pipeline by June 30, 2024, subject to Section 4.2.",
    "Either party may terminate this Agreement upon 30 days written notice.",
]

print("\n=== Contract entity extraction ===")
for chunk in contract_chunks:
    ents = model.predict_entities(chunk, CONTRACT_LABELS, threshold=0.4)
    print(f"\n  Chunk: {chunk[:70]}...")
    for e in ents:
        print(f"    [{e['label']:18s}] '{e['text']}'  ({e['score']:.2f})")


# ── 3. Confidence threshold effect ───────────────────────────────────────────
# Lower threshold = more recall, more noise.
# Higher threshold = fewer entities, more precise.
# OVIR: start at 0.5, tune based on graph quality vs. noise tradeoff.

text = "The agreement between Initech Holdings and Globex was signed by Bill Lumbergh on March 1, 2023."
labels = ["organization", "person", "date"]

print("\n=== Threshold comparison ===")
for threshold in [0.3, 0.5, 0.7, 0.9]:
    ents = model.predict_entities(text, labels, threshold=threshold)
    names = [(e['text'], f"{e['score']:.2f}") for e in ents]
    print(f"  threshold={threshold}: {names}")


# ── 4. Latency benchmark ──────────────────────────────────────────────────────
print("\n=== Latency benchmark ===")
texts = contract_chunks * 5  # 20 chunks
labels = ["party", "date", "monetary_amount"]

t0 = time.perf_counter()
for t in texts:
    model.predict_entities(t, labels, threshold=0.5)
per_chunk = (time.perf_counter() - t0) * 1000 / len(texts)
print(f"  {len(texts)} chunks: {per_chunk:.1f}ms/chunk (single-threaded CPU)")
print(f"  At 10k chunks: ~{10000 * per_chunk / 1000:.0f}s offline (run once, reuse forever)")

print("\nDone. Move on to 02_batch_corpus.py")
