# DSPy Meta Prompt: KG Corpus Chunking & Classification Pipeline

Source article: [Splunk — Knowledge Graphs: What They Are and Why They Matter](https://www.splunk.com/en_us/blog/learn/knowledge-graphs.html) (Oct 2025)

---

## Problem

Raw corpora — mixed file types, no schema — can't be fed directly into a KG construction pipeline. You need a stable chunking layer that produces labeled fragments, then a classification layer that maps each fragment to a KG primitive. DSPy's outer optimizer loop can tune both the chunking boundary heuristics and the classification instructions without manual prompt engineering.

## Mental model: KG primitives as tagging targets

Five element types from the Splunk framing drive every tagging decision downstream:

| Tag | KG Element | What to look for in a chunk |
|-----|-----------|----------------------------|
| `ENTITY` | Node | Named thing — person, org, product, place, event |
| `RELATIONSHIP` | Edge | A verb phrase connecting two entities; includes direction and type (hierarchical, associative, causal, sequential) |
| `ATTRIBUTE` | Property | A key-value fact about a named entity (birthdate, income, location) |
| `SCHEMA_TYPE` | Ontology/class definition | A sentence that defines what a class *is*, assigns a type, or constrains a domain |
| `TAXONOMY_NODE` | Taxonomy entry | A category or subcategory label in a hierarchy; no directional relationship implied |

A chunk may carry multiple tags. The goal is NOT mutual exclusivity — it's exhaustive coverage so downstream triple extraction has clean signal.

---

## Chunking layer: CLI tools by file type

Run before DSPy. Output is a JSONL file: one object per chunk with `{id, source, type, text}`.

```bash
# .txt / .md  — paragraph boundaries
awk 'BEGIN{RS=""; ORS="\n---\n"} {print NR, $0}' corpus/*.txt

# .csv — row windows (50 rows = one chunk)
split -l 50 data.csv chunk_csv_
# then csvkit to reattach headers:
# for f in chunk_csv_*; do (head -1 data.csv; cat $f) > ${f}.csv; done

# .json — top-level array elements or object keys
jq -c '.[]' data.json           # array of objects → one chunk per object
jq -c 'to_entries[]' data.json  # object map → one chunk per key

# .jsonl — line windows
split -l 20 data.jsonl chunk_jsonl_

# .pdf — page-level, via pdftotext
pdftotext -layout doc.pdf - | awk '/^Form Feed/{n++; next} {print > "page_" n ".txt"}'

# .html / .docx — convert to plain text first
pandoc -t plain doc.docx -o doc.txt
lynx --dump page.html > page.txt
```

Emit JSONL with `jq -n` or a two-line Python wrapper. Chunk size target: 200–500 tokens. Overlap: 1–2 sentences at boundaries.

---

## DSPy signatures

```python
import dspy

class ClassifyChunk(dspy.Signature):
    """
    You are a knowledge graph analyst. Given a text chunk from a document corpus,
    identify which knowledge graph element types are present.

    Knowledge graph elements:
    - ENTITY: a named real-world object (person, org, product, place, event, concept)
    - RELATIONSHIP: a verb phrase connecting two entities, with direction and type
      (hierarchical, associative, causal, sequential, network)
    - ATTRIBUTE: a key-value fact describing an entity (name, date, quantity, location)
    - SCHEMA_TYPE: a definition or type constraint — sentences that say what something IS
    - TAXONOMY_NODE: a category label in a containment hierarchy (is-a, part-of)

    Return a JSON list of objects: [{tag, span, confidence}].
    If no KG primitive is detectable, return [{tag: "NONE", span: "", confidence: 1.0}].
    """
    chunk_text: str = dspy.InputField(desc="Raw text fragment from the corpus")
    source_file: str = dspy.InputField(desc="Origin filename and file type")
    tags: list[dict] = dspy.OutputField(desc="List of {tag, span, confidence} dicts")


class ExtractTriples(dspy.Signature):
    """
    You are a knowledge graph builder. Given a text chunk that has been pre-labeled
    with KG element tags, extract all valid (subject, predicate, object) triples.

    Rules:
    - subject and object must be ENTITY spans
    - predicate must be an explicit RELATIONSHIP span or be inferable from ATTRIBUTE context
    - do not fabricate entities; only extract what is stated in the chunk
    - normalize entity names: strip titles, lowercase, deduplicate
    - for ATTRIBUTE tags, emit (entity, has_attribute, "key=value")

    Return a JSON list: [{subject, predicate, object, evidence_span}].
    """
    chunk_text: str = dspy.InputField(desc="Raw text fragment")
    tags: list[dict] = dspy.InputField(desc="Pre-classified {tag, span, confidence} list")
    triples: list[dict] = dspy.OutputField(desc="List of {subject, predicate, object, evidence_span}")
```

---

## Outer loop (DSPy optimizer target)

```python
import dspy
from dspy.teleprompt import MIPROv2
import json, subprocess, pathlib

# --- 1. Chunking (runs once, outside the optimization loop) ---

def chunk_corpus(corpus_dir: str, output_jsonl: str):
    """
    Shell out to CLI tools to produce chunks.jsonl.
    Each line: {"id": str, "source": str, "file_type": str, "text": str}
    """
    chunks = []
    for path in pathlib.Path(corpus_dir).rglob("*"):
        if path.suffix == ".txt" or path.suffix == ".md":
            paragraphs = path.read_text().split("\n\n")
            for i, p in enumerate(paragraphs):
                if len(p.strip()) > 50:
                    chunks.append({"id": f"{path.stem}_{i}", "source": str(path),
                                   "file_type": path.suffix, "text": p.strip()})
        elif path.suffix == ".jsonl":
            for i, line in enumerate(path.read_text().splitlines()):
                chunks.append({"id": f"{path.stem}_{i}", "source": str(path),
                               "file_type": ".jsonl", "text": line})
        elif path.suffix == ".csv":
            result = subprocess.run(
                ["python3", "-c",
                 f"import csv,json,sys; r=list(csv.DictReader(open('{path}'))); "
                 f"[print(json.dumps(row)) for row in r]"],
                capture_output=True, text=True
            )
            for i, line in enumerate(result.stdout.splitlines()):
                chunks.append({"id": f"{path.stem}_{i}", "source": str(path),
                               "file_type": ".csv", "text": line})
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            items = data if isinstance(data, list) else [data]
            for i, item in enumerate(items):
                chunks.append({"id": f"{path.stem}_{i}", "source": str(path),
                               "file_type": ".json", "text": json.dumps(item)})

    with open(output_jsonl, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return chunks


# --- 2. DSPy module ---

class KGCorpusPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyChunk)
        self.extract  = dspy.ChainOfThought(ExtractTriples)

    def forward(self, chunk_text, source_file):
        classification = self.classify(chunk_text=chunk_text, source_file=source_file)
        triples        = self.extract(chunk_text=chunk_text, tags=classification.tags)
        return dspy.Prediction(tags=classification.tags, triples=triples.triples)


# --- 3. Metric ---

def kg_metric(example, prediction, trace=None):
    """
    Score: fraction of gold triples recovered + penalty for hallucinated entities.
    Gold examples should have 'gold_triples' and 'gold_tags' fields.
    """
    gold_set  = set(tuple(t.values()) for t in example.gold_triples)
    pred_set  = set(tuple(t.values()) for t in (prediction.triples or []))
    recall    = len(gold_set & pred_set) / max(len(gold_set), 1)
    precision = len(gold_set & pred_set) / max(len(pred_set), 1)
    f1        = 2 * recall * precision / max(recall + precision, 1e-9)
    return f1


# --- 4. Run the optimizer ---

def run_optimizer(trainset, program=None):
    lm = dspy.LM("openai/gpt-4o-mini", max_tokens=2048)
    dspy.configure(lm=lm)

    program = program or KGCorpusPipeline()

    optimizer = MIPROv2(
        metric=kg_metric,
        num_candidates=8,    # prompt candidates per module
        init_temperature=1.0,
        verbose=True,
    )

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        num_trials=20,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        requires_permission_to_run=False,
    )
    return optimized


# --- 5. Entry point ---

if __name__ == "__main__":
    import sys
    corpus_dir = sys.argv[1]   # e.g. ./data/raw_corpus
    output     = sys.argv[2]   # e.g. ./data/chunks.jsonl

    chunks = chunk_corpus(corpus_dir, output)
    print(f"Chunked {len(chunks)} fragments → {output}")

    # Load a small gold-labeled trainset (hand-annotated JSONL).
    # Each record: {chunk_text, source_file, gold_tags, gold_triples}
    trainset = [dspy.Example(**json.loads(l)).with_inputs("chunk_text", "source_file")
                for l in open("trainset.jsonl")]

    optimized = run_optimizer(trainset)
    optimized.save("optimized_kg_pipeline.json")
    print("Saved optimized program → optimized_kg_pipeline.json")
```

---

## What MIPRO optimizes

MIPRO rewrites the `"""docstring"""` instructions inside `ClassifyChunk` and `ExtractTriples` to maximize `kg_metric` over the trainset. The five KG element tags act as a fixed vocabulary constraint — the optimizer cannot invent new tags, only refine how it explains each one in the prompt.

**Key levers to expose to the optimizer:**

- The description of each tag (especially RELATIONSHIP subtypes — causal vs. sequential vs. hierarchical are easy to conflate)
- The tie-breaking rule when a span qualifies as both ATTRIBUTE and SCHEMA_TYPE
- The normalization instruction in `ExtractTriples` (lowercase vs. preserve case matters for entity dedup)

**What to keep fixed** (not optimizable):

- The JSON output schema `{tag, span, confidence}` — changing this breaks the pipeline contract
- The five-tag vocabulary — it maps directly to the KG primitive set in the Splunk model
- The chunking CLI layer — this runs before DSPy and is out of the loop

---

## Trainset construction shortcut

You don't need a large gold set to start. Ten manually annotated chunks per file type (50–70 total) is enough for MIPRO with `num_trials=20`. Annotate with:

```bash
# pull 10 random chunks per file type from chunks.jsonl
jq -c 'select(.file_type == ".csv")' chunks.jsonl | shuf | head -10
```

Open each in a text editor, add `gold_tags` and `gold_triples` by hand, save as `trainset.jsonl`.

---

## TL;DR

Chunk with CLI tools by file type → emit JSONL → feed chunks into a two-stage DSPy module (classify KG element → extract triples) → run MIPROv2 to tune the classification and extraction instructions against a small gold set.
