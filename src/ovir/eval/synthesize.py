import dspy
import json
import time

OLLAMA_URL = "http://localhost:11434"
# Using the outer loop LM
LM_NAME = "ollama_chat/qwen3.6:35b-mlx"

class SyntheticEvalGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(
            "context -> user_question: str, expected_entities: str, expected_answer: str"
        )

    def forward(self, context: str):
        return self.generate(context=context)

def generate_synthetic_dataset(corpus_chunks: list[str], output_path: str = "synthetic_eval.json"):
    """
    Generate a synthetic evaluation dataset from the corpus.
    This acts as the forcing function for the outer optimization loop.
    """
    print(f"=== Generating Synthetic Dataset using {LM_NAME} ===")
    lm = dspy.LM(LM_NAME, api_base=OLLAMA_URL, temperature=0.7)
    dspy.configure(lm=lm)
    
    generator = SyntheticEvalGenerator()
    dataset = []

    t0 = time.perf_counter()
    for idx, chunk in enumerate(corpus_chunks):
        print(f"Processing chunk {idx+1}/{len(corpus_chunks)}...")
        try:
            result = generator(context=chunk)
            dataset.append({
                "context": chunk,
                "question": result.user_question,
                "expected_entities": result.expected_entities,
                "expected_answer": result.expected_answer
            })
        except Exception as e:
            print(f"Error generating for chunk {idx+1}: {e}")

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Generated {len(dataset)} synthetic queries in {elapsed:.0f}ms.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Example Usage
    sample_corpus = [
        "ACME Corp agrees to pay Globex $2M annually under this MSA.",
        "Liability is capped at 12 months of fees paid by ACME Corp under Section 8.2."
    ]
    generate_synthetic_dataset(sample_corpus)
