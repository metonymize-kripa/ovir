import dspy
import numpy as np
import requests
import pysolr
import pickle
import time
from typing import Literal
from pathlib import Path
from cobweb_language_embedding import CobwebRetriever

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
RETRIEVER_PATH = Path("cobweb_retriever.pkl")

SOLR_KEYWORD_URL = "http://localhost:8983/solr/ovir_corpus"
SOLR_VECTOR_URL = "http://localhost:8984/solr/ovir_vectors"

def embed_one(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-9))[0]


class OVIRQueryModule(dspy.Module):
    def __init__(self):
        self.extract_entities = dspy.Predict("query -> entities: str")
        self.assess = dspy.Predict(
            "query, retrieved_context -> answer: str, is_sufficient: Literal['yes','no','escalate']"
        )

    def forward(self, query: str, retrieved_context: str):
        ent_result = self.extract_entities(query=query)
        assess_result = self.assess(query=query, retrieved_context=retrieved_context)
        return dspy.Prediction(
            entities=ent_result.entities,
            answer=assess_result.answer,
            is_sufficient=assess_result.is_sufficient,
        )


class OnlineRuntime:
    def __init__(self, lm_name="ollama_chat/gemma4:e4b"):
        lm = dspy.LM(lm_name, api_base=OLLAMA_URL, temperature=0.0)
        dspy.configure(lm=lm)

        self.module = OVIRQueryModule()
        self.solr_keyword = pysolr.Solr(SOLR_KEYWORD_URL, always_commit=False, timeout=10)
        self.solr_vector = pysolr.Solr(SOLR_VECTOR_URL, always_commit=False, timeout=10)

        with open(RETRIEVER_PATH, "rb") as f:
            saved = pickle.load(f)
        self.cobweb_retriever: CobwebRetriever = saved["retriever"]
        self.docs_by_text = {doc["text"]: doc for doc in saved["docs"]}

    def _solr_hybrid_search(self, query_text: str, scope_ids: list[str], top_k: int = 5):
        if not scope_ids:
            return []
            
        fq_str = "chunk_id:(" + " OR ".join(scope_ids) + ")"
        
        # BM25 Search
        bm25_results = self.solr_keyword.search(query_text, fq=fq_str, fl="chunk_id,chunk_text,score", rows=top_k*2)
        
        # Vector Search
        query_vec = embed_one(query_text)
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
        knn_query = f"{{!knn f=chunk_vector topK={top_k*2}}}{vec_str}"
        knn_results = self.solr_vector.search(knn_query, fq=fq_str, fl="chunk_id,chunk_text,score", rows=top_k*2)

        # Hybrid Scoring (30% BM25, 70% KNN)
        scores = {}
        bm25_max = max((d["score"] for d in bm25_results), default=1.0)
        for doc in bm25_results:
            scores[doc["chunk_id"]] = {"text": doc["chunk_text"], "score": 0.3 * (doc["score"] / bm25_max)}

        knn_max = max((d["score"] for d in knn_results), default=1.0)
        for doc in knn_results:
            cid = doc["chunk_id"]
            contrib = 0.7 * (doc["score"] / knn_max)
            if cid in scores:
                scores[cid]["score"] += contrib
            else:
                scores[cid] = {"text": doc["chunk_text"], "score": contrib}

        return sorted(scores.items(), key=lambda x: -x[1]["score"])[:top_k]


    def query(self, query_text: str, top_k: int = 3):
        t0 = time.perf_counter()
        
        # Step 1: Embed Query & Get Scope via Cobweb
        query_vec = embed_one(query_text)
        k_bound = min(top_k * 2, len(self.docs_by_text))
        cobweb_results = self.cobweb_retriever.query(query_vec, k=k_bound)
        
        scope_ids = []
        for chunk_text in cobweb_results:
            doc = self.docs_by_text.get(chunk_text)
            if doc:
                scope_ids.append(doc["chunk_id"])

        # Step 2: Solr Hybrid Search constrained by Cobweb scope
        hybrid_results = self._solr_hybrid_search(query_text, scope_ids, top_k=top_k)
        
        context_parts = []
        for cid, info in hybrid_results:
            context_parts.append(f"[{cid}] {info['text']}")
        retrieved_context = "\\n".join(context_parts)

        # Step 3: DSPy execution
        result = self.module(query=query_text, retrieved_context=retrieved_context)
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        trace = {
            "query": query_text,
            "entities_extracted": result.entities,
            "cobweb_scope_size": len(scope_ids),
            "solr_retrieved_chunks": len(hybrid_results),
            "is_sufficient": result.is_sufficient,
            "answer": result.answer,
            "latency_ms": round(latency_ms, 2)
        }
        
        return trace

if __name__ == "__main__":
    runtime = OnlineRuntime()
    print("=== Running OVIR Online Query ===")
    
    questions = [
        "What is ACME Corp's liability cap?",
        "Who is the CEO of Initech Holdings?",
        "When is the Globex delivery deadline?"
    ]
    
    for q in questions:
        print(f"\nQuery: {q}")
        trace = runtime.query(q, top_k=3)
        print(f"Entities: {trace['entities_extracted']}")
        print(f"Answer: {trace['answer']}")
        print(f"Latency: {trace['latency_ms']}ms")
        print(f"Cobweb Scope Size: {trace['cobweb_scope_size']}")
        print(f"Solr Chunks: {trace['solr_retrieved_chunks']}")

