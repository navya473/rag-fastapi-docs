"""
rerank.py
------------------
Two reranking backends (selectable via RERANKER env var):
  • cohere       – Cohere Rerank API (cloud, best quality)
  • cross-encoder – sentence-transformers CrossEncoder (local, no cost)

The public function `rerank()` accepts the retrieved docs list and
returns a shorter, re-scored list ready for the generator.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from retrieve import RetrievedDoc

load_dotenv()

RERANKER  = os.getenv("RERANKER", "cohere")           # cohere | cross-encoder
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))
COHERE_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
CE_MODEL     = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ─────────────────────────────────────────────
# Cohere reranker
# ─────────────────────────────────────────────
def _cohere_rerank(
    query: str,
    docs: list[RetrievedDoc],
    top_n: int,
) -> list[RetrievedDoc]:
    from helpers import get_cohere_client

    co = get_cohere_client()
    texts = [d.text for d in docs]

    response = co.rerank(
        model=COHERE_MODEL,
        query=query,
        documents=texts,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        doc = docs[result.index]
        doc.score = result.relevance_score
        reranked.append(doc)

    return reranked


# ─────────────────────────────────────────────
# Cross-encoder reranker (local)
# ─────────────────────────────────────────────
_ce_model = None

def _get_cross_encoder():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder(CE_MODEL)
    return _ce_model


def _cross_encoder_rerank(
    query: str,
    docs: list[RetrievedDoc],
    top_n: int,
) -> list[RetrievedDoc]:
    ce = _get_cross_encoder()
    pairs = [[query, d.text] for d in docs]
    scores = ce.predict(pairs).tolist()

    for doc, score in zip(docs, scores):
        doc.score = float(score)

    reranked = sorted(docs, key=lambda d: d.score, reverse=True)
    return reranked[:top_n]


# ─────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────
def rerank(
    query: str,
    docs: list[RetrievedDoc],
    top_n: int = TOP_K_RERANK,
    backend: str = RERANKER,
) -> list[RetrievedDoc]:
    """
    Rerank `docs` relative to `query` and return the top `top_n`.

    Args:
        query:    User query string.
        docs:     Candidate documents from hybrid retrieval.
        top_n:    Number of documents to keep.
        backend:  'cohere' or 'cross-encoder'.

    Returns:
        Re-scored, trimmed list of RetrievedDoc.
    """
    if not docs:
        return []

    # Never request more than we have
    top_n = min(top_n, len(docs))

    if backend == "cohere":
        return _cohere_rerank(query, docs, top_n)
    elif backend == "cross-encoder":
        return _cross_encoder_rerank(query, docs, top_n)
    else:
        raise ValueError(f"Unknown reranker backend '{backend}'. Use 'cohere' or 'cross-encoder'.")


def rerank_as_dicts(
    query: str,
    docs: list[RetrievedDoc],
    top_n: int = TOP_K_RERANK,
) -> list[dict]:
    reranked = rerank(query, docs, top_n)
    return [
        {"id": d.id, "text": d.text, "score": d.score, **d.metadata}
        for d in reranked
    ]
# ─────────────────────────────────────────────
# MAIN (test reranking)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from retrieve import retrieve

    query = input("Enter query: ")

    # Step 1: retrieve candidates
    retrieved_docs = retrieve(query, top_k=10)

    print("\n--- BEFORE RERANK ---")
    for d in retrieved_docs:
        print(f"\nScore: {d.score:.4f}")
        print(f"Source: {d.metadata.get('source')}")
        print(d.text[:200])

    # Step 2: rerank
    reranked_docs = rerank(query, retrieved_docs, top_n=5)

    print("\n\n=== AFTER RERANK ===")
    for d in reranked_docs:
        print(f"\nScore: {d.score:.4f}")
        print(f"Source: {d.metadata.get('source')}")
        print(d.text[:200])