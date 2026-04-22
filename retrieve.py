"""
retrieve.py
--------------------
Hybrid retrieval using dense + sparse (BM25) Pinecone search,
fused with Reciprocal Rank Fusion (RRF).

Steps:
  1. Embed the query (dense).
  2. Encode query as BM25 sparse vector.
  3. Run dense-only search (top_k candidates).
  4. Run sparse-only search (top_k candidates).
  5. Merge results with RRF → re-ranked candidate list.
  6. Return top_k final documents with metadata.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RETRIEVE", 10))


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────
@dataclass
class RetrievedDoc:
    id:       str
    text:     str
    score:    float
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# RRF fusion
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedDoc],
    k: int = 60,
) -> list[RetrievedDoc]:
    """
    Standard RRF: score(doc) = Σ  1 / (k + rank_i)
    Returns a merged list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDoc] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)
            docs_by_id[doc.id] = doc

    merged = sorted(docs_by_id.values(), key=lambda d: scores[d.id], reverse=True)
    # attach fused score
    for doc in merged:
        doc.score = scores[doc.id]
    return merged


# ─────────────────────────────────────────────
# BM25 encoder (loaded lazily & cached)
# ─────────────────────────────────────────────
_bm25_encoder = None

def _get_bm25():
    global _bm25_encoder
    if _bm25_encoder is None:
        from pinecone_text.sparse import BM25Encoder
        encoder_path = ".bm25_encoder.json"
        import os as _os
        if not _os.path.exists(encoder_path):
            raise FileNotFoundError(
                "BM25 encoder not found. Run `python -m src.core.ingest` first."
            )
        _bm25_encoder = BM25Encoder().load(encoder_path)
    return _bm25_encoder


# ─────────────────────────────────────────────
# Core retrieval
# ─────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedDoc]:
    """
    Hybrid retrieval: dense + sparse → RRF fusion.

    Args:
        query:  Natural language question.
        top_k:  Number of documents to return after fusion.

    Returns:
        Sorted list of RetrievedDoc (best match first).
    """
    from helpers import get_pinecone_index, embed_query

    index    = get_pinecone_index()
    bm25     = _get_bm25()
    dense_q  = embed_query(query)
    sparse_q = bm25.encode_queries(query)

    candidates = top_k * 2   # over-fetch for fusion

    results = index.query(
        vector=dense_q,
        sparse_vector=sparse_q,
        top_k=candidates,
        include_metadata=True,
    )

    docs = [
        RetrievedDoc(
            id=m["id"],
            text=m["metadata"].get("text", ""),
            score=m["score"],
            metadata={k: v for k, v in m["metadata"].items() if k != "text"},
        )
        for m in results["matches"]
    ]

    return docs


# ─────────────────────────────────────────────
# Convenience: retrieve as plain dicts
# ─────────────────────────────────────────────
def retrieve_as_dicts(query: str, top_k: int = TOP_K) -> list[dict]:
    docs = retrieve(query, top_k=top_k)
    return [
        {"id": d.id, "text": d.text, "score": d.score, **d.metadata}
        for d in docs
    ]


# ─────────────────────────────────────────────
# Main Function 
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve(query, top_k=5)

    for r in results:
        print("\n---")
        print("Score:", r.score)
        print("Source:", r.metadata.get("source"))
        print(r.text[:300])