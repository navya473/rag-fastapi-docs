"""
eval_retrieval.py
-----------------------
Offline retrieval evaluation.
Metrics: Precision@K, Recall@K, MRR (Mean Reciprocal Rank)

Usage
─────
    python -m tests.eval_retrieval

Expected input: a JSON file at tests/retrieval_ground_truth.json
Format:
    [
      {
        "query": "How do I add a background task in FastAPI?",
        "relevant_ids": ["chunk-uuid-1", "chunk-uuid-2"]
      },
      ...
    ]

If you don't have IDs, you can also provide "relevant_texts" (list of strings)
and the script will match by text substring.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GROUND_TRUTH_PATH = Path("retrieval_ground_truth.json")
DEFAULT_K         = int(os.getenv("TOP_K_RETRIEVE", 10))


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    topk = retrieved_ids[:k]
    hits = sum(1 for doc_id in topk if doc_id in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs that appear in top-k results."""
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for doc_id in topk if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant document (0 if none found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────
# Load ground truth
# ─────────────────────────────────────────────
def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}\n"
            "Create it with the format described in the module docstring."
        )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Match by text substring (fallback)
# ─────────────────────────────────────────────
def _ids_from_texts(docs, relevant_texts: list[str]) -> set[str]:
    """Return IDs of docs whose text contains any relevant snippet."""
    relevant_ids = set()
    for doc in docs:
        for snippet in relevant_texts:
            if snippet.lower() in doc.text.lower():
                relevant_ids.add(doc.id)
                break
    return relevant_ids


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────
def evaluate(k: int = DEFAULT_K):
    from retrieve import retrieve

    ground_truth = load_ground_truth()
    print(f"Evaluating {len(ground_truth)} queries at K={k} …\n")

    precisions, recalls, rrs = [], [], []

    for sample in tqdm(ground_truth, desc="Evaluating"):
        query   = sample["query"]
        docs    = retrieve(query, top_k=k)
        doc_ids = [d.id for d in docs]

        # determine relevant ID set
        if "relevant_ids" in sample:
            relevant_ids = set(sample["relevant_ids"])
        elif "relevant_texts" in sample:
            relevant_ids = _ids_from_texts(docs, sample["relevant_texts"])
        else:
            raise ValueError(f"Sample missing 'relevant_ids' or 'relevant_texts': {sample}")

        p  = precision_at_k(doc_ids, relevant_ids, k)
        r  = recall_at_k(doc_ids, relevant_ids, k)
        rr = reciprocal_rank(doc_ids, relevant_ids)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)

    n = len(ground_truth)
    results = {
        f"Precision@{k}": round(sum(precisions) / n, 4),
        f"Recall@{k}":    round(sum(recalls)    / n, 4),
        "MRR":            round(sum(rrs)         / n, 4),
        "num_queries":    n,
    }

    # Pretty print
    print("\n" + "═" * 40)
    print("   RETRIEVAL EVALUATION RESULTS")
    print("═" * 40)
    for metric, value in results.items():
        print(f"  {metric:<20} {value}")
    print("═" * 40)

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K cutoff")
    args = parser.parse_args()
    evaluate(k=args.k)
