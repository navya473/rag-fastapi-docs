"""
deep_eval.py
--------
End-to-end RAG evaluation using DeepEval.

Metrics evaluated:
  • Faithfulness          – are claims in the answer grounded in context?
  • Answer Relevancy      – does the answer address the question?
  • Contextual Precision  – are retrieved chunks relevant to the question?
  • Contextual Recall     – does context cover all ground-truth information?

Install:
    pip install deepeval

Usage:
    python deval.py
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from deepeval.models import GeminiModel

load_dotenv()

GROUND_TRUTH_PATH = Path("deep_eval_ground_truth.json")


# ─────────────────────────────────────────────
# Text cleaner
# ─────────────────────────────────────────────
_REPLACEMENTS = {
    "\u2018": "'",  "\u2019": "'",
    "\u201c": '"',  "\u201d": '"',
    "\u2013": "-",  "\u2014": "-",
    "\u2026": "...", "\u00a0": " ",
}

def clean(text: str | None) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")

def clean_list(lst: list) -> list[str]:
    return [clean(s) for s in lst]


# ─────────────────────────────────────────────
# Ground truth loader
# ─────────────────────────────────────────────
def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}\n"
            'Expected format: [{"question": "...", "ground_truth": "..."}, ...]'
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {k: clean(v) if isinstance(v, str) else v for k, v in row.items()}
        for row in data
    ]


# ─────────────────────────────────────────────
# RAG pipeline runner
# ─────────────────────────────────────────────
def run_pipeline(question: str) -> dict:
    """Run retrieve -> rerank -> generate for one question."""
    from generate import rag

    result = rag(question)
    docs   = result.get("sources", [])

    return {
        "answer":   clean(result.get("answer", "")),
        "contexts": clean_list([d["text"] for d in docs]),
    }


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
def evaluate():
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )
    from deepeval.models import GeminiModel
    from deepeval.evaluate import AsyncConfig

    gemini_judge = GeminiModel(
        model="gemini-3.1-flash-lite-preview",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=gemini_judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7, model=gemini_judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.7, model=gemini_judge, include_reason=True),
        ContextualRecallMetric(threshold=0.7, model=gemini_judge, include_reason=True),
    ]

    samples = load_ground_truth()
    print(f"\nRunning RAG pipeline for {len(samples)} sample(s)...\n")

    test_cases: list[LLMTestCase] = []

    for sample in tqdm(samples, desc="Running pipeline"):
        question     = clean(sample["question"])
        ground_truth = clean(sample.get("ground_truth", ""))
        result       = run_pipeline(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=ground_truth,
            retrieval_context=result["contexts"],
        )
        test_cases.append(test_case)

    # Sanity check
    print("\n--- Sanity check (first test case) ---")
    tc = test_cases[0]
    print(f"Input   : {tc.input[:100]}")
    print(f"Output  : {tc.actual_output[:100]}")
    print(f"Expected: {tc.expected_output[:100]}")
    print(f"Contexts: {len(tc.retrieval_context)} chunk(s)")
    print()

    print("Running DeepEval evaluation...")
    results = deepeval_evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )

    # Summary table
    print("\n" + "=" * 60)
    print("          DEEPEVAL EVALUATION RESULTS")
    print("=" * 60)

    metric_scores: dict[str, list[float]] = {}

    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            name  = metric_data.name
            score = metric_data.score
            if score is not None:
                metric_scores.setdefault(name, []).append(score)

    if not metric_scores:
        print("  No scores returned — check that your test cases are valid.")
    else:
        for name, scores in metric_scores.items():
            avg     = sum(scores) / len(scores)
            passing = sum(1 for s in scores if s >= 0.7)
            print(f"  {name:<30} avg={avg:.4f}   passed={passing}/{len(scores)}")

    print("=" * 60)

    # Save to CSV
    import csv
    output_path = Path("deepeval_results.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        metric_names = list(metric_scores.keys())
        writer.writerow(["question", "answer", "ground_truth"] + metric_names)

        for i, test_result in enumerate(results.test_results):
            tc  = test_cases[i]
            row = [tc.input, tc.actual_output, tc.expected_output]
            scores_for_row = {
                md.name: md.score
                for md in test_result.metrics_data
                if md.score is not None
            }
            row += [scores_for_row.get(m, "") for m in metric_names]
            writer.writerow(row)

    print(f"\nDetailed results saved to {output_path}")

    # Reasons for first test case
    print("\n--- Metric reasons (first test case) ---")
    for metric_data in results.test_results[0].metrics_data:
        print(f"\n{metric_data.name}:  score={metric_data.score:.4f}")
        if hasattr(metric_data, "reason") and metric_data.reason:
            print(f"  Reason: {metric_data.reason[:300]}")

    return results


if __name__ == "__main__":
    evaluate()