# examples/basic_eval.py
#
# Demonstrates rageval.evaluate() end to end using a mock pipeline.
#
# The mock pipeline uses the financialqa-mini dataset itself as its
# "knowledge base": for each query it returns the relevant document
# content as the retrieved chunk and the ground_truth_answer as the
# answer. This is a perfect-retrieval baseline — useful for verifying
# the library is wired correctly and for understanding what scores of
# 1.0 look like before introducing real pipelines.
#
# No API keys, no external services, no setup required.
#
# How to run (from project root with venv activated):
#   python examples/basic_eval.py

import sys
import os

# Allow running from project root without pip-installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rageval
from rageval.datasets.loader import load_dataset

# ------------------------------------------------------------------
# Build the mock knowledge base from the dataset
# ------------------------------------------------------------------

_dataset = load_dataset("rageval-financialqa-mini")

# Map each query to its relevant document content and ground truth answer.
# The mock pipeline does a simple exact-match lookup on query string.
_knowledge_base: dict[str, tuple[list[str], str]] = {}
for example in _dataset:
    relevant_chunks = [
        doc["content"]
        for doc in example["documents"].values()
        if not doc["is_distractor"]
    ]
    _knowledge_base[example["query"]] = (
        relevant_chunks,
        example["ground_truth_answer"],
    )


def mock_pipeline(query: str) -> tuple[list[str], str]:
    """
    Simulated RAG pipeline — returns the ground-truth relevant chunk
    and answer for each query. Represents a perfect retrieval baseline.
    """
    if query not in _knowledge_base:
        # Graceful fallback for any query not in the dataset.
        return [], "No answer available."
    return _knowledge_base[query]


# ------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running rageval evaluation on financialqa-mini...\n")

    results = rageval.evaluate(
        pipeline_fn=mock_pipeline,
        dataset="rageval-financialqa-mini",
        metrics=["latency", "retrieval_precision"],
        threshold=0.5,
    )

    results.report()
