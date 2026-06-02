# rageval/metrics/relevance.py
#
# Computes answer_relevance: how well the answer addresses the query,
# measured via BERTScore F1 between query and answer strings.
#
# Design decisions (see docs/decisions.md):
#   ADR-010: BERTScore F1 with distilbert-base-uncased. F1 balances precision
#            (answer covers query terms) and recall (query terms present in
#            answer). distilbert is faster than bert-base with acceptable
#            accuracy for relevance scoring.
#   ADR-006: pipeline_outputs is the cached evaluator output — index-aligned
#            with examples. This function never calls pipeline_fn.
#
# Algorithm per example:
#   1. Extract query from examples[i]["query"].
#   2. Extract answer from pipeline_outputs[i][1].
#   3. Empty answer → score 0.0, skip model call.
#   4. Call bert_score.score(cands=[answer], refs=[query], ...) and take F1.
#   5. BERTScore returns tensors; convert to plain float via .item().
#
# Aggregate score = mean answer_relevance across all examples.
#
# Note: bert_score is imported inside the function body (not at module level)
# to keep lazy-loading consistent with the rest of the codebase — no download
# on import.

from __future__ import annotations

from typing import Any

import numpy as np


def compute_answer_relevance(
    examples: list[dict[str, Any]],
    pipeline_outputs: list[tuple[list[str], str]],
) -> dict[str, Any]:
    """
    Compute answer relevance for each example and aggregate.

    Args:
        examples:         Dataset examples; each must have "query".
        pipeline_outputs: Cached (chunks, answer) tuples from the evaluator loop,
                          index-aligned with examples.

    Returns:
        Dict with "scores" (mean answer_relevance) and "raw" (per-example breakdown).
    """
    if len(examples) != len(pipeline_outputs):
        raise ValueError(
            f"examples and pipeline_outputs must have the same length, "
            f"got {len(examples)} and {len(pipeline_outputs)}."
        )

    # Deferred import — model downloads only when this metric is first used.
    import bert_score

    per_example: list[dict[str, Any]] = []

    # Separate examples into those needing scoring and those that are empty.
    indices_to_score: list[int] = []
    queries_to_score: list[str] = []
    answers_to_score: list[str] = []

    for i, (example, (_chunks, answer)) in enumerate(zip(examples, pipeline_outputs)):
        if not answer.strip():
            per_example.append({
                "query": example["query"],
                "score": 0.0,
                "answer": answer,
            })
        else:
            per_example.append(None)  # placeholder; filled in after batch score
            indices_to_score.append(i)
            queries_to_score.append(example["query"])
            answers_to_score.append(answer)

    if indices_to_score:
        # BERTScore: cands = answers, refs = queries.
        # Returns (precision, recall, f1) each as a tensor of shape (n,).
        _P, _R, F1 = bert_score.score(
            cands=answers_to_score,
            refs=queries_to_score,
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        for list_pos, dataset_idx in enumerate(indices_to_score):
            query = examples[dataset_idx]["query"]
            answer = pipeline_outputs[dataset_idx][1]
            f1_value = round(float(F1[list_pos].item()), 6)
            per_example[dataset_idx] = {
                "query": query,
                "score": f1_value,
                "answer": answer,
            }

    aggregate = float(np.mean([e["score"] for e in per_example]))

    return {
        "scores": {"answer_relevance": round(aggregate, 6)},
        "raw": {"answer_relevance": per_example},
    }
