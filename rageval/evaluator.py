# rageval/evaluator.py
#
# Core evaluate() function — wires dataset loading, pipeline execution,
# and metric computation into a single Results object.
#
# Design decisions (see docs/decisions.md):
#   ADR-006: Evaluator owns the pipeline call loop. pipeline_fn is called
#            exactly once per example. All (chunks, answer) outputs and
#            wall-clock durations are collected BEFORE any metric is called.
#   ADR-002: Results receives two merged dicts: scores (flat floats) and
#            raw (full per-example / percentile data).
#
# Pipeline call loop order:
#   1. load_dataset() — validates format before any pipeline calls
#   2. for each example: time pipeline_fn(), cache (chunks, answer)
#   3. compute_latency()            if "latency" in metrics
#   4. compute_retrieval_precision() if "retrieval_precision" in metrics
#   5. Return Results(scores, raw, dataset_name, n_examples)

from __future__ import annotations

import time
from typing import Any, Callable

from rageval.datasets.loader import load_dataset
from rageval.metrics.latency import compute_latency
from rageval.metrics.retrieval import compute_retrieval_precision
from rageval.report import Results

_SUPPORTED_METRICS = {"latency", "retrieval_precision"}


def evaluate(
    pipeline_fn: Callable[[str], tuple[list[str], str]],
    dataset: str | list[dict[str, Any]],
    metrics: list[str],
    threshold: float = 0.5,
) -> Results:
    """
    Evaluate a RAG pipeline against a dataset with the requested metrics.

    Args:
        pipeline_fn: User pipeline — fn(query) -> (chunks, answer).
        dataset:     Built-in dataset name or a list of example dicts.
        metrics:     Metric names to compute. Supported: "latency",
                     "retrieval_precision".
        threshold:   Cosine similarity threshold for retrieval_precision.

    Returns:
        Results object with .report(), .save(), and .to_json() methods.

    Raises:
        ValueError: Unknown metric name, unrecognised dataset, or invalid
                    dataset format.
        TypeError:  Wrong pipeline_fn signature or wrong dataset type.
    """
    # --- 1. Validate metrics list before doing any work ---
    unknown = [m for m in metrics if m not in _SUPPORTED_METRICS]
    if unknown:
        supported = ", ".join(f'"{m}"' for m in sorted(_SUPPORTED_METRICS))
        raise ValueError(
            f"Unsupported metric(s): {unknown}. "
            f"Supported metrics: {supported}."
        )

    # --- 2. Load and validate dataset before any pipeline calls ---
    dataset_name = dataset if isinstance(dataset, str) else "custom"
    examples = load_dataset(dataset)

    # --- 3. Pipeline call loop — collect everything before metrics ---
    queries: list[str] = []
    durations: list[float] = []
    pipeline_outputs: list[tuple[list[str], str]] = []

    for example in examples:
        query = example["query"]
        start = time.perf_counter()
        output = pipeline_fn(query)
        elapsed = time.perf_counter() - start

        queries.append(query)
        durations.append(elapsed)
        pipeline_outputs.append(output)

    # --- 4. Compute requested metrics ---
    scores: dict[str, float] = {}
    raw: dict[str, Any] = {}

    if "latency" in metrics:
        result = compute_latency(durations, queries)
        scores.update(result["scores"])
        raw.update(result["raw"])

    if "retrieval_precision" in metrics:
        result = compute_retrieval_precision(examples, pipeline_outputs, threshold)
        scores.update(result["scores"])
        raw.update(result["raw"])

    # --- 5. Return Results ---
    return Results(
        scores=scores,
        raw=raw,
        dataset=dataset_name,
        n_examples=len(examples),
    )
