# rageval/metrics/latency.py
#
# Computes latency percentiles from pre-collected timing data.
#
# Design decisions (see docs/decisions.md):
#   ADR-003: scores["latency"] stores p50, not mean. Mean is skewed by outliers.
#   ADR-005: Caller must use time.perf_counter() — monotonic, high-resolution.
#   ADR-006: evaluator.py owns the pipeline call loop and timing.
#            This function receives pre-collected durations and query strings;
#            it never calls pipeline_fn directly.
#
# Args:
#   durations: wall-clock seconds per example, in dataset order.
#   queries:   corresponding query strings, same order as durations.
#
# Returns:
#   {
#     "scores": {"latency": float},          # p50 in seconds
#     "raw": {
#       "latency": {
#         "p50": float,
#         "p90": float,
#         "p99": float,
#         "per_example": [{"query": str, "latency_seconds": float}, ...]
#       }
#     }
#   }

from __future__ import annotations

from typing import Any

import numpy as np


def compute_latency(
    durations: list[float],
    queries: list[str],
) -> dict[str, Any]:
    """
    Compute latency percentiles from pre-collected timing data.

    Args:
        durations: Wall-clock seconds per example (from time.perf_counter).
        queries:   Query string for each example, same order as durations.

    Returns:
        Dict with "scores" (p50 aggregate) and "raw" (all percentiles +
        per-example timings).
    """
    if len(durations) != len(queries):
        raise ValueError(
            f"durations and queries must have the same length, "
            f"got {len(durations)} and {len(queries)}."
        )

    per_example = [
        {"query": q, "latency_seconds": round(d, 6)}
        for q, d in zip(queries, durations)
    ]

    arr = np.array(durations, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))

    return {
        "scores": {"latency": round(p50, 6)},
        "raw": {
            "latency": {
                "p50": round(p50, 6),
                "p90": round(p90, 6),
                "p99": round(p99, 6),
                "per_example": per_example,
            }
        },
    }
