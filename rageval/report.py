# rageval/report.py
#
# Results object returned by rageval.evaluate().
#
# Design decisions (see docs/decisions.md):
#   ADR-002: scores holds flat aggregate floats (one per metric).
#            raw holds full per-example and percentile data.
#   ADR-003: scores["latency"] is p50; p90/p99 live in raw["latency"].
#
# scores shape:
#   {
#     "retrieval_precision": 0.82,
#     "latency": 1.4,            # p50 in seconds
#   }
#
# raw shape:
#   {
#     "retrieval_precision": [{"query": ..., "score": ...}, ...],
#     "latency": {"p50": 1.4, "p90": 2.1, "p99": 3.8, "per_example": [...]},
#   }

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for markdown table display and escape pipe characters."""
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


class Results:
    """Stores evaluation scores and produces markdown / JSON output."""

    def __init__(
        self,
        scores: dict[str, float],
        raw: dict[str, Any],
        dataset: str,
        n_examples: int,
    ) -> None:
        self.scores = scores          # flat aggregate floats, one per metric
        self.raw = raw                # full per-example + percentile data
        self.dataset = dataset        # dataset name or "<custom>"
        self.n_examples = n_examples
        self.evaluated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a markdown report to the console."""
        print(self._build_markdown())

    def save(self, path: str) -> None:
        """Save the markdown report to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_markdown())

    def to_json(self, path: str) -> None:
        """Save raw scores as JSON for programmatic access or comparison mode."""
        payload = {
            "evaluated_at": self.evaluated_at,
            "dataset": self.dataset,
            "n_examples": self.n_examples,
            "scores": self.scores,
            "raw": self.raw,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_markdown(self) -> str:
        lines = [
            "# Lupa — Evaluation Report",
            "",
            f"**Dataset:** {self.dataset}  ",
            f"**Examples:** {self.n_examples}  ",
            f"**Evaluated at:** {self.evaluated_at}",
            "",
            "## Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]

        for metric, score in self.scores.items():
            if metric == "latency":
                lines.append(f"| Latency (p50) | {score:.2f}s |")
            else:
                label = metric.replace("_", " ").title()
                lines.append(f"| {label} | {score:.3f} |")

        # Latency percentile detail when present
        if "latency" in self.raw:
            lat = self.raw["latency"]
            lines += [
                "",
                "## Latency Percentiles",
                "",
                "| p50 | p90 | p99 |",
                "|-----|-----|-----|",
                f"| {lat['p50']:.2f}s | {lat['p90']:.2f}s | {lat['p99']:.2f}s |",
            ]

        # Per-example breakdown for non-latency metrics
        for metric, breakdown in self.raw.items():
            if metric == "latency":
                continue
            if not isinstance(breakdown, list) or not breakdown:
                continue

            label = metric.replace("_", " ").title()
            has_chunks = "retrieved_chunks" in breakdown[0]

            if has_chunks:
                lines += [
                    "",
                    f"## {label} — Per-Example Breakdown",
                    "",
                    "| # | Query | Score | Top Retrieved Chunk |",
                    "|---|-------|-------|---------------------|",
                ]
                for i, entry in enumerate(breakdown, start=1):
                    query_preview = _truncate(entry["query"], 60)
                    chunks = entry.get("retrieved_chunks", [])
                    top_chunk = _truncate(chunks[0], 80) if chunks else "—"
                    lines.append(
                        f"| {i} | {query_preview} | {entry['score']:.3f} | {top_chunk} |"
                    )
            else:
                lines += [
                    "",
                    f"## {label} — Per-Example Breakdown",
                    "",
                    "| # | Query | Score |",
                    "|---|-------|-------|",
                ]
                for i, entry in enumerate(breakdown, start=1):
                    query_preview = _truncate(entry["query"], 60)
                    lines.append(f"| {i} | {query_preview} | {entry['score']:.3f} |")

        return "\n".join(lines) + "\n"
