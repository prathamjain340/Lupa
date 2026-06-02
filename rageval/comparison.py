# rageval/comparison.py
#
# compare() function and ComparisonReport object.
#
# Design decisions (see docs/decisions.md):
#   ADR-011: Latency direction is inverted for regression detection.
#            For latency, lower is better — an increase triggers a regression
#            warning. For all other metrics, higher is better. Direction is
#            determined by metric name: "latency" is the only inverted metric.
#
# JSON input format (output of Results.to_json()):
#   {
#     "evaluated_at": "...",
#     "dataset": "...",
#     "n_examples": 10,
#     "scores": {"retrieval_precision": 0.82, "latency": 1.4},
#     "raw": {
#       "retrieval_precision": [...],
#       "latency": {"p50": 1.4, "p90": 2.1, "p99": 3.8, ...}
#     }
#   }
#
# For latency, p50 is read from raw["latency"]["p50"] (per ADR-003, this
# is also stored in scores["latency"], but raw is the authoritative source).
# For quality metrics, aggregate scores are read directly from scores dict.

from __future__ import annotations

import json
from typing import Any


_METRIC_LABELS: dict[str, str] = {
    "retrieval_precision": "Retrieval Prec.",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "latency": "Latency p50",
}

# Metrics where lower is better. All others: higher is better (ADR-011).
_LOWER_IS_BETTER = {"latency"}

# Preferred display order when metrics are present in both result files.
_METRIC_ORDER = ["retrieval_precision", "faithfulness", "answer_relevance", "latency"]


def _load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Results file not found: {path!r}. "
            "Run rageval.evaluate(...).to_json(path) to create it first."
        )


class ComparisonReport:
    """Comparison between two pipeline evaluation result files."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        regressions: list[dict[str, Any]],
        regression_threshold: float,
        baseline_path: str,
        new_path: str,
    ) -> None:
        self.rows = rows                          # one entry per compared metric
        self.regressions = regressions            # subset of rows that regressed
        self.regression_threshold = regression_threshold
        self.baseline_path = baseline_path
        self.new_path = new_path

    def report(self) -> None:
        """Print comparison table to the console."""
        print(self._build_text())

    def save(self, path: str) -> None:
        """Save comparison report to a text/markdown file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_text())

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_text(self) -> str:
        COL_LABEL = 18
        COL_VAL = 10

        lines = [
            "Lupa — Pipeline Comparison Report",
            "━" * 44,
            f"{'':18}{'baseline':<{COL_VAL}}{'new':<{COL_VAL}}{'delta'}",
        ]

        for row in self.rows:
            label = row["label"]
            is_latency = row["is_latency"]

            if is_latency:
                b_str = f"{row['baseline']:.2f}s"
                n_str = f"{row['new']:.2f}s"
                if row["delta"] < 0:
                    arrow, direction = "↑", "faster"
                elif row["delta"] > 0:
                    arrow, direction = "↓", "slower"
                else:
                    arrow, direction = "=", "unchanged"
                delta_str = f"{arrow} {direction}"
            else:
                b_str = f"{row['baseline']:.3f}"
                n_str = f"{row['new']:.3f}"
                pct = row["pct_change"]
                arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "=")
                sign = "+" if pct >= 0 else ""
                delta_str = f"{arrow} {sign}{pct:.1f}%"

            if row["is_regression"]:
                delta_str += "  ⚠ regression"

            lines.append(
                f"{label:<{COL_LABEL}}{b_str:<{COL_VAL}}{n_str:<{COL_VAL}}{delta_str}"
            )

        if self.regressions:
            lines.append("")
            threshold_pct = self.regression_threshold * 100
            for reg in self.regressions:
                pct = abs(reg["pct_change"])
                verb = "increased" if reg["is_latency"] else "dropped"
                lines.append(
                    f"⚠ Regression detected: {reg['label']} {verb} {pct:.1f}%"
                    f" (threshold: {threshold_pct:.0f}%)"
                )

        return "\n".join(lines) + "\n"


def compare(
    baseline: str,
    new: str,
    save: str = None,
    regression_threshold: float = 0.03,
) -> ComparisonReport:
    """
    Compare two sets of evaluation results and return a ComparisonReport.

    Args:
        baseline:             Path to baseline JSON results file
                              (output of Results.to_json()).
        new:                  Path to new JSON results file.
        save:                 Optional path to save the report as a text file.
        regression_threshold: Fraction threshold to flag a regression.
                              Default 0.03 = 3%. For latency, an *increase*
                              above this threshold is a regression.

    Returns:
        ComparisonReport with .report() and .save() methods.

    Raises:
        FileNotFoundError: If either results file does not exist.
    """
    baseline_data = _load_json(baseline)
    new_data = _load_json(new)

    baseline_scores: dict[str, float] = baseline_data["scores"]
    new_scores: dict[str, float] = new_data["scores"]
    baseline_raw: dict[str, Any] = baseline_data.get("raw", {})
    new_raw: dict[str, Any] = new_data.get("raw", {})

    # Ordered list of metrics present in both result files.
    in_order = [m for m in _METRIC_ORDER if m in baseline_scores and m in new_scores]
    extras = [
        m for m in baseline_scores
        if m in new_scores and m not in in_order
    ]
    metrics_to_compare = in_order + extras

    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for metric in metrics_to_compare:
        is_latency = metric in _LOWER_IS_BETTER

        if is_latency and "latency" in baseline_raw:
            b_val = baseline_raw["latency"].get("p50", baseline_scores[metric])
            n_val = new_raw.get("latency", {}).get("p50", new_scores[metric])
        else:
            b_val = baseline_scores[metric]
            n_val = new_scores[metric]

        delta = n_val - b_val
        pct_change = (delta / b_val * 100) if b_val != 0 else 0.0

        # Regression direction (ADR-011): latency increase = bad, score drop = bad.
        if is_latency:
            is_regression = pct_change > regression_threshold * 100
        else:
            is_regression = pct_change < -(regression_threshold * 100)

        label = _METRIC_LABELS.get(metric, metric.replace("_", " ").title())

        row: dict[str, Any] = {
            "metric": metric,
            "label": label,
            "baseline": b_val,
            "new": n_val,
            "delta": delta,
            "pct_change": pct_change,
            "is_regression": is_regression,
            "is_latency": is_latency,
        }
        rows.append(row)
        if is_regression:
            regressions.append(row)

    report = ComparisonReport(
        rows=rows,
        regressions=regressions,
        regression_threshold=regression_threshold,
        baseline_path=baseline,
        new_path=new,
    )

    if save is not None:
        report.save(save)

    return report
