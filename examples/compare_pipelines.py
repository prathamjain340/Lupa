# examples/compare_pipelines.py
#
# Demonstrates comparison mode: compare two pipeline result JSON files
# and print a report showing improvements, regressions, and latency changes.
#
# Run: python examples/compare_pipelines.py
#
# This example builds two mock result dicts that simulate a realistic
# pipeline iteration — retrieval and faithfulness improved, but answer
# relevance regressed slightly (just over the 3% threshold) and latency
# improved. The compare() call catches the regression automatically.

import json
import os
import shutil
import sys

# Force UTF-8 output on Windows — required for ━ ↑ ↓ ⚠ in the report.
sys.stdout.reconfigure(encoding="utf-8")

import rageval

# ------------------------------------------------------------------
# Build mock result dicts matching the Results.to_json() output format.
# ------------------------------------------------------------------

BASELINE = {
    "evaluated_at": "2026-06-01T10:00:00Z",
    "dataset": "rageval-financialqa-mini",
    "n_examples": 10,
    "scores": {
        "retrieval_precision": 0.74,
        "faithfulness": 0.68,
        "answer_relevance": 0.83,
        "latency": 1.4,
    },
    "raw": {
        "latency": {"p50": 1.4, "p90": 2.3, "p99": 3.1, "per_example": []},
        "retrieval_precision": [],
        "faithfulness": [],
        "answer_relevance": [],
    },
}

NEW = {
    "evaluated_at": "2026-06-02T10:00:00Z",
    "dataset": "rageval-financialqa-mini",
    "n_examples": 10,
    "scores": {
        "retrieval_precision": 0.81,
        "faithfulness": 0.71,
        "answer_relevance": 0.80,   # dropped 3.6% — triggers regression warning
        "latency": 1.1,
    },
    "raw": {
        "latency": {"p50": 1.1, "p90": 1.8, "p99": 2.5, "per_example": []},
        "retrieval_precision": [],
        "faithfulness": [],
        "answer_relevance": [],
    },
}

# ------------------------------------------------------------------
# Write temp files, run comparison, clean up.
# ------------------------------------------------------------------

scratch_dir = os.path.join(os.path.dirname(__file__), "..", "scratch")
os.makedirs(scratch_dir, exist_ok=True)

baseline_path = os.path.join(scratch_dir, "baseline_results.json")
new_path = os.path.join(scratch_dir, "new_results.json")

with open(baseline_path, "w", encoding="utf-8") as f:
    json.dump(BASELINE, f, indent=2)

with open(new_path, "w", encoding="utf-8") as f:
    json.dump(NEW, f, indent=2)

report = rageval.compare(baseline_path, new_path, regression_threshold=0.03)
report.report()

# Clean up scratch files.
shutil.rmtree(scratch_dir)
