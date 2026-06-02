# tests/test_comparison.py
#
# Pytest suite for comparison mode (rageval.compare / ComparisonReport).
#
# All tests use pytest's tmp_path fixture for file I/O — no hardcoded
# paths, no manual cleanup. Each test writes its own minimal JSON files
# that match the Results.to_json() output format.
#
# Run: pytest tests/test_comparison.py -v

import json
import os

import pytest

from rageval.comparison import ComparisonReport, compare


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _write_results(tmp_path, filename: str, scores: dict, latency_p50: float = None) -> str:
    """
    Write a minimal Results JSON file to tmp_path and return its path.

    If latency_p50 is provided it is written into both scores["latency"]
    and raw["latency"]["p50"], matching the Results.to_json() contract.
    """
    if latency_p50 is not None:
        scores = {**scores, "latency": latency_p50}

    raw = {}
    if latency_p50 is not None:
        raw["latency"] = {"p50": latency_p50, "p90": latency_p50 * 1.5, "p99": latency_p50 * 2.0, "per_example": []}

    payload = {
        "evaluated_at": "2026-06-01T00:00:00Z",
        "dataset": "test",
        "n_examples": 2,
        "scores": scores,
        "raw": raw,
    }
    path = str(tmp_path / filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_compare_returns_comparison_report(tmp_path):
    baseline = _write_results(tmp_path, "b.json", {"retrieval_precision": 0.70})
    new      = _write_results(tmp_path, "n.json", {"retrieval_precision": 0.75})

    result = compare(baseline, new)

    assert isinstance(result, ComparisonReport)
    assert len(result.rows) == 1
    assert result.rows[0]["metric"] == "retrieval_precision"


def test_compare_detects_regression(tmp_path):
    """A metric dropping more than 3% must appear in report.regressions."""
    # answer_relevance drops from 0.83 → 0.80: -3.6%, exceeds 3% threshold.
    baseline = _write_results(tmp_path, "b.json", {"answer_relevance": 0.83})
    new      = _write_results(tmp_path, "n.json", {"answer_relevance": 0.80})

    result = compare(baseline, new, regression_threshold=0.03)

    assert len(result.regressions) == 1
    assert result.regressions[0]["metric"] == "answer_relevance"
    assert result.regressions[0]["is_regression"] is True


def test_compare_no_regression_when_improvement(tmp_path):
    """All metrics improving must produce an empty regressions list."""
    baseline = _write_results(
        tmp_path, "b.json",
        {"retrieval_precision": 0.70, "faithfulness": 0.65, "answer_relevance": 0.75},
    )
    new = _write_results(
        tmp_path, "n.json",
        {"retrieval_precision": 0.80, "faithfulness": 0.72, "answer_relevance": 0.82},
    )

    result = compare(baseline, new, regression_threshold=0.03)

    assert result.regressions == []
    assert all(not row["is_regression"] for row in result.rows)


def test_compare_latency_regression_on_increase(tmp_path):
    """Latency increasing by more than the threshold must be flagged as regression."""
    # latency increases from 1.0s → 1.2s: +20%, far above 3% threshold.
    baseline = _write_results(tmp_path, "b.json", {}, latency_p50=1.0)
    new      = _write_results(tmp_path, "n.json", {}, latency_p50=1.2)

    result = compare(baseline, new, regression_threshold=0.03)

    latency_rows = [r for r in result.rows if r["metric"] == "latency"]
    assert len(latency_rows) == 1
    assert latency_rows[0]["is_regression"] is True
    assert any(r["metric"] == "latency" for r in result.regressions)


def test_compare_latency_improvement_on_decrease(tmp_path):
    """Latency decreasing must not be flagged as regression."""
    # latency decreases from 1.0s → 0.8s: -20%, a clear improvement.
    baseline = _write_results(tmp_path, "b.json", {}, latency_p50=1.0)
    new      = _write_results(tmp_path, "n.json", {}, latency_p50=0.8)

    result = compare(baseline, new, regression_threshold=0.03)

    latency_rows = [r for r in result.rows if r["metric"] == "latency"]
    assert len(latency_rows) == 1
    assert latency_rows[0]["is_regression"] is False
    assert result.regressions == []


def test_compare_saves_file(tmp_path):
    """Passing save= must create a non-empty report file at that path."""
    baseline   = _write_results(tmp_path, "b.json", {"faithfulness": 0.68})
    new        = _write_results(tmp_path, "n.json", {"faithfulness": 0.71})
    save_path  = str(tmp_path / "report.txt")

    compare(baseline, new, save=save_path)

    assert os.path.exists(save_path)
    with open(save_path, encoding="utf-8") as f:
        content = f.read()
    assert "Lupa" in content
    assert "Faithfulness" in content


def test_compare_missing_file_raises(tmp_path):
    """Passing a nonexistent path must raise FileNotFoundError immediately."""
    real_file    = _write_results(tmp_path, "real.json", {"faithfulness": 0.70})
    missing_file = str(tmp_path / "does_not_exist.json")

    with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
        compare(missing_file, real_file)

    with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
        compare(real_file, missing_file)
