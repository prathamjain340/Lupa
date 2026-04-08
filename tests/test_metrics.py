# tests/test_metrics.py
#
# Pytest suite for Week 1 metrics and dataset loader.
#
# Retrieval tests use the REAL embedding model (all-MiniLM-L6-v2) and
# real document content from financialqa-mini. No mocked embeddings —
# mocking would make tests pass trivially and hide scoring regressions.
#
# Run: pytest tests/test_metrics.py -v

import numpy as np
import pytest

from rageval.datasets.loader import load_dataset
from rageval.metrics.latency import compute_latency
from rageval.metrics.retrieval import compute_retrieval_precision


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="session")
def financialqa_mini():
    """Load financialqa-mini once for the whole test session."""
    return load_dataset("rageval-financialqa-mini")


@pytest.fixture(scope="session")
def perfect_pipeline_outputs(financialqa_mini):
    """
    Pipeline outputs where every query returns only its relevant
    (non-distractor) document content. Expect precision = 1.0.
    """
    outputs = []
    for example in financialqa_mini:
        relevant_chunks = [
            doc["content"]
            for doc in example["documents"].values()
            if not doc["is_distractor"]
        ]
        outputs.append((relevant_chunks, example["ground_truth_answer"]))
    return outputs


@pytest.fixture(scope="session")
def distractor_pipeline_outputs(financialqa_mini):
    """
    Pipeline outputs where every query returns only distractor document
    content. Expect precision = 0.0.
    """
    outputs = []
    for example in financialqa_mini:
        distractor_chunks = [
            doc["content"]
            for doc in example["documents"].values()
            if doc["is_distractor"]
        ]
        outputs.append((distractor_chunks, example["ground_truth_answer"]))
    return outputs


@pytest.fixture(scope="session")
def empty_chunks_pipeline_outputs(financialqa_mini):
    """Pipeline outputs where every query returns zero chunks."""
    return [([], example["ground_truth_answer"]) for example in financialqa_mini]


# ==================================================================
# Latency tests
# ==================================================================

def test_latency_returns_correct_keys():
    durations = [0.1, 0.2, 0.3]
    queries = ["q1", "q2", "q3"]
    result = compute_latency(durations, queries)

    assert "latency" in result["scores"]
    assert "latency" in result["raw"]
    lat = result["raw"]["latency"]
    assert "p50" in lat
    assert "p90" in lat
    assert "p99" in lat
    assert "per_example" in lat


def test_latency_p50_is_median():
    durations = [0.1, 0.5, 0.2, 0.8, 0.3]
    queries = [f"q{i}" for i in range(len(durations))]
    result = compute_latency(durations, queries)

    expected_p50 = float(np.percentile(durations, 50))
    assert result["scores"]["latency"] == pytest.approx(expected_p50, rel=1e-6)
    assert result["raw"]["latency"]["p50"] == pytest.approx(expected_p50, rel=1e-6)


def test_latency_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_latency([0.1, 0.2], ["q1"])


def test_latency_per_example_count():
    durations = [0.1, 0.2, 0.3, 0.4, 0.5]
    queries = [f"q{i}" for i in range(len(durations))]
    result = compute_latency(durations, queries)

    assert len(result["raw"]["latency"]["per_example"]) == len(durations)


# ==================================================================
# Retrieval precision tests
# ==================================================================

def test_retrieval_perfect_pipeline_scores_one(
    financialqa_mini, perfect_pipeline_outputs
):
    """Returning only relevant chunks should yield precision = 1.0."""
    result = compute_retrieval_precision(
        financialqa_mini, perfect_pipeline_outputs, threshold=0.5
    )
    assert result["scores"]["retrieval_precision"] == pytest.approx(1.0, abs=0.05)


def test_retrieval_distractor_pipeline_scores_zero(
    financialqa_mini, distractor_pipeline_outputs
):
    """Returning only distractor chunks should yield precision = 0.0."""
    result = compute_retrieval_precision(
        financialqa_mini, distractor_pipeline_outputs, threshold=0.5
    )
    assert result["scores"]["retrieval_precision"] == pytest.approx(0.0, abs=0.05)


def test_retrieval_empty_chunks_scores_zero(
    financialqa_mini, empty_chunks_pipeline_outputs
):
    """Zero retrieved chunks should yield precision = 0.0 without error."""
    result = compute_retrieval_precision(
        financialqa_mini, empty_chunks_pipeline_outputs, threshold=0.5
    )
    assert result["scores"]["retrieval_precision"] == pytest.approx(0.0, abs=1e-6)


def test_retrieval_length_mismatch_raises(financialqa_mini):
    with pytest.raises(ValueError, match="same length"):
        compute_retrieval_precision(financialqa_mini, [], threshold=0.5)


def test_retrieval_per_example_shape(financialqa_mini, perfect_pipeline_outputs):
    """Each per-example entry must have all four required keys (ADR-004)."""
    result = compute_retrieval_precision(
        financialqa_mini, perfect_pipeline_outputs, threshold=0.5
    )
    breakdown = result["raw"]["retrieval_precision"]
    assert len(breakdown) == len(financialqa_mini)
    for entry in breakdown:
        assert "query" in entry
        assert "score" in entry
        assert "retrieved_chunks" in entry
        assert "relevant_doc_ids" in entry


# ==================================================================
# Dataset loader tests
# ==================================================================

def test_load_builtin_dataset_returns_ten_examples():
    examples = load_dataset("rageval-financialqa-mini")
    assert len(examples) == 10


def test_load_unknown_dataset_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown built-in dataset"):
        load_dataset("rageval-does-not-exist")


def test_load_custom_dataset_passthrough():
    custom = [
        {
            "query": "What is 2+2?",
            "ground_truth_answer": "4",
            "relevant_document_ids": ["doc_001"],
            "documents": {
                "doc_001": {"content": "2+2 equals 4.", "is_distractor": False}
            },
        }
    ]
    result = load_dataset(custom)
    assert result == custom


def test_load_empty_dataset_raises_valueerror():
    with pytest.raises(ValueError, match="empty"):
        load_dataset([])


def test_load_missing_field_raises_valueerror():
    bad_example = [
        {
            # "query" is intentionally missing
            "ground_truth_answer": "some answer",
            "relevant_document_ids": ["doc_001"],
            "documents": {
                "doc_001": {"content": "some content", "is_distractor": False}
            },
        }
    ]
    with pytest.raises(ValueError, match="query"):
        load_dataset(bad_example)
