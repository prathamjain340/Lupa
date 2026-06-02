# rageval/metrics/faithfulness.py
#
# Computes faithfulness: fraction of answer sentences that are entailed by
# the retrieved context, per example.
#
# Design decisions (see docs/decisions.md):
#   ADR-009: NLI model is cross-encoder/nli-deberta-v3-small, loaded via
#            get_nli_model(). CrossEncoder.predict() returns logits for
#            [contradiction, entailment, neutral] — index 1 is entailment.
#   ADR-006: pipeline_outputs is the cached evaluator output — index-aligned
#            with examples. This function never calls pipeline_fn.
#
# Algorithm per example:
#   1. Get answer from pipeline_outputs[i][1].
#   2. Concatenate all retrieved chunks from pipeline_outputs[i][0] into
#      a single premise string (joined with " ").
#   3. Split the answer into sentences with a regex on sentence-ending
#      punctuation. Filter out blank sentences.
#   4. For each sentence, run CrossEncoder.predict([(premise, sentence)]).
#      A sentence is "entailed" if argmax of the 3-class logits == 1.
#   5. faithfulness = entailed_count / total_sentences.
#   6. No chunks retrieved → 0.0. Empty answer → 0.0.
#
# Aggregate score = mean faithfulness across all examples.

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rageval.models.loader import get_nli_model

# Label index for "entailment" in cross-encoder/nli-deberta-v3-small output.
# Label order: [contradiction=0, entailment=1, neutral=2]
_ENTAILMENT_IDX = 1


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries; drop blank fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def compute_faithfulness(
    examples: list[dict[str, Any]],
    pipeline_outputs: list[tuple[list[str], str]],
) -> dict[str, Any]:
    """
    Compute faithfulness for each example and aggregate.

    Args:
        examples:         Dataset examples; each must have "query".
        pipeline_outputs: Cached (chunks, answer) tuples from the evaluator loop,
                          index-aligned with examples.

    Returns:
        Dict with "scores" (mean faithfulness) and "raw" (per-example breakdown).
    """
    if len(examples) != len(pipeline_outputs):
        raise ValueError(
            f"examples and pipeline_outputs must have the same length, "
            f"got {len(examples)} and {len(pipeline_outputs)}."
        )

    model = get_nli_model()
    per_example: list[dict[str, Any]] = []

    for example, (chunks, answer) in zip(examples, pipeline_outputs):
        query = example["query"]

        if not chunks or not answer.strip():
            per_example.append({
                "query": query,
                "score": 0.0,
                "answer": answer,
                "retrieved_chunks": chunks,
            })
            continue

        premise = " ".join(chunks)
        sentences = _split_sentences(answer)

        if not sentences:
            per_example.append({
                "query": query,
                "score": 0.0,
                "answer": answer,
                "retrieved_chunks": chunks,
            })
            continue

        # Build (premise, hypothesis) pairs for batch NLI prediction.
        pairs = [(premise, sentence) for sentence in sentences]
        # logits shape: (n_sentences, 3) — [contradiction, entailment, neutral]
        logits = model.predict(pairs)
        entailed_count = int(np.sum(np.argmax(logits, axis=1) == _ENTAILMENT_IDX))
        score = entailed_count / len(sentences)

        per_example.append({
            "query": query,
            "score": round(score, 6),
            "answer": answer,
            "retrieved_chunks": chunks,
        })

    aggregate = float(np.mean([e["score"] for e in per_example]))

    return {
        "scores": {"faithfulness": round(aggregate, 6)},
        "raw": {"faithfulness": per_example},
    }
