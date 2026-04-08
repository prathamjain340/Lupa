# rageval/metrics/retrieval.py
#
# Computes retrieval_precision: fraction of retrieved chunks that are
# semantically relevant to a non-distractor document, per example.
#
# Design decisions (see docs/decisions.md):
#   ADR-004: Per-example entries include query, score, retrieved_chunks,
#            and relevant_doc_ids (ground truth IDs for debuggability).
#   ADR-006: pipeline_outputs is the cached evaluator output — index-aligned
#            with examples. This function never calls pipeline_fn.
#   ADR-008: Embedding model loaded via get_embedding_model(), not inline.
#
# Algorithm per example:
#   1. Retrieve chunks from pipeline_outputs[i][0].
#   2. Collect all document contents + distractor flags from examples[i].
#   3. Encode all chunks and all document contents with the embedding model
#      (normalize_embeddings=True so cosine sim = dot product).
#   4. For each chunk, find the document with the highest cosine similarity.
#   5. Chunk is "relevant" if best_sim >= threshold AND that doc is not a
#      distractor.
#   6. precision = relevant / total_retrieved. Zero chunks → 0.0.
#
# Aggregate score = mean precision across all examples.

from __future__ import annotations

from typing import Any

import numpy as np

from rageval.models.loader import get_embedding_model


def compute_retrieval_precision(
    examples: list[dict[str, Any]],
    pipeline_outputs: list[tuple[list[str], str]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute retrieval precision for each example and aggregate.

    Args:
        examples:         Dataset examples; each must have "query" and "documents".
        pipeline_outputs: Cached (chunks, answer) tuples from the evaluator loop,
                          index-aligned with examples.
        threshold:        Cosine similarity threshold to count a chunk as relevant.

    Returns:
        Dict with "scores" (mean precision) and "raw" (per-example breakdown).
    """
    if len(examples) != len(pipeline_outputs):
        raise ValueError(
            f"examples and pipeline_outputs must have the same length, "
            f"got {len(examples)} and {len(pipeline_outputs)}."
        )

    model = get_embedding_model()
    per_example: list[dict[str, Any]] = []

    for i, (example, (chunks, _answer)) in enumerate(zip(examples, pipeline_outputs)):
        query = example["query"]
        ground_truth_doc_ids = example["relevant_document_ids"]

        if not chunks:
            per_example.append({
                "query": query,
                "score": 0.0,
                "retrieved_chunks": [],
                "relevant_doc_ids": ground_truth_doc_ids,
            })
            continue

        # Build parallel lists of document contents and their distractor flags.
        doc_contents: list[str] = []
        doc_is_distractor: list[bool] = []
        for doc in example["documents"].values():
            doc_contents.append(doc["content"])
            doc_is_distractor.append(doc["is_distractor"])

        # Encode chunks and documents; normalize so dot product = cosine sim.
        chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
        doc_embeddings = model.encode(doc_contents, normalize_embeddings=True)

        # Shape: (n_chunks, n_docs)
        similarity_matrix = np.dot(chunk_embeddings, doc_embeddings.T)

        relevant_count = 0
        for chunk_sims in similarity_matrix:
            best_doc_idx = int(np.argmax(chunk_sims))
            best_sim = float(chunk_sims[best_doc_idx])
            if best_sim >= threshold and not doc_is_distractor[best_doc_idx]:
                relevant_count += 1

        precision = relevant_count / len(chunks)

        per_example.append({
            "query": query,
            "score": round(precision, 6),
            "retrieved_chunks": chunks,
            "relevant_doc_ids": ground_truth_doc_ids,
        })

    aggregate = float(np.mean([e["score"] for e in per_example]))

    return {
        "scores": {"retrieval_precision": round(aggregate, 6)},
        "raw": {"retrieval_precision": per_example},
    }
