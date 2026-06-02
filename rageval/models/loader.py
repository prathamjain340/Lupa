# rageval/models/loader.py
#
# Lazy model loading — models are instantiated on first use, never on import.
#
# Design decisions (see docs/decisions.md):
#   ADR-008: Embedding model centralised here so the model name is configured
#            in one place. retrieval.py calls get_embedding_model() rather than
#            instantiating SentenceTransformer inline.
#   ADR-009: NLI model uses CrossEncoder for premise-hypothesis entailment
#            scoring. Separate cache from embedding model — different class,
#            different lifecycle.
#
# Adding a new model:
#   1. Add a module-level _cache variable (None sentinel).
#   2. Add a getter function that instantiates on first call and caches.
#   3. Import SentenceTransformer / CrossEncoder / AutoModel inside the getter,
#      not at module top-level, so importing rageval.models.loader never
#      triggers a download.

from __future__ import annotations

_embedding_model = None
_nli_model = None


def get_embedding_model():
    """
    Return the shared SentenceTransformer instance, loading it on first call.

    Model: all-MiniLM-L6-v2 — 80 MB, fast on CPU, good semantic similarity.
    The first call triggers a download if the model is not cached locally.
    Subsequent calls return the already-loaded instance.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_nli_model():
    """
    Return the shared CrossEncoder NLI instance, loading it on first call.

    Model: cross-encoder/nli-deberta-v3-small — ~180 MB, fast on CPU.
    Scores (premise, hypothesis) pairs directly as entailment/neutral/contradiction.
    The first call triggers a download if the model is not cached locally.
    Subsequent calls return the already-loaded instance.
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model
