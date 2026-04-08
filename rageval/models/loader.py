# rageval/models/loader.py
#
# Lazy model loading — models are instantiated on first use, never on import.
#
# Design decisions (see docs/decisions.md):
#   ADR-008: Embedding model centralised here so the model name is configured
#            in one place. retrieval.py calls get_embedding_model() rather than
#            instantiating SentenceTransformer inline.
#
# Adding a new model in future weeks:
#   1. Add a module-level _cache variable (None sentinel).
#   2. Add a getter function that instantiates on first call and caches.
#   3. Import SentenceTransformer / AutoModel inside the getter, not at
#      module top-level, so importing rageval.models.loader never triggers
#      a download.

from __future__ import annotations

_embedding_model = None


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
