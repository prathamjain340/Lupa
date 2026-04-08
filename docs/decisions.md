# Architecture Decision Log

## ADR-001 — Deferred imports in __init__.py
**Decision:** Deferred imports inside each function, not at module level.
**Reason:** Stubs are pass statements during development. Eager imports
would make the package unimportable until all modules are implemented.
Deferred imports mean errors surface only when the unimplemented function
is called, not on import.

## ADR-002 — Results object uses two separate dicts: scores and raw
**Decision:** scores dict holds flat aggregate floats (one per metric).
raw dict holds full per-example and percentile data.
**Reason:** Report logic should never need to know the shape of a specific
metric's output. Flat scores dict keeps the markdown report rendering
consistent regardless of how many metrics are added. Full data lives in
raw for programmatic access via .to_json().

## ADR-003 — Latency representative score is p50, not mean
**Decision:** scores["latency"] stores p50 value.
**Reason:** Mean latency is skewed by outliers. P50 is the honest
middle-ground number. P90 and P99 are preserved in raw["latency"]
for users who care about tail latency behavior.

## ADR-008 — retrieval_precision uses all-MiniLM-L6-v2 via models/loader.py
**Decision:** Embedding model loaded through get_embedding_model()
in models/loader.py, not instantiated inline in retrieval.py.
Model: all-MiniLM-L6-v2 — small (80MB), fast on CPU, sufficient
for semantic similarity scoring.
**Reason:** Centralising model loading means the model name is
configured in one place. Lazy loading means the 80MB download
only happens when retrieval_precision is first used, not on import.

## ADR-007 — loader uses importlib.resources not __file__
**Decision:** Built-in dataset files located via importlib.resources.
**Reason:** __file__ path construction breaks when the package is
installed as a wheel or zip. importlib.resources works correctly
in all installation contexts including pip install.

## ADR-006 — Evaluator owns the pipeline call loop, metrics receive cached outputs
**Decision:** evaluator.py calls pipeline_fn once per example,
records wall-clock time, caches (chunks, answer). Latency metric
receives pre-collected durations. All other metrics receive cached
(chunks, answer) outputs.
**Reason:** Option B (each metric calls pipeline_fn independently)
multiplies API costs and wall-clock time by the number of metrics.
A pipeline with 4 metrics would make 4x LLM API calls per example.
Option A makes exactly one call per example regardless of how many
metrics are requested.

## ADR-005 — Latency uses time.perf_counter not time.time
**Decision:** Wall-clock timing uses time.perf_counter().
**Reason:** perf_counter is monotonic (never goes backwards) and
has higher resolution than time.time(). time.time() can jump
backward on clock adjustments, corrupting latency measurements.

## ADR-004 — retrieval_precision per-example entries include retrieved chunks
**Decision:** Each per-example entry stores query, score,
retrieved_chunks (full in raw/JSON), and relevant_doc_ids from
ground truth.
**Reason:** A precision score alone doesn't tell you why retrieval
failed. Showing which chunks were actually retrieved lets users
immediately see wrong-quarter or wrong-metric distractor problems.
Chunks truncated to 80 chars in .report() table, full content in
.to_json().
