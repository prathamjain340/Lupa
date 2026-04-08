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

## ADR-004 — retrieval_precision per-example entries include retrieved chunks
**Decision:** Each per-example entry stores query, score,
retrieved_chunks (full in raw/JSON), and relevant_doc_ids from
ground truth.
**Reason:** A precision score alone doesn't tell you why retrieval
failed. Showing which chunks were actually retrieved lets users
immediately see wrong-quarter or wrong-metric distractor problems.
Chunks truncated to 80 chars in .report() table, full content in
.to_json().
