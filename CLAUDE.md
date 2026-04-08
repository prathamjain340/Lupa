# Lupa — Claude Code Context

---

## What this project is
Lupa is a RAG evaluation framework. Package name: rageval.
pip install rageval

Full spec in SPEC.md — read it at the start of every session
before writing any code.

---

## My role as architect
I make all design decisions. You implement what I decide.

If you face any decision not already resolved in SPEC.md or in the
"Architecture decisions" section below — STOP and ask me before
implementing. Do not make silent design choices. Ever.

If something could reasonably be done two ways, show me both options
with tradeoffs and ask which I prefer.

---

## How we work
- Read SPEC.md fully before starting any session
- Build one file at a time
- Show me the complete file after writing it
- Wait for my confirmation before moving to the next file
- Add a comment block at the top of each non-trivial file explaining
  what it does and any decisions made in it
- Write tests alongside implementation, not after
- Commit after each working component with conventional commit format

---

## Architecture decisions (locked — do not reopen)

**Interface:** Function-based primary
```python
fn(query: str) -> tuple[list[str], str]
```
Structured class-based interface is a future v2 addition, not v1.

**Faithfulness metric:** NLI model primary (DeBERTa cross-encoder),
BERTScore secondary. No LLM API calls for evaluation.

**Datasets:** Hand-curated built-ins + generate_dataset() utility.
Week 1 ships financialqa_mini.json with 10 examples.

**Output v1:** JSON + Markdown only. HTML dashboard is v2.

**Comparison regression threshold:** 3% default, configurable.

**Dependencies:** sentence-transformers, torch, transformers,
bert-score, numpy, rich. NO langchain. NO openai in core.

**Model loading:** Lazy — models download only when first metric
using them is called, never on import.

**Metric output contract:** Every metric must return both an
aggregate score (float 0-1) and a per-example breakdown (list).

**Error handling:** Fail loudly. Wrong function signature, missing
dataset fields, unsupported metric names — raise clear exceptions
with helpful messages, not silent failures.

---

## Dataset format reference
```json
{
  "query": "string",
  "ground_truth_answer": "string",
  "relevant_document_ids": ["doc_001"],
  "documents": {
    "doc_001": {
      "content": "string",
      "is_distractor": false
    },
    "doc_002": {
      "content": "string",
      "is_distractor": true
    }
  }
}
```

---

## Current build stage
Week 1 — build in this exact order:

1. Full folder structure (stubs for Week 2+ files)
2. Results object — report.py
3. latency metric — metrics/latency.py
4. Dataset loader — datasets/loader.py
5. financialqa_mini.json — 10 examples (I will provide these)
6. retrieval_precision metric — metrics/retrieval.py
7. evaluator.py — wires everything together
8. examples/basic_eval.py
9. tests/test_metrics.py

Do not start Week 2 files until I say so.

---

## Commit style
```
feat: add retrieval_precision metric
feat: add Results object with json and markdown output
test: add tests for latency metric
fix: handle empty retrieved chunks in retrieval_precision
chore: add pyproject.toml with pinned dependencies
```

---

## Environment
venv at Lupa/venv/ — always use this environment
All dependencies pinned in pyproject.toml
