# Lupa — RAG Evaluation Framework
## Package name: rageval | pip install rageval

---

## What we're building
Framework-agnostic RAG evaluation library. User provides a pipeline
function, a dataset, and metrics. Library returns scored results with
a markdown report.

No LangChain. No LlamaIndex. No OpenAI API required for core metrics.
Fully local evaluation.

---

## Pipeline function contract
```python
fn(query: str) -> tuple[list[str], str]
# returns: (list of retrieved chunk strings, answer string)
```

User wraps their pipeline — whatever framework they use — into this
shape. That's the entire integration surface.

---

## Complete API

```python
# Core evaluation
rageval.evaluate(
    pipeline_fn,          # fn(query) -> (chunks, answer)
    dataset,              # str (built-in name) or list[dict] (custom)
    metrics,              # list of metric names
    threshold=0.5         # similarity threshold for retrieval_precision
) -> Results

# Compare two pipeline versions
rageval.compare(
    baseline,             # path to baseline JSON results
    new,                  # path to new JSON results
    save=None,            # optional path to save comparison report
    regression_threshold=0.03  # flag if any metric drops more than 3%
) -> ComparisonReport

# Generate a custom dataset from your own documents
rageval.generate_dataset(
    documents,            # list[str]
    n_questions=50,
    llm="openai"          # or "anthropic" — user provides API key
) -> list[dict]

# Results methods
Results.report()          # prints markdown report to console
Results.save(path)        # saves markdown file
Results.to_json(path)     # saves raw scores as JSON

# ComparisonReport methods  
ComparisonReport.report() # prints comparison table to console
ComparisonReport.save(path)
```

---

## All Metrics

| Metric | Method | API Required | Week |
|--------|--------|--------------|------|
| latency | wall-clock p50/p90/p99 | No | 1 |
| retrieval_precision | sentence-transformers similarity, threshold-based | No | 1 |
| faithfulness | NLI primary (DeBERTa cross-encoder), BERTScore secondary | No | 2 |
| answer_relevance | BERTScore between query and answer | No | 2 |

Every metric returns:
- A single aggregate score 0-1 for the full dataset
- A per-example breakdown so bad queries are identifiable

---

## Dataset Format

Each example in a dataset:

```json
{
  "query": "What was the company revenue in Q3 2023?",
  "ground_truth_answer": "The company reported Q3 2023 revenue of $4.2 billion.",
  "relevant_document_ids": ["doc_001"],
  "documents": {
    "doc_001": {
      "content": "In Q3 2023, the company reported revenue of $4.2 billion, up 12% year over year.",
      "is_distractor": false
    },
    "doc_002": {
      "content": "The company reported Q2 2023 revenue of $3.9 billion.",
      "is_distractor": true
    },
    "doc_003": {
      "content": "Annual revenue targets for 2023 were set at $16 billion.",
      "is_distractor": true
    }
  }
}
```

Distractor documents look relevant but do not answer the query.
They are used to test whether retrieval pulls the right chunk vs
a plausible-looking wrong one.

---

## Built-in Datasets

Three hand-curated datasets shipped with the library:

| Name | Domain | Examples | Purpose |
|------|--------|----------|---------|
| rageval-financialqa | Financial docs, earnings, filings | 75 | Finance RAG pipelines |
| rageval-technicaldocs | API docs, technical references | 75 | Dev tool RAG pipelines |
| rageval-general | General knowledge, mixed topics | 75 | General purpose testing |

Week 1 ships `rageval-financialqa-mini` — 10 hand-crafted examples
from the financial domain, used for development and testing.

---

## Comparison Mode Output

```
Lupa — Pipeline Comparison Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      baseline    new       delta
Retrieval Precision     0.74      0.81      ↑ +9.5%
Faithfulness            0.68      0.71      ↑ +4.4%
Answer Relevance        0.83      0.80      ↓ -3.6%  ⚠ regression
Latency p50             1.4s      1.1s      ↑ faster
Latency p99             3.1s      2.8s      ↑ faster

⚠ Regression detected: Answer Relevance dropped 3.6% (threshold: 3%)
Tip: Retrieval improved but answers may be less directly addressing
queries — check generation prompt or context window size.
```

Regression threshold defaults to 3%, configurable per run.

---

## Folder Structure

```
Lupa/
├── CLAUDE.md
├── SPEC.md
├── README.md
├── .gitignore
├── LICENSE
├── pyproject.toml
├── rageval/
│   ├── __init__.py           # exports: evaluate, compare, generate_dataset
│   ├── evaluator.py          # core evaluate() function
│   ├── comparison.py         # compare() function + ComparisonReport
│   ├── report.py             # Results object, markdown + json output
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── latency.py        # wall-clock timing, percentiles
│   │   ├── retrieval.py      # retrieval_precision
│   │   ├── faithfulness.py   # NLI + BERTScore (Week 2)
│   │   └── relevance.py      # answer relevance BERTScore (Week 2)
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── loader.py         # load built-in and custom datasets
│   │   ├── generator.py      # generate_dataset() utility (Week 3)
│   │   └── builtin/
│   │       ├── financialqa_mini.json    # 10 examples, Week 1
│   │       ├── financialqa.json         # 75 examples, Week 3
│   │       ├── technicaldocs.json       # 75 examples, Week 3
│   │       └── general.json             # 75 examples, Week 3
│   └── models/
│       ├── __init__.py
│       └── loader.py         # lazy model loading — NLI, sentence-transformers
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py       # unit tests for each metric
│   ├── test_datasets.py      # dataset loading and format validation
│   └── test_comparison.py    # comparison mode tests
├── examples/
│   ├── basic_eval.py         # minimal working example
│   ├── compare_pipelines.py  # comparison mode example
│   └── custom_dataset.py     # using your own dataset
└── docs/
    └── quickstart.md
```

---

## Dependencies

```toml
[project]
dependencies = [
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "bert-score>=0.3.13",
    "numpy>=1.24.0",
    "rich>=13.0.0",
]
```

NO langchain. NO openai in core. All evaluation runs locally.

---

## Design Principles

- Fail loudly — clear error messages for wrong function signatures,
  missing fields, unsupported metric names
- Lazy load ML models — nothing downloads on import, only when
  the relevant metric is first used
- Every metric returns score 0-1 AND per-example breakdown
- No side effects on import
- Framework agnostic — works with any RAG pipeline
- Reproducible — same inputs always produce same scores

---

## Build Order by Week

### Week 1 — Core skeleton + retrieval
1. Full folder structure (stubs for unimplemented files)
2. Results object in report.py — stores scores, .report(), .to_json()
3. latency metric — wall-clock timing, p50/p90/p99
4. Dataset loader — loads financialqa_mini.json, validates format
5. financialqa_mini.json — 10 hand-crafted examples with distractors
6. retrieval_precision metric — sentence-transformers similarity
7. evaluator.py — wires everything together, evaluate() function
8. examples/basic_eval.py — end to end working example
9. tests/test_metrics.py — tests for latency + retrieval_precision

Milestone: `rageval.evaluate(fn, "rageval-financialqa-mini",
["retrieval_precision", "latency"])` runs and returns a report.

### Week 2 — Generation quality metrics
1. models/loader.py — lazy loading for NLI model + sentence-transformers
2. faithfulness metric — NLI primary, BERTScore secondary
3. answer_relevance metric — BERTScore
4. Update evaluator.py to support new metrics
5. tests for faithfulness and answer_relevance with known examples

Milestone: All 4 metrics working on financialqa_mini.

### Week 3 — Comparison mode + full datasets
1. comparison.py — compare() function, regression detection
2. ComparisonReport object with .report() and .save()
3. financialqa.json — 75 curated examples
4. technicaldocs.json — 75 curated examples
5. general.json — 75 curated examples
6. examples/compare_pipelines.py
7. tests/test_comparison.py

Milestone: `rageval.compare(baseline, new)` works with regression warnings.

### Week 4 — Polish + generator + launch prep
1. generator.py — generate_dataset() utility with LLM
2. examples/custom_dataset.py
3. docs/quickstart.md
4. pyproject.toml finalized, package installable via pip
5. README polished with real benchmark numbers from actual runs
6. Loss curve / benchmark charts for transformer repo (separate)
7. Reddit post draft

Milestone: v0.1 published to PyPI, Reddit post live.

---

## Environment
Fresh venv at Lupa/venv/
Pin all dependency versions explicitly in pyproject.toml
