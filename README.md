# Lupa

**RAG evaluation that actually tells you what's wrong.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/badge/pip%20install-lupa--rageval-orange)

---

Most developers evaluate RAG by eyeballing outputs. When something breaks,
they don't know if it's retrieval or generation. Lupa tells you exactly where.

---

## Install

```bash
pip install lupa-rageval
```

---

## Quick Start

```python
import rageval

def my_pipeline(query: str) -> tuple[list[str], str]:
    # Wrap your existing RAG pipeline — any framework works.
    chunks = retriever.search(query)
    answer = llm.generate(query, chunks)
    return chunks, answer

results = rageval.evaluate(
    my_pipeline,
    dataset="rageval-financialqa-mini",
    metrics=["retrieval_precision", "faithfulness", "answer_relevance", "latency"],
)

results.report()       # prints markdown table to console
results.save("report.md")
results.to_json("results.json")
```

---

## Metrics

| Metric | What it measures | API required? |
|---|---|---|
| `retrieval_precision` | Fraction of retrieved chunks that are semantically relevant (not distractors) | No |
| `faithfulness` | Fraction of answer sentences entailed by the retrieved context | No |
| `answer_relevance` | How well the answer addresses the query (BERTScore F1) | No |
| `latency` | Wall-clock time per query — reports p50, p90, p99 | No |

All metrics run fully locally using sentence-transformers and BERTScore.
No evaluation API calls. No per-query cost.

---

## Real Benchmark Results

**Pipeline:** GPT-4o-mini + text-embedding-3-small + Chroma  
**Dataset:** rageval-financialqa-mini (10 financial QA examples)

| Metric | Score |
|---|---|
| Retrieval Precision | 0.300 |
| Faithfulness | 0.300 |
| Answer Relevance | 0.883 |
| Latency p50 | 1.58s |
| Latency p90 | 2.33s |
| Latency p99 | 3.20s |

Retrieval precision of 0.30 revealed the pipeline was retrieving wrong-quarter
documents (e.g. Q1 2024 instead of Q1 2023) — exactly the distractor failure
mode the dataset is designed to catch. Answer relevance stayed high at 0.88,
showing the LLM answered well given its context, but that context was often wrong.

---

## Why Lupa?

- **Framework agnostic** — one function signature works with LangChain, LlamaIndex,
  raw API calls, or anything else. Lupa never touches your pipeline internals.
- **Fully local** — all four metrics run on-device. No LLM API calls for evaluation,
  no per-query cost, no data leaving your machine.
- **Diagnostic** — every metric returns a per-example breakdown. See exactly which
  queries failed retrieval and which answers were hallucinated.
- **Comparable** — `compare()` diffs two result files and flags regressions
  automatically. Safe to run in CI before merging a prompt change.

---

## Comparing Pipeline Versions

```python
# Evaluate baseline
baseline = rageval.evaluate(baseline_pipeline, dataset, metrics)
baseline.to_json("baseline.json")

# Evaluate new version
new = rageval.evaluate(new_pipeline, dataset, metrics)
new.to_json("new.json")

# Compare — flags any metric that dropped more than 3%
report = rageval.compare("baseline.json", "new.json")
report.report()
```

```
Lupa — Pipeline Comparison Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  baseline  new       delta
Retrieval Prec.   0.740     0.810     ↑ +9.5%
Faithfulness      0.680     0.710     ↑ +4.4%
Answer Relevance  0.830     0.800     ↓ -3.6%  ⚠ regression
Latency p50       1.40s     1.10s     ↑ faster

⚠ Regression detected: Answer Relevance dropped 3.6% (threshold: 3%)
```

---

## Generating Your Own Dataset

Build an evaluation dataset from your own documents:

```python
dataset = rageval.generate_dataset(
    documents=my_docs,    # list of document strings from your corpus
    n_questions=50,
    llm="openai",         # or "anthropic"
    output_path="my_dataset.json",
)
```

Requires `pip install openai` or `pip install anthropic`.
The LLM generates questions, ground-truth answers, and distractor documents automatically.

---

## Links

- [Quickstart Guide](docs/quickstart.md)
- [PyPI](https://pypi.org/project/rageval)

---

## License

MIT
