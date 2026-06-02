# Lupa Quickstart

Lupa is a RAG evaluation framework. Give it a pipeline function and a dataset;
it returns scores and a diagnostic report — no API keys required for evaluation.

---

## Installation

```bash
pip install rageval
```

---

## Quick Start

Wrap your RAG pipeline in a single function and pass it to `evaluate()`:

```python
import rageval

def my_pipeline(query: str) -> tuple[list[str], str]:
    # Your retrieval + generation logic here.
    # Return: (list of retrieved chunk strings, answer string)
    chunks = retriever.search(query)
    answer = llm.generate(query, chunks)
    return chunks, answer

results = rageval.evaluate(
    my_pipeline,
    dataset="rageval-financialqa-mini",
    metrics=["retrieval_precision", "faithfulness", "answer_relevance", "latency"],
)

results.report()     # prints markdown table to console
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
| `latency` | Wall-clock time per query — p50, p90, p99 | No |

All metrics run fully locally. No OpenAI or Anthropic key needed for evaluation.

---

## Built-in Datasets

| Name | Domain | Examples |
|---|---|---|
| `rageval-financialqa-mini` | Financial earnings, filings | 10 |

Pass the name as a string to `evaluate()`. Full datasets (75 examples each) coming soon.

You can also pass your own list of examples directly:

```python
results = rageval.evaluate(my_pipeline, dataset=my_examples, metrics=["faithfulness"])
```

See [Dataset format](#dataset-format) at the bottom of this page for the required schema.

---

## Comparing Pipeline Versions

Evaluate baseline and new, save both to JSON, then compare:

```python
# Run baseline
baseline_results = rageval.evaluate(baseline_pipeline, dataset, metrics)
baseline_results.to_json("baseline.json")

# Run new version
new_results = rageval.evaluate(new_pipeline, dataset, metrics)
new_results.to_json("new.json")

# Compare — flags any metric that dropped more than 3%
report = rageval.compare("baseline.json", "new.json")
report.report()
```

Output:
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

The regression threshold is configurable: `rageval.compare(..., regression_threshold=0.05)`.

---

## Generating Your Own Dataset

Build an evaluation dataset from your own documents:

```python
dataset = rageval.generate_dataset(
    documents=my_docs,   # list of document strings
    n_questions=50,
    llm="openai",        # or "anthropic"
    output_path="my_dataset.json",
)

results = rageval.evaluate(my_pipeline, dataset=dataset, metrics=["retrieval_precision"])
```

Requires `pip install openai` or `pip install anthropic` depending on provider.
The LLM generates questions, ground-truth answers, and distractor documents automatically.

---

## Why Lupa?

- **Framework agnostic.** One function signature works with LangChain, LlamaIndex,
  raw API calls, or anything else. Lupa never touches your pipeline internals.
- **Fully local.** All four metrics run on-device using sentence-transformers and
  BERTScore. No evaluation API calls, no per-query cost.
- **Diagnostic, not just a score.** Every metric returns a per-example breakdown.
  You can see exactly which queries failed retrieval and which answers were hallucinated.
- **Comparable across versions.** `compare()` diffs two result files and flags
  regressions automatically — safe to run in CI before merging a prompt change.

---

## Dataset Format

Each example in a custom dataset must have this shape:

```json
{
  "query": "What was revenue in Q3 2023?",
  "ground_truth_answer": "Revenue was $4.2 billion.",
  "relevant_document_ids": ["doc_001"],
  "documents": {
    "doc_001": { "content": "Q3 2023 revenue was $4.2 billion.", "is_distractor": false },
    "doc_002": { "content": "Q2 2023 revenue was $3.9 billion.", "is_distractor": true }
  }
}
```

`relevant_document_ids` lists the document IDs that correctly answer the query.
All other documents are treated as distractors for retrieval precision scoring.
