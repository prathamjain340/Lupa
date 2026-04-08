# Lupa

**RAG evaluation that actually tells you what's wrong.**

Lupa is a framework-agnostic RAG evaluation library. Point it at your 
pipeline, give it a dataset, get back a report that tells you exactly 
where things are breaking down.

```python
pip install rageval
```

```python
import rageval

results = rageval.evaluate(
    pipeline_fn=my_rag_fn,
    dataset="rageval-financialqa",
    metrics=["retrieval_precision", "faithfulness", "latency"]
)

results.report()
```

## Why Lupa?

Most RAG evaluation tools either require expensive LLM API calls for 
every evaluation, lock you into a specific framework, or give you a 
single number with no diagnostic detail.

Lupa is different:
- **Framework agnostic** — works with LangChain, LlamaIndex, or your 
  own custom pipeline
- **Fully local** — no OpenAI API required for core metrics
- **Diagnostic** — tells you if failure is in retrieval or generation
- **Comparable** — track metric changes across pipeline versions

## Metrics

| Metric | Method | API Required |
|---|---|---|
| Retrieval Precision | Sentence-transformers similarity | No |
| Faithfulness | NLI model (DeBERTa) | No |
| Answer Relevance | BERTScore | No |
| Latency | Wall-clock p50/p90/p99 | No |

## Status

🚧 Active development — v0.1 coming soon

## License

MIT