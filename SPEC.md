# Lupa — RAG Evaluation Framework
## Package name: rageval | pip install rageval

### What we're building
Framework-agnostic RAG evaluation library. User provides a pipeline 
function, a dataset, and metrics. Library returns scored results with 
a markdown report.

### Pipeline function contract
fn(query: str) -> tuple[list[str], str]
# returns: (list of retrieved chunk strings, answer string)

### v1 Metrics
- retrieval_precision: sentence-transformers similarity, threshold-based
- faithfulness: NLI primary (DeBERTa cross-encoder), BERTScore secondary  
- answer_relevance: BERTScore between query and answer
- latency: wall-clock p50/p90/p99 across dataset

### Dataset format (each example)
{
  "query": str,
  "ground_truth_answer": str,
  "relevant_document_ids": list[str],
  "documents": {
    "doc_id": {
      "content": str,
      "is_distractor": bool
    }
  }
}

### Core API
rageval.evaluate(pipeline_fn, dataset, metrics) -> Results
rageval.compare(baseline_path, new_path, save=None) -> ComparisonReport
rageval.generate_dataset(documents, n_questions, llm) -> Dataset

Results.report()        # prints markdown to console
Results.save(path)      # saves markdown
Results.to_json(path)   # saves raw scores

### Folder structure
rageval/
  __init__.py
  evaluator.py
  metrics/
    retrieval.py
    faithfulness.py
    relevance.py
    latency.py
  datasets/
    loader.py
    generator.py
    builtin/
      financialqa_mini.json
  comparison.py
  report.py
  models/
    loader.py

### Dependencies
sentence-transformers, torch, transformers, bert-score, numpy, rich
NO langchain, NO openai in core

### Week 1 scope
1. Full folder structure with all files (stubs okay for metrics not 
   yet implemented)
2. evaluate() function — accepts pipeline_fn + dataset + metrics list
3. Results object with .report() and .to_json()
4. latency metric — fully working
5. retrieval_precision metric — fully working  
6. financialqa_mini dataset — 10 hand-crafted examples with 
   distractors
7. basic_eval.py example that runs end to end
8. tests/test_metrics.py with tests for retrieval precision

### Design principles
- Fail loudly with clear error messages (wrong function signature etc)
- Lazy load ML models (don't download on import, only when needed)
- Every metric returns a score 0-1 and a per-example breakdown
- No side effects on import

### Environment
Fresh venv at Lupa/venv/ — assume clean install, pin all dependency 
versions explicitly in pyproject.toml