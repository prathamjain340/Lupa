# rageval/datasets/generator.py
#
# generate_dataset() — create a Lupa-format evaluation dataset from raw
# documents using an LLM to produce question-answer pairs and distractors.
#
# Design decisions (see docs/decisions.md):
#   ADR-012: LLM generates both questions and distractor documents from the
#            same source material. Distractors are plausible-but-wrong
#            versions of the source doc (wrong numbers, dates, quarters).
#   ADR-001: openai / anthropic are imported inside the function, never at
#            module level, so importing rageval never requires these optional
#            dependencies.
#
# Algorithm:
#   1. Compute n_per_doc = ceil(n_questions / len(documents)).
#   2. For each document, send one LLM call asking for n_per_doc Q&A pairs
#      with 2 distractors each. Response format is {"questions": [...]}.
#   3. Parse JSON response and convert each item to Lupa dataset schema,
#      using "doc_001" for the source document and "doc_002"/"doc_003" for
#      the two distractors.
#   4. Trim the accumulated list to exactly n_questions.
#   5. Optionally save to output_path before returning.
#
# Models used:
#   openai:     gpt-4o-mini  — fast, cheap, reliable JSON output
#   anthropic:  claude-haiku-4-5-20251001 — fast, cheap, reliable JSON output

from __future__ import annotations

import json
import math
from typing import Any


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def _build_prompt(document: str, n: int) -> str:
    return f"""You are building an evaluation dataset for a RAG system.

Given the source document below, generate exactly {n} question-answer pairs.

Rules:
- Each question must be answerable ONLY from the source document.
- Each answer must be factually grounded in the source document.
- For each question, write exactly 2 distractor documents: plausible-looking
  passages that seem relevant but contain subtly wrong details (wrong numbers,
  wrong dates, wrong quarter, wrong percentage, wrong company name, etc.).
- Distractors must NOT correctly answer the question.

Source document:
{document}

Respond with ONLY valid JSON — no markdown fences, no explanation.
Use this exact structure:

{{
  "questions": [
    {{
      "query": "the question",
      "ground_truth_answer": "answer grounded in the source document",
      "distractor_1": "plausible-looking passage with wrong details",
      "distractor_2": "another plausible-looking passage with wrong details"
    }}
  ]
}}"""


# ------------------------------------------------------------------
# LLM call helpers
# ------------------------------------------------------------------

def _call_openai(client: Any, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OpenAI returned invalid JSON: {exc}\nRaw response:\n{raw}"
        ) from exc


def _call_anthropic(client: Any, prompt: str) -> dict[str, Any]:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Anthropic returned invalid JSON: {exc}\nRaw response:\n{raw}"
        ) from exc


# ------------------------------------------------------------------
# Schema conversion
# ------------------------------------------------------------------

def _to_example(document: str, item: dict[str, Any]) -> dict[str, Any]:
    """Convert one LLM-produced item into the Lupa dataset schema."""
    return {
        "query": item["query"],
        "ground_truth_answer": item["ground_truth_answer"],
        "relevant_document_ids": ["doc_001"],
        "documents": {
            "doc_001": {"content": document, "is_distractor": False},
            "doc_002": {"content": item["distractor_1"], "is_distractor": True},
            "doc_003": {"content": item["distractor_2"], "is_distractor": True},
        },
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_dataset(
    documents: list[str],
    n_questions: int = 50,
    llm: str = "openai",
    api_key: str = None,
    output_path: str = None,
) -> list[dict[str, Any]]:
    """
    Generate a Lupa evaluation dataset from raw documents using an LLM.

    For each document the LLM produces question-answer pairs where the
    answer is grounded in that document, plus two distractor documents
    with plausible-but-wrong details. Results are trimmed to n_questions.

    Args:
        documents:    List of source document strings. Must not be empty.
        n_questions:  Target number of examples in the returned dataset.
        llm:          LLM provider — "openai" or "anthropic".
        api_key:      API key. If None, read from the provider's standard
                      env var (OPENAI_API_KEY or ANTHROPIC_API_KEY).
        output_path:  If provided, save the dataset as JSON to this path
                      before returning.

    Returns:
        list[dict] — examples in Lupa dataset format, ready to pass to
        rageval.evaluate() as a custom dataset.

    Raises:
        ValueError:   Empty documents list, or unsupported llm value.
        ImportError:  Required LLM library (openai / anthropic) not installed.
    """
    if not documents:
        raise ValueError(
            "documents list is empty — provide at least one document string."
        )

    if llm not in ("openai", "anthropic"):
        raise ValueError(
            f"llm must be 'openai' or 'anthropic', got {llm!r}. "
            "These are the only supported providers."
        )

    # Lazy import — only the provider the user actually requested.
    if llm == "openai":
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for llm='openai'. "
                "Install it with: pip install openai"
            )
        client = _openai.OpenAI(api_key=api_key)
        call_fn = _call_openai

    else:  # anthropic
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for llm='anthropic'. "
                "Install it with: pip install anthropic"
            )
        client = _anthropic.Anthropic(api_key=api_key)
        call_fn = _call_anthropic

    n_per_doc = math.ceil(n_questions / len(documents))
    all_examples: list[dict[str, Any]] = []

    for doc in documents:
        prompt = _build_prompt(doc, n_per_doc)
        response = call_fn(client, prompt)
        items = response.get("questions", [])
        for item in items:
            all_examples.append(_to_example(doc, item))

    # Trim to requested size in case ceil() over-generated.
    all_examples = all_examples[:n_questions]

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_examples, f, indent=2)

    return all_examples
