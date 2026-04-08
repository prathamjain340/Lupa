# rageval/datasets/loader.py
#
# Loads built-in and custom datasets, validates format.
#
# Design decisions (see docs/decisions.md):
#   ADR-007: Built-in files located via importlib.resources, not __file__.
#            __file__ breaks under pip-installed wheels/zips.
#
# Built-in dataset names → JSON filenames in rageval/datasets/builtin/:
#   "rageval-financialqa-mini" → financialqa_mini.json
#
# Expected example format (each dict in the list):
#   {
#     "query": str,
#     "ground_truth_answer": str,
#     "relevant_document_ids": [str, ...],
#     "documents": {
#       "<doc_id>": {"content": str, "is_distractor": bool},
#       ...
#     }
#   }

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

# Maps public dataset names to their JSON filenames in the builtin/ package.
_BUILTIN_DATASETS: dict[str, str] = {
    "rageval-financialqa-mini": "financialqa_mini.json",
}


def load_dataset(name: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Load a built-in dataset by name, or validate and return a custom one.

    Args:
        name: Either a built-in dataset name string or a list of example dicts.

    Returns:
        List of validated example dicts.

    Raises:
        ValueError: If the name is not a recognised built-in, or if any
                    example fails format validation.
        TypeError:  If name is neither a str nor a list.
    """
    if isinstance(name, str):
        examples = _load_builtin(name)
    elif isinstance(name, list):
        examples = name
    else:
        raise TypeError(
            f"dataset must be a built-in name (str) or a list of dicts, "
            f"got {type(name).__name__}."
        )

    _validate(examples)
    return examples


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_builtin(name: str) -> list[dict[str, Any]]:
    if name not in _BUILTIN_DATASETS:
        supported = ", ".join(f'"{k}"' for k in _BUILTIN_DATASETS)
        raise ValueError(
            f'Unknown built-in dataset "{name}". '
            f"Supported datasets: {supported}."
        )

    filename = _BUILTIN_DATASETS[name]
    data = files("rageval.datasets.builtin").joinpath(filename).read_text(encoding="utf-8")
    return json.loads(data)


def _validate(examples: list[dict[str, Any]]) -> None:
    if not examples:
        raise ValueError("Dataset is empty — must contain at least one example.")

    for i, example in enumerate(examples):
        _check_field(example, i, "query", str)
        _check_field(example, i, "ground_truth_answer", str)
        _check_field(example, i, "relevant_document_ids", list)
        _check_field(example, i, "documents", dict)

        for doc_id, doc in example["documents"].items():
            if not isinstance(doc, dict):
                raise ValueError(
                    f"Example {i}: documents['{doc_id}'] must be a dict, "
                    f"got {type(doc).__name__}."
                )
            if "content" not in doc:
                raise ValueError(
                    f"Example {i}: documents['{doc_id}'] is missing required field 'content'."
                )
            if "is_distractor" not in doc:
                raise ValueError(
                    f"Example {i}: documents['{doc_id}'] is missing required field 'is_distractor'."
                )


def _check_field(
    example: dict[str, Any],
    index: int,
    field: str,
    expected_type: type,
) -> None:
    if field not in example:
        raise ValueError(
            f"Example {index}: missing required field '{field}'."
        )
    if not isinstance(example[field], expected_type):
        raise ValueError(
            f"Example {index}: field '{field}' must be {expected_type.__name__}, "
            f"got {type(example[field]).__name__}."
        )
