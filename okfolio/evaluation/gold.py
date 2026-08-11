from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .contracts import (
    EvidenceAtomId,
    FactLabel,
    GoldAnnotation,
    GoldQuestion,
)


_EVIDENCE_ATOM_PATTERN = r"^[^:\s]+:p[0-9]+:[sb][^:\s]+$"

_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fact_id": {"type": "string", "minLength": 1},
        "claim": {"type": "string", "minLength": 1},
        "weight": {"type": "number", "exclusiveMinimum": 0},
        "critical": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["fact_id", "claim"],
    "additionalProperties": False,
}

GOLD_QUESTION_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://okfolio.dev/schemas/rag-gold-question-v1.json",
    "title": "OKFolio RAG Gold Question",
    "type": "object",
    "properties": {
        "question_id": {"type": "string", "minLength": 1},
        "question": {"type": "string", "minLength": 1},
        "question_type": {"type": "string", "minLength": 1},
        "answerable": {"type": "boolean"},
        "scope": {"type": "object"},
        "required_facts": {
            "type": "array",
            "items": {"$ref": "#/$defs/fact"},
        },
        "forbidden_facts": {
            "type": "array",
            "items": {"$ref": "#/$defs/fact"},
        },
        "evidence_sets": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": _EVIDENCE_ATOM_PATTERN,
                },
            },
        },
        "reference_answer": {"type": ["string", "null"]},
        "annotation": {
            "type": "object",
            "properties": {
                "author": {"type": "string", "minLength": 1},
                "reviewer": {"type": ["string", "null"], "minLength": 1},
                "status": {"type": "string", "minLength": 1},
            },
            "required": ["author", "status"],
            "additionalProperties": False,
        },
    },
    "required": [
        "question_id",
        "question",
        "question_type",
        "answerable",
        "scope",
        "required_facts",
        "forbidden_facts",
        "evidence_sets",
        "reference_answer",
        "annotation",
    ],
    "allOf": [
        {
            "if": {"properties": {"answerable": {"const": True}}},
            "then": {
                "properties": {
                    "required_facts": {"minItems": 1},
                    "evidence_sets": {"minItems": 1},
                }
            },
            "else": {"properties": {"evidence_sets": {"maxItems": 0}}},
        }
    ],
    "additionalProperties": False,
    "$defs": {"fact": _FACT_SCHEMA},
}

_VALIDATOR = Draft202012Validator(GOLD_QUESTION_JSON_SCHEMA)


class GoldDataError(ValueError):
    """Raised when a gold JSONL row violates the stable evaluation contract."""


def gold_question_schema() -> dict[str, Any]:
    """Return a caller-safe copy of the public JSON Schema."""

    return copy.deepcopy(GOLD_QUESTION_JSON_SCHEMA)


def load_gold_jsonl(path: Path) -> tuple[GoldQuestion, ...]:
    """Load and validate UTF-8 JSONL with line-specific error messages."""

    if not path.is_file():
        raise FileNotFoundError(path)
    questions: list[GoldQuestion] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise GoldDataError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise GoldDataError(f"{path}:{line_number}: row must be a JSON object")
        errors = sorted(_VALIDATOR.iter_errors(payload), key=lambda item: list(item.path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "<row>"
            raise GoldDataError(
                f"{path}:{line_number}: {location}: {error.message}"
            )
        try:
            question = _parse_question(payload)
        except (TypeError, ValueError) as exc:
            raise GoldDataError(f"{path}:{line_number}: {exc}") from exc
        if question.question_id in seen_ids:
            raise GoldDataError(
                f"{path}:{line_number}: duplicate question_id {question.question_id!r}"
            )
        seen_ids.add(question.question_id)
        questions.append(question)
    if not questions:
        raise GoldDataError(f"{path}: no gold questions found")
    return tuple(questions)


def _parse_question(payload: Mapping[str, Any]) -> GoldQuestion:
    required = tuple(_parse_fact(item) for item in payload["required_facts"])
    forbidden = tuple(_parse_fact(item) for item in payload["forbidden_facts"])
    evidence_sets = tuple(
        tuple(EvidenceAtomId.parse(atom) for atom in evidence_set)
        for evidence_set in payload["evidence_sets"]
    )
    annotation = payload["annotation"]
    return GoldQuestion(
        question_id=payload["question_id"].strip(),
        question=payload["question"].strip(),
        question_type=payload["question_type"].strip(),
        answerable=payload["answerable"],
        scope=dict(payload["scope"]),
        required_facts=required,
        forbidden_facts=forbidden,
        evidence_sets=evidence_sets,
        reference_answer=payload["reference_answer"],
        annotation=GoldAnnotation(
            author=annotation["author"].strip(),
            reviewer=(
                annotation.get("reviewer", "").strip()
                if annotation.get("reviewer") is not None
                else None
            ),
            status=annotation["status"].strip(),
        ),
    )


def _parse_fact(payload: Mapping[str, Any]) -> FactLabel:
    return FactLabel(
        fact_id=payload["fact_id"].strip(),
        claim=payload["claim"].strip(),
        weight=float(payload.get("weight", 1.0)),
        critical=bool(payload.get("critical", True)),
        reason=str(payload.get("reason", "")).strip(),
    )
