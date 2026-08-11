from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .okf import OKFValidationError, normalize_slug


DISCOVER_SCHEMA_VERSION = "4"
COMPILE_SCHEMA_VERSION = "3"
PRESERVE_SCHEMA_VERSION = "2"
ENRICH_SCHEMA_VERSION = "6"

TAXONOMY = frozenset(
    {"数据口径", "分析框架", "政策建议", "国际比较", "术语解释"}
)


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ConceptRef:
    concept_id: str
    type: str
    title: str
    description: str
    source: str
    evidence: tuple[str, ...]
    asset_hints: tuple[str, ...]
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    evidence_block_ids: tuple[str, ...] = ()
    # Optional metadata is deliberately appended so existing callers and
    # serialized OKF files remain valid.  These fields guide matching and
    # provenance; they do not replace the verbatim evidence contract.
    semantic_signature: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    ref_family_hint: str = ""
    ref_version_id: str = ""
    document_family_id: str = ""
    document_version_id: str = ""


@dataclass(frozen=True)
class DraftConcept:
    ref: ConceptRef
    title: str
    description: str
    body: str


@dataclass(frozen=True)
class AssetPlacement:
    asset_id: str
    concept_id: str
    anchor: str
    position: Literal["before", "after"]
    reason: str


@dataclass(frozen=True)
class LinkSuggestion:
    target_id: str
    anchor: str
    reason: str
    occurrence: int | None = None


@dataclass(frozen=True)
class RelationAudit:
    status: Literal["linked", "no_links"]
    links: tuple[LinkSuggestion, ...]


def parse_discovery(
    response: str,
    source_name: str,
    evidence_catalog: dict[str, str],
    asset_ids: set[str],
    min_concepts: int = 1,
    required_types: set[str] | frozenset[str] = frozenset(),
) -> tuple[ConceptRef, ...]:
    payload = _object(response)
    _exact_fields(payload, {"concepts"}, "discovery response")
    concepts = _list(payload.get("concepts"), "concepts")
    if len(concepts) < min_concepts:
        raise ContractError(
            f"concepts must contain at least {min_concepts} concepts"
        )

    parsed: list[ConceptRef] = []
    seen: set[str] = set()
    for index, value in enumerate(concepts):
        item = _mapping(value, f"concepts[{index}]")
        _exact_fields(
            item,
            {"id", "type", "title", "description", "evidence", "asset_hints"},
            f"concepts[{index}]",
            optional={"semantic_signature", "scope", "ref_family_hint"},
        )
        concept_id = _concept_id(item.get("id"))
        if concept_id in seen:
            raise ContractError(f"duplicate concept id: {concept_id}")
        seen.add(concept_id)

        concept_type = _text(item.get("type"), f"concepts[{index}].type")
        if concept_type not in TAXONOMY:
            raise ContractError(f"invalid concept type: {concept_type}")
        evidence_ids = _text_tuple(
            item.get("evidence"), f"concepts[{index}].evidence"
        )
        if not evidence_ids:
            raise ContractError("evidence must contain at least one evidence ID")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ContractError("evidence IDs must be unique within a concept")
        unknown_evidence = set(evidence_ids) - set(evidence_catalog)
        if unknown_evidence:
            raise ContractError(
                f"unknown evidence ID: {sorted(unknown_evidence)[0]}"
            )
        hints = _text_tuple(
            item.get("asset_hints"), f"concepts[{index}].asset_hints"
        )
        unknown = set(hints) - asset_ids
        if unknown:
            raise ContractError(f"unknown asset hint: {sorted(unknown)[0]}")
        signature = item.get("semantic_signature", {})
        if not isinstance(signature, dict):
            raise ContractError("semantic_signature must be an object")
        scope = item.get("scope", {})
        if not isinstance(scope, dict):
            raise ContractError("scope must be an object")
        family_hint = item.get("ref_family_hint", "")
        if not isinstance(family_hint, str):
            raise ContractError("ref_family_hint must be a string")
        parsed.append(
            ConceptRef(
                concept_id=concept_id,
                type=concept_type,
                title=_text(item.get("title"), f"concepts[{index}].title"),
                description=_text(
                    item.get("description"), f"concepts[{index}].description"
                ),
                source=source_name,
                evidence=tuple(evidence_catalog[item] for item in evidence_ids),
                asset_hints=hints,
                semantic_signature=dict(signature),
                scope=dict(scope),
                ref_family_hint=family_hint,
            )
        )
    missing_types = required_types - {item.type for item in parsed}
    if missing_types:
        raise ContractError(
            "missing required concept types: " + ", ".join(sorted(missing_types))
        )
    return tuple(parsed)


def discovery_json_schema(min_concepts: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "minItems": min_concepts,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "type": {"type": "string", "enum": sorted(TAXONOMY)},
                        "title": {"type": "string", "minLength": 1},
                        "description": {"type": "string", "minLength": 1},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "asset_hints": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "semantic_signature": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "scope": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "ref_family_hint": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "type",
                        "title",
                        "description",
                        "evidence",
                        "asset_hints",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["concepts"],
        "additionalProperties": False,
    }


def draft_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
            "sections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string", "minLength": 1},
                        "paragraphs": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "required": ["heading", "paragraphs", "bullets"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["title", "description", "sections"],
        "additionalProperties": False,
    }


def placements_json_schema(asset_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "minItems": asset_count,
                "maxItems": asset_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string", "minLength": 1},
                        "concept_id": {"type": "string", "minLength": 1},
                        "anchor_id": {"type": "string", "minLength": 1},
                        "position": {
                            "type": "string",
                            "enum": ["before", "after"],
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "asset_id",
                        "concept_id",
                        "anchor_id",
                        "position",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["placements"],
        "additionalProperties": False,
    }


def relation_json_schema(anchor_ids: set[str] | None = None) -> dict[str, Any]:
    anchor_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if anchor_ids:
        anchor_id_schema["enum"] = sorted(anchor_ids)
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["linked", "no_links"]},
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "anchor_id": anchor_id_schema,
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["anchor_id", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "links"],
        "additionalProperties": False,
    }


def parse_draft(response: str, ref: ConceptRef) -> DraftConcept:
    payload = _object(response)
    _exact_fields(payload, {"title", "description", "sections"}, "draft response")
    sections = _list(payload.get("sections"), "sections")
    if not sections:
        raise ContractError("sections must contain at least one section")
    rendered: list[str] = []
    for index, value in enumerate(sections):
        section = _mapping(value, f"sections[{index}]")
        _exact_fields(
            section,
            {"heading", "paragraphs", "bullets"},
            f"sections[{index}]",
        )
        heading = _draft_fragment(
            section.get("heading"), f"sections[{index}].heading"
        )
        paragraphs = [
            _draft_fragment(item, f"sections[{index}].paragraphs[{position}]")
            for position, item in enumerate(
                _list(section.get("paragraphs"), f"sections[{index}].paragraphs")
            )
        ]
        if not paragraphs:
            raise ContractError(
                f"sections[{index}].paragraphs must not be empty"
            )
        bullets = [
            _draft_fragment(item, f"sections[{index}].bullets[{position}]")
            for position, item in enumerate(
                _list(section.get("bullets"), f"sections[{index}].bullets")
            )
        ]
        parts = [f"## {heading}", *paragraphs]
        if bullets:
            parts.append("\n".join(f"- {item}" for item in bullets))
        rendered.append("\n\n".join(parts))
    body = "\n\n".join(rendered)
    return DraftConcept(
        ref=ref,
        title=_text(payload.get("title"), "title"),
        description=_text(payload.get("description"), "description"),
        body=body,
    )


def _draft_fragment(value: Any, label: str) -> str:
    text = _text(value, label)
    if _contains_asset_or_concept_link(text):
        raise ContractError(f"{label} contains an asset or concept link")
    if "\n" in text or re.search(r"#{1,6}\s", text):
        raise ContractError(f"{label} contains raw Markdown structure")
    return text


def parse_placements(
    response: str,
    asset_ids: set[str],
    concept_ids: set[str],
    anchor_catalog: dict[tuple[str, str], str],
) -> tuple[AssetPlacement, ...]:
    payload = _object(response)
    _exact_fields(payload, {"placements"}, "preservation response")
    values = _list(payload.get("placements"), "placements")
    parsed: list[AssetPlacement] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _mapping(value, f"placements[{index}]")
        _exact_fields(
            item,
            {"asset_id", "concept_id", "anchor_id", "position", "reason"},
            f"placements[{index}]",
        )
        asset_id = _text(item.get("asset_id"), f"placements[{index}].asset_id")
        if asset_id not in asset_ids:
            raise ContractError(f"unknown asset id: {asset_id}")
        if asset_id in seen:
            raise ContractError(f"asset must appear exactly once: {asset_id}")
        seen.add(asset_id)
        concept_id = _text(
            item.get("concept_id"), f"placements[{index}].concept_id"
        )
        if concept_id not in concept_ids:
            raise ContractError(f"unknown concept id: {concept_id}")
        anchor_id = _text(
            item.get("anchor_id"), f"placements[{index}].anchor_id"
        )
        anchor = anchor_catalog.get((concept_id, anchor_id))
        if anchor is None:
            raise ContractError(
                f"unknown anchor id for concept {concept_id}: {anchor_id}"
            )
        position = _text(item.get("position"), f"placements[{index}].position")
        if position not in {"before", "after"}:
            raise ContractError(f"invalid position: {position}")
        parsed.append(
            AssetPlacement(
                asset_id=asset_id,
                concept_id=concept_id,
                anchor=anchor,
                position=position,
                reason=_text(item.get("reason"), f"placements[{index}].reason"),
            )
        )
    if seen != asset_ids:
        missing = sorted(asset_ids - seen)
        raise ContractError(
            f"each source asset must appear exactly once; missing: {', '.join(missing)}"
        )
    return tuple(parsed)


def parse_relation_audit(
    response: str,
    anchor_catalog: dict[str, tuple[str, str, int]],
    current_body: str,
) -> RelationAudit:
    payload = _object(response)
    _exact_fields(payload, {"status", "links"}, "relation response")
    status = _text(payload.get("status"), "status")
    if status not in {"linked", "no_links"}:
        raise ContractError(f"invalid relation status: {status}")
    values = _list(payload.get("links"), "links")
    intents: list[tuple[int, str, str, int, str]] = []
    seen_anchors: set[str] = set()
    seen_targets: set[str] = set()
    for index, value in enumerate(values):
        item = _mapping(value, f"links[{index}]")
        _exact_fields(item, {"anchor_id", "reason"}, f"links[{index}]")
        anchor_id = _text(item.get("anchor_id"), f"links[{index}].anchor_id")
        anchor_spec = anchor_catalog.get(anchor_id)
        if anchor_spec is None:
            raise ContractError(f"unknown relation anchor: {anchor_id}")
        target_id, anchor, occurrence = anchor_spec
        if anchor_id in seen_anchors:
            raise ContractError(f"duplicate relation anchor: {anchor_id}")
        seen_anchors.add(anchor_id)
        if target_id in seen_targets:
            raise ContractError(f"duplicate relation target: {target_id}")
        seen_targets.add(target_id)
        intents.append(
            (
                index,
                target_id,
                anchor,
                occurrence,
                _text(item.get("reason"), f"links[{index}].reason"),
            )
        )
    if status == "no_links" and intents:
        raise ContractError("no_links status requires an empty links list")
    if status == "linked" and not intents:
        raise ContractError("linked status requires at least one suggestion")
    if status == "no_links":
        return RelationAudit(status="no_links", links=())

    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, LinkSuggestion]] = []
    ranked_intents = sorted(
        intents,
        key=lambda item: (
            -len(item[2]),
            item[0],
        ),
    )
    for index, target_id, anchor, occurrence, reason in ranked_intents:
        span = _relation_span(current_body, anchor, occurrence)
        if any(_ranges_overlap(span, existing) for existing in occupied):
            continue
        occupied.append(span)
        selected.append(
            (
                index,
                LinkSuggestion(target_id, anchor, reason, occurrence),
            )
        )
    selected.sort(key=lambda item: item[0])
    return RelationAudit(
        status="linked", links=tuple(item[1] for item in selected)
    )


def _relation_span(body: str, anchor: str, occurrence: int) -> tuple[int, int]:
    matches = tuple(re.finditer(re.escape(anchor), body))
    if occurrence < 0 or occurrence >= len(matches):
        raise ContractError(
            f"relation anchor occurrence does not exist: {anchor}/{occurrence}"
        )
    return matches[occurrence].start(), matches[occurrence].end()


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _object(response: str) -> dict[str, Any]:
    try:
        value = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError) as error:
        raise ContractError("model response must be valid JSON with no extra text") from error
    return _mapping(value, "response")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    items = _list(value, label)
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(items))


def _exact_fields(
    value: dict[str, Any],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    allowed = expected | set(optional)
    if not expected.issubset(actual) or not actual.issubset(allowed):
        missing = sorted(expected - actual)
        extra = sorted(actual - allowed)
        raise ContractError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _concept_id(value: Any) -> str:
    concept_id = _text(value, "concept id")
    if concept_id.endswith(".md"):
        concept_id = concept_id[:-3]
    try:
        normalized = normalize_slug(f"{concept_id}.md")[:-3]
    except OKFValidationError as error:
        raise ContractError(f"invalid concept id: {concept_id}") from error
    if concept_id != normalized:
        raise ContractError(
            f"concept id must already be normalized: {concept_id}; expected {normalized}"
        )
    return concept_id


def _contains_asset_or_concept_link(body: str) -> bool:
    return any(
        (
            re.search(r"!\[[^\]]*\]\([^)]+\)", body),
            re.search(r"<table\b", body, flags=re.IGNORECASE),
            re.search(r"^\s*\|.*\|\s*$", body, flags=re.MULTILINE),
            re.search(r"\[\[[^\]]+\]\]", body),
            re.search(r"\[[^\]]+\]\([^)]*concepts/[^)]+\)", body),
        )
    )
