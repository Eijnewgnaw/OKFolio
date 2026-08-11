from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .contracts import ContractError


DiscoveryMode = Literal["heading", "llm", "hybrid"]
AssetPolicy = Literal["auto", "human_review"]
QualityDecision = Literal["pass", "recompile", "human_review"]


@dataclass(frozen=True)
class AgentPolicy:
    quality_threshold: float = 0.80
    max_recompile_attempts: int = 2
    max_component_refs: int = 24
    max_component_chars: int = 42_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0 and 1")
        if not 0 <= self.max_recompile_attempts <= 3:
            raise ValueError("max_recompile_attempts must be between 0 and 3")
        if self.max_component_refs < 2:
            raise ValueError("max_component_refs must be at least 2")
        if self.max_component_chars < 8_000:
            raise ValueError("max_component_chars must be at least 8000")


@dataclass(frozen=True)
class SourcePlan:
    discovery_mode: DiscoveryMode
    refine_discovery: bool
    asset_policy: AssetPolicy
    reason: str


@dataclass(frozen=True)
class GroupDecision:
    ref_ids: tuple[str, ...]
    title: str
    description: str
    reason: str


@dataclass(frozen=True)
class CompileGroup:
    group_id: str
    ref_ids: tuple[str, ...]
    title: str
    description: str
    reason: str


@dataclass(frozen=True)
class QualityAudit:
    score: float
    decision: QualityDecision
    issues: tuple[str, ...]
    recompile_instructions: str


def source_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "discovery_mode": {
                "type": "string",
                "enum": ["heading", "llm", "hybrid"],
            },
            "refine_discovery": {"type": "boolean"},
            "asset_policy": {
                "type": "string",
                "enum": ["auto", "human_review"],
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "discovery_mode",
            "refine_discovery",
            "asset_policy",
            "reason",
        ],
        "additionalProperties": False,
    }


def parse_source_plan(
    response: str,
    *,
    structured_section_count: int,
    asset_count: int,
) -> SourcePlan:
    payload = _object(response, "source plan")
    _exact(
        payload,
        {"discovery_mode", "refine_discovery", "asset_policy", "reason"},
        "source plan",
    )
    mode = _enum(
        payload.get("discovery_mode"),
        {"heading", "llm", "hybrid"},
        "discovery_mode",
    )
    refine = payload.get("refine_discovery")
    if not isinstance(refine, bool):
        raise ContractError("refine_discovery must be a boolean")
    asset_policy = _enum(
        payload.get("asset_policy"),
        {"auto", "human_review"},
        "asset_policy",
    )
    reason = _text(payload.get("reason"), "reason")
    if mode in {"heading", "hybrid"} and structured_section_count < 2:
        raise ContractError(
            "heading or hybrid discovery requires at least two structured sections"
        )
    if mode == "hybrid" and not refine:
        raise ContractError("hybrid discovery requires refine_discovery=true")
    if asset_count == 0 and asset_policy == "human_review":
        raise ContractError("asset_policy cannot require review when there are no assets")
    return SourcePlan(mode, refine, asset_policy, reason)  # type: ignore[arg-type]


def group_plan_schema(ref_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(ref_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "ref_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "string",
                                "enum": sorted(ref_ids),
                            },
                        },
                        "title": {"type": "string", "minLength": 1},
                        "description": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["ref_ids", "title", "description", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }


def parse_group_plan(
    response: str,
    *,
    refs: Mapping[str, Mapping[str, Any]],
    candidate_pairs: set[tuple[str, str]],
) -> tuple[GroupDecision, ...]:
    decisions, _recovered = _parse_group_plan(
        response,
        refs=refs,
        candidate_pairs=candidate_pairs,
        demote_invalid_joint_groups=False,
    )
    return decisions


def recover_group_plan(
    response: str,
    *,
    refs: Mapping[str, Mapping[str, Any]],
    candidate_pairs: set[tuple[str, str]],
) -> tuple[tuple[GroupDecision, ...], tuple[str, ...]]:
    """Preserve exact Ref coverage while demoting invalid joint groups.

    This recovery remains strict about JSON shape, known IDs, overlap, and
    complete coverage.  It only converts a semantically invalid multi-Ref
    decision into independently compiled singleton groups after the model has
    already failed the normal contract-repair attempt.
    """
    return _parse_group_plan(
        response,
        refs=refs,
        candidate_pairs=candidate_pairs,
        demote_invalid_joint_groups=True,
    )


def _parse_group_plan(
    response: str,
    *,
    refs: Mapping[str, Mapping[str, Any]],
    candidate_pairs: set[tuple[str, str]],
    demote_invalid_joint_groups: bool,
) -> tuple[tuple[GroupDecision, ...], tuple[str, ...]]:
    payload = _object(response, "group plan")
    _exact(payload, {"groups"}, "group plan")
    values = payload.get("groups")
    if not isinstance(values, list) or not values:
        raise ContractError("groups must be a non-empty array")

    expected = set(refs)
    seen: set[str] = set()
    decisions: list[GroupDecision] = []
    recovered: set[str] = set()
    forced_singletons: set[str] = set()
    if demote_invalid_joint_groups:
        memberships: list[set[str]] = []
        membership_counts: dict[str, int] = {}
        for raw in values:
            raw_ids = raw.get("ref_ids") if isinstance(raw, dict) else None
            if not isinstance(raw_ids, list):
                continue
            member_ids = {
                item.strip()
                for item in raw_ids
                if isinstance(item, str) and item.strip()
            }
            memberships.append(member_ids)
            for ref_id in member_ids:
                membership_counts[ref_id] = (
                    membership_counts.get(ref_id, 0) + 1
                )
        overlaps = {
            ref_id for ref_id, count in membership_counts.items() if count > 1
        }
        if overlaps:
            for member_ids in memberships:
                if member_ids & overlaps:
                    forced_singletons.update(member_ids)

    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise ContractError(f"groups[{index}] must be an object")
        _exact(
            raw,
            {"ref_ids", "title", "description", "reason"},
            f"groups[{index}]",
        )
        raw_ids = raw.get("ref_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ContractError(f"groups[{index}].ref_ids must be non-empty")
        ref_ids = tuple(_text(item, f"groups[{index}].ref_ids") for item in raw_ids)
        if len(ref_ids) != len(set(ref_ids)):
            if not demote_invalid_joint_groups:
                raise ContractError(f"groups[{index}] contains duplicate ref_ids")
            forced_singletons.update(ref_ids)
            ref_ids = tuple(dict.fromkeys(ref_ids))
        unknown = set(ref_ids) - expected
        if unknown:
            raise ContractError(f"unknown ref_id: {sorted(unknown)[0]}")
        overlap = seen & set(ref_ids)
        if overlap:
            if not demote_invalid_joint_groups:
                raise ContractError(
                    f"ref_id appears in multiple groups: {sorted(overlap)[0]}"
                )
            forced_singletons.update(ref_ids)
        seen.update(ref_ids)
        if forced_singletons & set(ref_ids):
            recovered.update(ref_ids)
            continue
        if len(ref_ids) > 1:
            types = {str(refs[ref_id]["type"]) for ref_id in ref_ids}
            articles = {str(refs[ref_id]["article_id"]) for ref_id in ref_ids}
            invalid_reason = ""
            if len(types) != 1:
                invalid_reason = "a joint group must contain one concept type"
            elif len(articles) < 2:
                invalid_reason = (
                    "a joint group must contain refs from at least two Articles"
                )
            elif not _connected(ref_ids, candidate_pairs):
                invalid_reason = (
                    "a joint group must be connected by deterministic candidate edges"
                )
            if invalid_reason:
                if not demote_invalid_joint_groups:
                    raise ContractError(invalid_reason)
                recovered.update(ref_ids)
                for ref_id in ref_ids:
                    ref = refs[ref_id]
                    decisions.append(
                        GroupDecision(
                            ref_ids=(ref_id,),
                            title=str(ref.get("title") or ref_id),
                            description=str(
                                ref.get("description")
                                or f"{ref.get('title') or ref_id}。"
                            ),
                            reason=(
                                "联合分组未通过代码合同，保留该 Ref 并独立编译。"
                            ),
                        )
                    )
                continue
        decisions.append(
            GroupDecision(
                ref_ids=tuple(sorted(ref_ids)),
                title=_text(raw.get("title"), f"groups[{index}].title"),
                description=_text(
                    raw.get("description"), f"groups[{index}].description"
                ),
                reason=_text(raw.get("reason"), f"groups[{index}].reason"),
            )
        )
    for ref_id in sorted(forced_singletons):
        if ref_id not in expected:
            continue
        ref = refs[ref_id]
        decisions.append(
            GroupDecision(
                ref_ids=(ref_id,),
                title=str(ref.get("title") or ref_id),
                description=str(
                    ref.get("description") or f"{ref.get('title') or ref_id}。"
                ),
                reason="模型分组发生重复归属，保留该 Ref 并独立编译。",
            )
        )
        recovered.add(ref_id)
    missing = expected - seen
    if missing:
        if not demote_invalid_joint_groups:
            raise ContractError(f"group plan omitted ref_id: {sorted(missing)[0]}")
        recovered.update(missing)
        for ref_id in sorted(missing):
            ref = refs[ref_id]
            decisions.append(
                GroupDecision(
                    ref_ids=(ref_id,),
                    title=str(ref.get("title") or ref_id),
                    description=str(
                        ref.get("description") or f"{ref.get('title') or ref_id}。"
                    ),
                    reason="模型分组漏项，保留该 Ref 并独立编译。",
                )
            )
    return (
        tuple(sorted(decisions, key=lambda item: item.ref_ids)),
        tuple(sorted(recovered)),
    )


def quality_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "decision": {
                "type": "string",
                "enum": ["pass", "recompile", "human_review"],
            },
            "issues": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "recompile_instructions": {"type": "string"},
        },
        "required": [
            "score",
            "decision",
            "issues",
            "recompile_instructions",
        ],
        "additionalProperties": False,
    }


def parse_quality_audit(response: str, threshold: float) -> QualityAudit:
    payload = _object(response, "quality audit")
    _exact(
        payload,
        {"score", "decision", "issues", "recompile_instructions"},
        "quality audit",
    )
    raw_score = payload.get("score")
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        raise ContractError("score must be a number")
    score = float(raw_score)
    if not 0.0 <= score <= 1.0:
        raise ContractError("score must be between 0 and 1")
    decision = _enum(
        payload.get("decision"),
        {"pass", "recompile", "human_review"},
        "decision",
    )
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raise ContractError("issues must be an array")
    issues = tuple(_text(item, "issues") for item in raw_issues)
    instructions = payload.get("recompile_instructions")
    if not isinstance(instructions, str):
        raise ContractError("recompile_instructions must be a string")
    instructions = instructions.strip()
    if decision == "pass" and score < threshold:
        raise ContractError("pass decision does not meet the quality threshold")
    if decision == "recompile" and not instructions:
        raise ContractError("recompile decision requires recompile_instructions")
    return QualityAudit(score, decision, issues, instructions)  # type: ignore[arg-type]


def _connected(
    ref_ids: tuple[str, ...], candidate_pairs: set[tuple[str, str]]
) -> bool:
    remaining = set(ref_ids)
    reached = {min(remaining)}
    while True:
        additions = {
            candidate
            for candidate in remaining - reached
            if any(tuple(sorted((candidate, current))) in candidate_pairs for current in reached)
        }
        if not additions:
            break
        reached.update(additions)
    return reached == remaining


def _object(response: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ContractError(f"{label} must be strict JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be an object")
    return payload


def _exact(payload: Mapping[str, Any], fields: set[str], label: str) -> None:
    actual = set(payload)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise ContractError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty text")
    return value.strip()


def _enum(value: Any, allowed: set[str], label: str) -> str:
    text = _text(value, label)
    if text not in allowed:
        raise ContractError(f"{label} is invalid: {text}")
    return text
