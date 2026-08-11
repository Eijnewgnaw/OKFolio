"""Generic document and ConceptRef identity/version helpers.

The compiler keeps source history immutable.  This module only classifies the
relationship between two snapshots; it never deletes an old Ref or decides
that a time- or scenario-specific statement is globally superseded.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


UPDATE_STATUSES = frozenset(
    {
        "unchanged",
        "revised",
        "added",
        "not_repeated",
        "temporal_variant",
        "scenario_variant",
        "retracted",
        "superseded",
    }
)


@dataclass(frozen=True)
class DocumentIdentity:
    """Stable family/version identity plus extensible publication metadata."""

    document_family_id: str
    document_version_id: str
    source: str = ""
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_family_id": self.document_family_id,
            "document_version_id": self.document_version_id,
            "source": self.source,
            "title": self.title,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RefUpdate:
    """One old/new pair or one side of an update reconciliation."""

    status: str
    old_ref_id: str | None = None
    new_ref_id: str | None = None
    ref_family_id: str = ""
    reason: str = ""
    scope: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in UPDATE_STATUSES:
            raise ValueError(f"unsupported ref update status: {self.status}")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = dict(self.scope)
        return payload


def reconcile_refs(
    old_refs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    new_refs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Classify a new snapshot against an old snapshot deterministically.

    Matching uses an explicit ``ref_family_id``/``ref_family_hint`` when
    available, then the semantic signature and finally a conservative
    type/title slot.  Scope is a compatibility signal, not a hard global
    validity rule: disjoint time or scenario values yield variants while the
    old record remains in history.
    """
    old = [_mapping(item) for item in old_refs]
    new = [_mapping(item) for item in new_refs]
    candidates: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, item in enumerate(new):
        candidates.setdefault(_semantic_key(item), []).append((index, item))

    used_new: set[int] = set()
    updates: list[RefUpdate] = []
    for item in old:
        key = _semantic_key(item)
        choices = [pair for pair in candidates.get(key, []) if pair[0] not in used_new]
        if not choices:
            updates.append(
                RefUpdate(
                    status=_explicit_missing_status(item),
                    old_ref_id=_ref_id(item),
                    ref_family_id=key,
                    reason="新快照未重复该 Ref；历史记录保留，不自动删除或全局失效。",
                    scope=_scope(item),
                )
            )
            continue
        index, matched = min(choices, key=lambda pair: _match_cost(item, pair[1]))
        used_new.add(index)
        updates.append(_classify_pair(item, matched, key))

    for index, item in enumerate(new):
        if index in used_new:
            continue
        updates.append(
            RefUpdate(
                status="added",
                new_ref_id=_ref_id(item),
                ref_family_id=_semantic_key(item),
                reason="新快照出现了旧快照没有的语义槽位。",
                scope=_scope(item),
            )
        )

    counts: dict[str, int] = {}
    for update in updates:
        counts[update.status] = counts.get(update.status, 0) + 1
    return {
        "schema": "okfolio.ref-reconciliation.v1",
        "old_count": len(old),
        "new_count": len(new),
        "current_ref_ids": [
            _ref_id(item)
            for item in new
            if _ref_id(item)
        ],
        "history_retained": True,
        "counts": dict(sorted(counts.items())),
        "updates": [item.as_dict() for item in updates],
    }


def semantic_key(ref: Mapping[str, Any] | Any) -> str:
    """Public stable slot key used by candidate/reconciliation experiments."""
    return _semantic_key(_mapping(ref))


def scope_compatibility(
    left: Mapping[str, Any] | Any, right: Mapping[str, Any] | Any
) -> str:
    """Return ``same``, ``overlap``, ``temporal_variant`` or ``scenario_variant``."""
    return _scope_relation(_scope(_mapping(left)), _scope(_mapping(right)))


def _classify_pair(
    old: Mapping[str, Any], new: Mapping[str, Any], key: str
) -> RefUpdate:
    relation = _scope_relation(_scope(old), _scope(new))
    if _explicit_status(new) in {"retracted", "superseded"}:
        status = _explicit_status(new)
        reason = "新快照显式声明了该 Ref 的生命周期状态。"
    elif relation == "temporal_variant":
        status = "temporal_variant"
        reason = "语义槽位相同，但适用时间范围不重叠；两版并存。"
    elif relation == "scenario_variant":
        status = "scenario_variant"
        reason = "语义槽位相同，但地区、对象或场景不重叠；两版并存。"
    elif _evidence_fingerprint(old) == _evidence_fingerprint(new) and relation in {"same", "overlap", "unknown"}:
        status = "unchanged"
        reason = "语义槽位、适用范围和证据均未变化。"
    else:
        status = "revised"
        reason = "同一语义槽位和适用范围内的证据或表述发生变化。"
    return RefUpdate(
        status=status,
        old_ref_id=_ref_id(old),
        new_ref_id=_ref_id(new),
        ref_family_id=key,
        reason=reason,
        scope=_scope(new),
    )


def _mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("refs must be mappings or dataclass records")


def _ref_id(ref: Mapping[str, Any]) -> str | None:
    value = ref.get("ref_id") or ref.get("concept_id") or ref.get("id")
    return str(value) if value else None


def _explicit_status(ref: Mapping[str, Any]) -> str:
    status = str(ref.get("status") or "").strip().lower()
    return status if status in {"retracted", "superseded"} else ""


def _explicit_missing_status(ref: Mapping[str, Any]) -> str:
    status = _explicit_status(ref)
    return status or "not_repeated"


def _semantic_key(ref: Mapping[str, Any]) -> str:
    for field in ("ref_family_id", "ref_family_hint"):
        value = str(ref.get(field) or "").strip()
        if value:
            return value
    signature = ref.get("semantic_signature")
    if isinstance(signature, Mapping):
        for field in ("key", "slot", "name", "indicator", "policy"):
            value = str(signature.get(field) or "").strip()
            if value:
                return _normalize_key(value)
    type_name = str(ref.get("type") or "")
    title = str(ref.get("title") or "")
    return _normalize_key(f"{type_name}:{title}")


def _normalize_key(value: str) -> str:
    value = re.sub(r"\s+", "", value).casefold()
    value = re.sub(r"[：:，,。；;、/\\()（）\[\]{}]", "-", value)
    return value.strip("-")


def _scope(ref: Mapping[str, Any]) -> dict[str, Any]:
    value = ref.get("scope")
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items() if item not in (None, "", [], {})}
    # Accept common frontmatter names as a generic fallback.
    fields = ("valid_from", "valid_to", "published_at", "time", "geography", "region", "scenario", "object")
    return {field: ref[field] for field in fields if ref.get(field) not in (None, "", [], {})}


def _scope_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if not left or not right:
        return "unknown"
    time_relation = _dimension_relation(left, right, ("time", "valid_from", "valid_to", "published_at"))
    scenario_relation = _dimension_relation(left, right, ("geography", "region", "scenario", "object"))
    if time_relation == "disjoint":
        return "temporal_variant"
    if scenario_relation == "disjoint":
        return "scenario_variant"
    if time_relation == "same" and scenario_relation == "same":
        return "same"
    return "overlap"


def _dimension_relation(left: Mapping[str, Any], right: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    values_left = _dimension_values(left, fields)
    values_right = _dimension_values(right, fields)
    if not values_left or not values_right:
        return "unknown"
    if values_left & values_right:
        return "same"
    intervals_left = _intervals(values_left)
    intervals_right = _intervals(values_right)
    if intervals_left and intervals_right and any(_interval_overlap(a, b) for a in intervals_left for b in intervals_right):
        return "same"
    return "disjoint"


def _dimension_values(scope: Mapping[str, Any], fields: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for field in fields:
        value = scope.get(field)
        if isinstance(value, (list, tuple, set)):
            values.update(str(item).strip().casefold() for item in value if str(item).strip())
        elif value not in (None, ""):
            values.add(str(value).strip().casefold())
    return values


def _intervals(values: set[str]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for value in values:
        match = re.fullmatch(r"(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?\s*(?:-|~|至|到)\s*(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?", value)
        if match:
            left_year, right_year = int(match.group(1)), int(match.group(4))
            intervals.append((left_year, right_year))
            continue
        years = [int(item) for item in re.findall(r"\b(19\d{2}|20\d{2})\b", value)]
        if len(years) == 1:
            intervals.append((years[0], years[0]))
        elif len(years) >= 2:
            intervals.append((min(years), max(years)))
    return intervals


def _interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _evidence_fingerprint(ref: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = ref.get("evidence") or ()
    if not isinstance(evidence, (list, tuple)):
        evidence = (evidence,)
    return tuple(re.sub(r"\s+", "", str(item)) for item in evidence)


def _match_cost(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[int, int, str]:
    relation = _scope_relation(_scope(old), _scope(new))
    priority = {"same": 0, "overlap": 1, "temporal_variant": 2, "scenario_variant": 3}.get(relation, 4)
    return priority, abs(len(_evidence_fingerprint(old)) - len(_evidence_fingerprint(new))), str(_ref_id(new) or "")
