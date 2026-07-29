"""Deterministic candidate retrieval and constrained Ref clustering.

The module deliberately contains no LLM calls.  Similarity only reduces the
number of Ref pairs that require an LLM judgement; it never creates a merge by
itself.  Keeping this layer pure makes a partially completed large run
repeatable and testable.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


KIND_BY_TYPE = {
    "数据口径": "Evidence",
    "分析框架": "Topic",
    "政策建议": "Proposition",
    "国际比较": "Topic",
    "术语解释": "Entity",
}

_STOPWORDS = frozenset(
    {
        "研究", "报告", "分析", "建议", "发展", "推进", "工作", "问题",
        "建设", "提升", "政策", "我国", "地方", "相关", "通过", "对于",
    }
)


@dataclass(frozen=True)
class CandidateEdge:
    edge_id: str
    left_ref_id: str
    right_ref_id: str
    signals: dict[str, Any]
    candidate_version: str = "lexical-v2"

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "left_ref_id": self.left_ref_id,
            "right_ref_id": self.right_ref_id,
            "signals": self.signals,
            "candidate_version": self.candidate_version,
        }


def kind_for(ref: Mapping[str, Any]) -> str:
    return KIND_BY_TYPE[str(ref["type"])]


def candidate_edges(refs: Iterable[Mapping[str, Any]], *, top_k: int = 8, minimum_score: float = 0.06) -> tuple[list[CandidateEdge], dict[str, str]]:
    """Return bounded cross-article candidate pairs and every Ref's state."""
    items = sorted(refs, key=lambda item: str(item["ref_id"]))
    terms = {str(item["ref_id"]): _terms(item) for item in items}
    by_term: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        for term in terms[str(item["ref_id"])]:
            by_term[term].append(item)

    selected: dict[str, CandidateEdge] = {}
    states: dict[str, str] = {str(item["ref_id"]): "no_candidate" for item in items}
    for item in items:
        ref_id = str(item["ref_id"])
        compatible: dict[str, Mapping[str, Any]] = {}
        for term in terms[ref_id]:
            for other in by_term[term]:
                other_id = str(other["ref_id"])
                if other_id == ref_id or other["article_id"] == item["article_id"]:
                    continue
                if not _compatible(item, other):
                    continue
                compatible[other_id] = other
        scored = []
        for other in compatible.values():
            score, signals = _score(item, other, terms[ref_id], terms[str(other["ref_id"])])
            if score >= minimum_score:
                scored.append((score, str(other["ref_id"]), signals))
        for score, other_id, signals in sorted(scored, key=lambda row: (-row[0], row[1]))[:top_k]:
            left, right = sorted((ref_id, other_id))
            edge_id = f"edge:{left}:{right}"
            selected.setdefault(
                edge_id,
                CandidateEdge(
                    edge_id=edge_id,
                    left_ref_id=left,
                    right_ref_id=right,
                    signals={**signals, "score": round(score, 4)},
                ),
            )
            states[ref_id] = "candidates"
            states[other_id] = "candidates"
    return sorted(selected.values(), key=lambda edge: edge.edge_id), states


def validate_judgements(candidates: Iterable[Mapping[str, Any]], judgements: Iterable[Mapping[str, Any]]) -> None:
    expected = {str(edge["edge_id"]) for edge in candidates}
    actual = [str(item["edge_id"]) for item in judgements]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate edge judgement")
    if set(actual) != expected:
        raise ValueError("judgements do not cover exactly the candidate edge set")
    if any(item.get("decision") not in {"same", "complementary", "related", "separate"} for item in judgements):
        raise ValueError("invalid edge judgement decision")


def build_clusters(refs: Iterable[Mapping[str, Any]], judgements: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build conservative cross-document themes from compatible mergeable edges."""
    by_id = {str(item["ref_id"]): item for item in refs}
    mergeable = sorted(
        (item for item in judgements if item["decision"] in {"same", "complementary"}),
        key=lambda item: str(item["edge_id"]),
    )
    separate = {
        tuple(sorted((str(item["left_ref_id"]), str(item["right_ref_id"]))))
        for item in judgements if item["decision"] == "separate"
    }
    parent = {ref_id: ref_id for ref_id in by_id}

    def root(ref_id: str) -> str:
        while parent[ref_id] != ref_id:
            parent[ref_id] = parent[parent[ref_id]]
            ref_id = parent[ref_id]
        return ref_id

    def members(ref_id: str) -> set[str]:
        group = root(ref_id)
        return {candidate for candidate in by_id if root(candidate) == group}

    judged_mergeable = {
        tuple(sorted((str(item["left_ref_id"]), str(item["right_ref_id"]))))
        for item in mergeable
    }
    for edge in mergeable:
        left, right = str(edge["left_ref_id"]), str(edge["right_ref_id"])
        left_group, right_group = members(left), members(right)
        if left_group == right_group:
            continue
        representative = min(left_group | right_group)
        incoming = right if representative in left_group else left
        if tuple(sorted((representative, incoming))) not in judged_mergeable:
            continue
        if any(pair in separate for first in left_group for second in right_group for pair in [tuple(sorted((first, second)))]):
            continue
        if any(not _compatible(by_id[first], by_id[second]) for first in left_group for second in right_group):
            continue
        parent[root(right)] = root(left)

    groups: dict[str, list[str]] = defaultdict(list)
    for ref_id in sorted(by_id):
        groups[root(ref_id)].append(ref_id)
    clusters = []
    for members_list in sorted(groups.values(), key=lambda group: group[0]):
        first = by_id[members_list[0]]
        digest = hashlib.sha256("\0".join(members_list).encode()).hexdigest()[:10]
        clusters.append(
            {
                "id": f"{kind_for(first).lower()}-{digest}",
                "kind": kind_for(first),
                "type": first["type"],
                "title": first["title"],
                "description": first["description"],
                "ref_ids": members_list,
                "article_count": len({str(by_id[ref_id]["article_id"]) for ref_id in members_list}),
            }
        )
    return clusters


def _compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["type"] == right["type"] and kind_for(left) == kind_for(right)


def _terms(ref: Mapping[str, Any]) -> set[str]:
    text = f"{ref.get('title', '')} {ref.get('description', '')}"
    words = re.findall(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]{2,}", text)
    values: set[str] = set()
    for word in words:
        if word in _STOPWORDS:
            continue
        values.add(word.lower())
        if re.fullmatch(r"[\u3400-\u9fff]{2,}", word):
            values.update(word[index:index + 2] for index in range(len(word) - 1))
    return values


def _score(left: Mapping[str, Any], right: Mapping[str, Any], left_terms: set[str], right_terms: set[str]) -> tuple[float, dict[str, Any]]:
    intersection = left_terms & right_terms
    union = left_terms | right_terms
    lexical = len(intersection) / len(union) if union else 0.0
    title_overlap = bool(_terms({"title": left["title"], "description": ""}) & _terms({"title": right["title"], "description": ""}))
    return lexical, {"shared_terms": sorted(intersection)[:12], "lexical": round(lexical, 4), "title_overlap": title_overlap}
