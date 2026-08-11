#!/usr/bin/env python3
"""Judge cross-Concept relations for a completed Agent compiler run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from kmpro_wiki.agentwiki.config import Settings
from kmpro_wiki.agentwiki.llm import OpenAICompatibleClient


DEFAULT_BATCH_SIZE = 15

# ``related`` is intentionally retained as a conservative fallback.  The
# other labels describe the relationship rather than merely its existence, so
# the resulting graph can be filtered without asking a model again.
RELATION_TYPES = (
    "defines",
    "supports",
    "constrains",
    "causes",
    "recommends",
    "compares",
    "extends",
    "related",
)
RELATION_DIRECTIONS = (
    "left_to_right",
    "right_to_left",
    "bidirectional",
)


class CompletionClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str: ...


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _schema(size: int) -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["related", "separate"],
            },
            "reason": {"type": "string", "minLength": 1},
            "relation_type": {
                "type": "string",
                "enum": ["none", *RELATION_TYPES],
            },
            "direction": {
                "type": "string",
                "enum": ["none", *RELATION_DIRECTIONS],
            },
            "evidence_ref_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "items": {"type": "string"},
            },
        },
        "required": [
            "decision",
            "reason",
            "relation_type",
            "direction",
            "evidence_ref_ids",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "judgements": {
                "type": "array",
                "minItems": size,
                "maxItems": size,
                "items": item,
            }
        },
        "required": ["judgements"],
        "additionalProperties": False,
    }


def _prompt(
    edges: list[dict[str, Any]],
    refs: dict[str, dict[str, Any]],
    ref_to_group: dict[str, str],
) -> str:
    cards: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for position, edge in enumerate(edges, start=1):
        left_id = edge["left_ref_id"]
        right_id = edge["right_ref_id"]
        for ref_id in (left_id, right_id):
            ref = refs[ref_id]
            cards[ref_id] = {
                "ref_id": ref_id,
                "concept_id": ref_to_group[ref_id],
                "article_id": ref["article_id"],
                "type": ref["type"],
                "title": ref["title"],
                "description": ref["description"],
                "evidence": [
                    text[:600] for text in ref.get("evidence", [])[:2]
                ],
                "scope": ref.get("scope", {}),
                "section_path": ref.get("section_path", []),
                "page_start": ref.get("page_start"),
                "page_end": ref.get("page_end"),
            }
        pairs.append(
            {
                "position": position,
                "left_ref_id": left_id,
                "right_ref_id": right_id,
                "allowed_evidence_ref_ids": [left_id, right_id],
                "signals": edge.get("signals", {}),
            }
        )
    return (
        "STAGE: judge_final_concept_relations\n"
        "你是智库知识关系审计员。候选 Ref 已经分别编译到不同的最终 "
        "Concept；本阶段只判断两个最终 Concept 之间是否存在值得进入知识图谱"
        "和 Markdown 链接的实质关系。\n\n"
        "判断规则：\n"
        "- related：指标定义、因果/约束、问题与建议、同一政策链条、可相互"
        "补充的国际比较等关系，阅读另一 Concept 会明显增加理解；\n"
        "- separate：仅共享城市、行业泛词或背景词，不能增加实质理解。\n"
        "不要因为候选召回分数高就判 related。不要改写、合并 Concept。"
        "只依据 RefCard 和逐字证据。\n\n"
        "对 related，必须给出：relation_type（defines=定义/口径，supports="
        "证据支撑，constrains=约束条件，causes=因果，recommends=问题到建议，"
        "compares=比较，extends=补充展开，related=无法再细分的实质关联）、"
        "direction（left_to_right、right_to_left 或 bidirectional），以及从"
        "allowed_evidence_ref_ids 选择 1–2 个真正支持关系的 Ref ID。\n"
        "对 separate，relation_type 和 direction 必须为 none，"
        "evidence_ref_ids 必须为空数组。\n\n"
        "输出严格 JSON。judgements 必须与候选数组等长，第 N 项只对应 "
        "position=N；每项只能包含 decision 和 reason，不得复制 ID。\n\n"
        "候选数组：\n"
        f"{json.dumps(pairs, ensure_ascii=False)}\n\n"
        "RefCard：\n"
        f"{json.dumps(list(cards.values()), ensure_ascii=False)}"
    )


def _parse_response(
    response: str,
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = json.loads(response)
    if set(payload) != {"judgements"}:
        raise ValueError("relation response must contain only judgements")
    records = payload["judgements"]
    if not isinstance(records, list) or len(records) != len(edges):
        raise ValueError("relation response count does not match candidate count")
    parsed: list[dict[str, Any]] = []
    for edge, item in zip(edges, records, strict=True):
        required = {
            "decision",
            "reason",
            "relation_type",
            "direction",
            "evidence_ref_ids",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("relation judgement fields are invalid")
        if item["decision"] not in {"related", "separate"}:
            raise ValueError("relation judgement decision is invalid")
        reason = item["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("relation judgement reason is empty")
        relation_type = item["relation_type"]
        direction = item["direction"]
        evidence_ref_ids = item["evidence_ref_ids"]
        if not isinstance(evidence_ref_ids, list) or len(evidence_ref_ids) > 2:
            raise ValueError("relation evidence refs are invalid")
        allowed_ids = {edge["left_ref_id"], edge["right_ref_id"]}
        if (
            any(not isinstance(ref_id, str) for ref_id in evidence_ref_ids)
            or not set(evidence_ref_ids).issubset(allowed_ids)
            or len(set(evidence_ref_ids)) != len(evidence_ref_ids)
        ):
            raise ValueError("relation evidence refs must come from the edge")
        if item["decision"] == "related":
            if relation_type not in RELATION_TYPES:
                raise ValueError("related judgement needs a relation type")
            if direction not in RELATION_DIRECTIONS:
                raise ValueError("related judgement needs a direction")
            if not evidence_ref_ids:
                raise ValueError("related judgement needs evidence refs")
        elif (
            relation_type != "none"
            or direction != "none"
            or evidence_ref_ids
        ):
            raise ValueError("separate judgement cannot carry a relation")
        parsed.append(
            {
                "edge_id": edge["edge_id"],
                "left_ref_id": edge["left_ref_id"],
                "right_ref_id": edge["right_ref_id"],
                "decision": item["decision"],
                "reason": reason.strip(),
                "relation_type": relation_type,
                "direction": direction,
                "evidence_ref_ids": evidence_ref_ids,
                "signals": edge.get("signals", {}),
            }
        )
    return parsed


def judge_relations(
    run_dir: Path,
    client: CompletionClient,
    *,
    resume: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_event: callable = print,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 30:
        raise ValueError("batch_size must be between 1 and 30")
    manifest = _load(run_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Agent run must be complete before relation judging")
    refs = {
        item["ref_id"]: item
        for item in _load(run_dir / "refs.json")["refs"]
    }
    groups = _load(run_dir / "groups.json")["groups"]
    ref_to_group = {
        ref_id: group["group_id"]
        for group in groups
        for ref_id in group["ref_ids"]
    }
    candidates = _load(run_dir / "candidates.json")["edges"]
    cross_edges = [
        edge
        for edge in candidates
        if ref_to_group[edge["left_ref_id"]]
        != ref_to_group[edge["right_ref_id"]]
    ]
    output = run_dir / "relations.json"
    if output.exists() and not resume:
        raise ValueError(f"relation output already exists: {output}")
    saved = {}
    if resume and output.exists():
        saved = {
            item["edge_id"]: item
            for item in _load(output).get("judgements", [])
        }
    valid_ids = {edge["edge_id"] for edge in cross_edges}
    saved = {
        edge_id: item
        for edge_id, item in saved.items()
        if edge_id in valid_ids
    }
    missing = [
        edge for edge in cross_edges if edge["edge_id"] not in saved
    ]

    def persist() -> None:
        judgements = [saved[key] for key in sorted(saved)]
        _write_json_atomic(
            output,
            {
                "status": (
                    "complete"
                    if len(judgements) == len(cross_edges)
                    else "running"
                ),
                "candidate_edges": len(candidates),
                "intra_concept_edges": len(candidates) - len(cross_edges),
                "cross_concept_edges": len(cross_edges),
                "judgements": judgements,
                "related": sum(
                    item["decision"] == "related" for item in judgements
                ),
                "separate": sum(
                    item["decision"] == "separate" for item in judgements
                ),
            },
        )

    def judge_batch(batch: list[dict[str, Any]], label: str) -> None:
        try:
            on_event(f"relations.start batch={label} edges={len(batch)}")
            response = client.complete(
                _prompt(batch, refs, ref_to_group),
                json_schema_name="final_concept_relations",
                json_schema=_schema(len(batch)),
            )
            parsed = _parse_response(response, batch)
        except Exception as error:
            if len(batch) == 1:
                raise
            midpoint = len(batch) // 2
            on_event(
                f"relations.split batch={label} edges={len(batch)} "
                f"reason={type(error).__name__}"
            )
            judge_batch(batch[:midpoint], f"{label}.1")
            judge_batch(batch[midpoint:], f"{label}.2")
            return
        for item in parsed:
            saved[item["edge_id"]] = item
        persist()
        on_event(
            f"relations.done batch={label} total={len(saved)}/"
            f"{len(cross_edges)}"
        )

    for batch_number, start in enumerate(
        range(0, len(missing), batch_size),
        start=1,
    ):
        judge_batch(
            missing[start : start + batch_size],
            str(batch_number),
        )
    persist()
    result = _load(output)
    if result["status"] != "complete":
        raise RuntimeError("relation judging did not cover every candidate edge")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.openai_model:
        raise ValueError("OPENAI_MODEL is required")
    client = OpenAICompatibleClient(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.openai_model,
        timeout=settings.openai_timeout_seconds,
        max_attempts=settings.openai_max_attempts,
        on_event=print,
        enable_thinking=settings.openai_enable_thinking,
        max_tokens=settings.openai_max_tokens,
    )
    result = judge_relations(
        args.run_dir,
        client,
        resume=args.resume,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
