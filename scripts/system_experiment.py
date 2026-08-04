#!/usr/bin/env python3
"""Resumable Article -> Ref -> Cluster -> Concept system experiment.

This command is deliberately separate from the legacy per-source compiler.
Intermediate Ref, candidate and judgement files are durable run assets: a
global clustering retry must never require re-running successful discovery.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kmpro_wiki.agentwiki.assets import inventory_assets, strip_missing_image_references
from kmpro_wiki.agentwiki.config import Settings
from kmpro_wiki.agentwiki.contracts import ConceptRef
from kmpro_wiki.agentwiki.global_cluster import (
    CandidateEdge,
    build_clusters,
    candidate_edges,
    kind_for,
    validate_judgements,
)
from kmpro_wiki.agentwiki.llm import OpenAICompatibleClient, LLMError
from kmpro_wiki.agentwiki.spatial_graph import build_spatial_graph
from kmpro_wiki.agentwiki.stages import discover_concepts


JUDGE_BATCH_SIZE = 15
JOINT_CONCEPT_MAX_TOKENS = 2048
JOINT_CONCEPT_MAX_BODY_CHARS = 1800


@dataclass(frozen=True)
class RefRecord:
    ref_id: str
    article_id: str
    concept_id: str
    type: str
    title: str
    description: str
    evidence: tuple[str, ...]
    asset_hints: tuple[str, ...]


def main() -> int:
    settings = Settings.from_env()
    output = settings.data_dir / "system-experiment"
    resume = "--resume" in sys.argv
    dry_run = "--dry-run" in sys.argv
    if output.exists() and not resume:
        shutil.rmtree(output)
    (output / "articles").mkdir(parents=True, exist_ok=True)
    (output / "concepts").mkdir(exist_ok=True)
    _write_manifest(output, {"status": "running", "started_at": _now(), "dry_run": dry_run})

    client = None if dry_run else _client(settings)
    refs, articles = _discover_or_resume(settings, output, client, resume)
    candidates, states = _load_or_make_candidates(output, refs, resume)
    print(f"system.candidates refs={len(refs)} edges={len(candidates)} states={_state_counts(states)}")

    judgements = _load_or_judge(output, client, refs, candidates, resume, dry_run)
    if dry_run:
        _write_manifest(output, {"status": "dry_run_complete", "finished_at": _now(), "articles": len(articles), "refs": len(refs), "candidates": len(candidates)})
        print(f"system.dry_run articles={len(articles)} refs={len(refs)} candidates={len(candidates)}")
        return 0

    validate_judgements([edge.as_dict() for edge in candidates], judgements)
    clusters = build_clusters([asdict(ref) for ref in refs], judgements)
    _write_json(output / "clusters.json", {"clusters": clusters})
    print(f"system.clusters refs={len(refs)} clusters={len(clusters)} merged={sum(1 for cluster in clusters if len(cluster['ref_ids']) > 1)}")

    publishable_clusters = clusters
    concepts = _compile_concepts(output, client, refs, publishable_clusters, resume)
    _write_json(output / "concepts.json", {"concepts": concepts})
    _write_graph(output / "graph.html", articles, refs, concepts, judgements)
    _write_manifest(output, {
        "status": "complete", "finished_at": _now(), "articles": len(articles),
        "refs": len(refs), "candidates": len(candidates), "judgements": len(judgements),
        "clusters": len(clusters), "concepts": len(concepts),
    })
    print(f"system.done articles={len(articles)} refs={len(refs)} candidates={len(candidates)} clusters={len(clusters)} publishable_clusters={len(publishable_clusters)} concepts={len(concepts)}")
    return 0


def _client(settings: Settings) -> OpenAICompatibleClient:
    if not settings.openai_api_key or not settings.openai_model:
        raise ValueError(
            "OPENAI_MODEL and OPENAI_API_KEY are required unless --dry-run is used"
        )
    return OpenAICompatibleClient(
        settings.openai_base_url, settings.openai_api_key, settings.openai_model,
        timeout=settings.openai_timeout_seconds, max_attempts=settings.openai_max_attempts,
        on_event=print, enable_thinking=settings.openai_enable_thinking,
        max_tokens=min(settings.openai_max_tokens, JOINT_CONCEPT_MAX_TOKENS),
    )


def _discover_or_resume(settings: Settings, output: Path, client: OpenAICompatibleClient | None, resume: bool) -> tuple[list[RefRecord], list[dict[str, Any]]]:
    refs_path = output / "refs.json"
    if resume and refs_path.exists():
        saved = json.loads(refs_path.read_text(encoding="utf-8"))
        articles = list(saved["articles"])
        refs = [_parse_ref(item) for item in saved["refs"]]
        print(f"system.resume articles={len(articles)} refs={len(refs)}")
    else:
        articles, refs = [], []
    processed = {item["source"] for item in articles}
    discover = (settings.prompts_dir / "discover.md").read_text(encoding="utf-8")
    sources = sorted((settings.data_dir / "sources").glob("*.md"), key=lambda path: path.name)
    for position, source in enumerate(sources, start=1):
        if source.name in processed:
            continue
        if client is None:
            raise ValueError("cannot discover new Article in --dry-run mode")
        content = strip_missing_image_references(source.read_text(encoding="utf-8"), settings.data_dir / "sources" / "images")
        article_id = f"article-{position:02d}"
        assets = inventory_assets(content, settings.data_dir / "sources" / "images")
        title = _title(content, source.stem)
        print(f"system.discovery.start article={article_id} source={source.name}")
        found = discover_concepts(client, discover, title=title, source_name=source.name, source_content=content, assets=assets)
        article_refs = [_ref_record(article_id, ref) for ref in found]
        refs.extend(article_refs)
        articles.append({"article_id": article_id, "source": source.name, "title": title, "ref_count": len(article_refs), "asset_count": len(assets)})
        _write_article(output / "articles" / f"{article_id}.md", title, source.name, content, article_refs)
        _write_json(refs_path, {"articles": articles, "refs": [asdict(ref) for ref in refs]})
        print(f"system.discovery.done article={article_id} refs={len(article_refs)}")
    return refs, articles


def _load_or_make_candidates(output: Path, refs: list[RefRecord], resume: bool) -> tuple[list[CandidateEdge], dict[str, str]]:
    path = output / "candidates.json"
    if resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        edges = [CandidateEdge(**item) for item in payload["edges"]]
        return edges, dict(payload["states"])
    edges, states = candidate_edges([asdict(ref) for ref in refs])
    _write_json(path, {"edges": [edge.as_dict() for edge in edges], "states": states, "algorithm": edges[0].candidate_version if edges else "metadata-aware-v3", "top_k": 8, "minimum_score": 0.06})
    return edges, states


def _load_or_judge(output: Path, client: OpenAICompatibleClient | None, refs: list[RefRecord], candidates: list[CandidateEdge], resume: bool, dry_run: bool) -> list[dict[str, Any]]:
    path = output / "judgements.json"
    saved: dict[str, dict[str, Any]] = {}
    if resume and path.exists():
        saved = {item["edge_id"]: item for item in json.loads(path.read_text(encoding="utf-8"))["judgements"]}
        print(f"system.judgements.resume completed={len(saved)}")
    if dry_run:
        return list(saved.values())
    if client is None:
        raise AssertionError("client required")
    by_id = {ref.ref_id: ref for ref in refs}
    missing = [edge for edge in candidates if edge.edge_id not in saved]
    failures: list[dict[str, Any]] = []

    def judge_batch(batch: list[CandidateEdge], label: str) -> None:
        try:
            print(f"system.judge.start batch={label} edges={len(batch)}")
            response = json.loads(client.complete(_judge_prompt(batch, by_id), json_schema_name="edge_judgements", json_schema=_judge_schema(len(batch))))
            _require_exact(response, {"judgements"}, "edge judgement response")
            records = response["judgements"]
            _validate_batch(batch, records)
        except LLMError as error:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                print(f"system.judge.split batch={label} edges={len(batch)} reason={type(error).__name__}")
                judge_batch(batch[:midpoint], f"{label}.1")
                judge_batch(batch[midpoint:], f"{label}.2")
                return
            failures.append({"edge_id": batch[0].edge_id, "error": str(error)})
            print(f"system.judge.deferred batch={label} edge={batch[0].edge_id} reason={type(error).__name__}")
            return
        for edge, item in zip(batch, records, strict=True):
            saved[edge.edge_id] = {
                **item,
                "edge_id": edge.edge_id,
                "left_ref_id": edge.left_ref_id,
                "right_ref_id": edge.right_ref_id,
            }
        _write_json(path, {"judgements": [saved[key] for key in sorted(saved)]})
        print(f"system.judge.done batch={label} total={len(saved)}")

    for batch_no, start in enumerate(range(0, len(missing), JUDGE_BATCH_SIZE), start=1):
        batch = missing[start:start + JUDGE_BATCH_SIZE]
        judge_batch(batch, str(batch_no))
    failures_path = output / "judgement_failures.json"
    if failures:
        _write_json(failures_path, {"failures": failures})
        raise LLMError(f"{len(failures)} edge judgements deferred for resume")
    if failures_path.exists():
        failures_path.unlink()
    return [saved[key] for key in sorted(saved)]


def _judge_prompt(edges: list[CandidateEdge], refs: dict[str, RefRecord]) -> str:
    cards: dict[str, dict[str, Any]] = {}
    for edge in edges:
        for ref_id in (edge.left_ref_id, edge.right_ref_id):
            ref = refs[ref_id]
            cards[ref_id] = {
                "ref_id": ref.ref_id,
                "article_id": ref.article_id,
                "kind": kind_for(asdict(ref)),
                "type": ref.type,
                "title": ref.title,
                "description": ref.description,
                "evidence": [text[:500] for text in ref.evidence[:2]],
            }
    payload = [{"position": position, "left_ref_id": edge.left_ref_id, "right_ref_id": edge.right_ref_id, "signals": edge.signals} for position, edge in enumerate(edges, start=1)]
    return """STAGE: judge_edges
你是智库知识审计员。按候选边给出的 position 顺序逐条判断 Ref 对：same 表示同一知识单元的重复表述；complementary 表示不同但互补、可联合编译为同一跨文档主题 Concept；related 表示有实质关联但不足以共同编译；separate 表示不能合并。不可因为同一地区或泛主题而选择 same/complementary。只依据提供的 RefCard 和逐字证据。输出严格 JSON。`judgements` 必须与候选边数组等长，**第 N 个 judgement 只对应 position=N 的候选边**。每个 judgement 只能包含 decision 和 reason；不要输出、复制或改写任何 ID、position 或 Ref 字段。\n\n候选边（数组顺序即绑定顺序）：\n""" + json.dumps(payload, ensure_ascii=False) + "\n\nRefCard：\n" + json.dumps(list(cards.values()), ensure_ascii=False)


def _validate_batch(batch: list[CandidateEdge], records: Any) -> None:
    if not isinstance(records, list) or len(records) != len(batch):
        raise ValueError("edge judgement batch must contain exactly one record per candidate")
    for item in records:
        _require_exact(item, {"decision", "reason"}, "edge judgement")
        if item["decision"] not in {"same", "complementary", "related", "separate"}:
            raise ValueError("edge judgement decision is invalid")


def _compile_concepts(output: Path, client: OpenAICompatibleClient, refs: list[RefRecord], clusters: list[dict[str, Any]], resume: bool) -> list[dict[str, Any]]:
    path = output / "concepts.json"
    saved: dict[str, dict[str, Any]] = {}
    if resume and path.exists():
        allowed = {cluster["id"] for cluster in clusters}
        saved = {item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))["concepts"] if item["id"] in allowed}
    by_id = {ref.ref_id: ref for ref in refs}
    failures: list[dict[str, str]] = []
    for cluster in clusters:
        if cluster["id"] in saved and (output / "concepts" / f"{cluster['id']}.md").exists():
            _write_concept(output / "concepts" / f"{cluster['id']}.md", saved[cluster["id"]])
            continue
        members = [by_id[ref_id] for ref_id in cluster["ref_ids"]]
        if len(members) == 1:
            concept = _compile_singleton(cluster, members[0])
            saved[cluster["id"]] = concept
            _write_concept(output / "concepts" / f"{cluster['id']}.md", concept)
            _write_json(path, {"concepts": [saved[key] for key in sorted(saved)]})
            print(f"system.compile.singleton cluster={cluster['id']}")
            continue
        prompt = _compile_prompt(cluster, members)
        print(f"system.compile.start cluster={cluster['id']} refs={len(members)}")
        try:
            payload = json.loads(client.complete(prompt, json_schema_name="joint_concept", json_schema=_concept_schema()))
        except LLMError as error:
            failures.append({"cluster_id": cluster["id"], "error": str(error)})
            print(f"system.compile.deferred cluster={cluster['id']} reason={type(error).__name__}")
            continue
        _require_exact(payload, {"title", "description", "body"}, f"concept {cluster['id']}")
        concept = {**cluster, **payload, "articles": sorted({item.article_id for item in members})}
        saved[cluster["id"]] = concept
        _write_concept(output / "concepts" / f"{cluster['id']}.md", concept)
        _write_json(path, {"concepts": [saved[key] for key in sorted(saved)]})
        print(f"system.compile.done cluster={cluster['id']}")
    failures_path = output / "compile_failures.json"
    if failures:
        _write_json(failures_path, {"failures": failures})
    elif failures_path.exists():
        failures_path.unlink()
    return [saved[cluster["id"]] for cluster in clusters if cluster["id"] in saved]


def _compile_singleton(cluster: dict[str, Any], ref: RefRecord) -> dict[str, Any]:
    evidence = []
    for position, text in enumerate(ref.evidence, start=1):
        evidence.append(f"### 证据 {position}\n\n{text.strip()}")
    body = f"## 概念摘要\n\n{ref.description.strip()}\n\n## 原文证据\n\n" + "\n\n".join(evidence)
    return {
        **cluster,
        "title": ref.title,
        "description": ref.description,
        "body": body,
        "articles": [ref.article_id],
    }


def _compile_prompt(cluster: dict[str, Any], members: list[RefRecord]) -> str:
    evidence = [{"ref_id": item.ref_id, "article_id": item.article_id, "evidence": list(item.evidence)} for item in members]
    return f"""STAGE: compile_joint_concept
你是智库知识工程师。将同一 Cluster 的 Ref 编译为一个可独立引用的 Concept。只能使用提供的逐字证据；保留关键数字、时间和政策名，不补充事实。不同文章证据不一致时明确适用范围，不要强行统一。直接作答，不输出分析过程。输出严格 JSON：title、description、body。body 最多 {JOINT_CONCEPT_MAX_BODY_CHARS} 个字符，使用 2—4 个 Markdown 二级标题、短段落和必要表格；不得写 frontmatter、图片、链接或来源说明。\n\nCluster：\n""" + json.dumps(cluster, ensure_ascii=False) + "\n\n证据：\n" + json.dumps(evidence, ensure_ascii=False)


def _ref_record(article_id: str, ref: ConceptRef) -> RefRecord:
    return RefRecord(f"{article_id}:{ref.concept_id}", article_id, ref.concept_id, ref.type, ref.title, ref.description, ref.evidence, ref.asset_hints)


def _parse_ref(item: dict[str, Any]) -> RefRecord:
    return RefRecord(**{**item, "evidence": tuple(item["evidence"]), "asset_hints": tuple(item["asset_hints"])})


def _write_article(path: Path, title: str, source: str, content: str, refs: list[RefRecord]) -> None:
    links = "\n".join(f"- `{item.ref_id}`：{item.title}" for item in refs)
    path.write_text(f"---\ntitle: {title}\nsource: {source}\n---\n\n# {title}\n\n## ConceptRef\n\n{links}\n\n## 原文\n\n{content}\n", encoding="utf-8")


def _write_concept(path: Path, concept: dict[str, Any]) -> None:
    yaml_sources = "\n".join(f"  - {json.dumps(article, ensure_ascii=False)}" for article in concept["articles"])
    yaml_refs = "\n".join(f"  - {json.dumps(ref_id, ensure_ascii=False)}" for ref_id in concept["ref_ids"])
    source_links = "\n".join(f"- [Article: {article}](../articles/{article}.md)" for article in concept["articles"])
    ref_links = "\n".join(f"- `{ref_id}`" for ref_id in concept["ref_ids"])
    source_label = "多来源联合编译" if len(concept["articles"]) > 1 else "单来源证据编译"
    frontmatter = (
        "---\n"
        f"type: {json.dumps(concept['type'], ensure_ascii=False)}\n"
        f"title: {json.dumps(concept['title'], ensure_ascii=False)}\n"
        f"description: {json.dumps(concept['description'], ensure_ascii=False)}\n"
        f"kind: {json.dumps(concept['kind'], ensure_ascii=False)}\n"
        f"source: {json.dumps(source_label, ensure_ascii=False)}\n"
        f"sources:\n{yaml_sources}\n"
        f"provenance_refs:\n{yaml_refs}\n"
        "---\n"
    )
    path.write_text(
        f"{frontmatter}\n{concept['body'].strip()}\n\n## 溯源\n\n{source_links}\n\n## ConceptRef\n\n{ref_links}\n",
        encoding="utf-8",
    )


def _write_graph(path: Path, articles: list[dict[str, Any]], refs: list[RefRecord], concepts: list[dict[str, Any]], judgements: list[dict[str, Any]]) -> None:
    path.write_text(
        build_spatial_graph(
            articles,
            [asdict(item) for item in refs],
            concepts,
            judgements,
        ),
        encoding="utf-8",
    )


def _judge_schema(size: int) -> dict[str, Any]:
    item = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["same", "complementary", "related", "separate"]}, "reason": {"type": "string", "minLength": 1}}, "required": ["decision", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {"judgements": {"type": "array", "minItems": size, "maxItems": size, "items": item}}, "required": ["judgements"], "additionalProperties": False}


def _concept_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"title": {"type": "string", "minLength": 1}, "description": {"type": "string", "minLength": 1}, "body": {"type": "string", "minLength": 1, "maxLength": JOINT_CONCEPT_MAX_BODY_CHARS}}, "required": ["title", "description", "body"], "additionalProperties": False}


def _write_manifest(output: Path, values: dict[str, Any]) -> None:
    path = output / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    _write_json(path, {**old, **values, "updated_at": _now()})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _title(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+)$", content)
    return match.group(1).strip() if match else fallback


def _state_counts(states: dict[str, str]) -> dict[str, int]:
    return {state: list(states.values()).count(state) for state in sorted(set(states.values()))}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_exact(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"invalid {label} fields")


if __name__ == "__main__":
    raise SystemExit(main())
