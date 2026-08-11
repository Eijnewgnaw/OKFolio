#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from okfolio.agentwiki.assets import AssetError, SourceAsset, inventory_assets
from okfolio.agentwiki.okf import (
    ConceptDocument,
    parse_concept_markdown,
    rewrite_image_paths,
    validate_link_only_enrichment,
)
from okfolio.agentwiki.relations import classify_links
from okfolio.agentwiki.state import Manifest


LLM_EVENT_RE = re.compile(
    r"llm\.done elapsed_ms=(?P<elapsed>\d+).*?"
    r"prompt_tokens=(?P<prompt>\d+|unknown) "
    r"completion_tokens=(?P<completion>\d+|unknown) "
    r"total_tokens=(?P<total>\d+|unknown)"
)


def audit_bundle(data_dir: Path, *, event_log: Path | None = None) -> dict[str, Any]:
    sources_dir = data_dir / "sources"
    source_images_dir = sources_dir / "images"
    inputs: dict[str, dict[str, Any]] = {}
    source_assets: dict[str, tuple[SourceAsset, ...]] = {}
    for source in sorted(sources_dir.glob("*.md"), key=lambda item: item.name):
        try:
            assets = inventory_assets(
                source.read_text(encoding="utf-8"), source_images_dir
            )
        except AssetError as error:
            inputs[source.name] = {"status": "input_error", "error": str(error)}
            continue
        source_assets[source.name] = assets
        inputs[source.name] = {
            "status": "ready",
            "assets": len(assets),
            "images": sum(item.kind == "image" for item in assets),
            "html_tables": sum(item.kind == "html_table" for item in assets),
            "markdown_tables": sum(
                item.kind == "markdown_table" for item in assets
            ),
        }

    concepts = _load_concepts(data_dir / "wiki" / "concepts")
    concepts_by_source: dict[str, list[ConceptDocument]] = {}
    for concept in concepts.values():
        source_name = str(concept.frontmatter.get("source", ""))
        concepts_by_source.setdefault(source_name, []).append(concept)

    manifest = Manifest.load(data_dir / ".state" / "manifest.json")
    source_metrics = {
        name: {
            "status": state.status,
            "discovery_status": state.discovery_status,
            "concept_status": state.concept_status,
            "preservation_status": state.preservation_status,
            "relation_status": state.relation_status,
            "outputs": len(state.outputs),
        }
        for name, state in sorted(manifest.sources.items())
    }
    for source_name, assets in source_assets.items():
        generated = "\n".join(
            concept.body for concept in concepts_by_source.get(source_name, [])
        )
        expected = [rewrite_image_paths(item.raw) for item in assets]
        inputs[source_name]["asset_occurrences_ok"] = all(
            generated.count(raw) == expected.count(raw) for raw in set(expected)
        )

    graph_content = _read_text(data_dir / "outputs" / "graph.html")
    graph = {
        "nodes": len(re.findall(r"data-node=", graph_content)),
        "edges": len(re.findall(r"data-source=", graph_content)),
    }
    site_pages = len(list((data_dir / "outputs" / "site" / "concepts").glob("*.html")))
    llm_metrics = _llm_metrics(event_log)
    invariants = _stage_invariants(data_dir, source_assets)
    return {
        "inputs": inputs,
        "sources": source_metrics,
        "concepts": len(concepts),
        "taxonomy": _taxonomy(concepts),
        "links": classify_links(concepts),
        "graph": graph,
        "site_pages": site_pages,
        "stage_invariants": invariants,
        "llm": llm_metrics,
    }


def render_report(metrics: dict[str, Any]) -> str:
    links = metrics["links"]
    llm = metrics["llm"]
    lines = [
        "# OKF 四阶段实验自动审计",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Concept | {metrics['concepts']} |",
        f"| 同来源链接 | {links['same_source']} |",
        f"| 跨来源链接 | {links['cross_source']} |",
        f"| 断链 | {links['broken']} |",
        f"| 自链接 | {links['self']} |",
        f"| 图节点 | {metrics['graph']['nodes']} |",
        f"| 图边 | {metrics['graph']['edges']} |",
        f"| 站点 Concept 页 | {metrics['site_pages']} |",
        f"| LLM 调用 | {llm['calls']} |",
        f"| prompt tokens | {llm['prompt_tokens']} |",
        f"| completion tokens | {llm['completion_tokens']} |",
        f"| total tokens | {llm['total_tokens']} |",
        f"| 缓存命中 | {llm['cache_hits']} |",
        "",
        "## 输入预检",
        "",
        "| 来源 | 状态 | 资产 | 说明 |",
        "|---|---|---:|---|",
    ]
    for source_name, item in metrics["inputs"].items():
        lines.append(
            f"| {source_name} | {item['status']} | {item.get('assets', 0)} | "
            f"{item.get('error', '')} |"
        )
    lines.extend(
        [
            "",
            "## 阶段不变性",
            "",
            f"- 去除资产插入后等于阶段 B：{metrics['stage_invariants']['asset_only']}",
            f"- 去除链接语法后等于阶段 C：{metrics['stage_invariants']['link_only']}",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_invariants(
    data_dir: Path, source_assets: dict[str, tuple[SourceAsset, ...]]
) -> dict[str, bool | None]:
    asset_checks: list[bool] = []
    link_checks: list[bool] = []
    for source_name, assets in source_assets.items():
        cache_path = (
            data_dir
            / ".staging"
            / "sources"
            / hashlib.md5(source_name.encode("utf-8")).hexdigest()
            / "cache.json"
        )
        if not cache_path.exists():
            continue
        try:
            stages = json.loads(cache_path.read_text(encoding="utf-8"))["stages"]
            if "concept" in stages:
                draft_payload = stages["concept"]["payload"]
            else:
                draft_payload = [
                    stage["payload"]
                    for name, stage in sorted(stages.items())
                    if name.startswith("concept:")
                ]
                if not draft_payload:
                    raise KeyError("concept drafts")
            preservation = stages["preservation"]["payload"]
            relation = stages["relation"]["payload"]
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        current_concept_ids = {
            Path(item["filename"]).stem for item in preservation["concepts"]
        }
        draft_payload = [
            item
            for item in draft_payload
            if item["ref"]["concept_id"] in current_concept_ids
        ]
        asset_checks.append(
            _asset_only_check(
                draft_payload,
                preservation["concepts"],
                preservation["placements"],
                assets,
            )
        )
        link_checks.append(
            _link_only_check(preservation["concepts"], relation["concepts"])
        )
    return {
        "asset_only": all(asset_checks) if asset_checks else None,
        "link_only": all(link_checks) if link_checks else None,
    }


def _asset_only_check(
    draft_payload: list[dict[str, Any]],
    concept_payload: list[dict[str, str]],
    placements: list[dict[str, str]],
    assets: tuple[SourceAsset, ...],
) -> bool:
    drafts = {item["ref"]["concept_id"]: item["body"] for item in draft_payload}
    bodies = {
        Path(item["filename"]).stem: parse_concept_markdown(
            item["filename"], item["content"]
        ).body
        for item in concept_payload
    }
    asset_by_id = {item.asset_id: item for item in assets}
    try:
        for placement in reversed(placements):
            raw = asset_by_id[placement["asset_id"]].raw
            anchor = placement["anchor"]
            inserted = (
                f"{raw}\n\n{anchor}"
                if placement["position"] == "before"
                else f"{anchor}\n\n{raw}"
            )
            concept_id = placement["concept_id"]
            bodies[concept_id] = bodies[concept_id].replace(inserted, anchor, 1)
    except KeyError:
        return False
    return bodies == drafts


def _link_only_check(
    before_payload: list[dict[str, str]],
    after_payload: list[dict[str, str]],
) -> bool:
    before = {
        item["filename"]: parse_concept_markdown(item["filename"], item["content"])
        for item in before_payload
    }
    after = {
        item["filename"]: parse_concept_markdown(item["filename"], item["content"])
        for item in after_payload
    }
    if set(before) != set(after):
        return False
    try:
        for filename in before:
            validate_link_only_enrichment(before[filename], after[filename])
    except ValueError:
        return False
    return True


def _load_concepts(directory: Path) -> dict[str, ConceptDocument]:
    concepts: dict[str, ConceptDocument] = {}
    for path in sorted(directory.glob("*.md"), key=lambda item: item.name):
        concepts[path.name] = parse_concept_markdown(
            path.name, path.read_text(encoding="utf-8")
        )
    return concepts


def _taxonomy(concepts: dict[str, ConceptDocument]) -> dict[str, int]:
    result: dict[str, int] = {}
    for concept in concepts.values():
        value = str(concept.frontmatter.get("type", ""))
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _llm_metrics(event_log: Path | None) -> dict[str, int]:
    result = {
        "calls": 0,
        "elapsed_ms": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_hits": 0,
    }
    if event_log is None or not event_log.exists():
        return result
    content = event_log.read_text(encoding="utf-8")
    result["cache_hits"] = content.count("cache.hit ")
    for match in LLM_EVENT_RE.finditer(content):
        result["calls"] += 1
        result["elapsed_ms"] += int(match.group("elapsed"))
        for field, group in (
            ("prompt_tokens", "prompt"),
            ("completion_tokens", "completion"),
            ("total_tokens", "total"),
        ):
            value = match.group(group)
            if value != "unknown":
                result[field] += int(value)
    return result


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    metrics = audit_bundle(args.data_dir, event_log=args.event_log)
    if args.preflight:
        print(json.dumps(metrics["inputs"], ensure_ascii=False, indent=2))
        return 1 if any(
            item["status"] == "input_error" for item in metrics["inputs"].values()
        ) else 0
    output_dir = args.output_dir or args.data_dir / "outputs" / "experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
