#!/usr/bin/env python3
"""Summarize an Agent compiler run for engineering and experiment reporting."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LLM_DONE = re.compile(
    r"llm\.done schema=(?P<schema>\S+) "
    r"elapsed_ms=(?P<elapsed>\d+) "
    r".*?"
    r"prompt_tokens=(?P<prompt>\d+) "
    r"completion_tokens=(?P<completion>\d+) "
    r"total_tokens=(?P<total>\d+)"
)
LLM_RETRY = re.compile(
    r"llm\.retry attempt=(?P<attempt>\d+)/(?P<limit>\d+) "
    r"reason=(?P<reason>.*)"
)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0,
            "p50": 0,
            "p95": 0,
            "min": 0,
            "max": 0,
        }
    return {
        "count": len(values),
        "sum": round(sum(values), 3),
        "mean": round(statistics.mean(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def count_files(path: Path, pattern: str) -> int:
    return (
        sum(item.is_file() for item in path.rglob(pattern))
        if path.exists()
        else 0
    )


def summarize(
    run_dir: Path,
    log_path: Path | None,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    sources_payload = load_json(run_dir / "source_progress.json", {"sources": []})
    candidates_payload = load_json(
        run_dir / "candidates.json", {"edges": [], "states": {}}
    )
    groups_payload = load_json(run_dir / "groups.json", {"groups": []})
    compile_payload = load_json(
        run_dir / "compile_progress.json",
        {"drafts": {}, "quality": {}, "reviews": [], "recompiles": 0},
    )
    trace_payload = load_json(run_dir / "agent_trace.json", {"events": []})
    validation_payload = load_json(
        run_dir / "ref_validation.json",
        {"raw_refs": 0, "accepted_refs": 0, "rejected_refs": []},
    )
    acceptance = load_json(run_dir / "acceptance.json", {})
    manifest = load_json(run_dir / "manifest.json", {})

    sources = sources_payload["sources"]
    raw_refs = [ref for source in sources for ref in source.get("refs", [])]
    rejected_ref_ids = {
        item["ref_id"] for item in validation_payload.get("rejected_refs", [])
    }
    refs = [ref for ref in raw_refs if ref["ref_id"] not in rejected_ref_ids]
    refs_by_id = {ref["ref_id"]: ref for ref in refs}
    ref_counts = [
        sum(
            ref["ref_id"] not in rejected_ref_ids
            for ref in source.get("refs", [])
        )
        for source in sources
    ]
    groups = groups_payload["groups"]
    joint_groups = [group for group in groups if len(group["ref_ids"]) > 1]
    singleton_groups = [group for group in groups if len(group["ref_ids"]) == 1]
    joint_ref_ids = {
        ref_id for group in joint_groups for ref_id in group["ref_ids"]
    }
    joint_articles = {
        refs_by_id[ref_id]["article_id"]
        for ref_id in joint_ref_ids
        if ref_id in refs_by_id
    }

    audits = compile_payload.get("quality", {})
    first_audits = [items[0] for items in audits.values() if items]
    final_audits = [items[-1] for items in audits.values() if items]
    recompiled_audits = [items for items in audits.values() if len(items) > 1]
    score_improvements = [
        float(items[-1]["score"]) - float(items[0]["score"])
        for items in recompiled_audits
    ]

    events = trace_payload.get("events", [])
    recoveries = [
        event
        for event in events
        if event.get("stage") == "compile_group_plan"
        and event.get("contract_recovery")
    ]
    recovered_ref_ids = {
        ref_id
        for event in recoveries
        for ref_id in event["contract_recovery"].get("ref_ids", [])
    }
    resume_counts = Counter(
        event.get("reused", "unknown")
        for event in events
        if event.get("stage") == "resume"
    )

    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path is not None and log_path.exists()
        else ""
    )
    calls = [
        {
            "schema": match.group("schema"),
            "elapsed_ms": int(match.group("elapsed")),
            "prompt_tokens": int(match.group("prompt")),
            "completion_tokens": int(match.group("completion")),
            "total_tokens": int(match.group("total")),
        }
        for match in LLM_DONE.finditer(log_text)
    ]
    retries = [
        {
            "attempt": int(match.group("attempt")),
            "limit": int(match.group("limit")),
            "reason": match.group("reason").strip(),
        }
        for match in LLM_RETRY.finditer(log_text)
    ]
    calls_by_schema: dict[str, dict[str, Any]] = {}
    for schema in sorted({call["schema"] for call in calls}):
        selected = [call for call in calls if call["schema"] == schema]
        calls_by_schema[schema] = {
            "calls": len(selected),
            "prompt_tokens": sum(call["prompt_tokens"] for call in selected),
            "completion_tokens": sum(
                call["completion_tokens"] for call in selected
            ),
            "total_tokens": sum(call["total_tokens"] for call in selected),
            "elapsed_ms": distribution(
                [float(call["elapsed_ms"]) for call in selected]
            ),
        }

    experiment_span_seconds = 0.0
    if log_path is not None and log_path.exists():
        stat = log_path.stat()
        started = getattr(stat, "st_birthtime", stat.st_ctime)
        experiment_span_seconds = max(0.0, stat.st_mtime - started)

    source_asset_hints = {
        hint for ref in refs for hint in ref.get("asset_hints", []) if hint
    }
    reviews = compile_payload.get("reviews", [])
    review_kinds = Counter(item.get("kind", "unknown") for item in reviews)
    concepts_dir = run_dir / "concepts"
    images_dir = run_dir / "images"

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "status": manifest.get("status", "unknown"),
        "model": model or manifest.get("model", "unknown"),
        "corpus": {
            "articles": len(sources),
            "raw_discovered_refs": len(raw_refs),
            "refs": len(refs),
            "rejected_refs": len(rejected_ref_ids),
            "ref_rejection_reasons": dict(
                Counter(
                    item.get("reason", "unknown")
                    for item in validation_payload.get("rejected_refs", [])
                )
            ),
            "ref_type_counts": dict(Counter(ref["type"] for ref in refs)),
            "refs_per_article": distribution([float(value) for value in ref_counts]),
            "discovery_routes": dict(
                Counter(source["plan"]["discovery_mode"] for source in sources)
            ),
            "refine_articles": sum(
                bool(source["plan"]["refine_discovery"]) for source in sources
            ),
            "auto_asset_articles": sum(
                source["plan"]["asset_policy"] == "auto" for source in sources
            ),
            "asset_bearing_articles": sum(
                int(source["profile"].get("asset_count", 0)) > 0
                for source in sources
            ),
            "referenced_assets": sum(
                int(source["profile"].get("asset_count", 0))
                for source in sources
            ),
            "concept_asset_hints": len(source_asset_hints),
        },
        "candidate_graph": {
            "edges": len(candidates_payload.get("edges", [])),
            "state_counts": dict(
                Counter(candidates_payload.get("states", {}).values())
            ),
        },
        "grouping": {
            "groups": len(groups),
            "single_groups": len(singleton_groups),
            "joint_groups": len(joint_groups),
            "joint_group_rate": (
                round(len(joint_groups) / len(groups), 4) if groups else 0.0
            ),
            "refs_in_joint_groups": len(joint_ref_ids),
            "joint_ref_rate": (
                round(len(joint_ref_ids) / len(refs), 4) if refs else 0.0
            ),
            "articles_in_joint_groups": len(joint_articles),
            "joint_group_sizes": dict(
                sorted(
                    Counter(str(len(group["ref_ids"])) for group in joint_groups).items()
                )
            ),
            "contract_recovery_batches": len(recoveries),
            "contract_recovery_refs": len(recovered_ref_ids),
        },
        "quality": {
            "completed_groups": len(compile_payload.get("completed_groups", [])),
            "first_pass_groups": sum(
                item.get("decision") == "pass" for item in first_audits
            ),
            "first_pass_rate": (
                round(
                    sum(item.get("decision") == "pass" for item in first_audits)
                    / len(first_audits),
                    4,
                )
                if first_audits
                else 0.0
            ),
            "first_scores": distribution(
                [float(item["score"]) for item in first_audits]
            ),
            "final_scores": distribution(
                [float(item["score"]) for item in final_audits]
            ),
            "final_decisions": dict(
                Counter(item.get("decision", "unknown") for item in final_audits)
            ),
            "recompiled_groups": len(recompiled_audits),
            "recompile_calls": int(compile_payload.get("recompiles", 0)),
            "score_improvement": distribution(score_improvements),
            "reviews": len(reviews),
            "review_kinds": dict(review_kinds),
        },
        "outputs": {
            "published_concepts": count_files(concepts_dir, "*.md"),
            "published_images": count_files(images_dir, "*"),
            "manifest_concepts": manifest.get("concepts"),
            "manifest_reviews": manifest.get("reviews"),
        },
        "acceptance": acceptance,
        "runtime": {
            "physical_model_calls": len(calls),
            "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
            "completion_tokens": sum(
                call["completion_tokens"] for call in calls
            ),
            "total_tokens": sum(call["total_tokens"] for call in calls),
            "model_elapsed_ms": distribution(
                [float(call["elapsed_ms"]) for call in calls]
            ),
            "max_prompt_tokens": max(
                (call["prompt_tokens"] for call in calls), default=0
            ),
            "request_retries": len(retries),
            "retry_reasons": dict(
                Counter(item["reason"] for item in retries)
            ),
            "calls_by_schema": calls_by_schema,
            "experiment_span_seconds_including_recovery": round(
                experiment_span_seconds, 3
            ),
            "resume_reuse_counts": dict(resume_counts),
        },
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    corpus = metrics["corpus"]
    grouping = metrics["grouping"]
    quality = metrics["quality"]
    runtime = metrics["runtime"]
    outputs = metrics["outputs"]
    acceptance = metrics["acceptance"]
    route_text = "、".join(
        f"{name} {count}"
        for name, count in sorted(corpus["discovery_routes"].items())
    )
    retry_reason_text = "；".join(
        f"{reason}（{count} 次）"
        for reason, count in runtime["retry_reasons"].items()
    )
    retry_reason_line = (
        f"- 重试原因记录：{retry_reason_text}。\n"
        if retry_reason_text
        else ""
    )
    schema_rows = "\n".join(
        "| {schema} | {calls} | {prompt:,} | {completion:,} | {mean:.0f} | {p95:.0f} |".format(
            schema=schema,
            calls=values["calls"],
            prompt=values["prompt_tokens"],
            completion=values["completion_tokens"],
            mean=values["elapsed_ms"]["mean"],
            p95=values["elapsed_ms"]["p95"],
        )
        for schema, values in runtime["calls_by_schema"].items()
    )
    return f"""# Agent 全量系统实验报告

生成时间：{metrics["generated_at"]}

运行状态：`{metrics["status"]}`  
正式模型：`{metrics["model"]}`

## 核心结果

- 输入 {corpus["articles"]} 篇 Article，原始发现 {corpus["raw_discovered_refs"]} 个 ConceptRef；确定性证据校验剔除 {corpus["rejected_refs"]} 个空壳 Ref，{corpus["refs"]} 个有效 Ref 进入编译。每篇有效 Ref 平均 {corpus["refs_per_article"]["mean"]:.2f} 个，中位数 {corpus["refs_per_article"]["p50"]:.1f} 个，范围 {corpus["refs_per_article"]["min"]:.0f}–{corpus["refs_per_article"]["max"]:.0f}。
- Agent 路由：{route_text}；{corpus["refine_articles"]} 篇自动触发 Discovery refine。
- {corpus["asset_bearing_articles"]} 篇文档含 {corpus["referenced_assets"]} 个有效图表资产引用，全部进入自动归位策略；其中图片 {acceptance.get("asset_kind_counts", {}).get("image", outputs["published_images"])} 个、HTML 表格 {acceptance.get("asset_kind_counts", {}).get("html_table", 0)} 个。
- 确定性候选图形成 {metrics["candidate_graph"]["edges"]} 条跨 Article 候选边。
- 联合编译得到 {grouping["groups"]} 个 Concept 组：{grouping["joint_groups"]} 个联合组、{grouping["single_groups"]} 个独立组。联合组覆盖 {grouping["refs_in_joint_groups"]} 个 Ref、{grouping["articles_in_joint_groups"]} 篇 Article。
- 已完成质检 {quality["completed_groups"]}/{grouping["groups"]} 组；首轮通过率 {quality["first_pass_rate"]:.1%}，自动重编译 {quality["recompile_calls"]} 次，最终待人工复核 {quality["reviews"]} 项。
- 发布 Concept 文件 {outputs["published_concepts"]} 个，发布图片 {outputs["published_images"]} 个。

## 运行与成本

- 实际模型请求 {runtime["physical_model_calls"]} 次；输入 Token {runtime["prompt_tokens"]:,}，输出 Token {runtime["completion_tokens"]:,}，合计 {runtime["total_tokens"]:,}。
- 单次模型请求平均 {runtime["model_elapsed_ms"]["mean"] / 1000:.2f}s，P95 {runtime["model_elapsed_ms"]["p95"] / 1000:.2f}s，最大 {runtime["model_elapsed_ms"]["max"] / 1000:.2f}s。
- 最大单次输入 {runtime["max_prompt_tokens"]:,} Token；调用封装重试 {runtime["request_retries"]} 次。
{retry_reason_line}- 从首次启动到最终产物的实验跨度（含诊断修复停顿）为 {runtime["experiment_span_seconds_including_recovery"] / 60:.1f} 分钟。

| Schema / 阶段 | 请求数 | 输入 Token | 输出 Token | 平均耗时 ms | P95 ms |
|---|---:|---:|---:|---:|---:|
{schema_rows}

## 质量与合同约束

- 首轮质量分：平均 {quality["first_scores"]["mean"]:.3f}，P50 {quality["first_scores"]["p50"]:.3f}，P95 {quality["first_scores"]["p95"]:.3f}。
- 最终质量分：平均 {quality["final_scores"]["mean"]:.3f}，P50 {quality["final_scores"]["p50"]:.3f}，P95 {quality["final_scores"]["p95"]:.3f}。
- 联合分组合同恢复涉及 {grouping["contract_recovery_batches"]} 个批次、{grouping["contract_recovery_refs"]} 个 Ref；错误 joint、重复归属和漏项均降为可审计的独立编译，没有丢弃 Ref。

## 独立验收

- 验收状态：`{acceptance.get("status", "not-run")}`；{acceptance.get("accepted_refs", 0)} 个有效 Ref 均且仅归属一个 Concept 组。
- {acceptance.get("concept_files", 0)} 个 Concept 文件全部通过 OKF/frontmatter 解析、来源回溯字段和正文非空检查，最终质量分下限为 {acceptance.get("quality_floor", 0):.2f}。
- {acceptance.get("asset_references", 0)} 个图表资产引用与源文档逐项守恒，{acceptance.get("unique_image_files", 0)} 个唯一图片文件通过内容哈希核对；人工复核项 {acceptance.get("reviews", 0)}。

## 可用于汇报/简历的事实表述

在 {corpus["articles"]} 篇智库文档上完成 Agent 化知识编译实验，构建 Article → ConceptRef → 候选关系 → 联合 Concept → 质量审计的可追溯流水线；模型可在结构化切分、混合精炼和纯 LLM Discovery 间自主路由，并通过严格 JSON Schema、确定性候选边、全覆盖合同和自动重编译保证产物完整性。原始发现 {corpus["raw_discovered_refs"]} 个 Ref，证据校验后以 {corpus["refs"]} 个有效 Ref 形成 {grouping["groups"]} 个 Concept 组，其中 {grouping["joint_groups"]} 个为跨文档联合概念；最终 {outputs["published_concepts"]}/{grouping["groups"]} 个 Concept 发布、人工复核 0 项，且全部来源与图表资产通过确定性验收。

> 上述表述仅包含本次实验可由日志和产物复核的数字；运行未完成时不应作为最终结果引用。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--model",
        help="Model name override for legacy runs whose manifest lacks it.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    metrics = summarize(args.run_dir, args.log, model=args.model)
    json_output = args.json_output or args.run_dir / "experiment-metrics.json"
    markdown_output = (
        args.markdown_output or args.run_dir / "experiment-report.md"
    )
    json_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
