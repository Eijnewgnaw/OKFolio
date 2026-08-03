#!/usr/bin/env python3
"""Run a deterministic, data-free four-stage OKFolio experiment.

This is a local contract/integration probe, not a model-quality benchmark. It
keeps the real discovery, candidate, cluster, provenance graph, and update
classification functions in the loop while using nine small synthetic
Articles so the public repository can reproduce it without private data or a
provider key.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmpro_wiki.agentwiki.contracts import ConceptRef
from kmpro_wiki.agentwiki.global_cluster import build_clusters, candidate_edges
from kmpro_wiki.agentwiki.spatial_graph import build_graph_data, build_spatial_graph
from kmpro_wiki.agentwiki.stages import build_evidence_catalog
from kmpro_wiki.agentwiki.versioning import reconcile_refs, semantic_key
from kmpro_wiki.agentwiki.agentic import discover_from_headings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local-experiment"),
        help="Ignored local output directory.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "r0" / "concepts").mkdir(parents=True)
    (output / "r1").mkdir(parents=True)

    articles: list[dict[str, str]] = []
    refs: list[dict[str, object]] = []
    for index in range(1, 10):
        source = f"sample-{index:02d}.md"
        year = 2024 + (index % 2)
        geography = "成都市" if index % 3 else "四川省"
        content = _article(source, year, geography)
        article_path = output / "r0" / source
        article_path.write_text(content, encoding="utf-8")
        discovered = discover_from_headings(source, content, ())
        article_id = f"article-{index:02d}"
        articles.append(
            {
                "article_id": article_id,
                "title": f"区域经济样例报告 {index:02d}",
                "source": source,
                "document_family_id": f"sample-family-{index:02d}",
                "document_version_id": f"sample-v0-{index:02d}",
                "toc_entries": [item.title for item in discovered],
            }
        )
        for position, item in enumerate(discovered, start=1):
            family = (
                "financing-demand-index"
                if position == 1
                else "regional-coordination-policy"
            )
            ref = replace(
                item,
                concept_id=f"ref-{index:02d}-{position}",
                semantic_signature={"key": family},
                ref_family_hint=family,
                scope={"time": f"{year}年", "geography": geography, "object": "制造业"},
                ref_version_id=f"sample-v0-{index:02d}",
                document_family_id=f"sample-family-{index:02d}",
                document_version_id=f"sample-v0-{index:02d}",
            )
            refs.append(_ref_payload(ref, article_id))

    candidates, states = candidate_edges(refs, top_k=24, minimum_score=0.01)
    judgements = []
    for edge in candidates:
        left = _by_id(refs, edge.left_ref_id)
        right = _by_id(refs, edge.right_ref_id)
        same_slot = semantic_key(left) == semantic_key(right)
        decision = "same" if same_slot else ("related" if edge.signals["lexical"] >= 0.02 else "separate")
        judgements.append(
            {
                "edge_id": edge.edge_id,
                "left_ref_id": edge.left_ref_id,
                "right_ref_id": edge.right_ref_id,
                "decision": decision,
                "reason": "同一语义槽位" if same_slot else "跨主题但存在可追溯关联" if decision == "related" else "仅作候选排除",
            }
        )
    clusters = build_clusters(refs, judgements)
    concepts = [_concept_payload(cluster, refs) for cluster in clusters]
    graph = build_graph_data(articles, refs, concepts, judgements)
    (output / "r0" / "graph.html").write_text(build_spatial_graph(articles, refs, concepts, judgements), encoding="utf-8")
    (output / "r0" / "graph-data.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "r0" / "candidates.json").write_text(json.dumps({"edges": [edge.as_dict() for edge in candidates], "states": states}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "r0" / "judgements.json").write_text(json.dumps(judgements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for concept in concepts:
        _write_concept(output / "r0" / "concepts" / f"{concept['id']}.md", concept, refs)
    (output / "r0" / "bundle.json").write_text(json.dumps({"schema": "okfolio.bundle.v1", "articles": articles, "refs": refs, "concepts": concepts, "graph_stats": graph["stats"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # R1 is deliberately scoped to one changed Article, rather than treating
    # a corpus refresh as if every historical Ref belonged to that document.
    old_refs = [dict(item) for item in refs if item["article_id"] == "article-01"]
    new_refs = [dict(item) for item in old_refs if item["ref_id"] != "ref-01-2"]
    changed = next(item for item in new_refs if item["ref_id"] == "ref-01-1")
    changed["ref_id"] = "ref-01-1-v1"
    changed["evidence"] = ["补贴标准由30%调整为50%。"]
    changed["scope"] = {"time": "2026年", "geography": "成都市", "object": "制造业"}
    changed["document_version_id"] = "sample-v1-01"
    new_refs.append(
        {
            **old_refs[1],
            "ref_id": "ref-01-new",
            "ref_family_hint": "new-digital-indicator",
            "semantic_signature": {"key": "new-digital-indicator"},
            "title": "数字产业新指标",
            "description": "新增一个可独立引用的指标。",
            "evidence": ["新增数字产业指标。"],
        }
    )
    update = reconcile_refs(old_refs, new_refs)
    (output / "r1" / "reconciliation.json").write_text(json.dumps(update, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = _report(graph, candidates, clusters, update)
    (output / "EXPERIMENT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"output={output}")
    return 0


def _article(source: str, year: int, geography: str) -> str:
    return f"""---
title: 区域经济样例报告 {source[7:9]}
source_file: {source}
document_family_id: sample-family-{source[7:9]}
document_version_id: sample-v0-{source[7:9]}
published_at: {year}-06-01
geography: {geography}
---

# 区域经济样例报告

## 制造业融资需求指数

{year}年{geography}制造业融资需求指数用于观察企业融资需求变化，补贴标准为30%。

## 区域协同政策机制

{year}年{geography}建立跨区域协同机制，明确产业链协作和责任主体。
"""


def _ref_payload(ref: ConceptRef, article_id: str) -> dict[str, object]:
    payload = asdict(ref)
    payload["ref_id"] = payload.pop("concept_id")
    payload["article_id"] = article_id
    payload["evidence"] = list(payload["evidence"])
    payload["asset_hints"] = list(payload["asset_hints"])
    payload["section_path"] = list(payload["section_path"])
    payload["evidence_block_ids"] = list(payload["evidence_block_ids"])
    return payload


def _by_id(refs: list[dict[str, object]], ref_id: str) -> dict[str, object]:
    return next(item for item in refs if item["ref_id"] == ref_id)


def _concept_payload(cluster: dict[str, object], refs: list[dict[str, object]]) -> dict[str, object]:
    members = [_by_id(refs, ref_id) for ref_id in cluster["ref_ids"]]
    return {
        "id": cluster["id"],
        "type": cluster["type"],
        "title": cluster["title"],
        "description": cluster["description"],
        "ref_ids": list(cluster["ref_ids"]),
        "articles": sorted({str(item["article_id"]) for item in members}),
        "body": "\n\n".join(str(item["evidence"][0]) for item in members),
    }


def _write_concept(path: Path, concept: dict[str, object], refs: list[dict[str, object]]) -> None:
    lines = [
        "---",
        f"type: {concept['type']}",
        f"title: {concept['title']}",
        f"description: {concept['description']}",
        f"concept_refs: {json.dumps(concept['ref_ids'], ensure_ascii=False)}",
        f"articles: {json.dumps(concept['articles'], ensure_ascii=False)}",
        "---",
        "",
        str(concept["body"]),
        "",
        "## Evidence trail",
        "",
    ]
    for ref_id in concept["ref_ids"]:
        ref = _by_id(refs, ref_id)
        lines.append(f"- `{ref_id}` · {ref['source']} · {ref['scope']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(graph: dict[str, object], candidates: list[object], clusters: list[dict[str, object]], update: dict[str, object]) -> str:
    counts = update["counts"]
    return """# Local four-stage experiment

This run uses nine synthetic, structured Articles to exercise the public
pipeline without private documents, model weights, endpoints, or API keys.

| Stage | Result |
|---|---:|
| R0 Articles | 9 |
| R0 ConceptRefs | 18 |
| Candidate edges | {candidates} |
| Concepts / clusters | {clusters} |
| Cross-document Concepts | {multi_source} |
| Graph relation groups | {relations} |
| R1 update statuses | {counts} |

The update experiment changes one Ref to a new year, omits one old Ref, and
adds one new semantic slot. Old records remain in history; the current view
uses the new snapshot. The expected temporal change is therefore classified
as `temporal_variant`, not as a destructive global overwrite.

Outputs: `r0/bundle.json`, `r0/graph.html`, `r0/concepts/`, and
`r1/reconciliation.json`.
""".format(
        candidates=len(candidates),
        clusters=len(clusters),
        multi_source=graph["stats"]["multi_source"],
        relations=graph["stats"]["relations"],
        counts=json.dumps(counts, ensure_ascii=False, sort_keys=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
