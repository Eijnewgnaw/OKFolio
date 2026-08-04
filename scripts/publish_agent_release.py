#!/usr/bin/env python3
"""Promote an accepted Agent run into a formal Bundle and graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kmpro_wiki.agentwiki.site import build_site

import yaml

from kmpro_wiki.agentwiki.assets import strip_missing_image_references
from kmpro_wiki.agentwiki.indexer import build_index
from kmpro_wiki.agentwiki.okf import (
    ConceptDocument,
    parse_concept_markdown,
    rewrite_image_paths,
)
from kmpro_wiki.agentwiki.explorer import build_explorer_html
from kmpro_wiki.agentwiki.spatial_graph import build_graph_data, build_spatial_graph


TYPE_KIND = {
    "数据口径": "metric",
    "分析框架": "topic",
    "政策建议": "proposition",
    "国际比较": "comparison",
    "术语解释": "entity",
}

RELATION_LABELS = {
    "defines": "定义/口径",
    "supports": "证据支撑",
    "constrains": "约束条件",
    "causes": "因果影响",
    "recommends": "问题到建议",
    "compares": "比较对标",
    "extends": "补充展开",
    "related": "实质关联",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ref_location(ref: dict[str, Any]) -> str:
    parts = [str(item) for item in ref.get("section_path") or () if str(item)]
    location = " › ".join(parts)
    start = ref.get("page_start")
    end = ref.get("page_end")
    if isinstance(start, int):
        pages = f"第 {start} 页" if end in {None, start} else f"第 {start}–{end} 页"
        location = f"{location} · {pages}" if location else pages
    return location


def _article_page(
    article: dict[str, Any],
    refs: list[dict[str, Any]],
    ref_to_group: dict[str, str],
    groups: dict[str, dict[str, Any]],
    content: str,
) -> str:
    metadata = yaml.safe_dump(
        {
            "title": article["title"],
            "source": article["source"],
            "article_id": article["article_id"],
            "concept_refs": [item["ref_id"] for item in refs],
        },
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    links = []
    seen: set[str] = set()
    for ref in refs:
        group_id = ref_to_group[ref["ref_id"]]
        if group_id in seen:
            continue
        seen.add(group_id)
        links.append(
            f"- [{groups[group_id]['title']}](../concepts/{group_id}.md)"
        )
    ref_lines = []
    for item in refs:
        location = _ref_location(item)
        suffix = f" — {location}" if location else ""
        ref_lines.append(
            f"- `{item['ref_id']}`：{item['title']}{suffix}"
        )
    return (
        f"---\n{metadata}\n---\n\n"
        f"# {article['title']}\n\n"
        "## 关联 Concept\n\n"
        f"{chr(10).join(links)}\n\n"
        "## ConceptRef\n\n"
        f"{chr(10).join(ref_lines)}\n\n"
        "## 原文\n\n"
        f"{content.strip()}\n"
    )


def _relation_pairs(
    judgements: list[dict[str, Any]],
    ref_to_group: dict[str, str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in judgements:
        if item["decision"] != "related":
            continue
        left = ref_to_group[item["left_ref_id"]]
        right = ref_to_group[item["right_ref_id"]]
        if left == right:
            continue
        pairs[tuple(sorted((left, right)))].append(item)
    return dict(pairs)


def _published_concept(
    document: ConceptDocument,
    *,
    concept_id: str,
    articles: list[str],
    refs: list[str],
    refs_by_id: dict[str, dict[str, Any]],
    article_titles: dict[str, str],
    related: dict[str, list[dict[str, Any]]],
    concept_titles: dict[str, str],
) -> str:
    body = document.body.rstrip()
    if related:
        lines = []
        for target in sorted(
            related,
            key=lambda item: (concept_titles[item], item),
        ):
            reasons = list(
                dict.fromkeys(item["reason"] for item in related[target])
            )
            relation_types = list(
                dict.fromkeys(
                    RELATION_LABELS.get(
                        str(item.get("relation_type") or "related"),
                        str(item.get("relation_type") or "related"),
                    )
                    for item in related[target]
                )
            )
            evidence_refs = list(
                dict.fromkeys(
                    ref_id
                    for item in related[target]
                    for ref_id in item.get("evidence_ref_ids", [])
                    if isinstance(ref_id, str) and ref_id
                )
            )
            evidence_suffix = (
                f"（依据 Ref：{', '.join(f'`{ref_id}`' for ref_id in evidence_refs)}）"
                if evidence_refs
                else ""
            )
            lines.append(
                f"- [{concept_titles[target]}](../concepts/{target}.md)"
                f" — {' / '.join(relation_types)}：{'；'.join(reasons)}"
                f"{evidence_suffix}"
            )
        body += "\n\n## 关联概念\n\n" + "\n".join(lines)
    source_lines = [
        f"- [Article：{article_titles[article_id]}]"
        f"(../articles/{article_id}.md)"
        for article_id in articles
    ]
    ref_lines = []
    for ref_id in refs:
        location = _ref_location(refs_by_id[ref_id])
        suffix = f" — {location}" if location else ""
        ref_lines.append(f"- `{ref_id}`{suffix}")
    body += (
        "\n\n## 溯源\n\n"
        + "\n".join(source_lines)
        + "\n\n## ConceptRef\n\n"
        + "\n".join(ref_lines)
    )
    frontmatter = dict(document.frontmatter)
    frontmatter["concept_refs"] = refs
    frontmatter["articles"] = articles
    frontmatter["relation_count"] = len(related)
    return ConceptDocument(
        filename=f"{concept_id}.md",
        frontmatter=frontmatter,
        body=body,
    ).render()


def _write_checksums(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_release(
    run_dir: Path,
    sources_dir: Path,
    release_dir: Path,
    *,
    version: str,
    replace: bool = False,
) -> dict[str, Any]:
    manifest = load(run_dir / "manifest.json")
    acceptance = load(run_dir / "acceptance.json")
    relations = load(run_dir / "relations.json")
    if manifest.get("status") != "complete":
        raise ValueError("Agent run is not complete")
    if acceptance.get("status") != "pass":
        raise ValueError("Agent run has not passed independent acceptance")
    if relations.get("status") != "complete":
        raise ValueError("relation judging is not complete")
    if release_dir.exists():
        if not replace:
            raise ValueError(f"release already exists: {release_dir}")
        shutil.rmtree(release_dir)

    wiki_dir = release_dir / "data" / "wiki"
    concepts_dir = wiki_dir / "concepts"
    articles_dir = wiki_dir / "articles"
    images_dir = wiki_dir / "images"
    outputs_dir = release_dir / "data" / "outputs"
    provenance_dir = release_dir / "data" / "provenance"
    for directory in (
        concepts_dir,
        articles_dir,
        images_dir,
        outputs_dir,
        provenance_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    refs = load(run_dir / "refs.json")["refs"]
    refs_by_id = {item["ref_id"]: item for item in refs}
    groups = load(run_dir / "groups.json")["groups"]
    groups_by_id = {item["group_id"]: item for item in groups}
    ref_to_group = {
        ref_id: group["group_id"]
        for group in groups
        for ref_id in group["ref_ids"]
    }
    source_progress = load(run_dir / "source_progress.json")["sources"]
    articles = [
        {
            "article_id": item["refs"][0]["article_id"],
            "source": item["source"],
            "title": item["profile"]["title"],
            "ref_count": len(item["refs"]),
            "asset_count": int(item["profile"]["asset_count"]),
        }
        for item in source_progress
    ]
    article_titles = {
        item["article_id"]: item["title"] for item in articles
    }
    refs_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_article[ref["article_id"]].append(ref)

    pairs = _relation_pairs(relations["judgements"], ref_to_group)
    related_by_concept: dict[str, dict[str, list[dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for (left, right), items in pairs.items():
        related_by_concept[left][right] = items
        related_by_concept[right][left] = items
    concept_titles = {
        group["group_id"]: group["title"] for group in groups
    }

    graph_concepts: list[dict[str, Any]] = []
    for group in groups:
        concept_id = group["group_id"]
        source_path = run_dir / "concepts" / f"{concept_id}.md"
        document = parse_concept_markdown(
            source_path.name,
            source_path.read_text(encoding="utf-8"),
        )
        article_ids = sorted(
            {
                refs_by_id[ref_id]["article_id"]
                for ref_id in group["ref_ids"]
            }
        )
        content = _published_concept(
            document,
            concept_id=concept_id,
            articles=article_ids,
            refs=group["ref_ids"],
            refs_by_id=refs_by_id,
            article_titles=article_titles,
            related=related_by_concept.get(concept_id, {}),
            concept_titles=concept_titles,
        )
        (concepts_dir / f"{concept_id}.md").write_text(
            content,
            encoding="utf-8",
        )
        published = parse_concept_markdown(f"{concept_id}.md", content)
        graph_concepts.append(
            {
                "id": concept_id,
                "title": published.frontmatter["title"],
                "type": published.frontmatter["type"],
                "kind": TYPE_KIND[published.frontmatter["type"]],
                "description": published.frontmatter["description"],
                "body": published.body,
                "articles": article_ids,
                "ref_ids": group["ref_ids"],
            }
        )

    for article in articles:
        source_path = sources_dir / article["source"]
        source_content = strip_missing_image_references(
            source_path.read_text(encoding="utf-8"),
            sources_dir / "images",
        )
        source_content = rewrite_image_paths(source_content)
        page = _article_page(
            article,
            refs_by_article[article["article_id"]],
            ref_to_group,
            groups_by_id,
            source_content,
        )
        (articles_dir / f"{article['article_id']}.md").write_text(
            page,
            encoding="utf-8",
        )

    for image in sorted((run_dir / "images").rglob("*")):
        if not image.is_file():
            continue
        target = images_dir / image.relative_to(run_dir / "images")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)

    graph_data = build_graph_data(
        articles,
        refs,
        graph_concepts,
        relations["judgements"],
    )
    graph_html = build_spatial_graph(
        articles,
        refs,
        graph_concepts,
        relations["judgements"],
    )
    (outputs_dir / "graph.html").write_text(graph_html, encoding="utf-8")
    (wiki_dir / "graph.html").write_text(graph_html, encoding="utf-8")
    explorer_html = build_explorer_html(
        graph_data,
        title="OKFolio Knowledge Explorer",
        scope_note=(
            f"Published Bundle · {len(articles)} Article · {len(refs)} Ref · "
            f"{len(groups)} Concept"
        ),
    )
    (outputs_dir / "explore.html").write_text(explorer_html, encoding="utf-8")
    (wiki_dir / "explore.html").write_text(explorer_html, encoding="utf-8")

    base_index = build_index(concepts_dir)
    base_index = re.sub(
        r"^# Knowledge Base Index",
        "# 决策参考知识库",
        base_index,
        count=1,
    )
    intro = (
        f"\n\n> {len(articles)} 篇 Article · {len(refs)} 个 ConceptRef · "
        f"{len(groups)} 个 Concept · {len(pairs)} 组语义关系\n\n"
        "[打开知识探索台](explore.html) · [打开三维知识图谱](graph.html)\n"
    )
    wiki_dir.joinpath("index.md").write_text(
        base_index.split("\n", 1)[0]
        + intro
        + "\n"
        + base_index.split("\n", 1)[1],
        encoding="utf-8",
    )
    source_lines = "\n".join(
        f"- `{item['article_id']}`：{item['title']}" for item in articles
    )
    wiki_dir.joinpath("log.md").write_text(
        "# Knowledge Base Update Log\n\n"
        f"## {datetime.now(UTC).date().isoformat()}\n\n"
        f"- **Release**: `{version}`\n"
        f"- **Model**: `{manifest.get('model', 'unknown')}`\n"
        f"- **Result**: {len(articles)} Article / {len(refs)} Ref / "
        f"{len(groups)} Concept / {len(pairs)} relation groups\n\n"
        "## Article 清单\n\n"
        f"{source_lines}\n",
        encoding="utf-8",
    )

    build_site(wiki_dir, outputs_dir / "site")
    for name in (
        "manifest.json",
        "refs.json",
        "candidates.json",
        "groups.json",
        "concepts.json",
        "relations.json",
        "relation-metrics.json",
        "ref_validation.json",
        "source_progress.json",
        "compile_progress.json",
        "asset_progress.json",
        "agent_trace.json",
        "review_queue.json",
        "acceptance.json",
        "experiment-metrics.json",
        "experiment-report.md",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, provenance_dir / name)

    release_manifest = {
        "version": version,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": run_dir.name,
        "model": manifest.get("model", "unknown"),
        "status": "complete",
        "articles": len(articles),
        "refs": len(refs),
        "concepts": len(groups),
        "joint_concepts": sum(
            len(group["ref_ids"]) > 1 for group in groups
        ),
        "candidate_edges": relations["candidate_edges"],
        "judged_cross_concept_edges": relations["cross_concept_edges"],
        "semantic_relation_groups": len(pairs),
        "semantic_relation_evidence": sum(len(items) for items in pairs.values()),
        "images": sum(path.is_file() for path in images_dir.rglob("*")),
        "remote_images": int(acceptance.get("remote_image_assets", 0)),
        "site_concept_pages": sum(
            path.is_file()
            for path in (outputs_dir / "site" / "concepts").glob("*.html")
        ),
        "site_article_pages": sum(
            path.is_file()
            for path in (outputs_dir / "site" / "articles").glob("*.html")
        ),
        "spatial_graph": True,
        "provenance": "Concept -> ConceptRef -> Article",
    }
    write_json(release_dir / "release-manifest.json", release_manifest)
    _write_checksums(release_dir)
    return release_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sources-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument(
        "--version",
        default=(
            Path(__file__).resolve().parents[1] / "VERSION"
        ).read_text(encoding="utf-8").strip(),
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = publish_release(
        args.run_dir,
        args.sources_dir,
        args.release_dir,
        version=args.version,
        replace=args.replace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
