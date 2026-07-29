#!/usr/bin/env python3
"""Deterministic acceptance audit for a promoted Agent release."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kmpro_wiki.agentwiki.okf import parse_concept_markdown


LINK_RE = re.compile(r"\[[^\]]+\]\((?P<target>[^)#?]+\.md)\)")
GRAPH_DATA_RE = re.compile(
    r"const DATA=(?P<payload>\{.*?\});\s*const COLORS=",
    re.DOTALL,
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_release(release_dir: Path) -> dict[str, Any]:
    manifest = load(release_dir / "release-manifest.json")
    data_dir = release_dir / "data"
    wiki_dir = data_dir / "wiki"
    provenance = data_dir / "provenance"
    refs = load(provenance / "refs.json")["refs"]
    relations = load(provenance / "relations.json")
    acceptance = load(provenance / "acceptance.json")
    by_ref = {item["ref_id"]: item for item in refs}

    concept_paths = sorted((wiki_dir / "concepts").glob("*.md"))
    article_paths = sorted((wiki_dir / "articles").glob("*.md"))
    assert len(concept_paths) == manifest["concepts"]
    assert len(article_paths) == manifest["articles"]
    assigned: list[str] = []
    relation_links: set[tuple[str, str]] = set()
    image_occurrences: Counter[str] = Counter()
    table_count = 0
    ref_to_concept: dict[str, str] = {}
    for path in concept_paths:
        document = parse_concept_markdown(
            path.name,
            path.read_text(encoding="utf-8"),
        )
        refs_for_concept = list(document.frontmatter["concept_refs"])
        articles = list(document.frontmatter["articles"])
        assert refs_for_concept
        assert articles
        assert set(articles) == {
            by_ref[ref_id]["article_id"] for ref_id in refs_for_concept
        }
        assert float(document.frontmatter["agent_quality_score"]) >= 0.80
        assigned.extend(refs_for_concept)
        for ref_id in refs_for_concept:
            ref_to_concept[ref_id] = path.stem
        for match in LINK_RE.finditer(document.body):
            target = Path(match.group("target")).name
            if target.startswith("article-"):
                assert (wiki_dir / "articles" / target).is_file()
            else:
                assert (wiki_dir / "concepts" / target).is_file()
                relation_links.add(tuple(sorted((path.stem, Path(target).stem))))
        image_occurrences.update(
            match.group("target") for match in IMAGE_RE.finditer(document.body)
        )
        table_count += len(TABLE_RE.findall(document.body))
    assert len(assigned) == len(set(assigned)) == len(refs)
    assert set(assigned) == set(by_ref)

    expected_pairs = {
        tuple(
            sorted(
                (
                    ref_to_concept[item["left_ref_id"]],
                    ref_to_concept[item["right_ref_id"]],
                )
            )
        )
        for item in relations["judgements"]
        if item["decision"] == "related"
    }
    expected_pairs = {
        pair for pair in expected_pairs if pair[0] != pair[1]
    }
    assert relation_links == expected_pairs

    graph_path = data_dir / "outputs" / "graph.html"
    graph_match = GRAPH_DATA_RE.search(
        graph_path.read_text(encoding="utf-8")
    )
    assert graph_match is not None
    graph = json.loads(graph_match.group("payload"))
    assert graph["stats"]["articles"] == manifest["articles"]
    assert graph["stats"]["refs"] == manifest["refs"]
    assert graph["stats"]["concepts"] == manifest["concepts"]
    assert graph["stats"]["relations"] == len(expected_pairs)
    assert len(graph["semantic_edges"]) == len(expected_pairs)

    image_paths = [
        path
        for path in (wiki_dir / "images").rglob("*")
        if path.is_file()
    ]
    assert len(image_paths) == manifest["images"]
    assert len(image_occurrences) == acceptance["unique_image_files"]
    asset_kind_counts = acceptance.get("asset_kind_counts", {})
    assert sum(image_occurrences.values()) == asset_kind_counts.get("image", 0)
    assert table_count == asset_kind_counts.get("html_table", 0)
    remote_images = {
        target
        for target in image_occurrences
        if target.startswith(("http://", "https://", "s3://"))
    }
    local_images = {
        target.removeprefix("../images/")
        for target in image_occurrences
        if target.startswith("../images/")
    }
    assert len(local_images) == manifest["images"]
    assert len(remote_images) == manifest.get("remote_images", 0)
    assert len(local_images) + len(remote_images) == len(image_occurrences)
    for target in local_images:
        assert (wiki_dir / "images" / target).is_file()

    site_dir = data_dir / "outputs" / "site"
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "graph.html").is_file()
    assert len(list((site_dir / "concepts").glob("*.html"))) == manifest[
        "site_concept_pages"
    ]
    assert len(list((site_dir / "articles").glob("*.html"))) == manifest[
        "site_article_pages"
    ]

    checksum_lines = (
        release_dir / "MANIFEST.sha256"
    ).read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        path = release_dir / relative
        assert path.is_file()
        assert sha256(path) == expected
        checked += 1

    return {
        "status": "pass",
        "version": manifest["version"],
        "articles": len(article_paths),
        "refs": len(refs),
        "concepts": len(concept_paths),
        "joint_concepts": manifest["joint_concepts"],
        "semantic_relation_groups": len(expected_pairs),
        "semantic_relation_evidence": manifest[
            "semantic_relation_evidence"
        ],
        "images": len(image_paths),
        "remote_images": len(remote_images),
        "html_tables": table_count,
        "site_concept_pages": manifest["site_concept_pages"],
        "site_article_pages": manifest["site_article_pages"],
        "checksummed_files": checked,
        "quality_floor": acceptance["quality_floor"],
        "reviews": acceptance["reviews"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_release(args.release_dir)
    output = args.output or args.release_dir / "acceptance.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
