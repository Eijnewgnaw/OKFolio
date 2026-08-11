#!/usr/bin/env python3
"""Deterministic acceptance audit for a completed Agent compiler run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path, PurePosixPath
from typing import Any

from okfolio.agentwiki.assets import (
    inventory_assets,
    strip_missing_image_references,
)
from okfolio.agentwiki.okf import parse_concept_markdown


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connected(ref_ids: list[str], pairs: set[tuple[str, str]]) -> bool:
    allowed = set(ref_ids)
    adjacency = {ref_id: set() for ref_id in ref_ids}
    for left, right in pairs:
        if left in allowed and right in allowed:
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached: set[str] = set()
    queue = deque([ref_ids[0]])
    while queue:
        current = queue.popleft()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(adjacency[current] - reached)
    return reached == allowed


def run_audit(run_dir: Path, sources_dir: Path) -> dict[str, Any]:
    manifest = load(run_dir / "manifest.json")
    validation = load(run_dir / "ref_validation.json")
    refs_payload = load(run_dir / "refs.json")
    candidates = load(run_dir / "candidates.json")
    groups_payload = load(run_dir / "groups.json")
    concepts_payload = load(run_dir / "concepts.json")
    review_queue = load(run_dir / "review_queue.json")
    asset_progress = load(run_dir / "asset_progress.json")
    source_progress = load(run_dir / "source_progress.json")

    assert manifest["status"] == "complete"
    refs = refs_payload["refs"]
    by_ref = {item["ref_id"]: item for item in refs}
    assert len(by_ref) == len(refs) == validation["accepted_refs"]
    assert manifest["articles"] == len(source_progress["sources"]) > 0
    quality_threshold = float(manifest["policy"]["quality_threshold"])

    groups = groups_payload["groups"]
    by_group = {item["group_id"]: item for item in groups}
    assert len(by_group) == len(groups) == manifest["groups"]
    assigned = [
        ref_id for group in groups for ref_id in group["ref_ids"]
    ]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(by_ref)

    pairs = {
        tuple(
            sorted((edge["left_ref_id"], edge["right_ref_id"]))
        )
        for edge in candidates["edges"]
    }
    joint_groups = 0
    for group in groups:
        ref_ids = group["ref_ids"]
        if len(ref_ids) == 1:
            continue
        joint_groups += 1
        assert len({by_ref[ref_id]["type"] for ref_id in ref_ids}) == 1
        assert len({by_ref[ref_id]["article_id"] for ref_id in ref_ids}) >= 2
        assert connected(ref_ids, pairs)

    concepts = concepts_payload["concepts"]
    assert len(concepts) == manifest["concepts"] == len(groups)
    assert all(item["status"] == "publishable" for item in concepts)
    assert review_queue["reviews"] == []
    assert not list((run_dir / "drafts").glob("*.md"))

    concept_paths = sorted((run_dir / "concepts").glob("*.md"))
    assert len(concept_paths) == len(groups)
    assert {path.stem for path in concept_paths} == set(by_group)
    rendered: dict[str, str] = {}
    for path in concept_paths:
        content = path.read_text(encoding="utf-8")
        concept = parse_concept_markdown(path.name, content)
        group = by_group[path.stem]
        assert concept.frontmatter["concept_refs"] == group["ref_ids"]
        assert set(concept.frontmatter["articles"]) == {
            by_ref[ref_id]["article_id"] for ref_id in group["ref_ids"]
        }
        assert (
            float(concept.frontmatter["agent_quality_score"])
            >= quality_threshold
        )
        for field in ("type", "title", "description", "source"):
            assert isinstance(concept.frontmatter.get(field), str)
            assert concept.frontmatter[field].strip()
        assert concept.body.strip()
        rendered[path.stem] = content.replace("../images/", "images/")

    source_images = sources_dir / "images"
    asset_counter: Counter[str] = Counter()
    asset_kind_counts: Counter[str] = Counter()
    local_image_targets: set[str] = set()
    remote_image_targets: set[str] = set()
    asset_bearing_sources = 0
    for item in source_progress["sources"]:
        path = sources_dir / item["source"]
        normalized = strip_missing_image_references(
            path.read_text(encoding="utf-8"), source_images
        )
        assets = inventory_assets(normalized, source_images)
        assert len(assets) == int(item["profile"]["asset_count"])
        if assets:
            asset_bearing_sources += 1
        for asset in assets:
            asset_counter[asset.raw] += 1
            asset_kind_counts[asset.kind] += 1
            if asset.target:
                if asset.target.startswith("images/"):
                    local_image_targets.add(asset.target)
                else:
                    remote_image_targets.add(asset.target)

    combined = "\n".join(rendered.values())
    for raw, expected in asset_counter.items():
        assert combined.count(raw) == expected, (
            raw,
            expected,
            combined.count(raw),
        )

    output_images = run_dir / "images"
    for target in local_image_targets:
        relative = PurePosixPath(target).relative_to("images")
        source_path = source_images.joinpath(*relative.parts)
        output_path = output_images.joinpath(*relative.parts)
        assert output_path.is_file()
        assert sha256(output_path) == sha256(source_path)

    assert len(asset_progress["processed_sources"]) == asset_bearing_sources
    assert asset_progress["reviews"] == []
    assert asset_progress["withheld"] == []

    return {
        "status": "pass",
        "articles": len(source_progress["sources"]),
        "raw_refs": validation["raw_refs"],
        "accepted_refs": len(refs),
        "rejected_refs": len(validation["rejected_refs"]),
        "candidate_edges": len(candidates["edges"]),
        "groups": len(groups),
        "joint_groups": joint_groups,
        "concept_files": len(concept_paths),
        "quality_floor": min(
            float(
                parse_concept_markdown(
                    path.name, path.read_text(encoding="utf-8")
                ).frontmatter["agent_quality_score"]
            )
            for path in concept_paths
        ),
        "asset_bearing_sources": asset_bearing_sources,
        "asset_references": sum(asset_counter.values()),
        "asset_kind_counts": dict(asset_kind_counts),
        "unique_image_files": (
            len(local_image_targets) + len(remote_image_targets)
        ),
        "local_image_files": len(local_image_targets),
        "remote_image_assets": len(remote_image_targets),
        "reviews": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sources-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "acceptance.json"
    try:
        result = run_audit(args.run_dir, args.sources_dir)
    except Exception as error:
        result = {
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
        }
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
