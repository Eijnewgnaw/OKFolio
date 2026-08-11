#!/usr/bin/env python3
"""Merge completed PDF shard manifests and activate all Article Markdown."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_complete(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "corpus-run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"PDF shard is not complete: {path}")
    if payload.get("activation_status") != "complete":
        raise ValueError(f"PDF shard is not activated for AgentWiki: {path}")
    return payload


def _sum_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        key
        for item in items
        for key, value in item.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return {
        key: sum(int(item.get(key) or 0) for item in items)
        for key in sorted(keys)
    }


def merge(primary: Path, secondary: Path) -> dict[str, Any]:
    primary = primary.resolve()
    secondary = secondary.resolve()
    manifests = [_load_complete(primary), _load_complete(secondary)]
    documents = [
        item
        for manifest in manifests
        for item in manifest.get("documents", [])
    ]
    hashes = [str(item["source_sha256"]) for item in documents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("PDF shard manifests overlap by source SHA-256")
    if any(
        item.get("status") != "complete"
        or item.get("activated_for_agentwiki") is not True
        for item in documents
    ):
        raise ValueError("combined corpus contains an incomplete activation")

    target_sources = primary / "normalized-sources"
    target_sources.mkdir(parents=True, exist_ok=True)
    secondary_sources = secondary / "normalized-sources"
    for source in sorted(
        path for path in secondary_sources.rglob("*") if path.is_file()
    ):
        relative = source.relative_to(secondary_sources)
        target = target_sources / relative
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"activation file collision: {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    source_files = sorted(target_sources.glob("*.md"))
    if len(source_files) != len(documents):
        raise ValueError(
            "activated Article count does not match completed PDF count"
        )

    shard_dir = primary / "corpus-shards"
    _write_json(shard_dir / "a.json", manifests[0])
    _write_json(shard_dir / "b.json", manifests[1])
    totals = _sum_totals([item["totals"] for item in manifests])
    result = {
        "schema_version": "okfolio.pdf-corpus-combined.v1",
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "shards": 2,
        "documents": documents,
        "totals": totals,
        "activated_articles": len(source_files),
    }
    _write_json(primary / "corpus-run-combined.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-data", type=Path, required=True)
    parser.add_argument("--secondary-data", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.primary_data, args.secondary_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
