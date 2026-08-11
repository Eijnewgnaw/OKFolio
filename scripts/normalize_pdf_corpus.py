#!/usr/bin/env python3
"""Replay structural normalization over completed PDF parser outputs.

This command performs no model calls and never edits the raw DocumentIR.  A
source is activated for AgentWiki only when every page role is resolved.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from okfolio.data_processing.activation import (
    activate_article,
)
from okfolio.data_processing.pipeline import render_article
from okfolio.data_processing.segmenter import segment_document
from okfolio.data_processing.structure import (
    document_from_dict,
    normalize_document_structure,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _article_path(directory: Path, manifest: dict[str, Any]) -> Path:
    raw_article = directory / "raw-article.md"
    if raw_article.is_file():
        return raw_article
    configured = Path(str(manifest.get("article_path") or ""))
    if configured.name:
        candidate = directory / configured.name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in directory.glob("*.md")
        if path.name not in {"raw-article.md", "normalized-article.md"}
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"cannot identify one Article Markdown under {directory}"
        )
    return candidates[0]


def _annotate_assets(
    path: Path,
    *,
    page_roles: dict[int, Any],
    block_pages: dict[str, int],
) -> None:
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assets = value.get("assets", [])
    for item in assets:
        if not isinstance(item, dict):
            continue
        page_idx = block_pages.get(str(item.get("block_id") or ""), -1)
        page = page_roles.get(page_idx)
        item.update(
            {
                "page_idx": page_idx,
                "page_number": page_idx + 1 if page_idx >= 0 else None,
                "page_role": page.role if page is not None else "content",
                "asset_policy": (
                    page.asset_policy if page is not None else "knowledge"
                ),
                "evidence_eligible": (
                    page.evidence_eligible if page is not None else True
                ),
            }
        )
    _write_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize completed PDF Articles before Agent discovery"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/app/runtime/data")),
    )
    parser.add_argument("--target-chars", type=int, default=12_000)
    parser.add_argument("--hard-max-chars", type=int, default=24_000)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    processed_dir = data_dir / "processed"
    jobs_dir = data_dir / "parser-jobs"
    sources_dir = data_dir / "normalized-sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    legacy_images = data_dir / "sources" / "images"
    if legacy_images.is_dir():
        shutil.copytree(
            legacy_images,
            sources_dir / "images",
            dirs_exist_ok=True,
        )
    summaries: list[dict[str, Any]] = []
    for directory in sorted(
        path for path in processed_dir.iterdir() if path.is_dir()
    ):
        manifest_path = directory / "manifest.json"
        document_path = directory / "document-ir.json"
        if not manifest_path.is_file() or not document_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        document = document_from_dict(
            json.loads(document_path.read_text(encoding="utf-8"))
        )
        job_dir = jobs_dir / directory.name
        normalization = normalize_document_structure(document, job_dir)
        normalized = normalization.document
        segments = segment_document(
            normalized,
            target_chars=args.target_chars,
            hard_max_chars=args.hard_max_chars,
        )
        article_path = _article_path(directory, manifest)
        source_article_name = str(
            manifest.get("source_article_name") or article_path.name
        )
        raw_article = directory / "raw-article.md"
        if article_path != raw_article and not raw_article.exists():
            shutil.copy2(article_path, raw_article)
        normalized_path = directory / "normalized-article.md"
        normalized_path.write_text(render_article(normalized), encoding="utf-8")
        normalized_ir_path = directory / "normalized-document-ir.json"
        structure_path = directory / "document-structure.json"
        segments_path = directory / "segments.json"
        _write_json(normalized_ir_path, normalized.to_dict())
        _write_json(structure_path, normalization.structure_manifest())
        _write_json(
            segments_path,
            {
                "schema_version": "okfolio.article-segments.v1",
                "document_id": normalized.document_id,
                "segments": [item.to_dict() for item in segments],
            },
        )
        page_roles = normalization.page_by_index()
        _annotate_assets(
            directory / "asset-manifest.json",
            page_roles=page_roles,
            block_pages={
                block.block_id: block.page_idx for block in document.blocks
            },
        )
        activated = activate_article(
            normalized_path,
            structure_path,
            sources_dir,
            ready=normalization.status == "complete",
            article_name=source_article_name,
        )
        manifest.update(
            {
                "article_path": str(normalized_path),
                "raw_article_path": str(raw_article),
                "normalized_document_ir_path": str(normalized_ir_path),
                "structure_path": str(structure_path),
                "segments_path": str(segments_path),
                "normalization_status": normalization.status,
                "source_article_name": source_article_name,
                "blocks": len(normalized.blocks),
                "segments": len(segments),
                "activated_for_agentwiki": activated,
            }
        )
        _write_json(manifest_path, manifest)
        summaries.append(
            {
                "document_id": document.document_id,
                "source_article_name": source_article_name,
                "status": normalization.status,
                "pages": document.page_count,
                "outline_entries": len(normalization.outline),
                "toc_entries": len(normalization.toc_entries),
                "excluded_blocks": len(normalization.excluded_block_ids),
                "unresolved_pages": [
                    page.page_number
                    for page in normalization.pages
                    if page.needs_review
                ],
                "activated": activated,
            }
        )

    report = {
        "schema_version": "okfolio.corpus-normalization.v1",
        "status": (
            "complete"
            if summaries and all(item["status"] == "complete" for item in summaries)
            else "needs_review"
        ),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "documents": summaries,
    }
    _write_json(data_dir / "normalization-report.json", report)
    corpus_path = data_dir / "corpus-run.json"
    if corpus_path.is_file():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        by_document_id = {
            item["document_id"]: item for item in summaries
        }
        for item in corpus.get("documents", []):
            result = item.get("result") or {}
            summary = by_document_id.get(result.get("document_id"))
            if summary is None:
                continue
            item["activated_for_agentwiki"] = summary["activated"]
            result["normalization_status"] = summary["status"]
        documents = corpus.get("documents", [])
        activated_documents = sum(
            item.get("activated_for_agentwiki") is True
            for item in documents
        )
        totals = corpus.setdefault("totals", {})
        totals["activated_documents"] = activated_documents
        totals["needs_review_documents"] = sum(
            item.get("status") == "complete"
            and item.get("activated_for_agentwiki") is not True
            for item in documents
        )
        corpus["activation_status"] = (
            "complete"
            if documents and activated_documents == len(documents)
            else "needs_review"
        )
        corpus["updated_at"] = report.get("finished_at")
        _write_json(corpus_path, corpus)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
