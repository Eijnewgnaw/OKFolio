"""End-to-end conversion of MinerU output into AgentWiki runtime inputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import replace
from pathlib import Path

from .mineru import MinerUContentAdapter
from .models import Block, DocumentIR, ProcessingResult
from .segmenter import segment_document
from .storage import AssetWriter, asset_key, content_type
from .structure import normalize_document_structure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    ascii_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return ascii_slug[:80] or hashlib.sha256(value.encode()).hexdigest()[:16]


def _document_title(document: DocumentIR, fallback: str) -> str:
    """Prefer an early MinerU title block over an opaque source filename."""
    generic = {
        "目录",
        "前言",
        "序言",
        "编委会",
        "专题篇",
        "总报告",
        "分报告",
        "附录",
    }
    candidates: list[tuple[int, int, str]] = []
    for block in document.blocks:
        if block.page_idx > 12:
            break
        if block.block_type != "title":
            continue
        title = re.sub(r"\s+", " ", block.content).strip(" -—·")
        if (
            len(title) < 4
            or len(title) > 120
            or title in generic
            or re.fullmatch(r"[\dⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ一二三四五六七八九十]+", title)
        ):
            continue
        candidates.append((block.page_idx, -len(title), title))
    if not candidates:
        return fallback
    early_page = min(item[0] for item in candidates)
    same_page = [item for item in candidates if item[0] == early_page]
    return min(same_page, key=lambda item: item[1])[2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def render_article(document: DocumentIR) -> str:
    lines = [
        "---",
        f"title: {json.dumps(document.title, ensure_ascii=False)}",
        f"source_file: {json.dumps(document.source_file, ensure_ascii=False)}",
        f"document_id: {document.document_id}",
        f"source_sha256: {document.source_sha256}",
        f"page_count: {document.page_count}",
        f"parser: {document.parser}",
        "---",
        "",
        f"# {document.title}",
        "",
    ]
    for block in document.blocks:
        if block.block_type == "title":
            level = max(2, min(6, (block.heading_level or 1) + 1))
            lines.extend([f"{'#' * level} {block.content}", ""])
        elif block.block_type in {"image", "chart"} and block.asset_uri:
            lines.extend(
                [f"![{block.content or block.block_type}]({block.asset_uri})", ""]
            )
        elif block.block_type == "table":
            lines.extend([block.content, ""])
        elif block.content:
            lines.extend([block.content, ""])
    return "\n".join(lines).rstrip() + "\n"


def process_mineru_output(
    pdf_path: Path,
    mineru_output_dir: Path,
    destination: Path,
    *,
    asset_writer: AssetWriter,
    target_chars: int = 12_000,
    hard_max_chars: int = 24_000,
) -> ProcessingResult:
    pdf = pdf_path.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF input not found: {pdf}")
    source_sha256 = _sha256(pdf)
    document_id = f"article-{source_sha256[:20]}"
    title = pdf.stem.strip() or document_id
    content_list = MinerUContentAdapter.find_content_list(mineru_output_dir)
    document = MinerUContentAdapter().load(
        content_list,
        document_id=document_id,
        source_file=pdf.name,
        source_sha256=source_sha256,
        title=title,
    )
    document = replace(document, title=_document_title(document, title))
    metadata_path = mineru_output_dir.resolve() / "document-metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        page_count = metadata.get("page_count")
        parser_name = metadata.get("parser")
        if isinstance(page_count, int) and page_count >= document.page_count:
            document = replace(document, page_count=page_count)
        if isinstance(parser_name, str) and parser_name.strip():
            document = replace(document, parser=parser_name.strip())
    normalization = normalize_document_structure(document, mineru_output_dir)
    page_roles = normalization.page_by_index()
    assets: list[dict[str, object]] = []
    updated_blocks: list[Block] = []
    content_root = content_list.parent
    for block in document.blocks:
        if not block.asset_path:
            updated_blocks.append(block)
            continue
        page = page_roles.get(block.page_idx)
        if page is not None and page.asset_policy == "exclude":
            updated_blocks.append(replace(block, asset_path=None, asset_uri=None))
            continue
        source = (content_root / block.asset_path).resolve()
        source.relative_to(content_root.resolve())
        if not source.is_file():
            raise FileNotFoundError(
                f"MinerU asset does not exist: {block.asset_path}"
            )
        key = asset_key(document_id, source)
        uri = asset_writer.write(
            key,
            source.read_bytes(),
            content_type=content_type(source),
        )
        updated_blocks.append(replace(block, asset_uri=uri))
        assets.append(
            {
                "block_id": block.block_id,
                "source_path": block.asset_path,
                "uri": uri,
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
                "page_idx": block.page_idx,
                "page_number": block.page_idx + 1,
                "page_role": page.role if page is not None else "content",
                "asset_policy": (
                    page.asset_policy if page is not None else "knowledge"
                ),
                "evidence_eligible": (
                    page.evidence_eligible if page is not None else True
                ),
            }
        )
    document = replace(document, blocks=tuple(updated_blocks))
    block_by_id = {block.block_id: block for block in document.blocks}
    normalized_blocks = tuple(
        replace(
            block,
            asset_path=block_by_id[block.block_id].asset_path,
            asset_uri=block_by_id[block.block_id].asset_uri,
        )
        for block in normalization.document.blocks
    )
    normalized_document = replace(
        normalization.document,
        blocks=normalized_blocks,
        parser=document.parser,
        page_count=document.page_count,
    )
    normalization = replace(normalization, document=normalized_document)
    segments = segment_document(
        normalized_document,
        target_chars=target_chars,
        hard_max_chars=hard_max_chars,
    )
    output = destination.resolve()
    output.mkdir(parents=True, exist_ok=True)
    article_path = output / f"{_safe_slug(pdf.stem)}.md"
    raw_article_path = output / "raw-article.md"
    document_ir_path = output / "document-ir.json"
    normalized_document_ir_path = output / "normalized-document-ir.json"
    segments_path = output / "segments.json"
    structure_path = output / "document-structure.json"
    asset_manifest_path = output / "asset-manifest.json"
    manifest_path = output / "manifest.json"
    raw_article_path.write_text(render_article(document), encoding="utf-8")
    article_path.write_text(
        render_article(normalized_document),
        encoding="utf-8",
    )
    _write_json(document_ir_path, document.to_dict())
    _write_json(normalized_document_ir_path, normalized_document.to_dict())
    _write_json(structure_path, normalization.structure_manifest())
    _write_json(
        segments_path,
        {
            "schema_version": "kmpro.article-segments.v1",
            "document_id": document_id,
            "segments": [segment.to_dict() for segment in segments],
        },
    )
    _write_json(
        asset_manifest_path,
        {
            "schema_version": "kmpro.asset-manifest.v1",
            "document_id": document_id,
            "assets": assets,
        },
    )
    result = ProcessingResult(
        document_id=document_id,
        article_path=str(article_path),
        raw_article_path=str(raw_article_path),
        document_ir_path=str(document_ir_path),
        normalized_document_ir_path=str(normalized_document_ir_path),
        segments_path=str(segments_path),
        structure_path=str(structure_path),
        asset_manifest_path=str(asset_manifest_path),
        manifest_path=str(manifest_path),
        normalization_status=normalization.status,
        blocks=len(normalized_document.blocks),
        segments=len(segments),
        pages=document.page_count,
        assets=len(assets),
    )
    _write_json(
        manifest_path,
        {
            "schema_version": "kmpro.processing-run.v1",
            "status": "complete",
            **result.to_dict(),
        },
    )
    return result
