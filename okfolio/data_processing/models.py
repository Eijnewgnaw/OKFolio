"""Stable internal representation between PDF parsing and AgentWiki."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Block:
    block_id: str
    block_type: str
    content: str
    page_idx: int
    bbox: tuple[int, int, int, int] | None
    reading_order: int
    heading_level: int | None = None
    heading_path: tuple[str, ...] = ()
    asset_path: str | None = None
    asset_uri: str | None = None
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox"] = list(self.bbox) if self.bbox is not None else None
        value["heading_path"] = list(self.heading_path)
        return value


@dataclass(frozen=True)
class ArticleSegment:
    segment_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    page_start: int
    page_end: int
    content: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        value["block_ids"] = list(self.block_ids)
        return value


@dataclass(frozen=True)
class DocumentIR:
    document_id: str
    source_file: str
    source_sha256: str
    title: str
    parser: str
    parser_output: str
    page_count: int
    blocks: tuple[Block, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "okfolio.document-ir.v1",
            "document_id": self.document_id,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "title": self.title,
            "parser": self.parser,
            "parser_output": self.parser_output,
            "page_count": self.page_count,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class ProcessingResult:
    document_id: str
    article_path: str
    raw_article_path: str
    document_ir_path: str
    normalized_document_ir_path: str
    segments_path: str
    structure_path: str
    asset_manifest_path: str
    manifest_path: str
    normalization_status: str
    blocks: int
    segments: int
    pages: int
    assets: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
