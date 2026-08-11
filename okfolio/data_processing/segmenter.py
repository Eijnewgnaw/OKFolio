"""Heading-first segmentation that never treats a page break as an article."""
from __future__ import annotations

import hashlib
from typing import Iterable

from .models import ArticleSegment, Block, DocumentIR


def _block_text(block: Block) -> str:
    if block.block_type == "title":
        level = block.heading_level or 1
        return f"{'#' * level} {block.content}".strip()
    if block.block_type in {"image", "chart"} and block.asset_uri:
        label = block.content or block.block_type
        return f"![{label}]({block.asset_uri})"
    return block.content.strip()


def _make_segment(
    document_id: str,
    ordinal: int,
    blocks: Iterable[Block],
) -> ArticleSegment:
    items = tuple(blocks)
    content = "\n\n".join(filter(None, (_block_text(block) for block in items)))
    pages = [block.page_idx for block in items if block.page_idx >= 0]
    seed = f"{document_id}:{ordinal}:{','.join(b.block_id for b in items)}"
    heading_path = next(
        (block.heading_path for block in reversed(items) if block.heading_path),
        (),
    )
    return ArticleSegment(
        segment_id="seg-" + hashlib.sha256(seed.encode()).hexdigest()[:20],
        ordinal=ordinal,
        heading_path=heading_path,
        block_ids=tuple(block.block_id for block in items),
        page_start=min(pages) if pages else -1,
        page_end=max(pages) if pages else -1,
        content=content,
        char_count=len(content),
    )


def segment_document(
    document: DocumentIR,
    *,
    target_chars: int = 12_000,
    hard_max_chars: int = 24_000,
) -> tuple[ArticleSegment, ...]:
    if target_chars < 500:
        raise ValueError("target_chars must be at least 500")
    if hard_max_chars < target_chars:
        raise ValueError("hard_max_chars must be >= target_chars")
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_chars = 0
    for block in document.blocks:
        block_chars = len(_block_text(block))
        begins_section = block.heading_level is not None and bool(current)
        would_overflow = current and current_chars + block_chars > hard_max_chars
        soft_boundary = begins_section
        target_boundary = (
            current
            and current_chars >= target_chars
            and current_chars + block_chars > target_chars
        )
        if would_overflow or soft_boundary or target_boundary:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(block)
        current_chars += block_chars + 2
    if current:
        groups.append(current)
    return tuple(
        _make_segment(document.document_id, ordinal, blocks)
        for ordinal, blocks in enumerate(groups, start=1)
    )
