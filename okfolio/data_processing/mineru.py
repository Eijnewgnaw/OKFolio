"""Normalize MinerU structured output into the stable OKFolio Block IR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .models import Block, DocumentIR


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (_text(item) for item in value))).strip()
    if isinstance(value, dict):
        preferred = (
            "text",
            "title_content",
            "paragraph_content",
            "table_body",
            "code_content",
            "math_content",
            "content",
        )
        for key in preferred:
            if key in value:
                candidate = _text(value[key])
                if candidate:
                    return candidate
    return str(value).strip()


def _content(item: dict[str, Any]) -> str:
    for key in (
        "text",
        "title_content",
        "paragraph_content",
        "table_body",
        "code_content",
        "math_content",
        "list_items",
        "content",
    ):
        if key in item:
            candidate = _text(item[key])
            if candidate:
                return candidate
    captions = _text(item.get("image_caption") or item.get("caption"))
    return captions


def _asset_path(item: dict[str, Any]) -> str | None:
    for key in ("img_path", "image_path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\\", "/")
    content = item.get("content")
    if isinstance(content, dict):
        return _asset_path(content)
    return None


def _bbox(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = item.get("bbox")
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(part, (int, float)) for part in value)
    ):
        return tuple(int(part) for part in value)
    return None


def _heading_level(item: dict[str, Any], block_type: str) -> int | None:
    for key in ("text_level", "level", "heading_level"):
        value = item.get(key)
        if isinstance(value, int) and 1 <= value <= 6:
            return value
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("level", "heading_level"):
            value = content.get(key)
            if isinstance(value, int) and 1 <= value <= 6:
                return value
    return 1 if block_type == "title" else None


def _flatten(payload: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("MinerU content list must be a JSON array")
    for item in payload:
        if not isinstance(item, dict):
            continue
        page_items = item.get("items") or item.get("blocks")
        if isinstance(page_items, list):
            page_idx = item.get("page_idx", item.get("page_index", -1))
            for child in page_items:
                if isinstance(child, dict):
                    normalized = dict(child)
                    normalized.setdefault("page_idx", page_idx)
                    yield normalized
        else:
            yield item


class MinerUContentAdapter:
    """Read either legacy content_list.json or page-grouped v2 output."""

    @staticmethod
    def find_content_list(output_dir: Path) -> Path:
        root = output_dir.resolve()
        legacy = sorted(root.rglob("*_content_list.json"))
        if legacy:
            return legacy[0]
        exact = sorted(root.rglob("content_list.json"))
        if exact:
            return exact[0]
        v2 = sorted(root.rglob("*_content_list_v2.json"))
        if v2:
            return v2[0]
        raise FileNotFoundError(f"MinerU content list not found under {root}")

    def load(
        self,
        content_list: Path,
        *,
        document_id: str,
        source_file: str,
        source_sha256: str,
        title: str,
    ) -> DocumentIR:
        payload = json.loads(content_list.read_text(encoding="utf-8"))
        headings: list[str] = []
        blocks: list[Block] = []
        for reading_order, item in enumerate(_flatten(payload)):
            block_type = str(item.get("type", "paragraph")).strip() or "paragraph"
            content = _content(item)
            asset_path = _asset_path(item)
            if not content and not asset_path:
                continue
            page_idx = item.get("page_idx", item.get("page_index", -1))
            page_idx = int(page_idx) if isinstance(page_idx, (int, float)) else -1
            level = _heading_level(item, block_type)
            if level is not None and content:
                headings = headings[: level - 1]
                headings.append(content)
            signature = json.dumps(
                [
                    document_id,
                    page_idx,
                    _bbox(item),
                    block_type,
                    content,
                    asset_path,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            content_hash = hashlib.sha256(
                (content + "\0" + (asset_path or "")).encode("utf-8")
            ).hexdigest()
            blocks.append(
                Block(
                    block_id="blk-" + hashlib.sha256(signature).hexdigest()[:20],
                    block_type=block_type,
                    content=content,
                    page_idx=page_idx,
                    bbox=_bbox(item),
                    reading_order=reading_order,
                    heading_level=level,
                    heading_path=tuple(headings),
                    asset_path=asset_path,
                    content_hash=content_hash,
                )
            )
        if not blocks:
            raise ValueError("MinerU output contains no readable blocks")
        page_indices = [block.page_idx for block in blocks if block.page_idx >= 0]
        page_count = max(page_indices) + 1 if page_indices else 0
        return DocumentIR(
            document_id=document_id,
            source_file=source_file,
            source_sha256=source_sha256,
            title=title,
            parser="mineru",
            parser_output=str(content_list),
            page_count=page_count,
            blocks=tuple(blocks),
        )
