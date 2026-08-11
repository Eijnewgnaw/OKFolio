"""Deterministic document-structure normalization after MinerU extraction.

The parser output is kept immutable.  This module derives a second, evidence
safe view that:

* classifies page roles;
* excludes covers, blank pages and tables of contents from discovery evidence;
* restores Chinese report heading levels;
* records an explicit outline and block-level provenance.

Ambiguous visual-only pages are never silently discarded.  They are preserved
in the raw view and marked for a later VLM or human role decision.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps, ImageStat

from .models import Block, DocumentIR


PageRole = Literal[
    "blank",
    "cover",
    "decorative",
    "toc",
    "section_divider",
    "content",
    "content_retry",
]

_EVIDENCE_ROLES = frozenset({"content"})
_STRUCTURAL_ROLES = frozenset({"section_divider"})
_KNOWN_PAGE_ROLES = frozenset(
    {
        "blank",
        "cover",
        "decorative",
        "toc",
        "section_divider",
        "content",
        "content_retry",
    }
)
_CN = "一二三四五六七八九十百零〇"
_BARE_HEADING_MARKER_RE = re.compile(
    rf"^(?:"
    rf"[{_CN}]{{1,3}}"
    rf"|[0-9]{{1,3}}"
    rf"|[（(][{_CN}0-9]{{1,3}}[）)]"
    rf"|[{_CN}0-9]{{1,3}}[）)]"
    rf")(?:[、.．])?$"
)
_FRONT_OR_PART = frozenset(
    {
        "编委会",
        "前言",
        "序言",
        "PREFACE",
        "后记",
        "附录",
        "综合篇",
        "专题篇",
        "区（市）县篇",
        "建言献策篇",
        "重大事件",
    }
)


@dataclass(frozen=True)
class StructurePolicy:
    """Corpus-independent thresholds for policy-report structure recovery."""

    toc_search_ratio: float = 0.2
    toc_search_min_pages: int = 20
    toc_search_max_pages: int = 80
    toc_span_ratio: float = 0.12
    toc_span_min_pages: int = 8
    toc_span_max_pages: int = 40
    headingless_toc_min_page_refs: int = 2
    headingless_toc_single_page_refs: int = 5
    toc_min_reference_density: float = 0.3
    toc_min_structural_lines: int = 2
    narrative_line_chars: int = 160

    def toc_search_limit(self, page_count: int) -> int:
        return min(
            page_count,
            self.toc_search_max_pages,
            max(
                self.toc_search_min_pages,
                math.ceil(page_count * self.toc_search_ratio),
            ),
        )

    def toc_span_limit(self, page_count: int) -> int:
        return min(
            page_count,
            self.toc_span_max_pages,
            max(
                self.toc_span_min_pages,
                math.ceil(page_count * self.toc_span_ratio),
            ),
        )


@dataclass(frozen=True)
class TocPageFeatures:
    page_refs: int
    structural_lines: int
    candidate_lines: int
    has_long_narrative: bool
    has_explicit_marker: bool

    @property
    def reference_density(self) -> float:
        return self.page_refs / max(1, self.candidate_lines)


@dataclass(frozen=True)
class PageDecision:
    page_idx: int
    role: PageRole
    confidence: float
    reason_codes: tuple[str, ...]
    evidence_eligible: bool
    asset_policy: Literal["exclude", "document", "knowledge", "review"]
    parser_finish_reason: str = ""
    image_sha256: str = ""
    ink_ratio: float | None = None
    entropy: float | None = None

    @property
    def page_number(self) -> int:
        return self.page_idx + 1

    @property
    def needs_review(self) -> bool:
        return self.role == "content_retry"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["page_number"] = self.page_number
        value["reason_codes"] = list(self.reason_codes)
        return value


@dataclass(frozen=True)
class OutlineEntry:
    heading_id: str
    title: str
    level: int
    path: tuple[str, ...]
    page_idx: int
    source: Literal["toc", "body"]
    printed_page: int | None = None
    matched_toc_id: str | None = None
    level_source: Literal[
        "parser",
        "toc_match",
        "numbering",
        "relative",
    ] = "relative"
    level_confidence: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = list(self.path)
        value["page_number"] = self.page_idx + 1
        return value


@dataclass(frozen=True)
class StructureNormalization:
    document: DocumentIR
    pages: tuple[PageDecision, ...]
    outline: tuple[OutlineEntry, ...]
    toc_entries: tuple[OutlineEntry, ...]
    excluded_block_ids: tuple[str, ...]

    @property
    def status(self) -> Literal["complete", "needs_review"]:
        return (
            "needs_review"
            if any(page.needs_review for page in self.pages)
            or self.quality_issues
            else "complete"
        )

    @property
    def quality_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if any(_is_bare_heading_marker(item.title) for item in self.outline):
            issues.append("orphan_heading_marker")
        if any(_is_noise_heading(item.title) for item in self.outline):
            issues.append("nonsemantic_heading")
        return tuple(issues)

    def page_by_index(self) -> dict[int, PageDecision]:
        return {item.page_idx: item for item in self.pages}

    def structure_manifest(self) -> dict[str, Any]:
        page_by_index = self.page_by_index()
        blocks = []
        for block in self.document.blocks:
            page = page_by_index.get(block.page_idx)
            blocks.append(
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "content": block.content,
                    "content_hash": block.content_hash,
                    "page_idx": block.page_idx,
                    "page_number": block.page_idx + 1,
                    "heading_level": block.heading_level,
                    "heading_path": list(block.heading_path),
                    "page_role": page.role if page is not None else "content",
                    "evidence_eligible": (
                        page is None
                        or (
                            page.evidence_eligible
                            and block.block_type != "title"
                        )
                    ),
                    "asset_uri": block.asset_uri,
                }
            )
        return {
            "schema_version": "okfolio.document-structure.v1",
            "status": self.status,
            "quality_issues": list(self.quality_issues),
            "document_id": self.document.document_id,
            "pages": [item.to_dict() for item in self.pages],
            "outline": [item.to_dict() for item in self.outline],
            "toc_entries": [item.to_dict() for item in self.toc_entries],
            "excluded_block_ids": list(self.excluded_block_ids),
            "blocks": blocks,
        }


def normalize_document_structure(
    document: DocumentIR,
    parser_output_dir: Path,
    *,
    policy: StructurePolicy | None = None,
) -> StructureNormalization:
    """Return a non-destructive, evidence-safe view of one parsed document."""
    policy = policy or StructurePolicy()
    page_records = _load_page_records(parser_output_dir)
    document = _overlay_recovered_pages(document, page_records)
    document, merged_marker_ids = _merge_fragmented_heading_markers(document)
    by_page: dict[int, list[Block]] = defaultdict(list)
    for block in document.blocks:
        by_page[block.page_idx].append(block)

    toc_pages = _detect_toc_pages(by_page, document.page_count, policy=policy)
    decisions = tuple(
        _classify_page(
            page_idx,
            by_page.get(page_idx, []),
            page_records.get(page_idx),
            parser_output_dir,
            toc_pages=toc_pages,
            page_count=document.page_count,
        )
        for page_idx in range(document.page_count)
    )
    page_by_index = {item.page_idx: item for item in decisions}
    toc_entries = _build_toc_entries(by_page, toc_pages)
    has_part_heading = any(
        _is_part_heading(block.content)
        for page_idx, blocks in by_page.items()
        if page_idx not in toc_pages
        for block in blocks
        if block.block_type == "title"
    )
    declared_levels = {
        block.heading_level
        for page_idx, blocks in by_page.items()
        if page_idx not in toc_pages
        for block in blocks
        if block.block_type == "title"
        and isinstance(block.heading_level, int)
    }
    trust_parser_levels = len(declared_levels) >= 2

    normalized: list[Block] = []
    excluded: list[str] = list(merged_marker_ids)
    outline: list[OutlineEntry] = []
    heading_stack: list[str] = []
    previous_title: Block | None = None
    previous_heading_level: int | None = None
    previous_heading_style: str | None = None
    anchor_level: int | None = None
    learned_style_levels: dict[str, int] = {}
    toc_matches: dict[str, list[OutlineEntry]] = defaultdict(list)
    for item in toc_entries:
        toc_matches[_heading_key(item.title)].append(item)

    for block in document.blocks:
        page = page_by_index.get(block.page_idx)
        role = page.role if page is not None else "content"
        if role not in _EVIDENCE_ROLES | _STRUCTURAL_ROLES:
            excluded.append(block.block_id)
            continue
        if role == "section_divider" and block.block_type != "title":
            excluded.append(block.block_id)
            continue

        if block.block_type == "title":
            if _is_noise_heading(block.content):
                excluded.append(block.block_id)
                previous_title = None
                continue
            if _is_bare_heading_marker(block.content):
                normalized.append(
                    replace(
                        block,
                        block_type="text",
                        heading_level=None,
                        heading_path=tuple(heading_stack),
                    )
                )
                previous_title = None
                continue
            if _looks_like_affiliation(block.content, previous_title, block):
                normalized.append(
                    replace(
                        block,
                        block_type="text",
                        heading_level=None,
                        heading_path=tuple(heading_stack),
                    )
                )
                previous_title = None
                continue
            key = _heading_key(block.content)
            matched = toc_matches.get(key, [])
            matched_toc = matched[0] if matched else None
            style = _heading_style(block.content)
            declared_level = (
                block.heading_level
                if (
                    isinstance(block.heading_level, int)
                    and (
                        trust_parser_levels
                        or block.block_id.startswith("blk-recovery-")
                    )
                )
                else None
            )
            level = _infer_heading_level(
                block.content,
                has_part_heading=has_part_heading,
                current_stack=tuple(heading_stack),
                toc_level=matched_toc.level if matched_toc is not None else None,
                declared_level=declared_level,
                previous_style=previous_heading_style,
                previous_level=previous_heading_level,
                anchor_level=anchor_level,
                learned_style_level=learned_style_levels.get(style),
                adjacent=previous_title is not None,
            )
            level = max(1, min(5, level))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(block.content.strip())
            updated = replace(
                block,
                heading_level=level,
                heading_path=tuple(heading_stack),
            )
            normalized.append(updated)
            outline.append(
                OutlineEntry(
                    heading_id=block.block_id,
                    title=block.content.strip(),
                    level=level,
                    path=tuple(heading_stack),
                    page_idx=block.page_idx,
                    source="body",
                    printed_page=(
                        matched_toc.printed_page
                        if matched_toc is not None
                        else None
                    ),
                    matched_toc_id=(
                        matched_toc.heading_id
                        if matched_toc is not None
                        else None
                    ),
                    level_source=_heading_level_source(
                        declared_level=declared_level,
                        matched_toc=matched_toc,
                        style=style,
                    ),
                    level_confidence=_heading_level_confidence(
                        declared_level=declared_level,
                        matched_toc=matched_toc,
                        style=style,
                    ),
                )
            )
            previous_title = block
            previous_heading_level = level
            previous_heading_style = style
            if declared_level is not None or matched_toc is not None:
                anchor_level = level
            elif style not in {"unnumbered", "bare_marker"}:
                learned_style_levels.setdefault(style, level)
                anchor_level = level
            continue

        normalized.append(replace(block, heading_path=tuple(heading_stack)))
        if block.content.strip():
            previous_title = None

    normalized_document = replace(document, blocks=tuple(normalized))
    return StructureNormalization(
        document=normalized_document,
        pages=decisions,
        outline=tuple(outline),
        toc_entries=toc_entries,
        excluded_block_ids=tuple(excluded),
    )


def _merge_fragmented_heading_markers(
    document: DocumentIR,
) -> tuple[DocumentIR, tuple[str, ...]]:
    """Join a standalone numbering marker to its adjacent title.

    MinerU can emit ``（一）`` and ``总体情况`` as two title blocks.  The raw
    DocumentIR remains unchanged on disk; this derived view removes the marker
    block and prefixes its text to the following title so discovery never sees
    a meaningless one-character heading.
    """
    blocks = list(document.blocks)
    normalized: list[Block] = []
    merged_ids: list[str] = []
    position = 0
    while position < len(blocks):
        marker = blocks[position]
        following = blocks[position + 1] if position + 1 < len(blocks) else None
        if (
            following is not None
            and marker.block_type == "title"
            and following.block_type == "title"
            and marker.page_idx == following.page_idx
            and _is_bare_heading_marker(marker.content)
            and not _is_bare_heading_marker(following.content)
        ):
            content = f"{marker.content.strip()} {following.content.strip()}"
            normalized.append(
                replace(
                    following,
                    content=content,
                    content_hash=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                )
            )
            merged_ids.append(marker.block_id)
            position += 2
            continue
        normalized.append(marker)
        position += 1
    return replace(document, blocks=tuple(normalized)), tuple(merged_ids)


def _is_bare_heading_marker(value: str) -> bool:
    return bool(_BARE_HEADING_MARKER_RE.fullmatch(value.strip()))


def _is_noise_heading(value: str) -> bool:
    return not bool(
        re.search(rf"[A-Za-z0-9{_CN}\u4e00-\u9fff]", value.strip())
    )


def document_from_dict(value: dict[str, Any]) -> DocumentIR:
    """Load a previously persisted DocumentIR for normalization replay."""
    blocks = tuple(
        Block(
            block_id=str(item["block_id"]),
            block_type=str(item["block_type"]),
            content=str(item.get("content") or ""),
            page_idx=int(item.get("page_idx") or 0),
            bbox=(
                tuple(int(part) for part in item["bbox"])
                if isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4
                else None
            ),
            reading_order=int(item.get("reading_order") or position),
            heading_level=(
                int(item["heading_level"])
                if isinstance(item.get("heading_level"), int)
                else None
            ),
            heading_path=tuple(item.get("heading_path") or ()),
            asset_path=item.get("asset_path"),
            asset_uri=item.get("asset_uri"),
            content_hash=str(item.get("content_hash") or ""),
        )
        for position, item in enumerate(value.get("blocks", []))
        if isinstance(item, dict)
    )
    return DocumentIR(
        document_id=str(value["document_id"]),
        source_file=str(value["source_file"]),
        source_sha256=str(value["source_sha256"]),
        title=str(value["title"]),
        parser=str(value.get("parser") or "mineru"),
        parser_output=str(value.get("parser_output") or ""),
        page_count=int(value.get("page_count") or 0),
        blocks=blocks,
    )


def _load_page_records(root: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted((root.resolve() / "page-results").glob("page-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, dict)
            and value.get("status") == "complete"
            and isinstance(value.get("page_idx"), int)
        ):
            records[int(value["page_idx"])] = value
    return records


def _overlay_recovered_pages(
    document: DocumentIR,
    page_records: dict[int, dict[str, Any]],
) -> DocumentIR:
    recovered = {
        page_idx: str(record.get("recovery_content") or "").strip()
        for page_idx, record in page_records.items()
        if record.get("page_role") == "content"
        and str(record.get("recovery_content") or "").strip()
    }
    if not recovered:
        return document
    blocks = [
        block for block in document.blocks if block.page_idx not in recovered
    ]
    next_order = max((block.reading_order for block in blocks), default=-1) + 1
    for page_idx, content in sorted(recovered.items()):
        for ordinal, (block_type, text, level) in enumerate(
            _recovery_blocks(content)
        ):
            signature = (
                f"{document.document_id}:{page_idx}:{ordinal}:"
                f"{block_type}:{text}"
            )
            blocks.append(
                Block(
                    block_id="blk-recovery-"
                    + hashlib.sha256(signature.encode()).hexdigest()[:16],
                    block_type=block_type,
                    content=text,
                    page_idx=page_idx,
                    bbox=None,
                    reading_order=next_order,
                    heading_level=level,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
            next_order += 1
    blocks.sort(key=lambda item: (item.page_idx, item.reading_order))
    return replace(document, blocks=tuple(blocks))


def _recovery_blocks(content: str) -> tuple[tuple[str, str, int | None], ...]:
    result: list[tuple[str, str, int | None]] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            result.append(("text", "\n".join(paragraph).strip(), None))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            text = "\n".join(table).strip()
            block_type = "table" if _is_markdown_table(table) else "text"
            result.append((block_type, text, None))
            table.clear()

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_table()
            flush_paragraph()
            continue
        match = re.match(r"^(#{1,6})[ \t]+(.+?)\s*$", stripped)
        if match:
            flush_table()
            flush_paragraph()
            result.append(
                ("title", match.group(2).strip(), len(match.group(1)))
            )
            continue
        if "|" in stripped:
            flush_paragraph()
            table.append(stripped)
            continue
        flush_table()
        paragraph.append(stripped)

    flush_table()
    flush_paragraph()
    return tuple(result)


def _is_markdown_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    separator = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    return bool(separator.match(lines[1]))


def _detect_toc_pages(
    by_page: dict[int, list[Block]],
    page_count: int,
    *,
    policy: StructurePolicy,
) -> set[int]:
    search_limit = policy.toc_search_limit(page_count)
    features = {
        page_idx: _toc_page_features(
            by_page.get(page_idx, []),
            policy=policy,
        )
        for page_idx in range(search_limit)
    }
    explicit_starts = [
        page_idx
        for page_idx in range(search_limit)
        if features[page_idx].has_explicit_marker
    ]
    if explicit_starts:
        start = min(explicit_starts)
    else:
        headingless_starts = [
            page_idx
            for page_idx in range(search_limit)
            if _is_headingless_toc_start(
                features[page_idx],
                features.get(page_idx + 1),
                policy=policy,
            )
        ]
        if not headingless_starts:
            return set()
        start = min(headingless_starts)

    pages: set[int] = set()
    span_limit = policy.toc_span_limit(page_count)
    for page_idx in range(start, min(page_count, start + span_limit)):
        feature = features.get(page_idx) or _toc_page_features(
            by_page.get(page_idx, []),
            policy=policy,
        )
        toc_like = (
            feature.has_explicit_marker
            or feature.page_refs >= 1
            or (
                feature.structural_lines >= policy.toc_min_structural_lines
                and not feature.has_long_narrative
            )
        )
        if (page_idx == start or toc_like) and not (
            feature.has_long_narrative and not feature.has_explicit_marker
        ):
            pages.add(page_idx)
            continue
        break
    return pages


def _toc_page_features(
    blocks: list[Block],
    *,
    policy: StructurePolicy,
) -> TocPageFeatures:
    lines = _toc_source_lines(blocks)
    page_refs = sum(printed_page is not None for _title, printed_page in lines)
    structural_lines = sum(
        _heading_style(title) != "unnumbered"
        for title, _printed_page in lines
        if title.upper() not in {"目录", "CONTENTS"}
    )
    exact = any(title.upper() in {"目录", "CONTENTS"} for title, _ in lines)
    has_long_narrative = any(
        block.block_type == "text"
        and len(block.content.strip()) >= policy.narrative_line_chars
        and _split_toc_printed_page(block.content)[1] is None
        for block in blocks
    )
    return TocPageFeatures(
        page_refs=page_refs,
        structural_lines=structural_lines,
        candidate_lines=len(lines),
        has_long_narrative=has_long_narrative,
        has_explicit_marker=exact,
    )


def _is_headingless_toc_start(
    feature: TocPageFeatures,
    next_feature: TocPageFeatures | None,
    *,
    policy: StructurePolicy,
) -> bool:
    if feature.has_long_narrative:
        return False
    if feature.page_refs < policy.headingless_toc_min_page_refs:
        return False
    if feature.reference_density < policy.toc_min_reference_density:
        return False
    if feature.structural_lines < policy.toc_min_structural_lines:
        return False
    if feature.page_refs >= policy.headingless_toc_single_page_refs:
        return True
    return bool(
        next_feature is not None
        and not next_feature.has_long_narrative
        and next_feature.page_refs >= policy.headingless_toc_min_page_refs
        and next_feature.reference_density >= policy.toc_min_reference_density
    )


def _toc_source_lines(
    blocks: list[Block],
) -> list[tuple[str, int | None]]:
    lines: list[tuple[str, int | None]] = []
    for block in blocks:
        content = block.content.strip()
        if not content:
            continue
        if block.block_type in {"title", "text"}:
            lines.append(_split_toc_printed_page(content))
            continue
        if block.block_type != "table":
            continue
        for row in re.findall(
            r"<tr\b[^>]*>(.*?)</tr>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            cells = [
                html.unescape(
                    re.sub(r"<[^>]+>", "", cell, flags=re.DOTALL)
                ).strip()
                for cell in re.findall(
                    r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
                    row,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            ]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            if re.fullmatch(r"\d{1,4}", cells[0]):
                lines.append((cells[1], int(cells[0])))
            else:
                lines.append((" ".join(cells), None))
    return lines


def _classify_page(
    page_idx: int,
    blocks: list[Block],
    record: dict[str, Any] | None,
    parser_output_dir: Path,
    *,
    toc_pages: set[int],
    page_count: int,
) -> PageDecision:
    del page_count
    external = str((record or {}).get("page_role") or "").strip()
    finish_reason = str((record or {}).get("finish_reason") or "")
    image_sha256 = ""
    ink_ratio: float | None = None
    entropy: float | None = None
    if record is not None and isinstance(record.get("image_path"), str):
        image = (parser_output_dir.resolve() / str(record["image_path"])).resolve()
        try:
            image.relative_to(parser_output_dir.resolve())
            if image.is_file():
                image_sha256 = _sha256(image)
                ink_ratio, entropy = _image_metrics(image)
        except ValueError:
            pass

    if page_idx in toc_pages:
        return _page_decision(
            page_idx,
            "toc",
            0.98,
            ("toc_heading_sequence",),
            finish_reason,
            image_sha256,
            ink_ratio,
            entropy,
        )
    if external in _KNOWN_PAGE_ROLES:
        return _page_decision(
            page_idx,
            external,
            float((record or {}).get("page_role_confidence") or 0.95),
            ("external_structured_role",),
            finish_reason,
            image_sha256,
            ink_ratio,
            entropy,
        )
    if finish_reason == "two_step_empty_page_preserved":
        if _is_blank_image(ink_ratio, entropy):
            return _page_decision(
                page_idx,
                "blank",
                0.995,
                ("empty_layout", "low_ink", "low_entropy"),
                finish_reason,
                image_sha256,
                ink_ratio,
                entropy,
            )
        if page_idx == 0:
            return _page_decision(
                page_idx,
                "cover",
                0.86,
                ("first_page", "empty_layout", "nonblank_visual"),
                finish_reason,
                image_sha256,
                ink_ratio,
                entropy,
            )
        return _page_decision(
            page_idx,
            "content_retry",
            0.5,
            ("empty_layout", "nonblank_visual", "requires_role_classifier"),
            finish_reason,
            image_sha256,
            ink_ratio,
            entropy,
        )
    titles = [block for block in blocks if block.block_type == "title"]
    body_chars = sum(
        len(block.content.strip())
        for block in blocks
        if block.block_type not in {"title", "image", "chart"}
    )
    if record is not None and page_idx == 0 and titles and body_chars < 200:
        return _page_decision(
            page_idx,
            "cover",
            0.9,
            ("first_page", "title_dominant"),
            finish_reason,
            image_sha256,
            ink_ratio,
            entropy,
        )
    if titles and body_chars == 0:
        return _page_decision(
            page_idx,
            "section_divider",
            0.83,
            ("title_dominant", "low_body_text"),
            finish_reason,
            image_sha256,
            ink_ratio,
            entropy,
        )
    return _page_decision(
        page_idx,
        "content",
        0.99,
        ("parsed_content",),
        finish_reason,
        image_sha256,
        ink_ratio,
        entropy,
    )


def _page_decision(
    page_idx: int,
    role: str,
    confidence: float,
    reasons: tuple[str, ...],
    finish_reason: str,
    image_sha256: str,
    ink_ratio: float | None,
    entropy: float | None,
) -> PageDecision:
    assert role in _KNOWN_PAGE_ROLES
    asset_policy: Literal["exclude", "document", "knowledge", "review"]
    if role == "blank":
        asset_policy = "exclude"
    elif role in {"cover", "decorative", "toc", "section_divider"}:
        asset_policy = "document"
    elif role == "content_retry":
        asset_policy = "review"
    else:
        asset_policy = "knowledge"
    return PageDecision(
        page_idx=page_idx,
        role=role,  # type: ignore[arg-type]
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        reason_codes=reasons,
        evidence_eligible=role in _EVIDENCE_ROLES,
        asset_policy=asset_policy,
        parser_finish_reason=finish_reason,
        image_sha256=image_sha256,
        ink_ratio=ink_ratio,
        entropy=entropy,
    )


def _image_metrics(path: Path) -> tuple[float, float]:
    with Image.open(path) as opened:
        gray = ImageOps.grayscale(opened)
        gray.thumbnail((256, 256))
        histogram = gray.histogram()
        total = sum(histogram) or 1
        ink = sum(histogram[:245])
        entropy = 0.0
        for count in histogram:
            if not count:
                continue
            probability = count / total
            entropy -= probability * math.log2(probability)
        # Force image decoding before the file handle closes.
        ImageStat.Stat(gray).mean
    return round(ink / total, 6), round(entropy, 6)


def _is_blank_image(
    ink_ratio: float | None,
    entropy: float | None,
) -> bool:
    return (
        ink_ratio is not None
        and entropy is not None
        and ink_ratio <= 0.003
        and entropy <= 1.2
    )


def _build_toc_entries(
    by_page: dict[int, list[Block]],
    toc_pages: set[int],
) -> tuple[OutlineEntry, ...]:
    raw: list[tuple[str, int, int | None, str]] = []
    for page_idx in sorted(toc_pages):
        for position, (title, printed_page) in enumerate(
            _toc_source_lines(by_page.get(page_idx, [])),
            start=1,
        ):
            if not title or title.upper() in {"目录", "CONTENTS"}:
                continue
            raw.append(
                (
                    title,
                    page_idx,
                    printed_page,
                    f"toc-{page_idx + 1:04d}-{position:03d}",
                )
            )
    has_part = any(_is_part_heading(item[0]) for item in raw)
    stack: list[str] = []
    entries: list[OutlineEntry] = []
    previous_style: str | None = None
    previous_level: int | None = None
    anchor_level: int | None = None
    learned_style_levels: dict[str, int] = {}
    for title, page_idx, printed_page, heading_id in raw:
        style = _heading_style(title)
        level = _infer_heading_level(
            title,
            has_part_heading=has_part,
            current_stack=tuple(stack),
            toc_level=None,
            previous_style=previous_style,
            previous_level=previous_level,
            anchor_level=anchor_level,
            learned_style_level=learned_style_levels.get(style),
            adjacent=True,
        )
        level = max(1, min(5, level))
        stack = stack[: level - 1]
        stack.append(title)
        entries.append(
            OutlineEntry(
                heading_id=heading_id,
                title=title,
                level=level,
                path=tuple(stack),
                page_idx=page_idx,
                source="toc",
                printed_page=printed_page,
                level_source=(
                    "numbering"
                    if style not in {"unnumbered", "bare_marker"}
                    else "relative"
                ),
                level_confidence=(
                    0.9
                    if style not in {"unnumbered", "bare_marker"}
                    else 0.68
                ),
            )
        )
        previous_style = style
        previous_level = level
        if style not in {"unnumbered", "bare_marker"}:
            learned_style_levels.setdefault(style, level)
            anchor_level = level
    return tuple(entries)


def _infer_heading_level(
    title: str,
    *,
    has_part_heading: bool,
    current_stack: tuple[str, ...],
    toc_level: int | None,
    declared_level: int | None = None,
    previous_style: str | None = None,
    previous_level: int | None = None,
    anchor_level: int | None = None,
    learned_style_level: int | None = None,
    adjacent: bool = False,
) -> int:
    if declared_level is not None:
        return declared_level
    if toc_level is not None:
        return toc_level
    style = _heading_style(title)
    if learned_style_level is not None:
        return learned_style_level

    base_levels = {
        "part": 1,
        "chapter": 2 if has_part_heading else 1,
        "section": 3 if has_part_heading else 2,
        "cn_enum": 2 if has_part_heading else 1,
        "cn_paren": 3 if has_part_heading else 2,
        "digit_enum": 4 if has_part_heading else 3,
        "digit_paren": 5 if has_part_heading else 4,
    }
    if style in base_levels:
        base = base_levels[style]
        if style in {
            "cn_enum",
            "cn_paren",
            "digit_enum",
            "digit_paren",
        } and anchor_level is not None:
            return max(base, min(5, anchor_level + 1))
        return base

    if style == "bare_marker":
        if anchor_level is not None:
            return min(5, anchor_level + 1)
        if previous_level is not None:
            return previous_level
        return max(1, min(5, len(current_stack)))

    if adjacent and previous_level is not None:
        if previous_style in {"unnumbered", "bare_marker"}:
            return previous_level
        if previous_style in {"part", "chapter", "section"} and re.fullmatch(
            rf"(?:第[{_CN}\d]+(?:篇|章|节))",
            str(current_stack[-1] if current_stack else ""),
        ):
            return previous_level
        return min(5, previous_level + 1)
    if anchor_level is not None:
        return min(5, anchor_level + 1)
    if current_stack:
        return max(1, min(5, len(current_stack)))
    return 1


def _heading_level_source(
    *,
    declared_level: int | None,
    matched_toc: OutlineEntry | None,
    style: str,
) -> Literal["parser", "toc_match", "numbering", "relative"]:
    if declared_level is not None:
        return "parser"
    if matched_toc is not None:
        return "toc_match"
    if style not in {"unnumbered", "bare_marker"}:
        return "numbering"
    return "relative"


def _heading_level_confidence(
    *,
    declared_level: int | None,
    matched_toc: OutlineEntry | None,
    style: str,
) -> float:
    if declared_level is not None:
        return 0.98
    if matched_toc is not None:
        return 0.96
    if style not in {"unnumbered", "bare_marker"}:
        return 0.9
    if style == "bare_marker":
        return 0.7
    return 0.62


def _heading_style(title: str) -> str:
    value = title.strip()
    if value.upper() in {"目录", "CONTENTS"}:
        return "toc"
    if _is_part_heading(value) or value in _FRONT_OR_PART:
        return "part"
    if re.match(rf"^第[{_CN}\d]+篇", value):
        return "part"
    if re.match(rf"^第[{_CN}\d]+章", value):
        return "chapter"
    if re.match(rf"^第[{_CN}\d]+节", value):
        return "section"
    if re.match(rf"^[{_CN}]+[、.．]", value):
        return "cn_enum"
    if re.match(rf"^[（(][{_CN}]+[）)]", value):
        return "cn_paren"
    if re.match(r"^\d+[、.．]", value):
        return "digit_enum"
    if re.match(r"^[（(]\d+[）)]", value):
        return "digit_paren"
    if re.fullmatch(rf"[{_CN}]+[）)]?", value) or re.fullmatch(
        r"\d+",
        value,
    ):
        return "bare_marker"
    if value in {"后记", "结语", "附录"}:
        return "part"
    return "unnumbered"


def _is_part_heading(title: str) -> bool:
    value = title.strip()
    return (
        value.endswith("篇")
        or bool(re.match(rf"^第[{_CN}\d]+篇", value))
        or value in {"重大事件"}
    )


def _looks_like_affiliation(
    title: str,
    previous_title: Block | None,
    current: Block,
) -> bool:
    value = title.strip()
    return bool(
        previous_title is not None
        and previous_title.page_idx == current.page_idx
        and re.match(r"^[（(].*[）)]$", value)
        and re.search(r"大学|研究院|研究所|学院|中心|公司", value)
        and (re.search(r"[;；]", value) or re.search(r"\d+[.．]", value))
    )


def _split_printed_page(title: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*?)(?:\s|·{2,})(\d{1,4})\s*$", title.strip())
    if not match:
        return title.strip(), None
    return match.group(1).strip(), int(match.group(2))


def _split_toc_printed_page(title: str) -> tuple[str, int | None]:
    clean, printed_page = _split_printed_page(title)
    if printed_page is not None:
        return clean, printed_page
    leading = re.match(r"^(\d{1,4})\s+(.+?)\s*$", title.strip())
    if leading:
        return leading.group(2).strip(), int(leading.group(1))
    return title.strip(), None


def _heading_key(title: str) -> str:
    value, _printed_page = _split_printed_page(title)
    value = re.sub(r"\s+", "", value)
    return value.strip("：:。.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
