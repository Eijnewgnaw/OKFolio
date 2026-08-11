"""Resumable PDF-page rendering and MinerU VLM orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PIL import Image

from .vlm import PageParseResult


class PageRenderer(Protocol):
    def page_count(self, pdf: Path) -> int: ...

    def render_page(
        self,
        pdf: Path,
        page_idx: int,
        destination: Path,
        *,
        dpi: int,
    ) -> None: ...


class PageParser(Protocol):
    model: str
    max_tokens: int

    def parse_image(
        self,
        image_path: Path,
        *,
        max_tokens: int | None = None,
    ) -> PageParseResult: ...


class PageRoleClassifier(Protocol):
    def classify(self, image_path: Path) -> object: ...


@dataclass(frozen=True)
class PDFPageRenderer:
    pdfinfo_command: str = "pdfinfo"
    pdftoppm_command: str = "pdftoppm"
    jpeg_quality: int = 85

    def page_count(self, pdf: Path) -> int:
        result = subprocess.run(
            [self.pdfinfo_command, str(pdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
        if not match:
            raise ValueError("pdfinfo output does not contain a page count")
        pages = int(match.group(1))
        if pages < 1:
            raise ValueError("PDF must contain at least one page")
        return pages

    def render_page(
        self,
        pdf: Path,
        page_idx: int,
        destination: Path,
        *,
        dpi: int,
    ) -> None:
        if page_idx < 0:
            raise ValueError("page_idx must be zero-based and non-negative")
        if dpi < 72 or dpi > 400:
            raise ValueError("render DPI must be between 72 and 400")
        destination.parent.mkdir(parents=True, exist_ok=True)
        prefix = destination.with_suffix("")
        page_number = page_idx + 1
        subprocess.run(
            [
                self.pdftoppm_command,
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-jpeg",
                "-jpegopt",
                f"quality={self.jpeg_quality}",
                "-r",
                str(dpi),
                str(pdf),
                str(prefix),
            ],
            check=True,
        )
        generated = prefix.with_suffix(".jpg")
        if generated != destination:
            os.replace(generated, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"page renderer produced no image: {destination}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page_record_path(root: Path, page_idx: int) -> Path:
    return root / "page-results" / f"page-{page_idx + 1:04d}.json"


def _load_complete_page(root: Path, page_idx: int) -> dict[str, object] | None:
    path = _page_record_path(root, page_idx)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("status") == "complete"
        and value.get("page_idx") == page_idx
    ):
        return value
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_job_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _completed_page_count(root: Path) -> int:
    count = 0
    for path in (root / "page-results").glob("page-*.json"):
        match = re.fullmatch(r"page-(\d{4,})\.json", path.name)
        if match is None:
            continue
        page_idx = int(match.group(1)) - 1
        if _load_complete_page(root, page_idx) is not None:
            count += 1
    return count


def _validate_resume(
    previous: dict[str, object],
    *,
    source_sha256: str,
    total_pages: int,
    model: str,
    has_page_results: bool,
) -> None:
    previous_hash = str(previous.get("source_sha256") or "")
    if previous_hash and previous_hash != source_sha256:
        raise ValueError(
            "parser job belongs to a different PDF; use a new output directory"
        )
    previous_pages = int(previous.get("page_count") or 0)
    if previous_pages and previous_pages != total_pages:
        raise ValueError(
            "parser job page count changed; use a new output directory"
        )
    previous_model = str(previous.get("model") or "")
    if has_page_results and previous_model and previous_model != model:
        raise ValueError(
            "parser job model changed after pages were persisted; "
            "use a new output directory"
        )


def _retry_delay(error: Exception, attempt: int) -> float:
    message = str(error).lower()
    if "429" in message or "请求过于频繁" in message:
        return float(min(60, 10 * attempt))
    if "timeout" in message or "timed out" in message:
        return float(min(30, 5 * attempt))
    return float(min(10, 2 * attempt))


def parse_pdf_with_vlm(
    pdf_path: Path,
    output_dir: Path,
    *,
    parser: PageParser,
    renderer: PageRenderer | None = None,
    page_start: int = 0,
    page_end: int | None = None,
    render_dpi: int = 160,
    max_attempts: int = 2,
    include_page_assets: bool = True,
    page_role_classifier: PageRoleClassifier | None = None,
) -> Path:
    """Create a MinerU-compatible content list with resumable page records."""
    pdf = pdf_path.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF input not found: {pdf}")
    if max_attempts < 1 or max_attempts > 5:
        raise ValueError("max_attempts must be between 1 and 5")
    engine = renderer or PDFPageRenderer()
    total_pages = engine.page_count(pdf)
    source_sha256 = _sha256(pdf)
    end = total_pages if page_end is None else page_end
    if page_start < 0 or end <= page_start or end > total_pages:
        raise ValueError("invalid zero-based page range")

    root = output_dir.resolve()
    pages_dir = root / "pages"
    root.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "job.json"
    previous_state = _load_job_state(state_path)
    completed_pages = _completed_page_count(root)
    _validate_resume(
        previous_state,
        source_sha256=source_sha256,
        total_pages=total_pages,
        model=parser.model,
        has_page_results=completed_pages > 0,
    )
    state: dict[str, object] = {
        "schema_version": "okfolio.pdf-worker.v1",
        "status": "running",
        "source_file": pdf.name,
        "source_sha256": source_sha256,
        "model": parser.model,
        "page_count": total_pages,
        "page_start": page_start,
        "page_end": end,
        "render_dpi": render_dpi,
        "resume_count": int(previous_state.get("resume_count") or 0)
        + int(bool(previous_state)),
        "completed_pages": completed_pages,
        "updated_at": _now(),
    }
    _write_json(state_path, state)

    page_records: list[dict[str, object]] = []
    current_page_idx: int | None = None
    try:
        for page_idx in range(page_start, end):
            current_page_idx = page_idx
            existing = _load_complete_page(root, page_idx)
            if existing is not None:
                page_records.append(existing)
                continue
            image_path = pages_dir / f"page-{page_idx + 1:04d}.jpg"
            if not image_path.is_file():
                engine.render_page(
                    pdf,
                    page_idx,
                    image_path,
                    dpi=render_dpi,
                )
            result: PageParseResult | None = None
            error_text = ""
            for attempt in range(1, max_attempts + 1):
                try:
                    token_limit = parser.max_tokens * (2 ** (attempt - 1))
                    result = parser.parse_image(
                        image_path,
                        max_tokens=token_limit,
                    )
                    if result.finish_reason == "length" and attempt < max_attempts:
                        continue
                    break
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"
                    if attempt == max_attempts:
                        raise
                    time.sleep(_retry_delay(error, attempt))
            if result is None:
                raise RuntimeError(error_text or "page parser returned no result")
            record: dict[str, object] = {
                "schema_version": "okfolio.page-result.v1",
                "page_idx": page_idx,
                "status": "complete",
                "attempts": attempt,
                "image_path": image_path.relative_to(root).as_posix(),
                **result.to_dict(),
            }
            if (
                page_role_classifier is not None
                and result.finish_reason == "two_step_empty_page_preserved"
            ):
                role = page_role_classifier.classify(image_path)
                to_dict = getattr(role, "to_dict", None)
                if not callable(to_dict):
                    raise TypeError(
                        "page role classifier result must provide to_dict()"
                    )
                record.update(to_dict())
            _write_json(_page_record_path(root, page_idx), record)
            page_records.append(record)
            completed_pages += 1
            state.update(
                {
                    "completed_pages": completed_pages,
                    "current_page": page_idx + 1,
                    "updated_at": _now(),
                }
            )
            _write_json(state_path, state)
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_page": (
                    current_page_idx + 1
                    if current_page_idx is not None
                    else None
                ),
                "completed_pages": _completed_page_count(root),
                "updated_at": _now(),
            }
        )
        _write_json(state_path, state)
        raise

    content_list: list[dict[str, object]] = []
    for record in sorted(page_records, key=lambda item: int(item["page_idx"])):
        page_idx = int(record["page_idx"])
        structured = _structured_content_items(
            record,
            root=root,
            page_idx=page_idx,
            include_assets=include_page_assets,
        )
        if structured:
            content_list.extend(structured)
            continue
        content = str(record["content"]).strip()
        block_type = "table" if _looks_like_markdown_table(content) else "text"
        content_list.append(
            {
                "type": block_type,
                "text": content,
                "page_idx": page_idx,
            }
        )
        if include_page_assets:
            content_list.append(
                {
                    "type": "image",
                    "img_path": str(record["image_path"]),
                    "image_caption": [f"原始 PDF 第 {page_idx + 1} 页"],
                    "page_idx": page_idx,
                }
            )
    content_path = root / f"{pdf.stem}_content_list.json"
    _write_json(content_path, content_list)
    _write_json(
        root / "document-metadata.json",
        {
            "schema_version": "okfolio.parser-metadata.v1",
            "parser": getattr(
                parser,
                "parser_name",
                "mineru-openai-compatible",
            ),
            "model": parser.model,
            "page_count": total_pages,
            "processed_page_start": page_start,
            "processed_page_end": end,
        },
    )
    state.update(
        {
            "status": "complete",
            "processed_pages": len(page_records),
            "completed_pages": _completed_page_count(root),
            "content_list": str(content_path),
            "finished_at": _now(),
            "updated_at": _now(),
        }
    )
    state.pop("error", None)
    state.pop("failed_page", None)
    _write_json(state_path, state)
    return content_path


def _structured_content_items(
    record: dict[str, object],
    *,
    root: Path,
    page_idx: int,
    include_assets: bool,
) -> list[dict[str, object]]:
    blocks = record.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return []
    image_relative = Path(str(record["image_path"]))
    image_path = (root / image_relative).resolve()
    image_path.relative_to(root.resolve())
    with Image.open(image_path) as opened:
        width, height = opened.size
        page_image = opened.convert("RGB")
    items: list[dict[str, object]] = []
    image_number = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "text")
        bbox = _pixel_bbox(block.get("bbox"), width=width, height=height)
        content = str(block.get("content") or "").strip()
        if block_type in {"image", "chart"}:
            if not include_assets or bbox is None:
                continue
            image_number += 1
            suffix = "chart" if block_type == "chart" else "image"
            asset_relative = (
                Path("images")
                / f"page-{page_idx + 1:04d}-{suffix}-{image_number:03d}.jpg"
            )
            asset_path = root / asset_relative
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            page_image.crop(bbox).save(asset_path, "JPEG", quality=90)
            items.append(
                {
                    "type": block_type,
                    "img_path": asset_relative.as_posix(),
                    "image_caption": [
                        content or f"原始 PDF 第 {page_idx + 1} 页图片"
                    ],
                    "bbox": list(bbox),
                    "page_idx": page_idx,
                }
            )
            continue
        if not content:
            continue
        item: dict[str, object] = {
            "type": block_type,
            "text": content,
            "page_idx": page_idx,
        }
        if bbox is not None:
            item["bbox"] = list(bbox)
        if block_type == "title":
            item["text_level"] = 1
        items.append(item)
    return items


def _pixel_bbox(
    value: object,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(part, (int, float)) for part in value)
    ):
        return None
    x1, y1, x2, y2 = (float(part) for part in value)
    if not all(0 <= part <= 1 for part in (x1, y1, x2, y2)):
        return None
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return left, top, right, bottom


def _looks_like_markdown_table(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return len(lines) >= 2 and sum("|" in line for line in lines[:4]) >= 2
