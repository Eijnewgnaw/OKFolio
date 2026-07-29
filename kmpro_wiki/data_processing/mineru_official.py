"""Official MinerU 2.5 two-step HTTP parser adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image
from mineru_vl_utils import MinerUClient

from .vlm import PageParseError, PageParseResult


def _server_url(api_base: str) -> str:
    """Preserve a gateway path while letting MinerU append /v1."""
    value = api_base.strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3].rstrip("/")
    return value + "/"


def _block_markdown(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "text")
    content = str(block.get("content") or "").strip()
    if not content:
        return ""
    if block_type == "title":
        return f"## {content}"
    if block_type == "table":
        return content
    if block_type in {"equation", "equation_block"}:
        return f"$$\n{content}\n$$"
    if block_type == "list_item":
        return f"- {content}"
    return content


class OfficialMinerUPageParser:
    """Run the official layout-detect then block-recognition protocol."""

    parser_name = "mineru-vl-utils-two-step"

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        max_concurrency: int = 8,
    ) -> None:
        if not api_base.strip():
            raise ValueError("MinerU API base must not be empty")
        if not api_key.strip():
            raise ValueError("MinerU API key must not be empty")
        if not model.strip():
            raise ValueError("MinerU model must not be empty")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = 4096
        self.client = MinerUClient(
            backend="http-client",
            server_url=_server_url(api_base),
            server_headers={"Authorization": f"Bearer {api_key}"},
            model_name=model,
            http_timeout=max(1, round(timeout)),
            connect_timeout=min(10, max(1, round(timeout))),
            max_concurrency=max_concurrency,
            max_connections=max_concurrency,
            max_keepalive_connections=max_concurrency,
            use_tqdm=False,
            skip_model_name_checking=True,
            abandon_paratext=True,
            image_analysis=False,
        )

    def parse_image(
        self,
        image_path: Path,
        *,
        max_tokens: int | None = None,
    ) -> PageParseResult:
        del max_tokens  # Official per-task sampling parameters own token limits.
        image = image_path.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"page image not found: {image}")
        started = time.monotonic()
        try:
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
                extracted = self.client.two_step_extract(
                    rgb,
                    image_analysis=False,
                )
        except Exception as error:
            raise PageParseError(
                "official MinerU two-step parsing failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        blocks = tuple(
            {
                "type": str(block.get("type") or "text"),
                "bbox": [float(value) for value in block.get("bbox", [])],
                "angle": block.get("angle"),
                "content": block.get("content"),
                "merge_prev": bool(block.get("merge_prev", False)),
            }
            for block in extracted
        )
        content = "\n\n".join(
            part for part in (_block_markdown(block) for block in blocks) if part
        )
        finish_reason = "two_step_complete"
        if not content and not any(
            block["type"] in {"image", "chart"} for block in blocks
        ):
            # A cover, separator or truly blank page can legitimately have no
            # detected layout. Preserve the rendered page as visual evidence
            # instead of failing the entire book or inventing OCR text.
            blocks = (
                {
                    "type": "image",
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "angle": 0,
                    "content": None,
                    "merge_prev": False,
                },
            )
            finish_reason = "two_step_empty_page_preserved"
        recognized = sum(
            1 for block in blocks if str(block.get("content") or "").strip()
        )
        return PageParseResult(
            content=content,
            model=self.model,
            finish_reason=finish_reason,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            blocks=blocks,
            request_count=1 + recognized,
        )
