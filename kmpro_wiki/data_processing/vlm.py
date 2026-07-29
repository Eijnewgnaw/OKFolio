"""OpenAI-compatible document VLM provider used by the PDF worker."""
from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_PAGE_PROMPT = """\
你是文档解析引擎。请忠实识别这一个 PDF 页面，不要总结、解释或补充。

要求：
1. 按正常阅读顺序输出页面全部可见正文。
2. 标题使用 Markdown 标题。
3. 表格尽可能使用 Markdown 表格；无法可靠还原时逐行保留单元格文字。
4. 保留数字、日期、文件名、机构名和政策名称。
5. 不要使用代码围栏，不要输出“识别结果”等前缀。
"""


class PageParseError(RuntimeError):
    """The remote VLM did not return a usable page parse."""


@dataclass(frozen=True)
class PageParseResult:
    content: str
    model: str
    finish_reason: str
    elapsed_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    blocks: tuple[dict[str, Any], ...] = ()
    request_count: int | None = 1

    def to_dict(self) -> dict[str, Any]:
        value = {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }
        if self.blocks:
            value["blocks"] = list(self.blocks)
        return value


def _strip_code_fence(value: str) -> str:
    content = value.strip()
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class OpenAICompatiblePageParser:
    """Parse one rendered PDF page through an OpenAI-compatible VLM."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        prompt: str = DEFAULT_PAGE_PROMPT,
        timeout: float = 180.0,
        max_tokens: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_base.strip():
            raise ValueError("MinerU API base must not be empty")
        if not api_key.strip():
            raise ValueError("MinerU API key must not be empty")
        if not model.strip():
            raise ValueError("MinerU model must not be empty")
        if max_tokens < 128:
            raise ValueError("MinerU max_tokens must be at least 128")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.prompt = prompt.strip()
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.transport = transport

    def parse_image(
        self,
        image_path: Path,
        *,
        max_tokens: int | None = None,
    ) -> PageParseResult:
        image = image_path.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"page image not found: {image}")
        media_type = mimetypes.guess_type(image.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        try:
            with httpx.Client(
                timeout=self.timeout,
                trust_env=False,
                transport=self.transport,
            ) as client:
                response = client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            raw = message.get("content")
            if not isinstance(raw, str) or not raw.strip():
                raw = message.get("reasoning_content") or message.get("reasoning")
            if not isinstance(raw, str) or not raw.strip():
                raise PageParseError("MinerU model returned empty page content")
            usage = body.get("usage") or {}
            finish_reason = str(choice.get("finish_reason") or "unknown")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise PageParseError(
                f"MinerU model returned an invalid response: {type(error).__name__}"
            ) from error
        return PageParseResult(
            content=_strip_code_fence(raw),
            model=str(body.get("model") or self.model),
            finish_reason=finish_reason,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
