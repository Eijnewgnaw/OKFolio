"""Constrained VLM classification for MinerU visual-only pages."""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from .vlm import PageParseError, _strip_code_fence


ResolvedRole = Literal["blank", "cover", "decorative", "content_retry"]

PAGE_ROLE_PROMPT = """\
你是PDF页面角色审计器。MinerU没有从该页识别出可靠布局，请只判断页面角色。

允许的 role：
- blank：纯空白或仅有不可见扫描噪声；
- cover：书籍/报告封面、封底，主要是题名、年份、机构；
- decorative：纯装饰、无知识内容的隔页或底纹；
- content_retry：存在正文、目录、表格、图表、章节标题或任何应重新解析的信息。

宁可输出 content_retry，也不能把可能有知识内容的页面判为 decorative。
只输出严格JSON：
{"role":"blank|cover|decorative|content_retry","confidence":0.0,"reason":"一句话理由"}
"""


@dataclass(frozen=True)
class PageRoleResult:
    role: ResolvedRole
    confidence: float
    reason: str
    model: str
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_role": self.role,
            "page_role_confidence": self.confidence,
            "page_role_reason": self.reason,
            "page_role_model": self.model,
            "page_role_elapsed_ms": self.elapsed_ms,
        }


class OpenAICompatiblePageRoleClassifier:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_base.strip() or not api_key.strip() or not model.strip():
            raise ValueError("page-role classifier requires API base, key and model")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.transport = transport

    def classify(self, image_path: Path) -> PageRoleResult:
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
                        {"type": "text", "text": PAGE_ROLE_PROMPT},
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
            "max_tokens": 256,
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
            message = body["choices"][0]["message"]
            raw = message.get("content") or message.get("reasoning_content")
            if not isinstance(raw, str):
                raise ValueError("missing classifier content")
            value = json.loads(_strip_code_fence(raw))
            if set(value) != {"role", "confidence", "reason"}:
                raise ValueError("unexpected classifier fields")
            role = str(value["role"])
            if role not in {"blank", "cover", "decorative", "content_retry"}:
                raise ValueError("invalid classifier role")
            confidence = float(value["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("classifier confidence must be between 0 and 1")
            reason = str(value["reason"]).strip()
            if not reason:
                raise ValueError("classifier reason must not be empty")
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise PageParseError(
                "page-role classifier returned an invalid response: "
                f"{type(error).__name__}"
            ) from error
        return PageRoleResult(
            role=role,  # type: ignore[arg-type]
            confidence=round(confidence, 4),
            reason=reason,
            model=str(body.get("model") or self.model),
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
