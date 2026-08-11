from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx


class LLMError(RuntimeError):
    pass


class LLMOutputTruncated(LLMError):
    """The provider stopped generation because the output token limit was hit."""

    def __init__(
        self,
        finish_reason: str,
        *,
        prompt_tokens: object = None,
        completion_tokens: object = None,
        total_tokens: object = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        super().__init__(
            "LLM output truncated "
            f"(finish_reason={finish_reason}, "
            f"completion_tokens={completion_tokens})"
        )


def _finish_reason(choice: dict[str, object]) -> str:
    value = choice.get("finish_reason")
    if value is None:
        value = choice.get("finishReason")
    return str(value) if value is not None else "unknown"


def _is_truncated_finish_reason(finish_reason: str) -> bool:
    normalized = "".join(
        character
        for character in finish_reason.casefold()
        if character.isalnum()
    )
    return normalized in {
        "length",
        "maxtokens",
        "maxtokensreached",
        "maxpredictedtokensreached",
    }


def render_compile_prompt(
    template: str, title: str, source_file: str, content: str
) -> str:
    return (
        template.replace("{title}", title)
        .replace("{source_file}", source_file)
        .replace("{content}", content)
    )


def render_enrich_prompt(template: str, content: str, index: str) -> str:
    return template.replace("{content}", content).replace("{index}", index)


class OpenAICompatibleClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 180.0,
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        on_event: Callable[[str], None] | None = None,
        enable_thinking: bool = False,
        send_chat_template_kwargs: bool = False,
        max_tokens: int = 32768,
        response_format: str = "json_object",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.on_event = on_event or (lambda _message: None)
        self.enable_thinking = enable_thinking
        self.send_chat_template_kwargs = send_chat_template_kwargs
        self.max_tokens = max_tokens
        if response_format not in {"json_object", "json_schema", "none"}:
            raise ValueError(
                "response_format must be json_object, json_schema, or none"
            )
        self.response_format = response_format

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        started = time.monotonic()
        url = f"{self.api_base}/chat/completions"
        if (json_schema_name is None) != (json_schema is None):
            raise ValueError(
                "json_schema_name and json_schema must be provided together"
            )
        request_prompt = prompt
        if (
            json_schema_name is not None
            and json_schema is not None
            and self.response_format != "json_schema"
        ):
            request_prompt = (
                f"{prompt}\n\n## 强制结构化输出约束\n\n"
                "只输出一个 JSON 对象，不要输出 Markdown 代码围栏或解释文字。"
                "该对象必须满足以下 JSON Schema；返回后还会由代码再次校验：\n"
                f"{json.dumps(json_schema, ensure_ascii=False, separators=(',', ':'))}"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request_prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.send_chat_template_kwargs:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if json_schema_name is not None and json_schema is not None:
            if self.response_format == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema_name,
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            elif self.response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}
        # OpenAI-compatible local runtimes commonly run without authentication.
        # Omit the header entirely when no key is configured; remote providers
        # can still enforce authentication and return their normal 401 response.
        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else {}
        )

        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                except httpx.RequestError as error:
                    if attempt == self.max_attempts:
                        raise LLMError(
                            f"LLM request failed after {attempt} attempts: {type(error).__name__}"
                        ) from error
                    self.on_event(
                        f"llm.retry attempt={attempt}/{self.max_attempts} "
                        f"reason={type(error).__name__}"
                    )
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_attempts:
                        self.on_event(
                            f"llm.retry attempt={attempt}/{self.max_attempts} "
                            f"reason=HTTP{response.status_code}"
                        )
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                        continue
                    raise LLMError(
                        f"LLM request failed after {attempt} attempts: HTTP {response.status_code}"
                    )

                try:
                    response.raise_for_status()
                    response_data = response.json()
                    choice = response_data["choices"][0]
                    message = choice["message"]
                    raw_content = message.get("content")
                    raw_reasoning = message.get("reasoning_content") or message.get(
                        "reasoning"
                    )
                    if isinstance(raw_content, str) and raw_content.strip():
                        content = raw_content
                        selected_field = "content"
                    else:
                        content = raw_reasoning
                        selected_field = (
                            "reasoning_content"
                            if message.get("reasoning_content") is not None
                            else "reasoning"
                        )
                except httpx.HTTPStatusError as error:
                    raise LLMError(
                        f"LLM request failed: HTTP {error.response.status_code}"
                    ) from error
                except (
                    httpx.HTTPError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise LLMError("LLM returned an invalid response") from error

                usage = response_data.get("usage") or {}
                elapsed_ms = round((time.monotonic() - started) * 1000)
                content_chars = (
                    len(raw_content) if isinstance(raw_content, str) else 0
                )
                reasoning_chars = (
                    len(raw_reasoning) if isinstance(raw_reasoning, str) else 0
                )
                finish_reason = _finish_reason(choice)
                event_fields = (
                    f"schema={json_schema_name or 'none'} "
                    f"elapsed_ms={elapsed_ms} "
                    f"thinking={str(self.enable_thinking).lower()} "
                    f"format={self.response_format} "
                    f"prompt_chars={len(request_prompt)} content_chars={content_chars} "
                    f"reasoning_chars={reasoning_chars} selected={selected_field} "
                    f"finish={finish_reason} "
                    f"prompt_tokens={usage.get('prompt_tokens', 'unknown')} "
                    f"completion_tokens={usage.get('completion_tokens', 'unknown')} "
                    f"total_tokens={usage.get('total_tokens', 'unknown')}"
                )
                if _is_truncated_finish_reason(finish_reason):
                    # Keep the physical-call telemetry compatible with existing
                    # run reports, then stop before any caller can parse or
                    # contract-repair the partial payload.
                    self.on_event(f"llm.done {event_fields}")
                    raise LLMOutputTruncated(
                        finish_reason,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                    )
                if not isinstance(content, str) or not content.strip():
                    raise LLMError("LLM returned empty content")
                self.on_event(f"llm.done {event_fields}")
                return content.strip()

        raise LLMError("LLM request exhausted without a response")


# Kept as a small source-compatible alias for integrations built against 0.1.x.
LLMClient = OpenAICompatibleClient
