from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _first(values: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name, "").strip()
        if value:
            return value
    return ""


def openai_base_url(values: Mapping[str, str] | None = None) -> str:
    """Return the configured OpenAI-compatible base URL."""
    source = os.environ if values is None else values
    return (_first(source, "OPENAI_BASE_URL", "LLM_API_BASE") or DEFAULT_OPENAI_BASE_URL).rstrip(
        "/"
    )


def openai_api_key(values: Mapping[str, str] | None = None) -> str:
    source = os.environ if values is None else values
    return _first(source, "OPENAI_API_KEY", "LLM_API_KEY")


def openai_model(values: Mapping[str, str] | None = None) -> str:
    source = os.environ if values is None else values
    return _first(source, "OPENAI_MODEL", "LLM_MODEL")


def provider_base_url(
    provider: str, values: Mapping[str, str] | None = None
) -> str:
    """Resolve an optional role-specific endpoint before the shared endpoint."""
    source = os.environ if values is None else values
    prefix = provider.upper()
    return (
        _first(source, f"{prefix}_BASE_URL", f"{prefix}_API_BASE")
        or openai_base_url(source)
    ).rstrip("/")


def provider_api_key(
    provider: str, values: Mapping[str, str] | None = None
) -> str:
    source = os.environ if values is None else values
    prefix = provider.upper()
    return _first(source, f"{prefix}_API_KEY") or openai_api_key(source)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    prompts_dir: Path
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: float = 900.0
    openai_max_attempts: int = 2
    openai_enable_thinking: bool = False
    openai_send_chat_template_kwargs: bool = False
    openai_max_tokens: int = 32768
    openai_response_format: str = "json_object"
    source_dir_override: Path | None = None

    @property
    def sources_dir(self) -> Path:
        return self.source_dir_override or self.data_dir / "sources"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        return cls(
            data_dir=Path(values.get("DATA_DIR", "/app/runtime/data")),
            prompts_dir=Path(values.get("PROMPTS_DIR", "/app/prompts")),
            openai_base_url=openai_base_url(values),
            openai_api_key=openai_api_key(values),
            openai_model=openai_model(values),
            openai_timeout_seconds=float(
                _first(values, "OPENAI_TIMEOUT_SECONDS") or "900"
            ),
            openai_max_attempts=int(
                _first(values, "OPENAI_MAX_ATTEMPTS") or "2"
            ),
            openai_enable_thinking=_as_bool(
                _first(values, "OPENAI_ENABLE_THINKING") or "false"
            ),
            openai_send_chat_template_kwargs=_as_bool(
                _first(
                    values,
                    "OPENAI_SEND_CHAT_TEMPLATE_KWARGS",
                    "LLM_SEND_CHAT_TEMPLATE_KWARGS",
                )
                or "false"
            ),
            openai_max_tokens=int(
                _first(values, "OPENAI_MAX_TOKENS") or "32768"
            ),
            openai_response_format=(
                _first(values, "OPENAI_RESPONSE_FORMAT") or "json_object"
            ),
            source_dir_override=(
                Path(values["SOURCES_DIR"])
                if values.get("SOURCES_DIR", "").strip()
                else None
            ),
        )
