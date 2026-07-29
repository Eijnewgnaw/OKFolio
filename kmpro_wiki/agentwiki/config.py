from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    prompts_dir: Path
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float = 900.0
    llm_max_attempts: int = 2
    llm_enable_thinking: bool = False
    llm_max_tokens: int = 32768
    llm_response_format: str = "json_object"
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
            llm_api_base=values.get("LLM_API_BASE", "").rstrip("/"),
            llm_api_key=values.get("LLM_API_KEY", ""),
            llm_model=values.get("LLM_MODEL", ""),
            llm_timeout_seconds=float(values.get("LLM_TIMEOUT_SECONDS", "900")),
            llm_max_attempts=int(values.get("LLM_MAX_ATTEMPTS", "2")),
            llm_enable_thinking=_as_bool(
                values.get("LLM_ENABLE_THINKING", "false")
            ),
            llm_max_tokens=int(values.get("LLM_MAX_TOKENS", "32768")),
            llm_response_format=values.get(
                "LLM_RESPONSE_FORMAT", "json_object"
            ).strip(),
            source_dir_override=(
                Path(values["SOURCES_DIR"])
                if values.get("SOURCES_DIR", "").strip()
                else None
            ),
        )
