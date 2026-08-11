#!/usr/bin/env python3
from __future__ import annotations

import sys

from okfolio.agentwiki.compiler import CompilationBatchError, Compiler
from okfolio.agentwiki.config import Settings
from okfolio.agentwiki.indexer import append_log
from okfolio.agentwiki.llm import OpenAICompatibleClient


def main() -> int:
    settings = Settings.from_env()
    if not settings.openai_model:
        print(
            "OPENAI_MODEL is required for model-backed compilation",
            file=sys.stderr,
        )
        return 2
    client = OpenAICompatibleClient(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.openai_model,
        timeout=settings.openai_timeout_seconds,
        max_attempts=settings.openai_max_attempts,
        on_event=print,
        enable_thinking=settings.openai_enable_thinking,
        max_tokens=settings.openai_max_tokens,
    )
    compiler = Compiler(settings, client, on_event=print)
    try:
        summary = compiler.run()
    except CompilationBatchError as error:
        print(str(error), file=sys.stderr)
        return 1
    append_log(settings.data_dir / "wiki" / "log.md", summary.compiled)
    print(
        f"compiled={len(summary.compiled)} skipped={len(summary.skipped)} failed=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
