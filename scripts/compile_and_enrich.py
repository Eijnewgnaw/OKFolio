#!/usr/bin/env python3
from __future__ import annotations

import sys

from kmpro_wiki.agentwiki.compiler import CompilationBatchError, Compiler
from kmpro_wiki.agentwiki.config import Settings
from kmpro_wiki.agentwiki.indexer import append_log
from kmpro_wiki.agentwiki.llm import LLMClient


def main() -> int:
    settings = Settings.from_env()
    if not settings.llm_api_base or not settings.llm_api_key or not settings.llm_model:
        print("LLM_API_BASE, LLM_API_KEY and LLM_MODEL are required", file=sys.stderr)
        return 2
    client = LLMClient(
        settings.llm_api_base,
        settings.llm_api_key,
        settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        on_event=print,
        enable_thinking=settings.llm_enable_thinking,
        max_tokens=settings.llm_max_tokens,
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
