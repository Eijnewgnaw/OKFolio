#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path

from okfolio.agentwiki.agent_contracts import AgentPolicy
from okfolio.agentwiki.agentic import AgentCompiler
from okfolio.agentwiki.config import Settings
from okfolio.agentwiki.llm import OpenAICompatibleClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated, auditable AgentWiki corpus compilation."
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Immutable output directory name under data/agent-runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a running or failed run with the same run-id and policy.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=float(os.environ.get("AGENT_QUALITY_THRESHOLD", "0.80")),
    )
    parser.add_argument(
        "--max-recompile-attempts",
        type=int,
        default=int(os.environ.get("AGENT_MAX_RECOMPILE_ATTEMPTS", "2")),
    )
    parser.add_argument(
        "--max-component-refs",
        type=int,
        default=int(os.environ.get("AGENT_MAX_COMPONENT_REFS", "24")),
    )
    parser.add_argument(
        "--max-component-chars",
        type=int,
        default=int(os.environ.get("AGENT_MAX_COMPONENT_CHARS", "42000")),
        help="Maximum total evidence characters in one grouping decision.",
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=int(os.environ.get("AGENT_COMPILE_WORKERS", "2")),
        choices=(1, 2, 3, 4),
        help="Bounded concurrent Concept compile and quality workers.",
    )
    arguments = parser.parse_args()

    settings = Settings.from_env()
    if not settings.openai_model:
        print(
            "OPENAI_MODEL is required for model-backed compilation",
            file=sys.stderr,
        )
        return 2
    run_id = arguments.run_id.strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    if arguments.resume and not arguments.run_id.strip():
        print("--resume requires an explicit --run-id", file=sys.stderr)
        return 2
    if "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        print("run-id must be one safe directory name", file=sys.stderr)
        return 2
    policy = AgentPolicy(
        quality_threshold=arguments.quality_threshold,
        max_recompile_attempts=arguments.max_recompile_attempts,
        max_component_refs=arguments.max_component_refs,
        max_component_chars=arguments.max_component_chars,
    )
    output = settings.data_dir / "agent-runs" / run_id
    event_log = output / "events.log"
    event_lock = Lock()

    def emit(message: str) -> None:
        print(message, flush=True)
        with event_lock:
            event_log.parent.mkdir(parents=True, exist_ok=True)
            with event_log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    client = OpenAICompatibleClient(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.openai_model,
        timeout=settings.openai_timeout_seconds,
        max_attempts=settings.openai_max_attempts,
        on_event=emit,
        enable_thinking=settings.openai_enable_thinking,
        send_chat_template_kwargs=settings.openai_send_chat_template_kwargs,
        max_tokens=settings.openai_max_tokens,
        response_format=settings.openai_response_format,
    )
    summary = AgentCompiler(
        settings,
        client,
        policy=policy,
        on_event=emit,
        compile_workers=arguments.compile_workers,
    ).run(output, resume=arguments.resume)
    print(
        f"status={summary.status} output={summary.output_dir} "
        f"articles={summary.articles} refs={summary.refs} "
        f"groups={summary.groups} concepts={summary.concepts} "
        f"reviews={summary.reviews} recompiles={summary.recompiles}"
    )
    return 0 if summary.status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
