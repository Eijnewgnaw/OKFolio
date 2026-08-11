#!/usr/bin/env python3
"""Run evidence-led Claim Review against an immutable AgentWiki run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okfolio.agentwiki.claim_review_run import (
    ClaimReviewRunError,
    ClaimReviewStageClients,
    ClaimReviewTemplates,
    run_claim_review,
)
from okfolio.agentwiki.llm import OpenAICompatibleClient


def _stage_clients(arguments: argparse.Namespace, make_client) -> object:
    """Build the completion client from CLI flags (kept importable for tests).

    ``--evidence-stage-thinking`` and ``--contract-stage-thinking`` route the
    stages with thinking enabled for the chosen evidence stages.  Plain mode
    still honours ``--coverage-max-tokens`` by routing all four stages with
    thinking disabled (so Coverage gets its own output budget without any
    thinking trace); without it a single plain client is used.
    """
    if arguments.evidence_stage_thinking:
        return ClaimReviewStageClients(
            contract=make_client(enable_thinking=True),
            coverage=make_client(
                enable_thinking=True,
                max_tokens=arguments.coverage_max_tokens,
            ),
            compile=make_client(enable_thinking=False),
            recompile=make_client(enable_thinking=False),
        )
    if arguments.contract_stage_thinking:
        return ClaimReviewStageClients(
            contract=make_client(enable_thinking=True),
            coverage=make_client(
                enable_thinking=False,
                max_tokens=arguments.coverage_max_tokens,
            ),
            compile=make_client(enable_thinking=False),
            recompile=make_client(enable_thinking=False),
        )
    if arguments.coverage_max_tokens is not None:
        return ClaimReviewStageClients(
            contract=make_client(enable_thinking=False),
            coverage=make_client(
                enable_thinking=False,
                max_tokens=arguments.coverage_max_tokens,
            ),
            compile=make_client(enable_thinking=False),
            recompile=make_client(enable_thinking=False),
        )
    return make_client(enable_thinking=arguments.enable_thinking)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an AgentWiki run, build Ref-only claim contracts, audit "
            "draft coverage, and write an independent reviewed run."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--seed-run",
        type=Path,
        help=(
            "Import matching validated pass or failed-recompile checkpoints "
            "from the same frozen source snapshot into a new output run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path)
    parser.add_argument(
        "--contract-prompt",
        type=Path,
        default=ROOT / "prompts" / "agent_claim_contract.md",
    )
    parser.add_argument(
        "--coverage-prompt",
        type=Path,
        default=ROOT / "prompts" / "agent_claim_coverage.md",
    )
    parser.add_argument(
        "--compile-prompt",
        type=Path,
        default=ROOT / "prompts" / "compile.md",
    )
    parser.add_argument(
        "--recompile-prompt",
        type=Path,
        default=ROOT / "prompts" / "agent_recompile.md",
    )
    parser.add_argument(
        "--draft-override-dir",
        type=Path,
        help=(
            "Directory of manually repaired drafts for a repair run. Each "
            "*.json file is named <group_id>.json and holds "
            "{\"title\", \"description\", \"body\"}; the override replaces "
            "the inherited/source draft for that group and is frozen in the "
            "run configuration. Never modifies the source run."
        ),
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--coverage-max-tokens",
        type=int,
        help=(
            "Optional output budget used only for Coverage. Defaults to "
            "--max-tokens."
        ),
    )
    parser.add_argument("--max-recompile-attempts", type=int, default=2)
    parser.add_argument(
        "--coverage-batch-size",
        type=int,
        default=12,
        help=(
            "Audit Coverage in deterministic contiguous sentence batches of "
            "at most this many sentences per model call. Must be at least 1. "
            "Defaults to 12."
        ),
    )
    parser.add_argument(
        "--send-chat-template-kwargs",
        action="store_true",
        help=(
            "Send chat_template_kwargs to compatible local runtimes. "
            "Disabled by default for standard OpenAI-compatible APIs."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Set chat_template_kwargs.enable_thinking=true when template kwargs "
            "are enabled. Disabled by default."
        ),
    )
    parser.add_argument(
        "--evidence-stage-thinking",
        action="store_true",
        help=(
            "Enable thinking only for Claim Contract and Coverage; keep "
            "Compile and Recompile bounded. Requires "
            "--send-chat-template-kwargs and cannot be combined with "
            "--enable-thinking."
        ),
    )
    parser.add_argument(
        "--contract-stage-thinking",
        action="store_true",
        help=(
            "Enable thinking only for Claim Contract. Coverage, Compile and "
            "Recompile remain bounded. Requires --send-chat-template-kwargs."
        ),
    )
    parser.add_argument(
        "--known-source-anomaly",
        action="append",
        default=[],
        help=(
            "Exact source-text anomaly to exclude from model-visible evidence; "
            "repeatable. Defaults to an empty vocabulary."
        ),
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="Probe one or more groups; repeatable and requires --allow-partial.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Probe mode: explicitly permit provenance fallback or unresolved "
            "groups. Formal mode is the default and requires every group to pass."
        ),
    )
    arguments = parser.parse_args()
    if arguments.coverage_batch_size < 1:
        parser.error("--coverage-batch-size must be at least 1")
    if arguments.evidence_stage_thinking and not arguments.send_chat_template_kwargs:
        parser.error("--evidence-stage-thinking requires --send-chat-template-kwargs")
    if arguments.evidence_stage_thinking and arguments.enable_thinking:
        parser.error(
            "--evidence-stage-thinking cannot be combined with --enable-thinking"
        )
    if arguments.contract_stage_thinking and not arguments.send_chat_template_kwargs:
        parser.error("--contract-stage-thinking requires --send-chat-template-kwargs")
    if arguments.contract_stage_thinking and (
        arguments.enable_thinking or arguments.evidence_stage_thinking
    ):
        parser.error(
            "--contract-stage-thinking cannot be combined with "
            "--enable-thinking or --evidence-stage-thinking"
        )
    if (
        arguments.coverage_max_tokens is not None
        and not arguments.evidence_stage_thinking
        and not arguments.contract_stage_thinking
        and arguments.enable_thinking
    ):
        parser.error(
            "--enable-thinking cannot be combined with --coverage-max-tokens "
            "(plain mode routes all stages with thinking disabled)"
        )

    templates = ClaimReviewTemplates(
        contract=arguments.contract_prompt.read_text(encoding="utf-8"),
        coverage=arguments.coverage_prompt.read_text(encoding="utf-8"),
        compile=arguments.compile_prompt.read_text(encoding="utf-8"),
        recompile=arguments.recompile_prompt.read_text(encoding="utf-8"),
    )
    draft_overrides: dict[str, dict[str, str]] = {}
    if arguments.draft_override_dir is not None:
        if not arguments.draft_override_dir.is_dir():
            parser.error(
                f"--draft-override-dir is not a directory: {arguments.draft_override_dir}"
            )
        for path in sorted(arguments.draft_override_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                parser.error(f"cannot read draft override {path.name}: {error}")
            draft_overrides[path.stem] = payload
    def on_event(message: str) -> None:
        print(message, flush=True)
        if arguments.output_dir.is_dir():
            with (arguments.output_dir / "events.log").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(message + "\n")

    def make_client(
        *, enable_thinking: bool, max_tokens: int | None = None
    ) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            arguments.api_base,
            os.environ.get(arguments.api_key_env, ""),
            arguments.model,
            timeout=arguments.timeout,
            max_tokens=max_tokens or arguments.max_tokens,
            response_format=arguments.response_format,
            on_event=on_event,
            send_chat_template_kwargs=arguments.send_chat_template_kwargs,
            enable_thinking=enable_thinking,
        )

    client = _stage_clients(arguments, make_client)
    try:
        result = run_claim_review(
            client,
            source_run=arguments.source_run,
            output_dir=arguments.output_dir,
            structures_dir=arguments.structures_dir,
            templates=templates,
            resume=arguments.resume,
            allow_partial=arguments.allow_partial,
            max_recompile_attempts=arguments.max_recompile_attempts,
            selected_group_ids=arguments.group_id,
            known_source_anomalies=arguments.known_source_anomaly,
            seed_run=arguments.seed_run,
            coverage_batch_size=arguments.coverage_batch_size,
            draft_overrides=draft_overrides,
        )
    except ClaimReviewRunError as error:
        print(
            json.dumps(
                {"status": "failed", "error": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
