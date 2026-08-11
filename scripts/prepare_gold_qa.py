#!/usr/bin/env python3
"""Prepare a deterministic human-annotation worksheet from structure sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from okfolio.evaluation.gold_sampling import (
    GoldSamplingQuota,
    prepare_gold_sampling_plan,
    write_gold_template_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample canonical evidence into blank Gold-QA annotation slots. "
            "Without --output this is a read-only dry run."
        )
    )
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--excerpt-chars", type=int, default=800)
    parser.add_argument("--single-per-book", type=int, default=3)
    parser.add_argument("--intra-per-book", type=int, default=2)
    parser.add_argument("--cross-per-book", type=int, default=1)
    parser.add_argument("--temporal-per-book", type=int, default=1)
    parser.add_argument("--unanswerable-per-book", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    quota = GoldSamplingQuota(
        single_evidence_fact=arguments.single_per_book,
        intra_document_synthesis=arguments.intra_per_book,
        cross_document_synthesis=arguments.cross_per_book,
        temporal_or_scenario=arguments.temporal_per_book,
        unanswerable=arguments.unanswerable_per_book,
    )
    plan = prepare_gold_sampling_plan(
        arguments.structures,
        quota=quota,
        seed=arguments.seed,
    )
    if arguments.output is not None:
        write_gold_template_jsonl(
            plan,
            arguments.output,
            excerpt_chars=arguments.excerpt_chars,
            overwrite=arguments.overwrite,
        )
    if arguments.audit_output is not None:
        if arguments.audit_output.exists() and not arguments.overwrite:
            raise FileExistsError(arguments.audit_output)
        arguments.audit_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.audit_output.write_text(
            json.dumps(plan.audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = dict(plan.audit)
    summary["mode"] = "write" if arguments.output is not None else "dry-run"
    summary["output"] = str(arguments.output) if arguments.output is not None else None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if plan.audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
