#!/usr/bin/env python3
"""Run or inspect the frozen, provider-neutral T0/T1/C1 RAG experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmpro_wiki.evaluation.experiment_runner import (
    ExperimentStateError,
    ThreeArmExperimentRunner,
    load_backend,
    load_experiment_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Generation contract:\n"
            "  The default local backend asks the model for "
            "okfolio.rag-answer.v1 JSON and deterministically maps only "
            "context_id + page citations. It never compares the answer with "
            "Gold facts.\n\n"
            "Semantic scoring:\n"
            "  Without a human-reviewed or independent-judge alignment plugin, "
            "answer_accuracy and joint_success_rate are written as null; "
            "retrieval, refusal, and citation diagnostics remain available."
        ),
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="frozen experiment JSON config"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or resumable experiment output directory",
    )
    parser.add_argument(
        "--stage",
        choices=("readiness", "index", "retrieve", "generate", "score", "all"),
        default="readiness",
        help=(
            "readiness: no-write validation; index/retrieve: retrieval stages; "
            "generate: fixed JSON answers; score: deterministic metrics with "
            "semantic fields provisional unless independently aligned; all: run in order"
        ),
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    try:
        if args.stage == "readiness":
            runner = ThreeArmExperimentRunner(config=config, output_dir=args.output)
            result = runner.readiness()
        else:
            backend = load_backend(config, args.output)
            runner = ThreeArmExperimentRunner(
                config=config, output_dir=args.output, backend=backend
            )
            result = runner.run(args.stage)
    except ExperimentStateError as error:
        print(
            json.dumps(
                {"status": "blocked", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
