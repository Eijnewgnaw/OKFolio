#!/usr/bin/env python3
"""Thin CLI adapter for the reproducible public four-stage experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from kmpro_wiki.agentwiki.local_experiment import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local-experiment"),
        help="Ignored local output directory.",
    )
    args = parser.parse_args()
    print(run(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
