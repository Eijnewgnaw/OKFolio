#!/usr/bin/env python3
"""Seed an immutable R1 AgentWiki run from a completed R0 snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from okfolio.agentwiki.update_runs import seed_update_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed an isolated AgentWiki document-update run."
    )
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("output_run", type=Path)
    args = parser.parse_args()
    print(json.dumps(seed_update_run(args.baseline_run, args.output_run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
