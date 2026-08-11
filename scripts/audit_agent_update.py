#!/usr/bin/env python3
"""Audit R0/R1 AgentWiki update semantics, provenance and cache reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from okfolio.agentwiki.update_runs import audit_update_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit an immutable AgentWiki document update."
    )
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("update_run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_update_run(args.baseline_run, args.update_run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
