#!/usr/bin/env python3
"""Audit corpus coverage before comparing traditional and Concept chunks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmpro_wiki.evaluation.corpus_readiness import audit_concept_rag_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_concept_rag_inputs(
            source_dir=args.sources,
            refs_path=args.refs,
            concepts_path=args.concepts,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"audit=failed error={type(error).__name__}: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready_for_full_comparison"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
