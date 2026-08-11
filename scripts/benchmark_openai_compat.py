#!/usr/bin/env python3
"""Measure an OpenAI-compatible streaming endpoint without sending documents."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from okfolio.evaluation.llm_benchmark import run_stream_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--include-usage",
        action="store_true",
        help="Request token usage when the compatible endpoint supports it.",
    )
    parser.add_argument(
        "--prompt",
        default="用一句话说明知识库检索的作用。",
        help="Small synthetic benchmark prompt; source documents are never loaded.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.base_url or not args.model:
        print("benchmark=skipped missing OPENAI_BASE_URL or OPENAI_MODEL")
        return 2

    try:
        result = run_stream_benchmark(
            api_base=args.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.model,
            prompt=args.prompt,
            requests=args.requests,
            warmup=args.warmup,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            include_usage=args.include_usage,
        )
    except (ValueError, httpx.HTTPError) as error:
        print(f"benchmark=failed error={type(error).__name__}: {error}")
        return 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"benchmark=ok output={args.output} requests={args.requests}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
