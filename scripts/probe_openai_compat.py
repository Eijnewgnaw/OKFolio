#!/usr/bin/env python3
"""Make one minimal OpenAI-compatible chat request and report only metadata.

The key is read from the environment and never printed.  This intentionally
does not invoke AgentWiki or send any source document.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kmpro_wiki.agentwiki.llm import LLMError, OpenAICompatibleClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.base_url or not args.model:
        print("probe=skipped missing OPENAI_BASE_URL or OPENAI_MODEL")
        return 2
    events: list[str] = []
    client = OpenAICompatibleClient(
        args.base_url,
        api_key,
        args.model,
        timeout=args.timeout,
        max_attempts=1,
        retry_delay=0,
        on_event=events.append,
        max_tokens=args.max_tokens,
    )
    try:
        result = client.complete("Reply with the single word OK.")
    except LLMError as error:
        print(f"probe=failed error={error}")
        return 1
    print(
        f"probe=ok endpoint=configured model={args.model} "
        f"response_chars={len(result)}"
    )
    if events:
        print(events[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
