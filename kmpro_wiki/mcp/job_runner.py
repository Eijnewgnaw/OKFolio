#!/usr/bin/env python3
"""Execute a prevalidated MCP background job and persist its final state."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: job_runner.py JOB_RECORD", file=sys.stderr)
        return 2
    record_path = Path(sys.argv[1]).resolve()
    record = load(record_path)
    log_path = Path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(record.get("environment", {}))
    record["status"] = "running"
    record["started_at"] = now()
    record["runner_pid"] = os.getpid()
    save(record_path, record)
    exit_code = 0
    error = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for index, command in enumerate(record["commands"], start=1):
                log.write(
                    f"[{now()}] command {index}/{len(record['commands'])}: "
                    + " ".join(command)
                    + "\n"
                )
                log.flush()
                completed = subprocess.run(
                    command,
                    cwd=record["cwd"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                exit_code = completed.returncode
                if exit_code != 0:
                    break
    except Exception as exc:  # pragma: no cover - catastrophic runner failure
        exit_code = 70
        error = f"{type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now()}] runner error: {error}\n")
    record = load(record_path)
    record["status"] = "complete" if exit_code == 0 else "failed"
    record["finished_at"] = now()
    record["exit_code"] = exit_code
    if error:
        record["error"] = error
    save(record_path, record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
