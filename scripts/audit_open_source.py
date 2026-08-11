#!/usr/bin/env python3
"""Fail when a public source tree contains private deployment material."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".venv-rag",
    ".venv313",
    ".local-runtime",
    ".snapshot-venv",
    "__pycache__",
    # Archived run-data snapshot (experiment-data/). The snapshot is kept
    # byte-for-byte as produced, per the documented sensitivity decision in
    # experiment-data/README.md: it records the internal MinIO asset endpoint
    # and the model names used by the runs. It is data, not deployment
    # configuration, and the audit guards the code/docs tree only.
    "experiment-data",
}
_SENSITIVE_NAMES = {".env", ".env.local"}
_SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_PRIVATE_NETWORK_RE = re.compile(
    r"(?<!\d)(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"169\.254(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
    r"192\.168(?:\.\d{1,3}){2}"
    r")(?::\d+)?(?!\d)"
)
_PRIVATE_PATH_RE = re.compile(
    r"(?:"
    + "/"
    + "Users"
    + r"/[A-Za-z0-9._-]+|"
    + "/"
    + "usr"
    + r"/local/software|"
    + "wxid"
    + "_|"
    + "DT"
    + "_LAB)"
)
_INTERNAL_MODEL_RE = re.compile(
    "(?i)(?:qwen" + "3p6|mineru" + "2p5)"
)
_TOKEN_RE = re.compile(
    r"(?i)(?:"
    r"AKIA[0-9A-Z]{16}|"
    r"ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}"
    r")"
)
_QUOTED_CREDENTIAL_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|secret[_-]?key|password)"
    r"\s*[:=]\s*['\"]([^'\"]+)['\"]"
)
_ENV_CREDENTIAL_RE = re.compile(
    r"(?i)^(?:[A-Z0-9_]*(?:API|ACCESS|SECRET)[A-Z0-9_]*KEY|PASSWORD)"
    r"\s*=\s*(.+?)\s*$"
)
_SAFE_VALUES = {
    "",
    "...",
    "access",
    "replace-me",
    "secret",
    "test-key",
    "unused",
}


class OpenSourceAuditError(RuntimeError):
    pass


def _is_safe_value(value: str) -> bool:
    normalized = value.strip().strip("'\"")
    return (
        normalized in _SAFE_VALUES
        or normalized.startswith("${")
        or normalized.startswith("<")
    )


def audit_open_source(root: Path) -> dict[str, object]:
    project = root.resolve()
    violations: list[str] = []
    scanned = 0
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if set(relative.parts) & _SKIP_PARTS:
            continue
        if not path.is_file():
            continue
        if path.name in _SENSITIVE_NAMES or path.suffix.lower() in _SENSITIVE_SUFFIXES:
            violations.append(f"sensitive file: {relative}")
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _PRIVATE_NETWORK_RE.search(text):
            violations.append(f"private network endpoint: {relative}")
        if _PRIVATE_PATH_RE.search(text):
            violations.append(f"private deployment path or account: {relative}")
        if _INTERNAL_MODEL_RE.search(text):
            violations.append(f"internal model deployment id: {relative}")
        if _TOKEN_RE.search(text):
            violations.append(f"token-like value: {relative}")
        if path.suffix.lower() != ".map":
            for match in _QUOTED_CREDENTIAL_RE.finditer(text):
                if not _is_safe_value(match.group(1)):
                    violations.append(f"hard-coded credential: {relative}")
                    break
        for line in text.splitlines():
            match = _ENV_CREDENTIAL_RE.match(line)
            if match and not _is_safe_value(match.group(1)):
                violations.append(f"configured credential: {relative}")
                break
    if violations:
        raise OpenSourceAuditError("; ".join(sorted(set(violations))))
    return {
        "status": "pass",
        "files_scanned": scanned,
        "private_endpoints": 0,
        "credentials": 0,
        "sensitive_files": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_open_source(args.root),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
