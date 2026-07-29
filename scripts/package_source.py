#!/usr/bin/env python3
"""Build a data-free OKFolio source distribution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
DIRECTORIES = ("kmpro_wiki", "scripts", "prompts", "tests")
ROOT_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docker-compose.yml",
    "README.md",
    "requirements.lock",
    "mkdocs.yml",
    "pytest.ini",
)
DOCUMENTS = (
    "docs/design/modular-architecture.md",
    "docs/operations/mcp-server.md",
    "docs/operations/mcp-capability-release.md",
    "docs/research/2026-07-28-超大PDF处理前沿成熟方案调研.md",
)
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
FORBIDDEN_PARTS = {
    "sources",
    "concepts",
    "provenance",
    "agent-runs",
    "system-experiment",
    "artifacts",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"__pycache__", ".pytest_cache", ".DS_Store"}
        or name.endswith((".pyc", ".pyo"))
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(project: Path, staging: Path) -> None:
    for relative in DIRECTORIES:
        source = project / relative
        if source.is_dir():
            shutil.copytree(source, staging / relative, ignore=_ignore)
    for relative in ROOT_FILES + DOCUMENTS:
        source = project / relative
        if not source.is_file():
            continue
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _module_manifest() -> dict[str, object]:
    return {
        "version": VERSION,
        "contains_data": False,
        "modules": {
            "data_processing": {
                "path": "kmpro_wiki/data_processing",
                "entrypoints": [
                    "scripts/process_pdf.py",
                    "scripts/process_pdf_corpus.py",
                ],
                "outputs": [
                    "document-ir.json",
                    "normalized-document-ir.json",
                    "segments.json",
                    "document-structure.json",
                    "asset-manifest.json",
                    "article markdown",
                ],
            },
            "agentwiki": {
                "path": "kmpro_wiki/agentwiki",
                "implementation": "kmpro_wiki/agentwiki",
                "entrypoint": "scripts/agent_compile.py",
            },
            "mcp": {
                "path": "kmpro_wiki/mcp",
                "implementation": "kmpro_wiki/mcp",
                "entrypoint": "python -m kmpro_wiki.mcp.server",
            },
        },
    }


def _source_tokens(source_dirs: Iterable[Path]) -> set[str]:
    tokens: set[str] = set()
    for directory in source_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.md"):
            tokens.add(path.name)
            if len(path.stem) >= 8:
                tokens.add(path.stem)
    return tokens


def audit(root: Path, *, source_tokens: set[str]) -> dict[str, object]:
    required = {
        "kmpro_wiki/data_processing/pipeline.py",
        "kmpro_wiki/data_processing/pdf_worker.py",
        "kmpro_wiki/data_processing/activation.py",
        "kmpro_wiki/data_processing/mineru_official.py",
        "kmpro_wiki/data_processing/page_role.py",
        "kmpro_wiki/data_processing/s3.py",
        "kmpro_wiki/data_processing/structure.py",
        "kmpro_wiki/data_processing/vlm.py",
        "kmpro_wiki/__init__.py",
        "kmpro_wiki/agentwiki/__init__.py",
        "kmpro_wiki/mcp/__init__.py",
        "kmpro_wiki/mcp/server.py",
        "kmpro_wiki/mcp/job_runner.py",
        "kmpro_wiki/agentwiki/agentic.py",
        "kmpro_wiki/mcp/service.py",
        "scripts/normalize_pdf_corpus.py",
        "scripts/resolve_page_roles.py",
    }
    files = [path for path in root.rglob("*") if path.is_file()]
    relative_files = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(required - relative_files)
    if missing:
        raise ValueError("missing modular files: " + ", ".join(missing))
    for path in files:
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden source/data file: {relative}")
        if set(relative.parts) & FORBIDDEN_PARTS:
            raise ValueError(f"forbidden data/result path: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {
            "",
            ".cfg",
            ".ini",
            ".json",
            ".md",
            ".py",
            ".sh",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in source_tokens:
            if token and token in text:
                raise ValueError(
                    f"{relative} contains private source filename token"
                )
    runtime_files = [
        path
        for path in (root / "runtime").rglob("*")
        if path.is_file()
    ]
    if runtime_files:
        raise ValueError("runtime directory is not empty")
    return {
        "status": "pass",
        "version": VERSION,
        "files": len(files),
        "contains_original_documents": False,
        "contains_generated_knowledge_assets": False,
        "runtime_files": 0,
        "required_files": len(required),
    }


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "MANIFEST.sha256":
            continue
        lines.append(f"{_sha256(path)}  {relative}")
    (root / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build(
    project: Path,
    output: Path,
    *,
    replace: bool = False,
) -> dict[str, object]:
    if output.exists() and not replace:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="kmpro-source-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        _copy(project, staging)
        (staging / "runtime/data").mkdir(parents=True)
        (staging / "runtime/releases").mkdir(parents=True)
        (staging / "VERSION").write_text(VERSION + "\n", encoding="utf-8")
        (staging / "module-manifest.json").write_text(
            json.dumps(_module_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tokens = _source_tokens(
            (
                project / "runtime/data/normalized-sources",
                project / "runtime/data/sources",
            )
        )
        acceptance = audit(staging, source_tokens=tokens)
        (staging / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        backup: Path | None = None
        if output.exists():
            if (output / "VERSION").read_text(encoding="utf-8").strip() != VERSION:
                raise ValueError("refusing to replace a different version")
            backup = output.with_name(f".{output.name}.replace-backup")
            if backup.exists():
                raise FileExistsError(f"stale replacement backup exists: {backup}")
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a data-free OKFolio source distribution"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"../okfolio-source-{VERSION}"),
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = build(project, args.output.resolve(), replace=args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
