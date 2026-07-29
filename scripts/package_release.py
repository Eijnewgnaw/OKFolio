#!/usr/bin/env python3
"""Create a generic, checksummed OKFolio release archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path


EXCLUDES = {
    ".git",
    ".vscode",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
    "artifacts",
    "data",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_source(project_root: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDES}

    shutil.copytree(project_root, target, ignore=ignore)


def write_manifest(release_dir: Path) -> None:
    manifest = release_dir / "RELEASE-MANIFEST.sha256"
    lines = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file() and path != manifest:
            lines.append(
                f"{sha256(path)}  {path.relative_to(release_dir).as_posix()}"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(release_dir: Path, archive: Path) -> dict[str, str | int]:
    release = release_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    required = [
        release / "release-manifest.json",
        release / "acceptance.json",
        release / "data" / "wiki" / "index.md",
        release / "data" / "outputs" / "graph.html",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("release is incomplete: " + ", ".join(missing))
    acceptance = json.loads(
        (release / "acceptance.json").read_text(encoding="utf-8")
    )
    if acceptance.get("status") != "pass":
        raise ValueError("release acceptance status is not pass")
    copy_source(project_root, release / "source")
    for source, target in (
        (project_root / "docker-compose.yml", release / "docker-compose.yml"),
        (project_root / "Dockerfile", release / "Dockerfile"),
    ):
        if source.is_file():
            shutil.copy2(source, target)
    write_manifest(release)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        handle.add(release, arcname=release.name, recursive=True)
    os.replace(temporary, archive)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(
        f"{sha256(archive)}  {archive.name}\n",
        encoding="utf-8",
    )
    return {
        "release": str(release),
        "archive": str(archive),
        "checksum": str(checksum),
        "bytes": archive.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    result = package(args.release_dir, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
