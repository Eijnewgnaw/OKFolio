#!/usr/bin/env python3
"""Privacy audit for the data-free OKFolio MCP capability package."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORBIDDEN_COMPONENTS = {
    "articles",
    "artifacts",
    "concepts",
    "outputs",
    "provenance",
    "sources",
    "system-experiment",
    "agent-runs",
    ".state",
    ".staging",
}
FORBIDDEN_BASENAMES = {
    "acceptance.json",
    "agent_trace.json",
    "concepts.json",
    "experiment-metrics.json",
    "graph.html",
    "groups.json",
    "ref_validation.json",
    "refs.json",
    "relation-metrics.json",
    "relations.json",
    "release-manifest.json",
}
FORBIDDEN_EXTENSIONS = {
    ".bmp",
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
}
TEXT_EXTENSIONS = {
    "",
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def source_tokens(source_dirs: Iterable[Path]) -> set[str]:
    tokens: set[str] = set()
    for directory in source_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.md"):
            tokens.add(path.name)
            if len(path.stem) >= 8:
                tokens.add(path.stem)
    return tokens


def _check_relative_path(relative: PurePosixPath) -> None:
    parts = set(relative.parts)
    if parts & FORBIDDEN_COMPONENTS:
        raise ValueError(f"forbidden data/result path: {relative}")
    if relative.name in FORBIDDEN_BASENAMES:
        raise ValueError(f"forbidden data/result file: {relative}")
    if relative.suffix.lower() in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"forbidden source asset: {relative}")


def _scan_text(label: str, content: bytes, tokens: set[str]) -> None:
    if not tokens or len(content) > 5 * 1024 * 1024:
        return
    text = content.decode("utf-8", errors="ignore")
    matched = sorted(token for token in tokens if token and token in text)
    if matched:
        raise ValueError(
            f"{label} contains source filename token: {matched[0]}"
        )


def audit_image_tar(path: Path, tokens: set[str]) -> dict[str, Any]:
    required = {
        "app/kmpro_wiki/mcp/server.py",
        "app/kmpro_wiki/mcp/service.py",
        "app/prompts/compile.md",
    }
    seen: set[str] = set()
    app_files = 0
    with tarfile.open(path, "r") as image:
        manifest_file = image.extractfile("manifest.json")
        if manifest_file is None:
            raise ValueError("Docker image archive has no manifest.json")
        manifests = json.load(manifest_file)
        if len(manifests) != 1:
            raise ValueError("Docker image archive must contain one image")
        for layer_name in manifests[0]["Layers"]:
            layer_member = image.getmember(layer_name)
            layer_file = image.extractfile(layer_member)
            if layer_file is None:
                raise ValueError(f"cannot read Docker layer: {layer_name}")
            with tarfile.open(fileobj=io.BytesIO(layer_file.read())) as layer:
                for member in layer.getmembers():
                    name = member.name.removeprefix("./").lstrip("/")
                    if not name.startswith("app/"):
                        continue
                    relative = PurePosixPath(name)
                    if name.startswith(
                        (
                            "app/data/",
                            "app/artifacts/",
                            "app/release-manifest",
                            "app/acceptance",
                            "app/MANIFEST",
                        )
                    ):
                        raise ValueError(
                            f"Docker image contains knowledge data: {name}"
                        )
                    if name.startswith("app/runtime/data/") and member.isfile():
                        raise ValueError(
                            f"Docker image runtime data is not empty: {name}"
                        )
                    if (
                        name.startswith("app/runtime/releases/")
                        and member.isfile()
                    ):
                        raise ValueError(
                            f"Docker image runtime releases are not empty: {name}"
                        )
                    if member.isfile():
                        app_files += 1
                        seen.add(name)
                        if relative.suffix.lower() in TEXT_EXTENSIONS:
                            handle = layer.extractfile(member)
                            if handle is not None:
                                _scan_text(name, handle.read(), tokens)
    missing = sorted(required - seen)
    if missing:
        raise ValueError(
            "Docker image is missing required capability files: "
            + ", ".join(missing)
        )
    return {
        "app_files": app_files,
        "runtime_data_files": 0,
        "runtime_release_files": 0,
        "required_capability_files": len(required),
    }


def audit_tree(
    root: Path,
    *,
    tokens: set[str] | None = None,
) -> dict[str, Any]:
    package = root.resolve()
    source_name_tokens = tokens or set()
    if not package.is_dir():
        raise FileNotFoundError(f"capability package not found: {package}")
    if (package / "data").exists() or (package / "artifacts").exists():
        raise ValueError("package root contains data or artifacts")
    files = [path for path in package.rglob("*") if path.is_file()]
    image_tars: list[Path] = []
    for path in files:
        relative = PurePosixPath(path.relative_to(package).as_posix())
        if relative.parts[:2] == ("runtime", "data"):
            raise ValueError(f"runtime data directory is not empty: {relative}")
        if relative.parts[:2] == ("runtime", "releases"):
            raise ValueError(
                f"runtime releases directory is not empty: {relative}"
            )
        if relative.parts[:1] == ("images",) and relative.suffix == ".tar":
            image_tars.append(path)
            continue
        _check_relative_path(relative)
        if path.suffix.lower() in TEXT_EXTENSIONS:
            _scan_text(str(relative), path.read_bytes(), source_name_tokens)
    if len(image_tars) != 1:
        raise ValueError("package must contain exactly one Docker image tar")
    image_audit = audit_image_tar(image_tars[0], source_name_tokens)
    return {
        "status": "pass",
        "contains_original_documents": False,
        "contains_generated_knowledge_assets": False,
        "package_files": len(files),
        "docker_image": image_audit,
    }


def audit_archive(
    archive: Path,
    *,
    tokens: set[str] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kmpro-mcp-audit-") as temporary:
        target = Path(temporary)
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe archive member: {member.name}")
            handle.extractall(target, filter="data")
        roots = [path for path in target.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("archive must contain one package root")
        return audit_tree(roots[0], tokens=tokens)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Audit a .tar.gz archive instead of a directory.",
    )
    parser.add_argument(
        "--forbidden-source-dir",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokens = source_tokens(args.forbidden_source_dir)
    result = (
        audit_archive(args.target, tokens=tokens)
        if args.archive
        else audit_tree(args.target, tokens=tokens)
    )
    if args.output:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
