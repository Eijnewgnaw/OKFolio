from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.audit_mcp_capability import audit_tree


REQUIRED_IMAGE_FILES = (
    "app/okfolio/mcp/server.py",
    "app/okfolio/mcp/service.py",
    "app/prompts/compile.md",
)


def _write_member(handle: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    handle.addfile(info, io.BytesIO(content))


def _write_image_tar(
    path: Path,
    *,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    layer_buffer = io.BytesIO()
    with tarfile.open(fileobj=layer_buffer, mode="w") as layer:
        for name in REQUIRED_IMAGE_FILES:
            _write_member(layer, name, b"# capability code\n")
        for name, content in (extra_files or {}).items():
            _write_member(layer, name, content)
    layer_bytes = layer_buffer.getvalue()
    manifest = json.dumps(
        [{"Config": "config.json", "RepoTags": [], "Layers": ["layer.tar"]}]
    ).encode()
    with tarfile.open(path, mode="w") as image:
        _write_member(image, "manifest.json", manifest)
        _write_member(image, "config.json", b"{}")
        _write_member(image, "layer.tar", layer_bytes)


def _empty_package(tmp_path: Path) -> Path:
    package = tmp_path / "capability"
    (package / "images").mkdir(parents=True)
    (package / "runtime/data").mkdir(parents=True)
    (package / "runtime/releases").mkdir(parents=True)
    _write_image_tar(package / "images/capability.tar")
    return package


def test_capability_audit_accepts_empty_runtime(tmp_path: Path) -> None:
    package = _empty_package(tmp_path)

    result = audit_tree(package)

    assert result["status"] == "pass"
    assert result["contains_original_documents"] is False
    assert result["contains_generated_knowledge_assets"] is False
    assert result["docker_image"]["runtime_data_files"] == 0


def test_capability_audit_rejects_generated_knowledge_path(
    tmp_path: Path,
) -> None:
    package = _empty_package(tmp_path)
    concepts = package / "runtime/data/concepts"
    concepts.mkdir(parents=True)
    (concepts / "leak.md").write_text("generated knowledge", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime data directory is not empty"):
        audit_tree(package)


def test_capability_audit_rejects_source_filename_fingerprint(
    tmp_path: Path,
) -> None:
    package = _empty_package(tmp_path)
    (package / "README.md").write_text(
        "accidentally mentions confidential-report-2026.md",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source filename token"):
        audit_tree(package, tokens={"confidential-report-2026.md"})


def test_capability_audit_rejects_data_inside_image(tmp_path: Path) -> None:
    package = _empty_package(tmp_path)
    _write_image_tar(
        package / "images/capability.tar",
        extra_files={"app/data/sources/confidential.md": b"secret"},
    )

    with pytest.raises(ValueError, match="contains knowledge data"):
        audit_tree(package)
