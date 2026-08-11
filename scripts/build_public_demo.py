#!/usr/bin/env python3
"""Build a static public demo without private infrastructure metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from kmpro_wiki.agentwiki.explorer import (
    build_explorer_html,
    extract_graph_data,
)


_PRIVATE_IPV4 = (
    r"(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"127(?:\.\d{1,3}){3}|"
    r"169\.254(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
    r"192\.168(?:\.\d{1,3}){2}"
    r")"
)
_PRIVATE_URL_RE = re.compile(
    rf"https?://{_PRIVATE_IPV4}(?::\d+)?[^\s\"'<>)]*",
    re.IGNORECASE,
)
_PRIVATE_IP_RE = re.compile(rf"(?<!\d){_PRIVATE_IPV4}(?::\d+)?(?!\d)")
_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".txt",
    ".xml",
}
_PUBLIC_MANIFEST_FIELDS = (
    "articles",
    "refs",
    "concepts",
    "joint_concepts",
    "candidate_edges",
    "judged_cross_concept_edges",
    "semantic_relation_groups",
    "semantic_relation_evidence",
    "site_concept_pages",
    "site_article_pages",
    "spatial_graph",
    "provenance",
)
_EXTERNAL_HIGHLIGHT_ASSETS = {
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css": (
        "data:text/css,"
    ),
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css": (
        "data:text/css,"
    ),
    (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/'
        '11.8.0/highlight.min.js"></script>'
    ): (
        "<script>window.hljs=window.hljs||"
        "{highlightAll:function(){}};</script>"
    ),
}
_PUBLIC_BRAND_REPLACEMENTS = {
    "KMPro Wiki": "OKFolio",
}


class PublicDemoError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicDemoError(f"manifest must be a JSON object: {path}")
    return value


def _relative_placeholder(path: Path, site_dir: Path) -> str:
    target = site_dir / "assets" / "private-asset-removed.svg"
    return Path(os.path.relpath(target, path.parent)).as_posix()


def _sanitize_site(
    site_dir: Path,
    private_metadata: dict[str, str],
) -> tuple[int, int, int]:
    url_replacements = 0
    metadata_replacements = 0
    external_asset_replacements = 0
    metadata_pairs = sorted(
        (
            (value, replacement)
            for value, replacement in private_metadata.items()
            if value and len(value) >= 4
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        replacement = _relative_placeholder(path, site_dir)
        sanitized, count = _PRIVATE_URL_RE.subn(replacement, text)
        url_replacements += count
        for private_value, public_value in metadata_pairs:
            count = sanitized.count(private_value)
            if count:
                sanitized = sanitized.replace(private_value, public_value)
                metadata_replacements += count
        for external_value, local_value in _EXTERNAL_HIGHLIGHT_ASSETS.items():
            count = sanitized.count(external_value)
            if count:
                sanitized = sanitized.replace(external_value, local_value)
                external_asset_replacements += count
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")
    return (
        url_replacements,
        metadata_replacements,
        external_asset_replacements,
    )


def _write_placeholder(site_dir: Path) -> None:
    path = site_dir / "assets" / "private-asset-removed.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" '
            'viewBox="0 0 960 540"><rect width="960" height="540" '
            'fill="#eef3f8"/><text x="480" y="270" text-anchor="middle" '
            'font-family="system-ui,sans-serif" font-size="30" fill="#52657a">'
            "Public demo asset</text></svg>\n"
        ),
        encoding="utf-8",
    )


def _add_explorer_link(site_dir: Path) -> None:
    """Make the explorer discoverable from an older MkDocs index page."""
    index = site_dir / "index.html"
    if not index.is_file():
        return
    content = index.read_text(encoding="utf-8")
    if "explore.html" in content:
        return
    banner = (
        '<p><a href="explore.html">Open Knowledge Explorer</a> · '
        '<a href="graph.html">Open 3D graph</a></p>'
    )
    if "</body>" in content:
        content = content.replace("</body>", banner + "\n</body>", 1)
    else:
        content += "\n" + banner + "\n"
    index.write_text(content, encoding="utf-8")


def _replace_public_brand(site_dir: Path) -> int:
    replacements = 0
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        branded = text
        for old_value, new_value in _PUBLIC_BRAND_REPLACEMENTS.items():
            count = branded.count(old_value)
            if count:
                branded = branded.replace(old_value, new_value)
                replacements += count
        if branded != text:
            path.write_text(branded, encoding="utf-8")
    return replacements


def _write_checksums(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_public_tree(root: Path) -> None:
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if _PRIVATE_IP_RE.search(text):
            violations.append(f"private IP: {path.relative_to(root)}")
        if re.search(
            r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?key)"
            r"\s*[=:]\s*(?!replace-me|test-key|\$\{)[^\s\"']+",
            text,
        ):
            violations.append(f"credential-like value: {path.relative_to(root)}")
    if violations:
        raise PublicDemoError("; ".join(violations))


def build_public_demo(
    release_dir: Path,
    output_dir: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    release = release_dir.resolve()
    output = output_dir.resolve()
    source_site = release / "data" / "outputs" / "site"
    source_manifest = release / "release-manifest.json"
    if not source_site.is_dir() or not source_manifest.is_file():
        raise PublicDemoError("release must contain a built site and manifest")
    if output == release or release in output.parents:
        raise PublicDemoError("public demo must be outside the source release")
    if output.exists():
        if not replace:
            raise PublicDemoError(f"output already exists: {output}")
        shutil.rmtree(output)
    source = _read_json(source_manifest)
    output.mkdir(parents=True)
    site_dir = output / "site"
    shutil.copytree(source_site, site_dir)
    # Older accepted releases predate the explorer page.  Derive the same
    # portable payload from their graph so the public showcase keeps one
    # consistent entry point without requiring private run artifacts.
    explorer = site_dir / "explore.html"
    graph = site_dir / "graph.html"
    if not explorer.is_file() and graph.is_file():
        graph_data = extract_graph_data(graph.read_text(encoding="utf-8"))
        explorer.write_text(
            build_explorer_html(
                graph_data,
                title="OKFolio Knowledge Explorer",
                scope_note=(
                    f"Public showcase · {graph_data['stats']['articles']} Article · "
                    f"{graph_data['stats']['refs']} Ref · "
                    f"{graph_data['stats']['concepts']} Concept"
                ),
            ),
            encoding="utf-8",
        )
    _add_explorer_link(site_dir)
    _write_placeholder(site_dir)
    brand_replacements = _replace_public_brand(site_dir)
    (
        url_replacements,
        metadata_replacements,
        external_asset_replacements,
    ) = _sanitize_site(
        site_dir,
        {
            str(source.get("model", "")): "OpenAI-compatible LLM",
            str(source.get("source_run", "")): "public-demo",
            str(source.get("version", "")): "0.1.0-demo",
        },
    )

    manifest = {
        "schema": "okfolio.public-demo.v1",
        "status": "complete",
        "version": "0.1.0-demo",
        "model": "OpenAI-compatible LLM",
        "data_policy": "public demo data; private infrastructure removed",
        "evidence_scope": (
            "feature demo generated from an audited public release; "
            "not a benchmark dataset"
        ),
        "removed_private_asset_urls": url_replacements,
        "removed_private_metadata_values": metadata_replacements,
        "localized_external_highlight_assets": external_asset_replacements,
        "renamed_legacy_brand_values": brand_replacements,
        **{
            field: source[field]
            for field in _PUBLIC_MANIFEST_FIELDS
            if field in source
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _audit_public_tree(output)
    _write_checksums(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = build_public_demo(
        args.release_dir,
        args.output_dir,
        replace=args.replace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
