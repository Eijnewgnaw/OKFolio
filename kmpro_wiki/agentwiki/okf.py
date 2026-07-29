from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class OKFValidationError(ValueError):
    pass


FILE_BLOCK_RE = re.compile(
    r"===FILE:\s*(?P<filename>[^\r\n=]+?)\s*===\s*\r?\n"
    r"(?P<content>.*?)\r?\n===END===",
    re.DOTALL,
)
FRONTMATTER_RE = re.compile(
    r"\A---\s*\r?\n(?P<yaml>.*?)\r?\n---\s*(?:\r?\n|\Z)(?P<body>.*)\Z",
    re.DOTALL,
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
IMAGE_REWRITE_RE = re.compile(r"(!\[[^\]]*\]\()images/([^)]+)(\))")
MARKDOWN_SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]+\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass(frozen=True)
class ConceptDocument:
    filename: str
    frontmatter: dict[str, Any]
    body: str

    def render(self) -> str:
        metadata = yaml.safe_dump(
            self.frontmatter,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip()
        return f"---\n{metadata}\n---\n{self.body.rstrip()}\n"


def parse_concept_markdown(filename: str, content: str) -> ConceptDocument:
    match = FRONTMATTER_RE.match(content.strip())
    if match is None:
        raise OKFValidationError(f"{filename}: missing or invalid YAML frontmatter")
    try:
        frontmatter = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as error:
        raise OKFValidationError(f"{filename}: invalid YAML frontmatter") from error
    if not isinstance(frontmatter, dict):
        raise OKFValidationError(f"{filename}: frontmatter must be a mapping")
    return ConceptDocument(
        filename=filename.strip(),
        frontmatter=frontmatter,
        body=match.group("body").strip(),
    )


def parse_compile_response(response: str) -> list[ConceptDocument]:
    matches = list(FILE_BLOCK_RE.finditer(response))
    if not matches:
        if "===FILE:" in response:
            raise OKFValidationError("compile response contains an unclosed file block")
        raise OKFValidationError("compile response contains no file blocks")

    end_markers = response.count("===END===")
    if response.count("===FILE:") != len(matches) or end_markers != len(matches):
        raise OKFValidationError("compile response contains an unclosed file block")
    return [
        parse_concept_markdown(match.group("filename"), match.group("content"))
        for match in matches
    ]


def normalize_slug(filename: str) -> str:
    candidate = unicodedata.normalize("NFKC", filename).strip()
    if (
        not candidate
        or candidate.startswith(("/", "\\"))
        or "/" in candidate
        or "\\" in candidate
        or ".." in candidate
    ):
        raise OKFValidationError(f"unsafe concept filename: {filename}")
    if not candidate.lower().endswith(".md"):
        raise OKFValidationError(f"concept filename must end with .md: {filename}")
    if candidate.lower() in {"index.md", "log.md"}:
        raise OKFValidationError(f"reserved concept filename: {filename}")

    stem = candidate[:-3].strip()
    stem = re.sub(r"\s+", "-", stem)
    stem = re.sub(r"[^\w\-\u3400-\u9fff]", "-", stem, flags=re.UNICODE)
    stem = stem.strip("-_")
    if not stem:
        raise OKFValidationError(f"empty concept slug: {filename}")
    return f"{stem}.md"


def validate_concept(concept: ConceptDocument, expected_source: str) -> None:
    normalized = normalize_slug(concept.filename)
    if normalized != concept.filename:
        raise OKFValidationError(
            f"concept filename is not normalized: {concept.filename}; expected {normalized}"
        )
    for field in ("type", "title", "description", "source"):
        value = concept.frontmatter.get(field)
        if not isinstance(value, str) or not value.strip():
            raise OKFValidationError(f"{concept.filename}: missing non-empty {field}")
    if concept.frontmatter["source"] != expected_source:
        raise OKFValidationError(
            f"{concept.filename}: source must be {expected_source}"
        )


def _markdown_tables(content: str) -> list[str]:
    lines = content.splitlines()
    tables: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        block: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index])
            index += 1
        if len(block) >= 2 and MARKDOWN_SEPARATOR_RE.match(block[1].strip()):
            tables.append("\n".join(lines[start:index]))
    return tables


def _local_images(content: str) -> list[str]:
    return [
        match.group("target")
        for match in IMAGE_RE.finditer(content)
        if match.group("target").startswith("images/")
    ]


def validate_preserved_assets(
    source_content: str,
    generated_contents: list[str],
    source_images_dir: Path,
) -> None:
    combined = "\n".join(generated_contents)

    for target in _local_images(source_content):
        relative = PurePosixPath(target).relative_to("images")
        if ".." in relative.parts:
            raise OKFValidationError(f"unsafe image reference: {target}")
        if not (source_images_dir / Path(*relative.parts)).is_file():
            raise OKFValidationError(f"referenced image does not exist: {target}")
        if target not in combined:
            raise OKFValidationError(f"generated concepts dropped image: {target}")

    for table in HTML_TABLE_RE.findall(source_content):
        if table not in combined:
            raise OKFValidationError("generated concepts dropped an HTML table")

    for table in _markdown_tables(source_content):
        if table not in combined:
            raise OKFValidationError("generated concepts dropped a Markdown table")


def restore_missing_assets(
    source_content: str,
    concepts: list[ConceptDocument],
) -> list[ConceptDocument]:
    if not concepts:
        return concepts

    combined = "\n".join(concept.render() for concept in concepts)
    source_assets = [
        match.group(0)
        for match in IMAGE_RE.finditer(source_content)
        if match.group("target").startswith("images/")
    ]
    source_assets.extend(HTML_TABLE_RE.findall(source_content))
    source_assets.extend(_markdown_tables(source_content))
    missing = [asset for asset in source_assets if asset not in combined]
    if not missing:
        return concepts

    recipient = next(
        (
            index
            for index, concept in enumerate(concepts)
            if concept.frontmatter.get("type") == "数据口径"
        ),
        0,
    )
    restored = list(concepts)
    addition = "\n\n".join(missing)
    body = (
        f"{restored[recipient].body.rstrip()}\n\n"
        f"## 原文图表（自动保真）\n\n{addition}"
    )
    restored[recipient] = replace(restored[recipient], body=body)
    return restored


def validate_link_only_enrichment(
    original: ConceptDocument,
    candidate: ConceptDocument,
) -> None:
    if candidate.frontmatter != original.frontmatter:
        raise OKFValidationError(
            f"{original.filename}: enrichment rewrote frontmatter"
        )

    def without_links(value: str) -> str:
        value = MARKDOWN_LINK_RE.sub(r"\1", value)
        return WIKI_LINK_RE.sub(r"\1", value)

    if without_links(candidate.body) != without_links(original.body):
        raise OKFValidationError(f"{original.filename}: enrichment rewrote body")


def rewrite_image_paths(content: str) -> str:
    return IMAGE_REWRITE_RE.sub(r"\1../images/\2\3", content)
