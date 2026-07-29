from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from .contracts import AssetPlacement, DraftConcept
from .okf import (
    HTML_TABLE_RE,
    IMAGE_RE,
    MARKDOWN_SEPARATOR_RE,
    ConceptDocument,
)


class AssetError(ValueError):
    pass


@dataclass(frozen=True)
class SourceAsset:
    asset_id: str
    kind: Literal["image", "html_table", "markdown_table"]
    raw: str
    target: str | None
    before: str
    after: str
    ordinal: int
    sha256: str | None


@dataclass(frozen=True)
class _AssetMatch:
    kind: Literal["image", "html_table", "markdown_table"]
    start: int
    end: int
    raw: str
    target: str | None = None


def inventory_assets(source_content: str, source_images_dir: Path) -> tuple[SourceAsset, ...]:
    matches: list[_AssetMatch] = []
    for match in IMAGE_RE.finditer(source_content):
        target = match.group("target")
        if not target.startswith("images/") and not _remote_image_target(target):
            continue
        matches.append(
            _AssetMatch("image", match.start(), match.end(), match.group(0), target)
        )
    for match in HTML_TABLE_RE.finditer(source_content):
        matches.append(
            _AssetMatch("html_table", match.start(), match.end(), match.group(0))
        )
    matches.extend(_markdown_table_matches(source_content))
    matches.sort(key=lambda item: (item.start, item.end))
    _reject_overlaps(matches)

    counters: Counter[str] = Counter()
    assets: list[SourceAsset] = []
    for match in matches:
        counters[match.kind] += 1
        ordinal = counters[match.kind]
        asset_id = f"{match.kind.replace('_', '-')}-{ordinal:03d}"
        checksum = None
        if match.kind == "image":
            assert match.target is not None
            if match.target.startswith("images/"):
                image_path = _resolve_local_image(
                    match.target, source_images_dir
                )
                if not image_path.is_file():
                    raise AssetError(
                        f"referenced image does not exist: {match.target}"
                    )
                checksum = _sha256_file(image_path)
        assets.append(
            SourceAsset(
                asset_id=asset_id,
                kind=match.kind,
                raw=match.raw,
                target=match.target,
                before=_neighbor(source_content[: match.start], reverse=True),
                after=_neighbor(source_content[match.end :], reverse=False),
                ordinal=ordinal,
                sha256=checksum,
            )
        )
    return tuple(assets)


def strip_missing_image_references(source_content: str, source_images_dir: Path) -> str:
    """Remove local-image tags whose binary is absent before pipeline ingestion.

    Missing binaries are not knowledge assets.  They must not create an asset
    record, block Ref extraction, or leak into Articles and final Concepts.
    The caller keeps the raw source file untouched and uses this normalized
    content for all downstream stages.
    """

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if not target.startswith("images/"):
            return match.group(0)
        image_path = _resolve_local_image(target, source_images_dir)
        return match.group(0) if image_path.is_file() else ""

    return IMAGE_RE.sub(replace, source_content)


def apply_asset_placements(
    drafts: list[DraftConcept] | tuple[DraftConcept, ...],
    assets: list[SourceAsset] | tuple[SourceAsset, ...],
    placements: list[AssetPlacement] | tuple[AssetPlacement, ...],
) -> tuple[ConceptDocument, ...]:
    asset_by_id = {asset.asset_id: asset for asset in assets}
    if len(asset_by_id) != len(assets):
        raise AssetError("source asset IDs must be unique")
    placement_counts = Counter(item.asset_id for item in placements)
    if set(placement_counts) != set(asset_by_id) or any(
        count != 1 for count in placement_counts.values()
    ):
        raise AssetError("every source asset requires exactly once placement")

    concepts = {
        item.ref.concept_id: ConceptDocument(
            filename=f"{item.ref.concept_id}.md",
            frontmatter={
                "type": item.ref.type,
                "title": item.title,
                "description": item.description,
                "source": item.ref.source,
            },
            body=item.body,
        )
        for item in drafts
    }
    if len(concepts) != len(drafts):
        raise AssetError("draft concept IDs must be unique")

    for placement in placements:
        asset = asset_by_id.get(placement.asset_id)
        if asset is None:
            raise AssetError(f"unknown source asset: {placement.asset_id}")
        concept = concepts.get(placement.concept_id)
        if concept is None:
            raise AssetError(f"unknown target concept: {placement.concept_id}")
        if concept.body.count(placement.anchor) != 1:
            raise AssetError(
                f"asset anchor must occur exactly once: {placement.anchor}"
            )
        if placement.position == "before":
            replacement = f"{asset.raw}\n\n{placement.anchor}"
        elif placement.position == "after":
            replacement = f"{placement.anchor}\n\n{asset.raw}"
        else:
            raise AssetError(f"invalid asset position: {placement.position}")
        concepts[placement.concept_id] = ConceptDocument(
            filename=concept.filename,
            frontmatter=concept.frontmatter,
            body=concept.body.replace(placement.anchor, replacement, 1),
        )

    ordered = tuple(concepts[item.ref.concept_id] for item in drafts)
    _validate_body_invariance(drafts, ordered, assets, placements)
    return ordered


def validate_asset_preservation(
    assets: list[SourceAsset] | tuple[SourceAsset, ...],
    concepts: list[ConceptDocument] | tuple[ConceptDocument, ...],
    source_images_dir: Path,
    *,
    baseline: list[DraftConcept] | tuple[DraftConcept, ...] = (),
) -> None:
    combined = "\n".join(concept.body for concept in concepts)
    previous = "\n".join(concept.body for concept in baseline)
    expected = Counter(asset.raw for asset in assets)
    for raw, count in expected.items():
        actual = combined.count(raw) - previous.count(raw)
        if actual != count:
            raise AssetError(
                "source asset count delta mismatch: "
                f"expected {count}, found {actual}"
            )
    for asset in assets:
        if asset.kind != "image":
            continue
        assert asset.target is not None
        if _remote_image_target(asset.target):
            if asset.sha256 is not None:
                raise AssetError("remote image must not claim a local checksum")
            continue
        assert asset.sha256 is not None
        image_path = _resolve_local_image(asset.target, source_images_dir)
        if not image_path.is_file():
            raise AssetError(f"referenced image does not exist: {asset.target}")
        if _sha256_file(image_path) != asset.sha256:
            raise AssetError(f"source image checksum changed: {asset.target}")


def _markdown_table_matches(content: str) -> list[_AssetMatch]:
    lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    matches: list[_AssetMatch] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start_index = index
        block: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index])
            index += 1
        if len(block) < 2 or not MARKDOWN_SEPARATOR_RE.match(block[1].strip()):
            continue
        start = offsets[start_index]
        end = offsets[index] if index < len(offsets) else len(content)
        raw = content[start:end].rstrip("\r\n")
        matches.append(_AssetMatch("markdown_table", start, start + len(raw), raw))
    return matches


def _resolve_local_image(target: str, source_images_dir: Path) -> Path:
    try:
        relative = PurePosixPath(target).relative_to("images")
    except ValueError as error:
        raise AssetError(f"unsafe image reference: {target}") from error
    if not relative.parts or ".." in relative.parts or "." in relative.parts:
        raise AssetError(f"unsafe image reference: {target}")
    return source_images_dir.joinpath(*relative.parts)


def _remote_image_target(target: str) -> bool:
    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    if parsed.scheme == "s3":
        return bool(parsed.netloc and parsed.path.strip("/"))
    return False


def _neighbor(content: str, *, reverse: bool) -> str:
    lines = content.splitlines()
    values = reversed(lines) if reverse else iter(lines)
    for line in values:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _reject_overlaps(matches: list[_AssetMatch]) -> None:
    previous_end = -1
    for match in matches:
        if match.start < previous_end:
            raise AssetError("source assets overlap")
        previous_end = match.end


def _validate_body_invariance(
    drafts: list[DraftConcept] | tuple[DraftConcept, ...],
    concepts: tuple[ConceptDocument, ...],
    assets: list[SourceAsset] | tuple[SourceAsset, ...],
    placements: list[AssetPlacement] | tuple[AssetPlacement, ...],
) -> None:
    asset_by_id = {item.asset_id: item for item in assets}
    restored = {item.filename[:-3]: item.body for item in concepts}
    for placement in reversed(placements):
        raw = asset_by_id[placement.asset_id].raw
        if placement.position == "before":
            inserted = f"{raw}\n\n{placement.anchor}"
        else:
            inserted = f"{placement.anchor}\n\n{raw}"
        restored[placement.concept_id] = restored[placement.concept_id].replace(
            inserted, placement.anchor, 1
        )
    for draft in drafts:
        if restored[draft.ref.concept_id] != draft.body:
            raise AssetError(
                f"asset placement rewrote draft body: {draft.ref.concept_id}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
