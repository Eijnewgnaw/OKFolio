from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .contracts import LinkSuggestion, RelationAudit
from .okf import HTML_TABLE_RE, ConceptDocument, validate_link_only_enrichment


class RelationError(ValueError):
    pass


@dataclass(frozen=True)
class RelationAnchor:
    target_id: str
    anchor_id: str
    text: str
    occurrence: int
    context: str


CONCEPT_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target>[^)#?]*concepts/[^)#?]+\.md)(?:#[^)]*)?\)"
)
_MARKDOWN_LINK_OR_IMAGE_RE = re.compile(r"!?\[[^\]]*\]\([^)]+\)")
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s.*$")
_FENCED_CODE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MARKDOWN_TABLE_RE = re.compile(
    r"(?m)^[ \t]*\|.*\|[ \t]*(?:\n^[ \t]*\|.*\|[ \t]*)+"
)
_TEMPORAL_RE = re.compile(
    r"(?:19|20)\d{2}年|年[一二三四1234]季度|[Qq][1-4]"
)
_WEAK_EDGE_RE = re.compile(r"^[与及和为在至对将把由从的于]|的$")
_LOW_INFORMATION_ANCHORS = {
    "有待提升",
    "持续提升",
    "持续走低",
    "稳固向好",
    "稳定运行",
    "结构性分化",
    "通过建立",
    "反映信用",
}


def build_relation_anchor_catalog(
    current: ConceptDocument,
    candidates: dict[str, ConceptDocument],
) -> tuple[RelationAnchor, ...]:
    protected = _protected_ranges(current.body)
    result: list[RelationAnchor] = []
    for target_id, target in sorted(candidates.items()):
        position = 0
        for text in _shared_metadata_phrases(current.body, target):
            for occurrence, match in enumerate(
                re.finditer(re.escape(text), current.body)
            ):
                span = (match.start(), match.end())
                if any(_overlaps(span, item) for item in protected):
                    continue
                position += 1
                context_start = max(0, match.start() - 30)
                context_end = min(len(current.body), match.end() + 30)
                result.append(
                    RelationAnchor(
                        target_id=target_id,
                        anchor_id=f"{target_id}--anchor-{position:03d}",
                        text=text,
                        occurrence=occurrence,
                        context=current.body[context_start:context_end],
                    )
                )
    metadata_by_target = {
        target_id: "。".join(
            str(target.frontmatter.get(field, ""))
            for field in ("title", "description")
        )
        for target_id, target in candidates.items()
    }
    return tuple(
        item
        for item in result
        if sum(
            item.text in metadata for metadata in metadata_by_target.values()
        )
        == 1
        and not _is_low_information_anchor(item.text)
    )


def validate_relation_anchor_selection(
    current: ConceptDocument, audit: RelationAudit
) -> None:
    protected = _protected_ranges(current.body)
    ranges: list[tuple[int, int]] = []
    for suggestion in audit.links:
        start, end = _resolve_suggestion_span(current.body, suggestion)
        if any(_overlaps((start, end), item) for item in protected):
            raise RelationError(
                f"relation anchor occurs in protected Markdown: {suggestion.anchor}"
            )
        if any(_overlaps((start, end), item) for item in ranges):
            raise RelationError("relation anchors overlap")
        ranges.append((start, end))


def apply_relation_audit(
    current: ConceptDocument,
    audit: RelationAudit,
    candidates: dict[str, ConceptDocument],
) -> ConceptDocument:
    if audit.status == "no_links":
        if audit.links:
            raise RelationError("no_links audit must not contain suggestions")
        return current
    if audit.status != "linked" or not audit.links:
        raise RelationError("linked audit requires at least one suggestion")

    validate_relation_anchor_selection(current, audit)
    replacements: list[tuple[int, int, str]] = []
    identities: set[tuple[str, str, int | None]] = set()
    ranges: list[tuple[int, int]] = []
    for suggestion in audit.links:
        identity = (
            suggestion.target_id,
            suggestion.anchor,
            suggestion.occurrence,
        )
        if identity in identities:
            raise RelationError("duplicate relation suggestion")
        identities.add(identity)
        target = candidates.get(suggestion.target_id)
        if target is None:
            raise RelationError(f"unknown target concept: {suggestion.target_id}")
        if target.filename == current.filename:
            raise RelationError(f"relation would create a self-link: {target.filename}")
        start, end = _resolve_suggestion_span(current.body, suggestion)
        ranges.append((start, end))
        replacements.append(
            (
                start,
                end,
                f"[{suggestion.anchor}](../concepts/{target.filename})",
            )
        )

    body = current.body
    for start, end, replacement in sorted(replacements, reverse=True):
        body = body[:start] + replacement + body[end:]
    updated = ConceptDocument(
        filename=current.filename,
        frontmatter=current.frontmatter,
        body=body,
    )
    try:
        validate_link_only_enrichment(current, updated)
    except ValueError as error:
        raise RelationError(str(error)) from error
    return updated


def classify_links(concepts: dict[str, ConceptDocument]) -> dict[str, int]:
    counts = {"same_source": 0, "cross_source": 0, "broken": 0, "self": 0}
    for source_filename, concept in concepts.items():
        for match in CONCEPT_LINK_RE.finditer(concept.body):
            target_filename = Path(match.group("target")).name
            target = concepts.get(target_filename)
            if target is None:
                counts["broken"] += 1
            elif target_filename == source_filename:
                counts["self"] += 1
            elif target.frontmatter.get("source") == concept.frontmatter.get("source"):
                counts["same_source"] += 1
            else:
                counts["cross_source"] += 1
    return counts


def _protected_ranges(body: str) -> tuple[tuple[int, int], ...]:
    matches = []
    for pattern in (
        _HEADING_RE,
        _FENCED_CODE_RE,
        _INLINE_CODE_RE,
        _MARKDOWN_LINK_OR_IMAGE_RE,
        HTML_TABLE_RE,
        _MARKDOWN_TABLE_RE,
    ):
        matches.extend((item.start(), item.end()) for item in pattern.finditer(body))
    return tuple(matches)


def _resolve_suggestion_span(
    body: str, suggestion: LinkSuggestion
) -> tuple[int, int]:
    anchor = suggestion.anchor
    occurrence = suggestion.occurrence
    matches = tuple(re.finditer(re.escape(anchor), body))
    if occurrence is None:
        if len(matches) != 1:
            raise RelationError(
                f"relation anchor must occur exactly once: {anchor}"
            )
        selected = matches[0]
    else:
        if occurrence < 0 or occurrence >= len(matches):
            raise RelationError(
                f"relation anchor occurrence does not exist: {anchor}"
            )
        selected = matches[occurrence]
    return selected.start(), selected.end()


def _shared_metadata_phrases(
    body: str, target: ConceptDocument
) -> tuple[str, ...]:
    metadata = "。".join(
        str(target.frontmatter.get(field, ""))
        for field in ("title", "description")
    )
    matches: set[str] = set()
    for sequence in re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", metadata):
        for length in range(4, min(12, len(sequence)) + 1):
            for start in range(len(sequence) - length + 1):
                value = sequence[start : start + length]
                if value in body:
                    matches.add(value)
    maximal = [
        value
        for value in matches
        if not any(value != other and value in other for other in matches)
    ]
    return tuple(sorted(maximal, key=lambda value: (-len(value), value)))


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _is_low_information_anchor(text: str) -> bool:
    return (
        text in _LOW_INFORMATION_ANCHORS
        or _TEMPORAL_RE.search(text) is not None
        or _WEAK_EDGE_RE.search(text) is not None
        or re.fullmatch(r"[0-9年月日季度Qq]+", text) is not None
    )
