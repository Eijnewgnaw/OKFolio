"""Evidence-led Concept review contracts.

This module is intentionally independent from :mod:`agentic`.  It defines
strict, serializable contracts for two bounded model decisions:

1. derive one canonical question and an auditable claim ledger from the
   ConceptRefs in an already selected compile group;
2. compare an existing Concept draft with that frozen ledger both claim by
   claim and draft sentence by draft sentence.

Large drafts are audited in deterministic contiguous sentence batches (each
batch reuses the same frozen Claim Contract and carries only its own
sentences); :func:`merge_claim_coverage_batches` folds the per-batch fragments
into one full coverage payload before the single-matrix parser runs.

The model never supplies a quality score or release decision.  Parsers verify
claim/sentence ID coverage, verbatim excerpts and high-confidence inference/OCR
warnings, then :func:`derive_review_decision` applies the deterministic rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Literal, Mapping, Sequence

from .contracts import ContractError


MemberRelation = Literal[
    "supports",
    "qualifies",
    "contrasts",
    "applies_to",
    "separate",
]
EvidenceDisposition = Literal["required", "duplicate", "context_only"]
ClaimKind = Literal[
    "fact",
    "metric",
    "time",
    "policy",
    "actor",
    "scope",
    "causal",
    "recommendation",
    "judgment",
]
CoverageStatus = Literal["covered", "omitted", "contradicted", "uncertain"]
SentenceAttributionStatus = Literal["supported", "unsupported", "uncertain"]
ReviewDecision = Literal["pass", "recompile", "human_review"]


MEMBER_RELATIONS = frozenset(
    {"supports", "qualifies", "contrasts", "applies_to", "separate"}
)
EVIDENCE_DISPOSITIONS = frozenset({"required", "duplicate", "context_only"})
CLAIM_KINDS = frozenset(
    {
        "fact",
        "metric",
        "time",
        "policy",
        "actor",
        "scope",
        "causal",
        "recommendation",
        "judgment",
    }
)
TYPE_SLOTS: Mapping[str, tuple[str, ...]] = {
    "数据口径": (
        "indicator",
        "definition",
        "calculation",
        "unit",
        "time",
        "region",
        "data_source",
        "boundary",
    ),
    "分析框架": (
        "subject",
        "core_judgment",
        "evidence",
        "problem",
        "cause",
        "constraint",
        "impact",
        "scope",
    ),
    "政策建议": (
        "target_problem",
        "measure",
        "implementer",
        "target_group",
        "implementation_path",
        "condition",
        "time",
        "expected_effect",
    ),
    "国际比较": (
        "comparison_subject",
        "measurement_basis",
        "country_or_region",
        "time",
        "benchmark",
        "difference",
        "applicability_limit",
    ),
    "术语解释": (
        "term",
        "definition",
        "components",
        "boundary",
        "application",
    ),
}
COVERAGE_STATUSES = frozenset(
    {"covered", "omitted", "contradicted", "uncertain"}
)
SENTENCE_ATTRIBUTION_STATUSES = frozenset(
    {"supported", "unsupported", "uncertain"}
)

# Keep the model-facing Claim Contract bounded.  These limits are deliberately
# generous for one atomic Chinese policy claim, while preventing an unconstrained
# grammar from consuming the entire completion budget.
MAX_CLAIMS_PER_REF = 8
MAX_CANONICAL_QUESTION_LENGTH = 160
MAX_MEMBER_CONTRIBUTION_LENGTH = 120
MAX_CLAIM_TEXT_LENGTH = 160
MAX_EVIDENCE_EXCERPT_LENGTH = 240
MAX_EVIDENCE_REASON_LENGTH = 120
SCOPE_KEYS = frozenset(
    {"region", "time", "object", "condition", "scenario", "actor"}
)
MAX_SCOPE_VALUE_LENGTH = 64
MAX_SCOPE_VALUES_PER_KEY = 4


@dataclass(frozen=True)
class SourceEvidenceBlock:
    block_id: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExcludedEvidenceFragment:
    text: str
    block_id: str | None
    page_number: int | None
    source_text_anomalies: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    ref_id: str
    text: str
    source_blocks: tuple[SourceEvidenceBlock, ...] = ()
    excluded_fragments: tuple[ExcludedEvidenceFragment, ...] = ()


@dataclass(frozen=True)
class MemberContribution:
    ref_id: str
    relation: MemberRelation
    contribution: str


@dataclass(frozen=True)
class ClaimObligation:
    claim_id: str
    ref_id: str
    evidence_id: str
    claim: str
    slot: str
    kind: ClaimKind
    evidence_excerpt: str
    evidence_block_ids: tuple[str, ...]
    page_numbers: tuple[int, ...]
    scope: Mapping[str, Any]
    source_text_anomalies: tuple[str, ...] = ()
    ocr_suspicions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceUnitReview:
    evidence_id: str
    disposition: EvidenceDisposition
    reason: str
    claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConceptClaimContract:
    group_id: str
    canonical_question: str
    members: tuple[MemberContribution, ...]
    claims: tuple[ClaimObligation, ...]
    evidence_units: tuple[EvidenceUnitReview, ...]
    schema_version: str = "okfolio.claim-contract.v1"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "canonical_question": self.canonical_question,
            "members": [asdict(item) for item in self.members],
            "claims": [
                {
                    **asdict(item),
                    "scope": dict(item.scope),
                }
                for item in self.claims
            ],
            "evidence_units": [asdict(item) for item in self.evidence_units],
        }


@dataclass(frozen=True)
class ClaimCoverageRow:
    claim_id: str
    status: CoverageStatus
    draft_excerpt: str
    finding: str


@dataclass(frozen=True)
class DraftSentence:
    sentence_id: str
    field: Literal["description", "body"]
    text: str


@dataclass(frozen=True)
class DraftSentenceAttribution:
    sentence_id: str
    status: SentenceAttributionStatus
    claim_ids: tuple[str, ...]
    draft_excerpt: str
    finding: str
    source_text_anomalies: tuple[str, ...] = ()
    ocr_suspicions: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnsupportedClaim:
    draft_excerpt: str
    finding: str


@dataclass(frozen=True)
class ScopeViolation:
    claim_ids: tuple[str, ...]
    draft_excerpt: str
    finding: str


@dataclass(frozen=True)
class ClaimCoverageMatrix:
    rows: tuple[ClaimCoverageRow, ...]
    sentence_attributions: tuple[DraftSentenceAttribution, ...]
    unsupported_claims: tuple[UnsupportedClaim, ...]
    scope_violations: tuple[ScopeViolation, ...]
    decision: ReviewDecision
    schema_version: str = "okfolio.claim-coverage.v2"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rows": [asdict(item) for item in self.rows],
            "sentence_attributions": [
                asdict(item) for item in self.sentence_attributions
            ],
            "unsupported_claims": [
                asdict(item) for item in self.unsupported_claims
            ],
            "scope_violations": [asdict(item) for item in self.scope_violations],
            "decision": self.decision,
        }


@dataclass(frozen=True)
class ClaimCoverageBatch:
    """One deterministic coverage fragment for a contiguous sentence batch.

    A batch carries only the claims the model could express, contradict or
    doubt from its own sentences; ``omitted`` rows are never produced by the
    model and are completed by code when batches are merged.  Batch index is
    part of the record so a resumed run can skip already persisted batches and
    so the merge is independent of the order in which batches completed.
    """

    batch_index: int
    rows: tuple[ClaimCoverageRow, ...]
    sentence_attributions: tuple[DraftSentenceAttribution, ...]
    unsupported_claims: tuple[UnsupportedClaim, ...]
    scope_violations: tuple[ScopeViolation, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "rows": [asdict(item) for item in self.rows],
            "sentence_attributions": [
                {
                    "sentence_id": item.sentence_id,
                    "status": item.status,
                    "claim_ids": list(item.claim_ids),
                    "draft_excerpt": item.draft_excerpt,
                    "finding": item.finding,
                }
                for item in self.sentence_attributions
            ],
            "unsupported_claims": [
                asdict(item) for item in self.unsupported_claims
            ],
            "scope_violations": [
                {**asdict(item), "claim_ids": list(item.claim_ids)}
                for item in self.scope_violations
            ],
        }


def build_evidence_units(
    refs: Sequence[Mapping[str, Any]],
    *,
    known_source_anomalies: Sequence[str] = (),
) -> tuple[EvidenceUnit, ...]:
    """Build one bounded model-facing evidence bundle per Ref.

    Real source blocks remain attached as a code-side provenance catalog.  The
    model classifies the Ref bundle once; individual claims are mapped back to
    matching blocks by :func:`parse_claim_contract`.
    """
    normalized = _normalize_refs(refs)
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    units: list[EvidenceUnit] = []
    for ref_id, ref in normalized.items():
        blocks = ref.get("evidence_blocks") or ()
        if blocks:
            source_blocks: list[SourceEvidenceBlock] = []
            model_fragments: list[str] = []
            excluded_fragments: list[ExcludedEvidenceFragment] = []
            for index, raw in enumerate(blocks):
                if not isinstance(raw, Mapping):
                    raise ContractError(
                        f"refs[{ref_id}].evidence_blocks[{index}] must be an object"
                    )
                block_id = _text(
                    raw.get("block_id"),
                    f"refs[{ref_id}].evidence_blocks[{index}].block_id",
                )
                page = raw.get("page_number")
                if page is not None and (
                    isinstance(page, bool) or not isinstance(page, int) or page < 1
                ):
                    raise ContractError("evidence block page_number must be positive")
                block = SourceEvidenceBlock(
                    block_id=block_id,
                    text=_text(
                        raw.get("content"),
                        f"refs[{ref_id}].evidence_blocks[{index}].content",
                    ),
                    page_number=page,
                )
                source_blocks.append(block)
                clean, excluded = _filter_source_anomaly_sentences(
                    block.text,
                    anomalies,
                )
                if clean:
                    model_fragments.append(clean)
                excluded_fragments.extend(
                    ExcludedEvidenceFragment(
                        text=fragment,
                        block_id=block.block_id,
                        page_number=block.page_number,
                        source_text_anomalies=_source_text_anomalies(
                            fragment,
                            anomalies,
                        ),
                    )
                    for fragment in excluded
                )
            bundle_text = "\n\n".join(model_fragments).strip()
            if not bundle_text:
                raise ContractError(
                    "evidence bundle has no clean model-visible text after "
                    f"source anomaly exclusion: {ref_id}"
                )
            digest = hashlib.sha256(
                _normalized_text(bundle_text).encode("utf-8")
            ).hexdigest()[:12]
            units.append(
                EvidenceUnit(
                    evidence_id=f"{ref_id}:bundle-{digest}",
                    ref_id=ref_id,
                    text=bundle_text,
                    source_blocks=tuple(source_blocks),
                    excluded_fragments=tuple(excluded_fragments),
                )
            )
            continue
        fallback_clean: list[str] = []
        fallback_excluded: list[ExcludedEvidenceFragment] = []
        for index, evidence in enumerate(ref["evidence"]):
            clean, excluded = _filter_source_anomaly_sentences(
                evidence,
                anomalies,
            )
            if clean:
                fallback_clean.append(clean)
            fallback_excluded.extend(
                ExcludedEvidenceFragment(
                    text=fragment,
                    block_id=None,
                    page_number=None,
                    source_text_anomalies=_source_text_anomalies(
                        fragment,
                        anomalies,
                    ),
                )
                for fragment in excluded
            )
        bundle_text = "\n\n".join(fallback_clean).strip()
        if not bundle_text:
            raise ContractError(
                "evidence bundle has no clean model-visible text after "
                f"source anomaly exclusion: {ref_id}"
            )
        digest = hashlib.sha256(
            _normalized_text(bundle_text).encode("utf-8")
        ).hexdigest()[:12]
        units.append(
            EvidenceUnit(
                evidence_id=f"{ref_id}:bundle-{digest}",
                ref_id=ref_id,
                text=bundle_text,
                excluded_fragments=tuple(fallback_excluded),
            )
        )
    ids = [item.evidence_id for item in units]
    if len(ids) != len(set(ids)):
        raise ContractError("evidence bundle IDs must be unique in a group")
    block_ids = [
        block.block_id for unit in units for block in unit.source_blocks
    ]
    if len(block_ids) != len(set(block_ids)):
        raise ContractError("evidence block IDs must be globally unique in a group")
    return tuple(units)


def build_draft_sentences(
    draft: Mapping[str, Any],
) -> tuple[DraftSentence, ...]:
    """Split auditable description/body prose into stable sentence records.

    Markdown headings, image-only lines and table separators are presentation,
    not factual prose.  List markers are removed, while each list item remains
    an independently auditable sentence.  We split on sentence punctuation and
    semicolons, but deliberately not commas to avoid fragmenting Chinese facts.
    """
    if not isinstance(draft, Mapping):
        raise ContractError("draft must be an object")
    records: list[DraftSentence] = []
    for field in ("description", "body"):
        raw = draft.get(field, "")
        if not isinstance(raw, str):
            raise ContractError(f"draft.{field} must be a string")
        fragments = _draft_sentence_fragments(raw, markdown=field == "body")
        for position, text in enumerate(fragments):
            digest = hashlib.sha256(
                f"{field}\0{position}\0{_normalized_text(text)}".encode("utf-8")
            ).hexdigest()[:16]
            records.append(
                DraftSentence(
                    sentence_id=f"sentence-{digest}",
                    field=field,  # type: ignore[arg-type]
                    text=text,
                )
            )
    if not records:
        raise ContractError("draft has no auditable description or body sentences")
    return tuple(records)


def claim_contract_json_schema(
    refs: Sequence[Mapping[str, Any]],
    *,
    known_source_anomalies: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the structured-output schema for one Concept claim contract."""
    normalized = _normalize_refs(refs)
    concept_type = _concept_type(normalized)
    ref_ids = tuple(normalized)
    evidence_ids = tuple(
        item.evidence_id
        for item in build_evidence_units(
            refs,
            known_source_anomalies=known_source_anomalies,
        )
    )
    scope_value = {
        "anyOf": [
            {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SCOPE_VALUE_LENGTH,
            },
            {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SCOPE_VALUES_PER_KEY,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SCOPE_VALUE_LENGTH,
                },
            },
        ]
    }
    claim = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CLAIM_TEXT_LENGTH,
            },
            "slot": {"type": "string", "enum": list(TYPE_SLOTS[concept_type])},
            "kind": {"type": "string", "enum": sorted(CLAIM_KINDS)},
            "evidence_excerpt": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_EVIDENCE_EXCERPT_LENGTH,
            },
            "scope": {
                "type": "object",
                "properties": {
                    key: scope_value for key in sorted(SCOPE_KEYS)
                },
                "additionalProperties": False,
            },
        },
        "required": ["claim", "slot", "kind", "evidence_excerpt", "scope"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "canonical_question": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CANONICAL_QUESTION_LENGTH,
            },
            "members": {
                "type": "array",
                "minItems": len(ref_ids),
                "maxItems": len(ref_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string", "enum": sorted(ref_ids)},
                        "relation": {
                            "type": "string",
                            "enum": sorted(MEMBER_RELATIONS),
                        },
                        "contribution": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_MEMBER_CONTRIBUTION_LENGTH,
                        },
                    },
                    "required": ["ref_id", "relation", "contribution"],
                    "additionalProperties": False,
                },
            },
            "evidence_units": {
                "type": "array",
                "minItems": len(evidence_ids),
                "maxItems": len(evidence_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence_id": {
                            "type": "string",
                            "enum": sorted(evidence_ids),
                        },
                        "disposition": {
                            "type": "string",
                            "enum": sorted(EVIDENCE_DISPOSITIONS),
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_EVIDENCE_REASON_LENGTH,
                        },
                        "claims": {
                            "type": "array",
                            "maxItems": MAX_CLAIMS_PER_REF,
                            "items": claim,
                        },
                    },
                    "required": [
                        "evidence_id",
                        "disposition",
                        "reason",
                        "claims",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["canonical_question", "members", "evidence_units"],
        "additionalProperties": False,
    }


def parse_claim_contract(
    response: str,
    *,
    group_id: str,
    refs: Sequence[Mapping[str, Any]],
    known_source_anomalies: Sequence[str] = (),
) -> ConceptClaimContract:
    """Parse a model-produced claim contract and enforce evidence coverage."""
    normalized_refs = _normalize_refs(refs)
    concept_type = _concept_type(normalized_refs)
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    evidence_units = build_evidence_units(
        refs,
        known_source_anomalies=anomalies,
    )
    by_evidence = {item.evidence_id: item for item in evidence_units}
    payload = _object(response, "claim contract")
    _exact_keys(
        payload,
        {"canonical_question", "members", "evidence_units"},
        "claim contract",
    )
    canonical_question = _bounded_text(
        payload["canonical_question"],
        "canonical_question",
        MAX_CANONICAL_QUESTION_LENGTH,
    )

    raw_members = _array(payload["members"], "members")
    members: list[MemberContribution] = []
    seen_members: set[str] = set()
    for index, raw in enumerate(raw_members):
        item = _mapping(raw, f"members[{index}]")
        _exact_keys(
            item,
            {"ref_id", "relation", "contribution"},
            f"members[{index}]",
        )
        ref_id = _text(item["ref_id"], f"members[{index}].ref_id")
        if ref_id not in normalized_refs:
            raise ContractError(f"unknown member ref_id: {ref_id}")
        if ref_id in seen_members:
            raise ContractError(f"duplicate member ref_id: {ref_id}")
        seen_members.add(ref_id)
        relation = _enum(
            item["relation"], MEMBER_RELATIONS, f"members[{index}].relation"
        )
        members.append(
            MemberContribution(
                ref_id=ref_id,
                relation=relation,  # type: ignore[arg-type]
                contribution=_bounded_text(
                    item["contribution"],
                    f"members[{index}].contribution",
                    MAX_MEMBER_CONTRIBUTION_LENGTH,
                ),
            )
        )
    expected_refs = set(normalized_refs)
    if seen_members != expected_refs:
        missing = sorted(expected_refs - seen_members)
        raise ContractError(
            "claim contract must cover every member Ref exactly once; "
            f"missing={missing}"
        )
    if len(members) == 1 and members[0].relation == "separate":
        raise ContractError("a singleton Concept cannot be separate from itself")

    raw_units = _array(payload["evidence_units"], "evidence_units")
    claims: list[ClaimObligation] = []
    reviews: list[EvidenceUnitReview] = []
    seen_units: set[str] = set()
    contributor_refs: set[str] = set()
    seen_claim_ids: set[str] = set()
    for index, raw in enumerate(raw_units):
        item = _mapping(raw, f"evidence_units[{index}]")
        _exact_keys(
            item,
            {"evidence_id", "disposition", "reason", "claims"},
            f"evidence_units[{index}]",
        )
        evidence_id = _text(
            item["evidence_id"], f"evidence_units[{index}].evidence_id"
        )
        if evidence_id not in by_evidence:
            raise ContractError(f"unknown evidence_id: {evidence_id}")
        if evidence_id in seen_units:
            raise ContractError(f"duplicate evidence_id: {evidence_id}")
        seen_units.add(evidence_id)
        unit = by_evidence[evidence_id]
        disposition = _enum(
            item["disposition"],
            EVIDENCE_DISPOSITIONS,
            f"evidence_units[{index}].disposition",
        )
        reason = _bounded_text(
            item["reason"],
            f"evidence_units[{index}].reason",
            MAX_EVIDENCE_REASON_LENGTH,
        )
        raw_claims = _array(item["claims"], f"evidence_units[{index}].claims")
        if len(raw_claims) > MAX_CLAIMS_PER_REF:
            raise ContractError(
                "each evidence bundle may produce at most "
                f"{MAX_CLAIMS_PER_REF} claims"
            )
        if disposition == "required" and not raw_claims:
            raise ContractError("required evidence must produce at least one claim")
        if disposition != "required" and raw_claims:
            raise ContractError(
                "duplicate or context_only evidence cannot produce required claims"
            )
        unit_claim_ids: list[str] = []
        for claim_index, raw_claim in enumerate(raw_claims):
            claim_item = _mapping(
                raw_claim,
                f"evidence_units[{index}].claims[{claim_index}]",
            )
            _exact_keys(
                claim_item,
                {"claim", "slot", "kind", "evidence_excerpt", "scope"},
                f"evidence_units[{index}].claims[{claim_index}]",
            )
            claim_text = _bounded_text(
                claim_item["claim"],
                f"evidence_units[{index}].claims[{claim_index}].claim",
                MAX_CLAIM_TEXT_LENGTH,
            )
            kind = _enum(
                claim_item["kind"],
                CLAIM_KINDS,
                f"evidence_units[{index}].claims[{claim_index}].kind",
            )
            slot = _enum(
                claim_item["slot"],
                frozenset(TYPE_SLOTS[concept_type]),
                f"evidence_units[{index}].claims[{claim_index}].slot",
            )
            excerpt = _bounded_text(
                claim_item["evidence_excerpt"],
                f"evidence_units[{index}].claims[{claim_index}].evidence_excerpt",
                MAX_EVIDENCE_EXCERPT_LENGTH,
            )
            if "..." in excerpt or "…" in excerpt:
                raise ContractError(
                    "evidence_excerpt must be one contiguous verbatim source "
                    "span and cannot join evidence with ellipses; split the "
                    "compound claim into one atomic claim per source span, "
                    "preserving each span's own year, number, and scope"
                )
            submitted_anomalies = _source_text_anomalies(excerpt, anomalies)
            if submitted_anomalies:
                raise ContractError(
                    f"evidence_units[{index}].claims[{claim_index}] uses known "
                    f"source-text anomaly {list(submitted_anomalies)}; do not "
                    "correct the source wording. Select a clean claim/excerpt "
                    "from the same evidence bundle, or emit no claim when this "
                    "fact is not required by the Concept type slots"
                )
            excerpt = _canonicalize_submitted_excerpt(unit, excerpt)
            if _normalized_text(excerpt) not in _normalized_text(unit.text):
                nearest = _nearest_verbatim_span(unit, excerpt)
                suggestion = (
                    f"; nearest_verbatim_span={nearest!r}; remove any contextual "
                    "title/framing not present in that span"
                    if nearest
                    else ""
                )
                raise ContractError(
                    "claim evidence excerpt is not verbatim evidence: "
                    f"{evidence_id}; submitted={excerpt[:160]!r}{suggestion}"
                )
            excerpt = _canonicalize_claim_excerpt(unit, claim_text, excerpt)
            source_anomalies = _source_text_anomalies(excerpt, anomalies)
            if source_anomalies:
                raise ContractError(
                    f"evidence_units[{index}].claims[{claim_index}] uses known "
                    f"source-text anomaly {list(source_anomalies)}; do not correct "
                    "the source wording. Select a clean claim/excerpt from the "
                    "same evidence bundle, or emit no claim when this fact is "
                    "not required by the Concept type slots"
                )
            _validate_temporal_qualifiers(
                claim_text,
                excerpt,
                (
                    f"evidence_units[{index}].claims[{claim_index}]"
                    f".evidence_excerpt ({evidence_id})"
                ),
            )
            _validate_hard_anchors(
                claim_text,
                excerpt,
                (
                    f"evidence_units[{index}].claims[{claim_index}]"
                    f".evidence_excerpt ({evidence_id})"
                ),
            )
            scope = _normalize_claim_scope(
                claim_item["scope"],
                label=f"evidence_units[{index}].claims[{claim_index}].scope",
            )
            scope = _validate_claim_scope(
                scope,
                claim=claim_text,
                excerpt=excerpt,
                evidence_context=unit.text,
                ref_scope=normalized_refs[unit.ref_id].get("scope") or {},
            )
            block_ids, page_numbers = _locate_claim_provenance(unit, excerpt)
            claim_id = _claim_id(
                group_id,
                unit.ref_id,
                evidence_id,
                claim_text,
                excerpt,
            )
            if claim_id in seen_claim_ids:
                raise ContractError(f"duplicate claim obligation: {claim_id}")
            seen_claim_ids.add(claim_id)
            unit_claim_ids.append(claim_id)
            contributor_refs.add(unit.ref_id)
            claims.append(
                ClaimObligation(
                    claim_id=claim_id,
                    ref_id=unit.ref_id,
                    evidence_id=evidence_id,
                    claim=claim_text,
                    slot=slot,
                    kind=kind,  # type: ignore[arg-type]
                    evidence_excerpt=excerpt,
                    evidence_block_ids=block_ids,
                    page_numbers=page_numbers,
                    scope=dict(scope),
                    source_text_anomalies=source_anomalies,
                    ocr_suspicions=_ocr_suspicions(excerpt),
                )
            )
        reviews.append(
            EvidenceUnitReview(
                evidence_id=evidence_id,
                disposition=disposition,  # type: ignore[arg-type]
                reason=reason,
                claim_ids=tuple(unit_claim_ids),
            )
        )
    expected_evidence = set(by_evidence)
    if seen_units != expected_evidence:
        missing = sorted(expected_evidence - seen_units)
        raise ContractError(
            "claim contract must classify every evidence unit exactly once; "
            f"missing={missing}"
        )
    active_refs = {
        item.ref_id for item in members if item.relation != "separate"
    }
    missing_contributions = sorted(active_refs - contributor_refs)
    if missing_contributions:
        raise ContractError(
            "every non-separate member must contribute a required claim; "
            f"missing={missing_contributions}"
        )
    if not claims:
        raise ContractError("claim contract must contain at least one required claim")
    return ConceptClaimContract(
        group_id=_text(group_id, "group_id"),
        canonical_question=canonical_question,
        members=tuple(members),
        claims=tuple(claims),
        evidence_units=tuple(reviews),
    )


def _coverage_schema(
    *,
    claim_ids: Sequence[str],
    sentence_ids: Sequence[str],
    rows_status_enum: Sequence[str],
    rows_min_items: int,
    rows_max_items: int,
    rows_excerpt_enum: Sequence[str],
    sentence_min_items: int,
    sentence_max_items: int,
    sentence_excerpt_enum: Sequence[str],
) -> dict[str, Any]:
    """Build one claim-coverage JSON schema with a caller-chosen row/attribution scope."""
    claim_id_schema = {"type": "string", "enum": sorted(claim_ids)}
    sentence_id_schema = {"type": "string", "enum": sorted(sentence_ids)}
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "minItems": rows_min_items,
                "maxItems": rows_max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": claim_id_schema,
                        "status": {
                            "type": "string",
                            "enum": sorted(rows_status_enum),
                        },
                        "draft_excerpt": {
                            "type": "string",
                            "enum": list(rows_excerpt_enum),
                        },
                        "finding": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "claim_id",
                        "status",
                        "draft_excerpt",
                        "finding",
                    ],
                    "additionalProperties": False,
                },
            },
            "sentence_attributions": {
                "type": "array",
                "minItems": sentence_min_items,
                "maxItems": sentence_max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence_id": sentence_id_schema,
                        "status": {
                            "type": "string",
                            "enum": sorted(SENTENCE_ATTRIBUTION_STATUSES),
                        },
                        "claim_ids": {
                            "type": "array",
                            "maxItems": len(claim_ids),
                            "items": claim_id_schema,
                        },
                        "draft_excerpt": {
                            "type": "string",
                            "enum": list(sentence_excerpt_enum),
                        },
                        "finding": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 160,
                        },
                    },
                    "required": [
                        "sentence_id",
                        "status",
                        "claim_ids",
                        "draft_excerpt",
                        "finding",
                    ],
                    "additionalProperties": False,
                },
            },
            "unsupported_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "draft_excerpt": {"type": "string", "minLength": 1},
                        "finding": {"type": "string", "minLength": 1},
                    },
                    "required": ["draft_excerpt", "finding"],
                    "additionalProperties": False,
                },
            },
            "scope_violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": claim_id_schema,
                        },
                        "draft_excerpt": {"type": "string", "minLength": 1},
                        "finding": {"type": "string", "minLength": 1},
                    },
                    "required": ["claim_ids", "draft_excerpt", "finding"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "rows",
            "sentence_attributions",
            "unsupported_claims",
            "scope_violations",
        ],
        "additionalProperties": False,
    }


def claim_coverage_json_schema(
    claim_ids: Sequence[str],
    sentence_ids: Sequence[str],
    draft_sentence_catalog: Sequence[DraftSentence],
) -> dict[str, Any]:
    """Return the structured-output schema for one claim coverage matrix."""
    unique_ids = tuple(dict.fromkeys(_text(item, "claim_id") for item in claim_ids))
    if not unique_ids:
        raise ValueError("at least one claim_id is required")
    unique_sentence_ids = tuple(
        dict.fromkeys(_text(item, "sentence_id") for item in sentence_ids)
    )
    if not unique_sentence_ids:
        raise ValueError("at least one sentence_id is required")
    catalog = tuple(draft_sentence_catalog)
    catalog_ids = tuple(item.sentence_id for item in catalog)
    if len(catalog_ids) != len(set(catalog_ids)) or set(catalog_ids) != set(
        unique_sentence_ids
    ):
        raise ValueError("draft sentence catalog must match sentence_ids exactly")
    exact_sentence_texts = tuple(
        dict.fromkeys(_text(item.text, "draft sentence text") for item in catalog)
    )
    return _coverage_schema(
        claim_ids=unique_ids,
        sentence_ids=unique_sentence_ids,
        rows_status_enum=COVERAGE_STATUSES,
        rows_min_items=len(unique_ids),
        rows_max_items=len(unique_ids),
        rows_excerpt_enum=("", *exact_sentence_texts),
        sentence_min_items=len(unique_sentence_ids),
        sentence_max_items=len(unique_sentence_ids),
        sentence_excerpt_enum=exact_sentence_texts,
    )


def chunk_draft_sentences(
    sentences: Sequence[DraftSentence],
    *,
    batch_size: int,
) -> tuple[tuple[DraftSentence, ...], ...]:
    """Split auditable draft sentences into deterministic contiguous batches.

    Batch membership is a pure function of the sentence catalog and the batch
    size, so the same draft always yields the same batches.  The union of
    batches equals the full model-visible sentence catalog in order, and every
    batch holds at most ``batch_size`` sentences.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("coverage batch size must be a positive integer")
    catalog = tuple(sentences)
    return tuple(
        catalog[position : position + batch_size]
        for position in range(0, len(catalog), batch_size)
    )


def claim_coverage_batch_json_schema(
    claim_ids: Sequence[str],
    batch_sentences: Sequence[DraftSentence],
) -> dict[str, Any]:
    """Return the structured-output schema for one deterministic sentence batch.

    The model assesses only the claims it can express, contradict or doubt
    from this batch's sentences, so ``rows`` may legitimately be empty and
    never contains ``omitted`` status (omitted rows are completed by code when
    batches are merged).  Sentence attributions must cover this batch exactly
    once, which keeps the full matrix's exact-once catalog contract intact.
    """
    unique_ids = tuple(dict.fromkeys(_text(item, "claim_id") for item in claim_ids))
    if not unique_ids:
        raise ValueError("at least one claim_id is required")
    batch = tuple(batch_sentences)
    batch_ids = tuple(item.sentence_id for item in batch)
    if not batch_ids:
        raise ValueError("a coverage batch requires at least one sentence")
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("a coverage batch cannot contain duplicate sentences")
    exact_sentence_texts = tuple(
        dict.fromkeys(_text(item.text, "draft sentence text") for item in batch)
    )
    return _coverage_schema(
        claim_ids=unique_ids,
        sentence_ids=batch_ids,
        rows_status_enum=("covered", "contradicted", "uncertain"),
        rows_min_items=0,
        rows_max_items=len(unique_ids),
        rows_excerpt_enum=exact_sentence_texts,
        sentence_min_items=len(batch_ids),
        sentence_max_items=len(batch_ids),
        sentence_excerpt_enum=exact_sentence_texts,
    )


def parse_claim_coverage(
    response: str,
    *,
    contract: ConceptClaimContract,
    draft: Mapping[str, Any],
    known_source_anomalies: Sequence[str] = (),
) -> ClaimCoverageMatrix:
    """Parse a coverage matrix and derive its non-model release decision."""
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    payload = _object(response, "claim coverage")
    _exact_keys(
        payload,
        {
            "rows",
            "sentence_attributions",
            "unsupported_claims",
            "scope_violations",
        },
        "claim coverage",
    )
    claims = {item.claim_id: item for item in contract.claims}
    draft_text = _draft_text(draft)
    draft_sentences = build_draft_sentences(draft)
    sentences = {item.sentence_id: item for item in draft_sentences}
    rows: list[ClaimCoverageRow] = []
    seen_claims: set[str] = set()
    for index, raw in enumerate(_array(payload["rows"], "rows")):
        item = _mapping(raw, f"rows[{index}]")
        _exact_keys(
            item,
            {"claim_id", "status", "draft_excerpt", "finding"},
            f"rows[{index}]",
        )
        claim_id = _text(item["claim_id"], f"rows[{index}].claim_id")
        if claim_id not in claims:
            raise ContractError(f"unknown coverage claim_id: {claim_id}")
        if claim_id in seen_claims:
            raise ContractError(f"duplicate coverage claim_id: {claim_id}")
        seen_claims.add(claim_id)
        status = _enum(
            item["status"], COVERAGE_STATUSES, f"rows[{index}].status"
        )
        excerpt = item["draft_excerpt"]
        if not isinstance(excerpt, str):
            raise ContractError(f"rows[{index}].draft_excerpt must be a string")
        excerpt = excerpt.strip()
        if status == "omitted":
            if excerpt:
                raise ContractError("omitted claims require an empty draft_excerpt")
        else:
            _validate_draft_excerpt(
                excerpt,
                draft_text,
                f"rows[{index}].draft_excerpt",
            )
            if excerpt not in {sentence.text for sentence in draft_sentences}:
                raise ContractError(
                    f"rows[{index}].draft_excerpt must equal one deterministic "
                    "draft sentence"
                )
        rows.append(
            ClaimCoverageRow(
                claim_id=claim_id,
                status=status,  # type: ignore[arg-type]
                draft_excerpt=excerpt,
                finding=_text(item["finding"], f"rows[{index}].finding"),
            )
        )
    expected_claims = set(claims)
    if seen_claims != expected_claims:
        missing = sorted(expected_claims - seen_claims)
        raise ContractError(
            "coverage matrix must assess every claim exactly once; "
            f"missing={missing}"
        )

    unsupported: list[UnsupportedClaim] = []
    for index, raw in enumerate(
        _array(payload["unsupported_claims"], "unsupported_claims")
    ):
        item = _mapping(raw, f"unsupported_claims[{index}]")
        _exact_keys(
            item,
            {"draft_excerpt", "finding"},
            f"unsupported_claims[{index}]",
        )
        excerpt = _text(
            item["draft_excerpt"],
            f"unsupported_claims[{index}].draft_excerpt",
        )
        _validate_draft_excerpt(
            excerpt,
            draft_text,
            f"unsupported_claims[{index}].draft_excerpt",
        )
        unsupported.append(
            UnsupportedClaim(
                draft_excerpt=excerpt,
                finding=_text(
                    item["finding"], f"unsupported_claims[{index}].finding"
                ),
            )
        )
    reported_excerpts = {
        _normalized_text(item.draft_excerpt) for item in unsupported
    }
    for anchor in _unsupported_draft_anchors(draft_text, contract.claims):
        if _normalized_text(anchor) in reported_excerpts:
            continue
        unsupported.append(
            UnsupportedClaim(
                draft_excerpt=anchor,
                finding=(
                    "代码门禁发现草稿硬锚点未出现在 Claim Contract 的逐字证据中。"
                ),
            )
        )
        reported_excerpts.add(_normalized_text(anchor))

    sentence_attributions: list[DraftSentenceAttribution] = []
    seen_sentences: set[str] = set()
    inference_rejected_claim_ids: set[str] = set()
    for index, raw in enumerate(
        _array(payload["sentence_attributions"], "sentence_attributions")
    ):
        item = _mapping(raw, f"sentence_attributions[{index}]")
        _exact_keys(
            item,
            {"sentence_id", "status", "claim_ids", "draft_excerpt", "finding"},
            f"sentence_attributions[{index}]",
        )
        sentence_id = _text(
            item["sentence_id"], f"sentence_attributions[{index}].sentence_id"
        )
        if sentence_id not in sentences:
            raise ContractError(f"unknown draft sentence_id: {sentence_id}")
        if sentence_id in seen_sentences:
            raise ContractError(f"duplicate draft sentence_id: {sentence_id}")
        seen_sentences.add(sentence_id)
        status = _enum(
            item["status"],
            SENTENCE_ATTRIBUTION_STATUSES,
            f"sentence_attributions[{index}].status",
        )
        attribution_ids = tuple(
            _text(value, f"sentence_attributions[{index}].claim_ids")
            for value in _array(
                item["claim_ids"], f"sentence_attributions[{index}].claim_ids"
            )
        )
        if len(attribution_ids) != len(set(attribution_ids)):
            raise ContractError("sentence attribution claim_ids must be unique")
        unknown = set(attribution_ids) - set(claims)
        if unknown:
            raise ContractError(
                f"unknown sentence attribution claim_id: {sorted(unknown)[0]}"
            )
        missing_supported_ids = status == "supported" and not attribution_ids
        if missing_supported_ids:
            # A sentence without a cited Claim cannot be supported.  Convert
            # the model's self-contradictory labels into a safe draft defect so
            # the normal recompile path can repair it.
            status = "unsupported"
        if status == "unsupported" and attribution_ids:
            attribution_ids = ()
        excerpt = _text(
            item["draft_excerpt"],
            f"sentence_attributions[{index}].draft_excerpt",
        )
        sentence = sentences[sentence_id]
        _validate_draft_excerpt(
            excerpt,
            draft_text,
            f"sentence_attributions[{index}].draft_excerpt",
        )
        if excerpt != sentence.text:
            raise ContractError(
                "sentence attribution excerpt must equal its deterministic "
                f"draft sentence: {sentence_id}"
            )
        finding = _bounded_text(
            item["finding"],
            f"sentence_attributions[{index}].finding",
            160,
        )
        if missing_supported_ids:
            finding = (
                "代码门禁将 supported 降级为 unsupported：该草稿句没有归因到"
                "任何 Claim ID。"
            )
        source_text_anomalies = _source_text_anomalies(excerpt, anomalies)
        ocr_suspicions = _ocr_suspicions(excerpt)
        inference_phrases = (
            _unsupported_inference_phrases(
                excerpt,
                tuple(claims[claim_id] for claim_id in attribution_ids),
            )
            if attribution_ids
            else ()
        )
        if (
            status == "uncertain"
            and inference_phrases
            and not source_text_anomalies
            and not ocr_suspicions
        ):
            # This is no longer a genuinely ambiguous attribution: the model
            # identified the candidate claims, while code can prove that the
            # sentence's strong inference wording is absent from every cited
            # claim's verbatim evidence.  Reclassify it as a correctable draft
            # defect; other uncertain cases remain human-review decisions.
            inference_rejected_claim_ids.update(attribution_ids)
            status = "unsupported"
            attribution_ids = ()
            finding = (
                "代码门禁将 uncertain 降级为 unsupported：强判断措辞"
                "未出现在归因 claim 的逐字证据中。"
            )
        sentence_attributions.append(
            DraftSentenceAttribution(
                sentence_id=sentence_id,
                status=status,  # type: ignore[arg-type]
                claim_ids=attribution_ids,
                draft_excerpt=excerpt,
                finding=finding,
                source_text_anomalies=source_text_anomalies,
                ocr_suspicions=ocr_suspicions,
            )
        )
        deterministic_unsupported = status == "unsupported" or (
            status == "supported"
            and bool(inference_phrases)
        )
        if deterministic_unsupported and _normalized_text(excerpt) not in reported_excerpts:
            unsupported.append(
                UnsupportedClaim(
                    draft_excerpt=excerpt,
                    finding=(
                        finding
                        if status == "unsupported"
                        else "代码门禁发现强判断措辞未出现在归因 claim 的原始证据中。"
                    ),
                )
            )
            reported_excerpts.add(_normalized_text(excerpt))
    expected_sentences = set(sentences)
    if seen_sentences != expected_sentences:
        missing = sorted(expected_sentences - seen_sentences)
        raise ContractError(
            "coverage matrix must audit every deterministic draft sentence "
            f"exactly once; missing={missing}"
        )

    # A row may name one primary sentence, while one claim can legitimately be
    # expressed across multiple draft sentences.  Therefore covered-claim
    # anchors are checked against the complete set of *supported* sentence
    # attributions for that claim, rather than against the row excerpt alone.
    for row_index, row in enumerate(rows):
        if row.status != "covered":
            continue
        supported = tuple(
            item
            for item in sentence_attributions
            if item.status == "supported" and row.claim_id in item.claim_ids
        )
        if not supported:
            if row.claim_id in inference_rejected_claim_ids:
                rows[row_index] = ClaimCoverageRow(
                    claim_id=row.claim_id,
                    status="omitted",
                    draft_excerpt="",
                    finding=(
                        f"{row.finding} 代码门禁降级为 omitted：承载该 claim "
                        "的草稿句含无逐字证据支持的强判断措辞。"
                    ),
                )
                continue
            # The row and sentence table are two views of the same semantic
            # audit.  When the model marks a claim covered but cannot name any
            # supported sentence, the safe interpretation is that the draft
            # omitted the claim.  Treat this as a correctable content defect,
            # not a malformed JSON contract that aborts the whole run.
            rows[row_index] = ClaimCoverageRow(
                claim_id=row.claim_id,
                status="omitted",
                draft_excerpt="",
                finding=(
                    f"{row.finding} 代码门禁降级为 omitted：没有任何受支持的"
                    "草稿句归因到该 claim。"
                ),
            )
            continue
        if row.draft_excerpt not in {
            item.draft_excerpt for item in supported
        }:
            if row.claim_id in inference_rejected_claim_ids:
                rows[row_index] = ClaimCoverageRow(
                    claim_id=row.claim_id,
                    status="omitted",
                    draft_excerpt="",
                    finding=(
                        f"{row.finding} 代码门禁降级为 omitted：row 所引用的"
                        "草稿句含无逐字证据支持的强判断措辞。"
                    ),
                )
                continue
            # The claim is supported, but the redundant primary-sentence
            # pointer is inconsistent.  Canonicalize it to the first supported
            # sentence so downstream provenance remains exact and deterministic.
            rows[row_index] = ClaimCoverageRow(
                claim_id=row.claim_id,
                status="covered",
                draft_excerpt=supported[0].draft_excerpt,
                finding=(
                    f"{row.finding} 代码门禁已将主句指针规范化为该 claim 的"
                    "首个受支持草稿句。"
                ),
            )
            row = rows[row_index]
        combined_excerpt = "\n".join(item.draft_excerpt for item in supported)
        try:
            _validate_temporal_qualifiers(
                claims[row.claim_id].claim,
                combined_excerpt,
                f"covered claim attribution {row.claim_id}",
            )
            _validate_hard_anchors(
                claims[row.claim_id].claim,
                combined_excerpt,
                f"covered claim attribution {row.claim_id}",
            )
        except ContractError as exc:
            # The response is structurally valid, but the draft does not
            # actually satisfy this frozen claim (for example, it uses a
            # shortened policy alias or drops an exact temporal qualifier).
            # Preserve the request and turn the mismatch into a deterministic
            # recompile signal instead of spending another model retry on the
            # same already-valid JSON shape.
            rows[row_index] = ClaimCoverageRow(
                claim_id=row.claim_id,
                status="omitted",
                draft_excerpt="",
                finding=(
                    f"{row.finding} 代码门禁降级为 omitted：{exc}"
                ),
            )

    scope_violations: list[ScopeViolation] = []
    for index, raw in enumerate(
        _array(payload["scope_violations"], "scope_violations")
    ):
        item = _mapping(raw, f"scope_violations[{index}]")
        _exact_keys(
            item,
            {"claim_ids", "draft_excerpt", "finding"},
            f"scope_violations[{index}]",
        )
        raw_ids = _array(
            item["claim_ids"], f"scope_violations[{index}].claim_ids"
        )
        violation_ids = tuple(
            _text(value, f"scope_violations[{index}].claim_ids")
            for value in raw_ids
        )
        if not violation_ids or len(violation_ids) != len(set(violation_ids)):
            raise ContractError(
                "scope violation claim_ids must be non-empty and unique"
            )
        unknown = set(violation_ids) - expected_claims
        if unknown:
            raise ContractError(
                f"unknown scope violation claim_id: {sorted(unknown)[0]}"
            )
        excerpt = _text(
            item["draft_excerpt"],
            f"scope_violations[{index}].draft_excerpt",
        )
        _validate_draft_excerpt(
            excerpt,
            draft_text,
            f"scope_violations[{index}].draft_excerpt",
        )
        scope_violations.append(
            ScopeViolation(
                claim_ids=violation_ids,
                draft_excerpt=excerpt,
                finding=_text(
                    item["finding"], f"scope_violations[{index}].finding"
                ),
            )
        )
    decision = derive_review_decision(
        contract,
        rows,
        sentence_attributions=sentence_attributions,
        unsupported_claims=unsupported,
        scope_violations=scope_violations,
    )
    return ClaimCoverageMatrix(
        rows=tuple(rows),
        sentence_attributions=tuple(sentence_attributions),
        unsupported_claims=tuple(unsupported),
        scope_violations=tuple(scope_violations),
        decision=decision,
    )


def parse_claim_coverage_batch(
    response: str,
    *,
    contract: ConceptClaimContract,
    batch_index: int,
    batch_sentences: Sequence[DraftSentence],
) -> ClaimCoverageBatch:
    """Parse one deterministic sentence batch's coverage fragment.

    The model sees only this batch's sentences together with the full frozen
    Claim Contract.  Rows may reference only contract claims and only
    ``covered``/``contradicted``/``uncertain`` statuses: a model-produced
    ``omitted`` row is rejected because the merge completes omitted rows
    deterministically.  Every excerpt must be verbatim text of this batch, and
    sentence_attributions must cover this batch's sentence_ids exactly once.
    A claim expressed by several sentences of the same batch yields one row
    per supporting sentence; these duplicates are converged to a single row
    with the most severe status (``contradicted`` > ``uncertain`` >
    ``covered``), ties keeping the row whose excerpt appears earlier in the
    batch's sentence order.  The full-matrix deterministic gates (hard
    anchors, temporal qualifiers, inference wording, row downgrades)
    intentionally run once on the merged matrix, where a claim may
    legitimately be supported across batches.
    """
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or batch_index < 0
    ):
        raise ValueError("batch index must be a non-negative integer")
    payload = _object(response, "claim coverage batch")
    _exact_keys(
        payload,
        {
            "rows",
            "sentence_attributions",
            "unsupported_claims",
            "scope_violations",
        },
        "claim coverage batch",
    )
    claims = {item.claim_id: item for item in contract.claims}
    batch = tuple(batch_sentences)
    sentences = {item.sentence_id: item for item in batch}
    batch_text = "\n".join(item.text for item in batch)

    row_order: list[str] = []
    rows_by_claim: dict[str, ClaimCoverageRow] = {}
    for index, raw in enumerate(_array(payload["rows"], "rows")):
        item = _mapping(raw, f"rows[{index}]")
        _exact_keys(
            item,
            {"claim_id", "status", "draft_excerpt", "finding"},
            f"rows[{index}]",
        )
        claim_id = _text(item["claim_id"], f"rows[{index}].claim_id")
        if claim_id not in claims:
            raise ContractError(f"unknown coverage claim_id: {claim_id}")
        status = _enum(
            item["status"], COVERAGE_STATUSES, f"rows[{index}].status"
        )
        if status == "omitted":
            raise ContractError(
                "batch coverage rows must not mark claims omitted; omitted "
                "rows are completed deterministically when batches are merged"
            )
        excerpt = item["draft_excerpt"]
        if not isinstance(excerpt, str):
            raise ContractError(f"rows[{index}].draft_excerpt must be a string")
        excerpt = excerpt.strip()
        _validate_draft_excerpt(
            excerpt,
            batch_text,
            f"rows[{index}].draft_excerpt",
        )
        if excerpt not in {sentence.text for sentence in batch}:
            raise ContractError(
                f"rows[{index}].draft_excerpt must equal one deterministic "
                "sentence of this batch"
            )
        row = ClaimCoverageRow(
            claim_id=claim_id,
            status=status,  # type: ignore[arg-type]
            draft_excerpt=excerpt,
            finding=_text(item["finding"], f"rows[{index}].finding"),
        )
        existing = rows_by_claim.get(claim_id)
        if existing is None:
            row_order.append(claim_id)
            rows_by_claim[claim_id] = row
            continue
        # A claim expressed by several sentences of the same batch may be
        # listed once per sentence (the model emits one row per supporting
        # sentence).  Mirroring the single-matrix semantics and the
        # cross-batch merge rule, converge duplicates deterministically:
        # severity first, then the row whose excerpt appears earlier in the
        # batch's sentence order (with that row's excerpt and finding).
        if _COVERAGE_ROW_PRIORITY[row.status] > _COVERAGE_ROW_PRIORITY[
            existing.status
        ]:
            rows_by_claim[claim_id] = row
            continue
        if _COVERAGE_ROW_PRIORITY[row.status] == _COVERAGE_ROW_PRIORITY[
            existing.status
        ] and _batch_sentence_position(
            row.draft_excerpt, batch
        ) < _batch_sentence_position(
            existing.draft_excerpt, batch
        ):
            rows_by_claim[claim_id] = row
    rows = [rows_by_claim[claim_id] for claim_id in row_order]

    sentence_attributions: list[DraftSentenceAttribution] = []
    seen_sentences: set[str] = set()
    for index, raw in enumerate(
        _array(payload["sentence_attributions"], "sentence_attributions")
    ):
        item = _mapping(raw, f"sentence_attributions[{index}]")
        _exact_keys(
            item,
            {"sentence_id", "status", "claim_ids", "draft_excerpt", "finding"},
            f"sentence_attributions[{index}]",
        )
        sentence_id = _text(
            item["sentence_id"], f"sentence_attributions[{index}].sentence_id"
        )
        if sentence_id not in sentences:
            raise ContractError(
                f"sentence_id is not part of this coverage batch: {sentence_id}"
            )
        if sentence_id in seen_sentences:
            raise ContractError(f"duplicate draft sentence_id: {sentence_id}")
        seen_sentences.add(sentence_id)
        status = _enum(
            item["status"],
            SENTENCE_ATTRIBUTION_STATUSES,
            f"sentence_attributions[{index}].status",
        )
        attribution_ids = tuple(
            _text(value, f"sentence_attributions[{index}].claim_ids")
            for value in _array(
                item["claim_ids"], f"sentence_attributions[{index}].claim_ids"
            )
        )
        if len(attribution_ids) != len(set(attribution_ids)):
            raise ContractError("sentence attribution claim_ids must be unique")
        unknown = set(attribution_ids) - set(claims)
        if unknown:
            raise ContractError(
                f"unknown sentence attribution claim_id: {sorted(unknown)[0]}"
            )
        sentence = sentences[sentence_id]
        excerpt = _text(
            item["draft_excerpt"],
            f"sentence_attributions[{index}].draft_excerpt",
        )
        if excerpt != sentence.text:
            raise ContractError(
                "sentence attribution excerpt must equal its deterministic "
                f"draft sentence: {sentence_id}"
            )
        finding = _bounded_text(
            item["finding"],
            f"sentence_attributions[{index}].finding",
            160,
        )
        sentence_attributions.append(
            DraftSentenceAttribution(
                sentence_id=sentence_id,
                status=status,  # type: ignore[arg-type]
                claim_ids=attribution_ids,
                draft_excerpt=excerpt,
                finding=finding,
            )
        )
    expected_batch = set(sentences)
    if seen_sentences != expected_batch:
        missing = sorted(expected_batch - seen_sentences)
        raise ContractError(
            "batch coverage must audit every batch sentence exactly once; "
            f"missing={missing}"
        )

    unsupported: list[UnsupportedClaim] = []
    for index, raw in enumerate(
        _array(payload["unsupported_claims"], "unsupported_claims")
    ):
        item = _mapping(raw, f"unsupported_claims[{index}]")
        _exact_keys(
            item,
            {"draft_excerpt", "finding"},
            f"unsupported_claims[{index}]",
        )
        excerpt = _text(
            item["draft_excerpt"],
            f"unsupported_claims[{index}].draft_excerpt",
        )
        _validate_draft_excerpt(
            excerpt,
            batch_text,
            f"unsupported_claims[{index}].draft_excerpt",
        )
        unsupported.append(
            UnsupportedClaim(
                draft_excerpt=excerpt,
                finding=_text(
                    item["finding"], f"unsupported_claims[{index}].finding"
                ),
            )
        )

    scope_violations: list[ScopeViolation] = []
    for index, raw in enumerate(
        _array(payload["scope_violations"], "scope_violations")
    ):
        item = _mapping(raw, f"scope_violations[{index}]")
        _exact_keys(
            item,
            {"claim_ids", "draft_excerpt", "finding"},
            f"scope_violations[{index}]",
        )
        raw_ids = _array(
            item["claim_ids"], f"scope_violations[{index}].claim_ids"
        )
        violation_ids = tuple(
            _text(value, f"scope_violations[{index}].claim_ids")
            for value in raw_ids
        )
        if not violation_ids or len(violation_ids) != len(set(violation_ids)):
            raise ContractError(
                "scope violation claim_ids must be non-empty and unique"
            )
        unknown = set(violation_ids) - set(claims)
        if unknown:
            raise ContractError(
                f"unknown scope violation claim_id: {sorted(unknown)[0]}"
            )
        excerpt = _text(
            item["draft_excerpt"],
            f"scope_violations[{index}].draft_excerpt",
        )
        _validate_draft_excerpt(
            excerpt,
            batch_text,
            f"scope_violations[{index}].draft_excerpt",
        )
        scope_violations.append(
            ScopeViolation(
                claim_ids=violation_ids,
                draft_excerpt=excerpt,
                finding=_text(
                    item["finding"], f"scope_violations[{index}].finding"
                ),
            )
        )

    return ClaimCoverageBatch(
        batch_index=batch_index,
        rows=tuple(rows),
        sentence_attributions=tuple(sentence_attributions),
        unsupported_claims=tuple(unsupported),
        scope_violations=tuple(scope_violations),
    )


# Batch rows carry one status per claim; when the same claim is legitimately
# expressed by several sentences (a single-matrix model would have seen all
# sentences at once and produced one overall judgment), duplicate rows are
# converged deterministically toward the most severe status so a defect can
# never hide behind a milder label.  The same rule applies within a batch
# (one row per supporting sentence) and across batches (claim spanning the
# batch boundary).
_COVERAGE_ROW_PRIORITY = {
    "contradicted": 3,
    "uncertain": 2,
    "covered": 1,
}


def _batch_sentence_position(
    excerpt: str,
    batch: Sequence[DraftSentence],
) -> int:
    """Return the batch position of the first sentence equal to ``excerpt``."""
    for position, sentence in enumerate(batch):
        if sentence.text == excerpt:
            return position
    return len(batch)


def merge_claim_coverage_batches(
    batches: Mapping[int, ClaimCoverageBatch],
    *,
    contract: ConceptClaimContract,
    draft: Mapping[str, Any],
    known_source_anomalies: Sequence[str] = (),
) -> ClaimCoverageMatrix:
    """Merge per-batch coverage fragments into one full Claim Coverage matrix.

    Batches are consumed in ascending batch index order so the merged result
    does not depend on the order in which batches completed.  Every frozen
    claim is assessed exactly once: a claim missing from every batch is
    completed as an ``omitted`` row by code.  A claim reported by several
    batches (the same claim can be expressed across the batch boundary) is
    converged to one row with the most severe status
    (``contradicted`` > ``uncertain`` > ``covered``); ties keep the earliest
    batch's row.  This mirrors the single-matrix semantics, where one model
    call sees all supporting sentences at once and emits one status per claim.
    Sentence attributions must still cover every draft sentence exactly once
    across batches: batches partition the sentence catalog, so a duplicated
    sentence_id remains a hard contract error.  The merged payload is then
    parsed by :func:`parse_claim_coverage`, so every deterministic gate
    (verbatim excerpts, exact-once catalogs, hard anchors, temporal
    qualifiers, inference wording, row downgrades) applies unchanged to the
    full Concept.
    """
    ordered = tuple(batches[index] for index in sorted(batches))
    expected_claims = {item.claim_id for item in contract.claims}
    expected_sentences = {
        item.sentence_id for item in build_draft_sentences(draft)
    }
    row_order: list[str] = []
    row_by_claim: dict[str, ClaimCoverageRow] = {}
    sentence_attributions: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    scope_violations: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    for batch in ordered:
        for row in batch.rows:
            existing = row_by_claim.get(row.claim_id)
            if existing is None:
                row_order.append(row.claim_id)
                row_by_claim[row.claim_id] = row
                continue
            if _COVERAGE_ROW_PRIORITY[row.status] > _COVERAGE_ROW_PRIORITY[
                existing.status
            ]:
                # Converge to the more severe status; equal status keeps the
                # earliest batch's row because batches arrive in index order.
                row_by_claim[row.claim_id] = row
        for item in batch.sentence_attributions:
            if item.sentence_id in seen_sentences:
                raise ContractError(
                    "draft sentence audited in more than one coverage batch: "
                    f"{item.sentence_id}"
                )
            seen_sentences.add(item.sentence_id)
            sentence_attributions.append(
                {
                    "sentence_id": item.sentence_id,
                    "status": item.status,
                    "claim_ids": list(item.claim_ids),
                    "draft_excerpt": item.draft_excerpt,
                    "finding": item.finding,
                }
            )
        unsupported_claims.extend(
            {"draft_excerpt": item.draft_excerpt, "finding": item.finding}
            for item in batch.unsupported_claims
        )
        scope_violations.extend(
            {
                "claim_ids": list(item.claim_ids),
                "draft_excerpt": item.draft_excerpt,
                "finding": item.finding,
            }
            for item in batch.scope_violations
        )
    missing_sentences = expected_sentences - seen_sentences
    if missing_sentences:
        raise ContractError(
            "coverage batches must audit every deterministic draft sentence "
            f"exactly once; missing={sorted(missing_sentences)}"
        )
    missing_claims = sorted(expected_claims - set(row_by_claim))
    rows = [
        {
            "claim_id": claim_id,
            "status": row_by_claim[claim_id].status,
            "draft_excerpt": row_by_claim[claim_id].draft_excerpt,
            "finding": row_by_claim[claim_id].finding,
        }
        for claim_id in row_order
    ]
    rows.extend(
        {
            "claim_id": claim_id,
            "status": "omitted",
            "draft_excerpt": "",
            "finding": (
                "代码门禁补全为 omitted：该 claim 未在任何批次的草稿句中被表达。"
            ),
        }
        for claim_id in missing_claims
    )
    payload = {
        "rows": rows,
        "sentence_attributions": sentence_attributions,
        "unsupported_claims": unsupported_claims,
        "scope_violations": scope_violations,
    }
    return parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=draft,
        known_source_anomalies=known_source_anomalies,
    )


def derive_review_decision(
    contract: ConceptClaimContract,
    rows: Sequence[ClaimCoverageRow],
    *,
    sentence_attributions: Sequence[DraftSentenceAttribution] = (),
    unsupported_claims: Sequence[UnsupportedClaim] = (),
    scope_violations: Sequence[ScopeViolation] = (),
) -> ReviewDecision:
    """Apply the deterministic Claim Review release gate."""
    if (
        any(item.relation == "separate" for item in contract.members)
        or any(item.status == "uncertain" for item in rows)
        or any(item.status == "uncertain" for item in sentence_attributions)
        or any(item.source_text_anomalies for item in contract.claims)
        or any(item.ocr_suspicions for item in contract.claims)
        or any(item.source_text_anomalies for item in sentence_attributions)
        or any(item.ocr_suspicions for item in sentence_attributions)
    ):
        return "human_review"
    if (
        any(item.status in {"omitted", "contradicted"} for item in rows)
        or unsupported_claims
        or scope_violations
    ):
        return "recompile"
    return "pass"


def _normalize_refs(
    refs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(refs):
        if not isinstance(raw, Mapping):
            raise ContractError(f"refs[{index}] must be an object")
        ref_id = _text(raw.get("ref_id"), f"refs[{index}].ref_id")
        if ref_id in normalized:
            raise ContractError(f"duplicate ref_id: {ref_id}")
        concept_type = _text(raw.get("type"), f"refs[{index}].type")
        if concept_type not in TYPE_SLOTS:
            raise ContractError(f"unsupported Concept type: {concept_type}")
        evidence = raw.get("evidence")
        evidence_blocks = raw.get("evidence_blocks")
        if evidence is not None and not isinstance(evidence, (list, tuple)):
            raise ContractError(f"refs[{index}].evidence must be an array")
        if evidence_blocks is not None and not isinstance(
            evidence_blocks, (list, tuple)
        ):
            raise ContractError(
                f"refs[{index}].evidence_blocks must be an array"
            )
        raw_scope = raw.get("scope")
        if raw_scope is not None and not isinstance(raw_scope, Mapping):
            raise ContractError(f"refs[{index}].scope must be an object")
        has_evidence = isinstance(evidence, (list, tuple)) and bool(evidence)
        has_blocks = isinstance(evidence_blocks, (list, tuple)) and bool(
            evidence_blocks
        )
        if not has_evidence and not has_blocks:
            raise ContractError(
                f"refs[{index}] requires evidence or evidence_blocks"
            )
        normalized[ref_id] = {
            **dict(raw),
            "ref_id": ref_id,
            "type": concept_type,
            "evidence": tuple(
                _text(item, f"refs[{index}].evidence")
                for item in (evidence or ())
            ),
            "evidence_blocks": tuple(evidence_blocks or ()),
        }
    if not normalized:
        raise ContractError("at least one Ref is required")
    return normalized


def _concept_type(refs: Mapping[str, Mapping[str, Any]]) -> str:
    types = {str(item["type"]) for item in refs.values()}
    if len(types) != 1:
        raise ContractError("a Claim Contract requires one Concept type")
    return next(iter(types))


def _claim_id(
    group_id: str,
    ref_id: str,
    evidence_id: str,
    claim: str,
    excerpt: str,
) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                group_id,
                ref_id,
                evidence_id,
                _normalized_text(claim),
                _normalized_text(excerpt),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"claim-{digest}"


def _locate_claim_provenance(
    unit: EvidenceUnit,
    excerpt: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not unit.source_blocks:
        return (), ()
    normalized_excerpt = _normalized_text(excerpt)
    matches = tuple(
        block
        for block in unit.source_blocks
        if normalized_excerpt in _normalized_text(block.text)
    )
    if not matches:
        adjacent_matches = tuple(
            (left, right)
            for left, right in zip(unit.source_blocks, unit.source_blocks[1:])
            if _source_blocks_are_contiguous(left, right)
            and normalized_excerpt
            in _normalized_text(f"{left.text}\n\n{right.text}")
        )
        if adjacent_matches:
            blocks = tuple(
                dict.fromkeys(
                    block
                    for pair in adjacent_matches
                    for block in pair
                )
            )
            return (
                tuple(block.block_id for block in blocks),
                tuple(
                    dict.fromkeys(
                        block.page_number
                        for block in blocks
                        if block.page_number is not None
                    )
                ),
            )
    if not matches:
        raise ContractError(
            "claim excerpt cannot be located in one source evidence block: "
            f"{unit.evidence_id}"
        )
    return (
        tuple(block.block_id for block in matches),
        tuple(
            dict.fromkeys(
                block.page_number
                for block in matches
                if block.page_number is not None
            )
        ),
    )


_MARKDOWN_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+(?:\.\d+)*[.)、．]\s*)"
)
_SENTENCE_PART_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
_IGNORABLE_EXCERPT_PUNCTUATION = frozenset(
    "，,。；;：:！？!?、·“”‘’\"'（）()【】[]"
)


def _excerpt_match_text_with_offsets(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    offsets: list[int] = []
    in_inline_math = False
    for index, character in enumerate(value):
        if value[index : index + 2] == r"\(":
            in_inline_math = True
        elif value[index : index + 2] == r"\)":
            in_inline_math = False
        if (
            character == "\\"
            and index + 1 < len(value)
            and value[index + 1] in "()%"
        ):
            continue
        if in_inline_math and character in "^{}":
            continue
        if character.isspace() or character in _IGNORABLE_EXCERPT_PUNCTUATION:
            continue
        characters.append(character)
        offsets.append(index)
    return "".join(characters), tuple(offsets)


def _canonicalize_submitted_excerpt(unit: EvidenceUnit, excerpt: str) -> str:
    """Recover only punctuation/whitespace variants as exact source text.

    A model may copy the right source span while normalizing Chinese quotes or
    commas.  This helper maps that formatting-only variant back to one
    contiguous source substring.  It deliberately does not fuzzy-match words,
    join blocks, or repair paraphrases.
    """
    if _normalized_text(excerpt) in _normalized_text(unit.text):
        return excerpt
    needle, _ = _excerpt_match_text_with_offsets(excerpt)
    if not needle:
        return excerpt
    candidates: list[tuple[int, int, int, str]] = []
    for block_count, block_position, source in _source_text_windows(unit):
        normalized, offsets = _excerpt_match_text_with_offsets(source)
        start = normalized.find(needle)
        while start >= 0:
            original_start = offsets[start]
            original_end = offsets[start + len(needle) - 1] + 1
            candidate = source[original_start:original_end].strip()
            if candidate and len(candidate) <= MAX_EVIDENCE_EXCERPT_LENGTH:
                candidates.append(
                    (block_count, block_position, original_start, candidate)
                )
            start = normalized.find(needle, start + 1)
    if not candidates:
        return excerpt
    return min(candidates, key=lambda item: item)[3]


def _source_text_windows(unit: EvidenceUnit) -> tuple[tuple[int, int, str], ...]:
    if not unit.source_blocks:
        return ((1, 0, unit.text),)
    windows: list[tuple[int, int, str]] = [
        (1, index, block.text)
        for index, block in enumerate(unit.source_blocks)
    ]
    windows.extend(
        (2, index, f"{left.text}\n\n{right.text}")
        for index, (left, right) in enumerate(
            zip(unit.source_blocks, unit.source_blocks[1:])
        )
        if _source_blocks_are_contiguous(left, right)
    )
    return tuple(windows)


def _nearest_verbatim_span(unit: EvidenceUnit, excerpt: str) -> str:
    """Return a long exact source fragment for repair guidance, never acceptance."""
    needle, _ = _excerpt_match_text_with_offsets(excerpt)
    if not needle:
        return ""
    candidates: list[tuple[int, int, int, str]] = []
    for block_count, block_position, source in _source_text_windows(unit):
        normalized, offsets = _excerpt_match_text_with_offsets(source)
        match = SequenceMatcher(None, needle, normalized, autojunk=False).find_longest_match()
        if match.size < 20:
            continue
        original_start = offsets[match.b]
        original_end = offsets[match.b + match.size - 1] + 1
        candidate = source[original_start:original_end].strip()
        if candidate and len(candidate) <= MAX_EVIDENCE_EXCERPT_LENGTH:
            candidates.append((-match.size, block_count, block_position, candidate))
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item)[3]


def _source_blocks_are_contiguous(
    left: SourceEvidenceBlock,
    right: SourceEvidenceBlock,
) -> bool:
    """Return whether two adjacent catalog blocks may span one PDF sentence."""
    if left.page_number is None or right.page_number is None:
        return False
    if not 0 <= right.page_number - left.page_number <= 1:
        return False
    # Only bridge an extraction/page boundary that visibly interrupts one
    # sentence.  Never concatenate two independently complete source sentences.
    return not left.text.rstrip().endswith(("。", "！", "？", "!", "?", "；", ";"))


def _draft_sentence_fragments(value: str, *, markdown: bool) -> tuple[str, ...]:
    fragments: list[str] = []
    for raw_line in value.splitlines() or [value]:
        line = raw_line.strip()
        if not line:
            continue
        if markdown and (
            re.match(r"^#{1,6}\s+", line)
            or line.startswith("![")
            or re.fullmatch(r"\|?[\s:|+-]+\|?", line)
        ):
            continue
        if markdown:
            line = _MARKDOWN_LIST_PREFIX_RE.sub("", line).strip()
        for match in _SENTENCE_PART_RE.finditer(line):
            sentence = match.group(0).strip()
            if sentence:
                fragments.append(sentence)
    return tuple(fragments)


_INFERENCE_ASSERTION_RE = re.compile(
    r"(?:标志着|意味着|表明|证明|反映出|彰显了?|确立了?|奠定了?|"
    r"开创了?|进入了?)[^，；。！？!?]{0,80}"
)


def _unsupported_inference_phrases(
    sentence: str,
    claims: Sequence[ClaimObligation],
) -> tuple[str, ...]:
    evidence = _normalized_text(
        "\n".join(item.evidence_excerpt for item in claims)
    )
    return tuple(
        match.group(0)
        for match in _INFERENCE_ASSERTION_RE.finditer(sentence)
        if _normalized_text(match.group(0)) not in evidence
    )


def normalize_known_source_anomalies(
    values: Sequence[str],
) -> tuple[str, ...]:
    """Return one canonical, run-configurable source-anomaly vocabulary."""
    if isinstance(values, (str, bytes)):
        raise ValueError("known source anomalies must be a sequence of strings")
    normalized: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"known source anomaly {index} must be a non-empty string"
            )
        normalized.add(value.strip())
    return tuple(sorted(normalized))


def _source_text_anomalies(
    value: str,
    known_source_anomalies: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return configured source anomalies found verbatim in ``value``."""
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    located = (
        (value.find(anomaly), anomaly)
        for anomaly in anomalies
        if anomaly in value
    )
    return tuple(anomaly for _position, anomaly in sorted(located))


def _filter_source_anomaly_sentences(
    value: str,
    known_source_anomalies: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Exclude complete anomalous sentences from the model-facing view."""
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    excluded: list[str] = []
    retained: list[str] = []
    cursor = 0
    for match in _SENTENCE_PART_RE.finditer(value):
        fragment = match.group(0).strip()
        if not fragment:
            continue
        if _source_text_anomalies(fragment, anomalies):
            retained.append(value[cursor : match.start()])
            excluded.append(fragment)
            cursor = match.end()
    if not excluded:
        return value.strip(), ()
    retained.append(value[cursor:])
    clean = "\n".join(item.strip() for item in retained if item.strip())
    return clean, tuple(excluded)


_OCR_SUSPICION_RE = re.compile(r"�|[\ue000-\uf8ff]")


def _ocr_suspicions(value: str) -> tuple[str, ...]:
    """Return only high-confidence extraction/OCR warnings."""
    return tuple(
        dict.fromkeys(match.group(0) for match in _OCR_SUSPICION_RE.finditer(value))
    )


def _draft_text(draft: Mapping[str, Any]) -> str:
    if not isinstance(draft, Mapping):
        raise ContractError("draft must be an object")
    values: list[str] = []
    for field in ("title", "description", "body"):
        value = draft.get(field, "")
        if not isinstance(value, str):
            raise ContractError(f"draft.{field} must be a string")
        values.append(value)
    text = "\n".join(values).strip()
    if not text:
        raise ContractError("draft must contain text")
    return text


def _validate_draft_excerpt(excerpt: str, draft: str, label: str) -> None:
    if not excerpt:
        raise ContractError(f"{label} must be non-empty")
    if _normalized_text(excerpt) not in _normalized_text(draft):
        raise ContractError(f"{label} is not verbatim text from the draft")


_HARD_ANCHOR_RE = re.compile(
    r"《[^》]+》|"
    r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?|"
    r"\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?"
)


def _hard_anchors(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0) for match in _HARD_ANCHOR_RE.finditer(value)))


def _unsupported_draft_anchors(
    draft: str,
    claims: Sequence[ClaimObligation],
) -> tuple[str, ...]:
    allowed = {
        _normalized_text(anchor)
        for claim in claims
        for source in (claim.claim, claim.evidence_excerpt)
        for anchor in _hard_anchors(source)
    }
    # Markdown heading/list ordinals are layout, not factual assertions.
    semantic_draft = re.sub(
        r"(?m)^\s*#{1,6}\s+(?:\d+(?:\.\d+)*[、.．)]?\s*)?",
        "",
        draft,
    )
    semantic_draft = re.sub(
        r"(?m)^\s*\d+[.)、．]\s+",
        "",
        semantic_draft,
    )
    return tuple(
        anchor
        for anchor in _hard_anchors(semantic_draft)
        if _normalized_text(anchor) not in allowed
    )


def _validate_hard_anchors(source: str, excerpt: str, label: str) -> None:
    missing = [
        anchor
        for anchor in _hard_anchors(source)
        if not _contains_valid_hard_anchor(excerpt, anchor)
    ]
    if missing:
        raise ContractError(
            f"{label} is missing hard anchors from the claim: {missing}; "
            f"claim={source[:160]!r}; excerpt={excerpt[:160]!r}"
        )


_TEMPORAL_QUALIFIER_RE = re.compile(
    r"(?:截至|截止(?:到)?)\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
)


def _temporal_qualifiers(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(0) for match in _TEMPORAL_QUALIFIER_RE.finditer(value))
    )


def _validate_temporal_qualifiers(source: str, excerpt: str, label: str) -> None:
    normalized_excerpt = _normalized_text(excerpt)
    missing = [
        item
        for item in _temporal_qualifiers(source)
        if _normalized_text(item) not in normalized_excerpt
    ]
    if missing:
        raise ContractError(
            f"{label} has unsupported temporal qualifier {missing}; remove the "
            "qualifier from the claim or cite an excerpt containing the exact phrase"
        )


def _canonicalize_claim_excerpt(
    unit: EvidenceUnit,
    claim: str,
    excerpt: str,
) -> str:
    """Safely expand a short excerpt to include claim hard anchors.

    The expansion is allowed only within one real source block and always
    returns one contiguous verbatim window.  It never joins blocks or rewrites
    the claim.  When no bounded source window exists, the original excerpt is
    returned so the existing hard-anchor/provenance gates reject it.
    """
    anchors = _hard_anchors(claim)
    missing_anchors = tuple(
        anchor
        for anchor in anchors
        if not _contains_valid_hard_anchor(excerpt, anchor)
    )
    qualifiers = _temporal_qualifiers(claim)
    missing_qualifiers = tuple(
        item
        for item in qualifiers
        if _normalized_text(item) not in _normalized_text(excerpt)
    )
    if (not missing_anchors and not missing_qualifiers) or not unit.source_blocks:
        return excerpt

    candidates: list[tuple[int, int, int, str]] = []
    for block_count, block_position, source in _source_text_windows(unit):
        if _normalized_text(excerpt) not in _normalized_text(source):
            continue
        if any(
            not _contains_valid_hard_anchor(source, anchor)
            for anchor in missing_anchors
        ):
            continue
        if any(
            _normalized_text(item) not in _normalized_text(source)
            for item in missing_qualifiers
        ):
            continue
        window = _shortest_contiguous_window(
            source,
            (excerpt, *missing_anchors, *missing_qualifiers),
        )
        if window is None or len(window) > MAX_EVIDENCE_EXCERPT_LENGTH:
            continue
        if any(
            not _contains_valid_hard_anchor(window, anchor)
            for anchor in anchors
        ):
            continue
        candidates.append((len(window), block_count, block_position, window))
    if not candidates:
        return excerpt
    return min(candidates, key=lambda item: item)[3]


def _shortest_contiguous_window(
    source: str,
    required_fragments: Sequence[str],
) -> str | None:
    """Return the shortest original substring containing every fragment."""
    normalized_source, offsets = _hard_anchor_text_with_offsets(source)
    if not normalized_source:
        return None
    occurrences: list[tuple[int, int, int]] = []
    for label, fragment in enumerate(required_fragments):
        normalized_fragment = _hard_anchor_normalized_text(fragment)
        if not normalized_fragment:
            return None
        start = normalized_source.find(normalized_fragment)
        if start < 0:
            return None
        while start >= 0:
            if (
                label > 0
                and _is_temporal_hard_anchor(fragment)
                and _inside_policy_title(source, offsets[start])
            ):
                start = normalized_source.find(normalized_fragment, start + 1)
                continue
            occurrences.append(
                (start, start + len(normalized_fragment), label)
            )
            start = normalized_source.find(normalized_fragment, start + 1)
    occurrences.sort(key=lambda item: (item[0], item[1], item[2]))

    required_labels = len(required_fragments)
    counts: dict[int, int] = {}
    covered = 0
    left = 0
    best: tuple[int, int] | None = None
    for right, (_start, _end, label) in enumerate(occurrences):
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1:
            covered += 1
        while covered == required_labels:
            window = occurrences[left : right + 1]
            normalized_start = window[0][0]
            normalized_end = max(item[1] for item in window)
            original_start = offsets[normalized_start]
            original_end = offsets[normalized_end - 1] + 1
            candidate = (original_start, original_end)
            if best is None or (
                candidate[1] - candidate[0], candidate[0], candidate[1]
            ) < (best[1] - best[0], best[0], best[1]):
                best = candidate

            left_label = occurrences[left][2]
            counts[left_label] -= 1
            if counts[left_label] == 0:
                covered -= 1
            left += 1
    if best is None:
        return None
    return source[best[0] : best[1]].strip()


def _hard_anchor_text_with_offsets(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(value):
        if character.isspace():
            continue
        if character == "\\" and index + 1 < len(value) and value[index + 1] in "()%":
            continue
        if character in "()" and index > 0 and value[index - 1] == "\\":
            continue
        characters.append(character)
        offsets.append(index)
    return "".join(characters), tuple(offsets)


def _hard_anchor_normalized_text(value: str) -> str:
    return _hard_anchor_text_with_offsets(value)[0]


_TEMPORAL_HARD_ANCHOR_RE = re.compile(
    r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
)


def _is_temporal_hard_anchor(value: str) -> bool:
    return bool(_TEMPORAL_HARD_ANCHOR_RE.fullmatch(value))


def _inside_policy_title(value: str, original_offset: int) -> bool:
    prefix = value[:original_offset]
    return prefix.rfind("《") > prefix.rfind("》")


def _contains_valid_hard_anchor(value: str, anchor: str) -> bool:
    normalized, offsets = _hard_anchor_text_with_offsets(value)
    needle = _hard_anchor_normalized_text(anchor)
    start = normalized.find(needle)
    while start >= 0:
        if not (
            _is_temporal_hard_anchor(anchor)
            and _inside_policy_title(value, offsets[start])
        ):
            return True
        start = normalized.find(needle, start + 1)
    return False


def _scope_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        return tuple(
            item
            for nested in value.values()
            for item in _scope_values(nested)
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(
            item for nested in value for item in _scope_values(nested)
        )
    return ()


def _validate_claim_scope(
    scope: Mapping[str, Any],
    *,
    claim: str,
    excerpt: str,
    evidence_context: str,
    ref_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove metadata-only scope noise while rejecting factual inventions.

    Scope is auxiliary retrieval metadata, so an unsupported value that appears
    only in ``scope`` can be discarded deterministically.  If the same value is
    present in the claim text, however, it has become an unsupported factual
    assertion and must still fail the contract.
    """
    claim_text = _excerpt_match_text_with_offsets(claim)[0]
    excerpt_text = _excerpt_match_text_with_offsets(excerpt)[0]
    context_text = _excerpt_match_text_with_offsets(evidence_context)[0]
    declared = {
        _excerpt_match_text_with_offsets(item)[0]
        for item in _scope_values(ref_scope)
    }
    sanitized: dict[str, Any] = {}
    for key, raw in scope.items():
        values = (raw,) if isinstance(raw, str) else tuple(raw)
        retained: list[str] = []
        for item in values:
            normalized = _excerpt_match_text_with_offsets(item)[0]
            if (
                normalized in excerpt_text
                or normalized in context_text
                or normalized in declared
            ):
                retained.append(item)
                continue
            if normalized in claim_text:
                raise ContractError(
                    "claim contains a scope value absent from evidence and Ref "
                    f"scope: {item}"
                )
        if not retained:
            continue
        sanitized[key] = retained[0] if isinstance(raw, str) else retained
    return sanitized


def _normalize_claim_scope(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    unknown = sorted(set(value) - SCOPE_KEYS)
    if unknown:
        raise ContractError(f"{label} contains unsupported keys: {unknown}")
    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        item_label = f"{label}.{key}"
        if isinstance(raw, str):
            normalized[key] = _bounded_text(
                raw, item_label, MAX_SCOPE_VALUE_LENGTH
            )
            continue
        if not isinstance(raw, list):
            raise ContractError(
                f"{item_label} must be short text or a short text array"
            )
        if not 1 <= len(raw) <= MAX_SCOPE_VALUES_PER_KEY:
            raise ContractError(
                f"{item_label} must contain between 1 and "
                f"{MAX_SCOPE_VALUES_PER_KEY} values"
            )
        items = [
            _bounded_text(nested, f"{item_label}[{index}]", MAX_SCOPE_VALUE_LENGTH)
            for index, nested in enumerate(raw)
        ]
        if len(items) != len(set(items)):
            raise ContractError(f"{item_label} must not contain duplicate values")
        normalized[key] = items
    return normalized


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _object(response: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError) as error:
        raise ContractError(f"{label} must be strict JSON") from error
    return _mapping(value, label)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty text")
    return value.strip()


def _bounded_text(value: Any, label: str, max_length: int) -> str:
    text = _text(value, label)
    if len(text) > max_length:
        raise ContractError(f"{label} exceeds maximum length {max_length}")
    return text


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    text = _text(value, label)
    if text not in allowed:
        raise ContractError(f"{label} is invalid: {text}")
    return text


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    item = _mapping(value, label)
    actual = set(item)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            f"{label} fields mismatch; missing={missing}, extra={extra}"
        )
