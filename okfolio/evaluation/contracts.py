from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


AtomKind = Literal["segment", "block"]

_ATOM_PATTERN = re.compile(
    r"^(?P<article>[^:\s]+):p(?P<page>[0-9]+):"
    r"(?P<kind>[sb])(?P<locator>[^:\s]+)$"
)


@dataclass(frozen=True, order=True)
class EvidenceAtomId:
    """Stable evidence identity shared by every RAG experiment arm.

    The canonical representation is ``article:p003:s007`` for a segment or
    ``article:p003:b012`` for a block.  Page numbers are one-based.  Retrieval
    units may change between experiment arms, but this identity must not.
    """

    article_id: str
    page: int
    atom_kind: AtomKind
    locator_id: str

    def __post_init__(self) -> None:
        if not self.article_id or any(
            character.isspace() or character == ":" for character in self.article_id
        ):
            raise ValueError("article_id must be non-empty and contain no ':' or space")
        if self.page < 1:
            raise ValueError("page must be one-based")
        if self.atom_kind not in {"segment", "block"}:
            raise ValueError("atom_kind must be 'segment' or 'block'")
        if not self.locator_id or any(
            character.isspace() or character == ":" for character in self.locator_id
        ):
            raise ValueError("locator_id must be non-empty and contain no ':' or space")

    @property
    def canonical(self) -> str:
        prefix = "s" if self.atom_kind == "segment" else "b"
        return f"{self.article_id}:p{self.page:03d}:{prefix}{self.locator_id}"

    def __str__(self) -> str:
        return self.canonical

    @classmethod
    def parse(cls, value: str) -> EvidenceAtomId:
        if not isinstance(value, str):
            raise TypeError("evidence atom id must be a string")
        match = _ATOM_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "evidence atom id must match article:p<page>:s<segment> "
                "or article:p<page>:b<block>"
            )
        return cls(
            article_id=match.group("article"),
            page=int(match.group("page")),
            atom_kind="segment" if match.group("kind") == "s" else "block",
            locator_id=match.group("locator"),
        )


@dataclass(frozen=True)
class FactLabel:
    fact_id: str
    claim: str
    weight: float = 1.0
    critical: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.fact_id.strip():
            raise ValueError("fact_id must be non-empty")
        if not self.claim.strip():
            raise ValueError("claim must be non-empty")
        if self.weight <= 0:
            raise ValueError("fact weight must be positive")


@dataclass(frozen=True)
class GoldAnnotation:
    author: str
    status: str
    reviewer: str | None = None

    def __post_init__(self) -> None:
        if not self.author.strip():
            raise ValueError("annotation author must be non-empty")
        if not self.status.strip():
            raise ValueError("annotation status must be non-empty")
        if self.reviewer is not None and not self.reviewer.strip():
            raise ValueError("annotation reviewer must be non-empty when present")


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    question: str
    question_type: str
    answerable: bool
    scope: Mapping[str, Any]
    required_facts: tuple[FactLabel, ...]
    forbidden_facts: tuple[FactLabel, ...]
    evidence_sets: tuple[tuple[EvidenceAtomId, ...], ...]
    reference_answer: str | None
    annotation: GoldAnnotation

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not self.question_type.strip():
            raise ValueError("question_type must be non-empty")
        fact_ids = [fact.fact_id for fact in (*self.required_facts, *self.forbidden_facts)]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_ids must be unique within a question")
        if self.answerable and not self.required_facts:
            raise ValueError("answerable questions require at least one required fact")
        if self.answerable and not self.evidence_sets:
            raise ValueError("answerable questions require at least one evidence set")
        if not self.answerable and self.evidence_sets:
            raise ValueError("unanswerable questions cannot have a complete evidence set")
        if any(not evidence_set for evidence_set in self.evidence_sets):
            raise ValueError("evidence sets must be non-empty")
        if any(
            len(evidence_set) != len(set(evidence_set))
            for evidence_set in self.evidence_sets
        ):
            raise ValueError("an evidence set cannot contain duplicate atoms")
        canonical_sets = [
            tuple(sorted(atom.canonical for atom in evidence_set))
            for evidence_set in self.evidence_sets
        ]
        if len(canonical_sets) != len(set(canonical_sets)):
            raise ValueError("evidence sets must be unique")

    @property
    def forbidden_claims(self) -> tuple[FactLabel, ...]:
        """Alias used by answer scorers and external evaluation adapters."""

        return self.forbidden_facts

    @property
    def all_gold_atoms(self) -> frozenset[EvidenceAtomId]:
        return frozenset(atom for evidence_set in self.evidence_sets for atom in evidence_set)


@dataclass(frozen=True)
class Citation:
    evidence_atom_id: EvidenceAtomId


@dataclass(frozen=True)
class AnswerPrediction:
    """Provider-neutral answer decisions used by deterministic scorers.

    Citation atoms and answerability can come from the strict structured-answer
    contract. Matching natural-language output to atomic Gold facts is
    deliberately outside this deterministic contract. Until a human assessor
    or separately calibrated judge supplies that alignment, both fact-ID fields
    remain empty and the generation trace is marked provisional.
    """

    question_id: str
    predicted_answerable: bool
    matched_required_fact_ids: tuple[str, ...] = ()
    asserted_forbidden_fact_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class RetrievedUnit:
    """A retrieval hit plus both search-time and generation-time provenance.

    ``retrieval_evidence_atom_ids`` records atoms represented by the item that
    was ranked (for example, a child chunk).  ``context_evidence_atom_ids``
    records atoms actually passed to the generator (for example, its parent).
    Keeping both prevents Parent-Child retrieval from conflating hit quality
    with final context coverage.
    """

    unit_id: str
    arm: str
    retrieval_text: str
    context_text: str
    retrieval_evidence_atom_ids: tuple[EvidenceAtomId, ...]
    context_evidence_atom_ids: tuple[EvidenceAtomId, ...]
    article_ids: tuple[str, ...]
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("unit_id must be non-empty")
        if not self.arm.strip():
            raise ValueError("arm must be non-empty")
        if not self.retrieval_text.strip():
            raise ValueError("retrieval_text must be non-empty")
        if not self.context_text.strip():
            raise ValueError("context_text must be non-empty")
        if not self.retrieval_evidence_atom_ids:
            raise ValueError("retrieval_evidence_atom_ids must be non-empty")
        if not self.context_evidence_atom_ids:
            raise ValueError("context_evidence_atom_ids must be non-empty")
        if not self.article_ids:
            raise ValueError("article_ids must be non-empty")
        if len(self.article_ids) != len(set(self.article_ids)):
            raise ValueError("article_ids must be unique")
        known_articles = set(self.article_ids)
        represented_articles = {
            atom.article_id
            for atom in (
                *self.retrieval_evidence_atom_ids,
                *self.context_evidence_atom_ids,
            )
        }
        if not represented_articles <= known_articles:
            raise ValueError("article_ids must cover every represented evidence atom")
