"""Deterministic, human-first preparation of RAG gold-QA annotation slots.

This module deliberately does *not* generate questions, facts, or answers.  It
only samples canonical source evidence into a balanced annotation worksheet.
The resulting JSONL rows use the field layout accepted by :mod:`.gold`, but
remain intentionally invalid gold data until a human fills the empty required
fields and the normal gold loader validates them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from .corpus import EvidenceAtom, build_t0_fixed_chunks


_TIME_OR_SCENARIO = re.compile(
    r"(?:19|20)\d{2}年?|第?[一二三四五六七八九十]+季度|"
    r"同比|环比|期间|截至|规划期|试点|地区|区域|城市|农村|行业|企业|人口"
)


@dataclass(frozen=True)
class GoldSamplingQuota:
    """Requested number of draft slots per book and question type."""

    single_evidence_fact: int = 3
    intra_document_synthesis: int = 2
    cross_document_synthesis: int = 1
    temporal_or_scenario: int = 1
    unanswerable: int = 1

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("gold sampling quotas cannot be negative")

    @property
    def per_book(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class GoldDraftSlot:
    """One annotation slot plus exact, non-generated evidence provenance."""

    question_id: str
    question_type: str
    answerable: bool
    primary_article_id: str
    evidence_atoms: tuple[EvidenceAtom, ...]
    selection_reason: str
    requires_scope_confirmation: bool = False

    def to_gold_template(self, *, excerpt_chars: int = 800) -> dict[str, Any]:
        """Return a human-fill row shaped like the stable gold contract.

        Empty strings are intentional: ``load_gold_jsonl`` must reject an
        untouched worksheet, preventing sampled source text from being
        mistaken for authored questions or answers.
        """

        if excerpt_chars < 1:
            raise ValueError("excerpt_chars must be positive")
        provenance = [
            {
                "evidence_atom_id": atom.atom_id.canonical,
                "article_id": atom.article_id,
                "source_file": atom.source_file,
                "page": atom.page,
                "block_id": atom.block_id,
                "block_type": atom.block_type,
                "heading_path": list(atom.heading_path),
                "content_hash": atom.content_hash,
                "source_excerpt": atom.text[:excerpt_chars],
                "excerpt_truncated": len(atom.text) > excerpt_chars,
            }
            for atom in self.evidence_atoms
        ]
        evidence_sets = (
            [[atom.atom_id.canonical for atom in self.evidence_atoms]]
            if self.answerable
            else []
        )
        required_facts = (
            [
                {
                    "fact_id": "fact-1",
                    "claim": "",
                    "weight": 1.0,
                    "critical": True,
                    "reason": "",
                }
            ]
            if self.answerable
            else []
        )
        return {
            "question_id": self.question_id,
            "question": "",
            "question_type": self.question_type,
            "answerable": self.answerable,
            "scope": {
                "primary_article_id": self.primary_article_id,
                "article_ids": sorted({atom.article_id for atom in self.evidence_atoms}),
                "candidate_provenance": provenance,
                "selection_reason": self.selection_reason,
                "requires_scope_confirmation": self.requires_scope_confirmation,
                "worksheet_state": "human_required",
            },
            "required_facts": required_facts,
            "forbidden_facts": [],
            "evidence_sets": evidence_sets,
            "reference_answer": "",
            "annotation": {"author": "", "reviewer": None, "status": "draft"},
        }


@dataclass(frozen=True)
class GoldSamplingPlan:
    slots: tuple[GoldDraftSlot, ...]
    audit: dict[str, Any]


def _stable_digest(seed: int, *parts: str) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atom_order(atoms: Iterable[EvidenceAtom], *, seed: int, salt: str) -> list[EvidenceAtom]:
    """Spread selection over pages before taking additional same-page atoms."""

    by_page: dict[int, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        by_page[atom.page].append(atom)
    for page_atoms in by_page.values():
        page_atoms.sort(
            key=lambda atom: _stable_digest(seed, salt, atom.atom_id.canonical)
        )
    pages = sorted(
        by_page,
        key=lambda page: _stable_digest(seed, salt, f"page:{page}"),
    )
    ordered: list[EvidenceAtom] = []
    depth = 0
    while True:
        added = False
        for page in pages:
            if depth < len(by_page[page]):
                ordered.append(by_page[page][depth])
                added = True
        if not added:
            return ordered
        depth += 1


@lru_cache(maxsize=None)
def _heading_tokens(atom: EvidenceAtom) -> frozenset[str]:
    text = " ".join(atom.heading_path).lower()
    ascii_words = re.findall(r"[a-z0-9]{2,}", text)
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", text))
    bigrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return frozenset((*ascii_words, *bigrams))


def _pair_candidates(atoms: Sequence[EvidenceAtom]) -> list[tuple[EvidenceAtom, EvidenceAtom]]:
    """Prefer pairs in one semantic heading and spanning distinct pages."""

    # Real books contain thousands of blocks.  Generating the Cartesian pair
    # set would be both wasteful and biased toward long chapters, so enumerate
    # only neighbouring evidence within each heading, then add page-spread
    # fallbacks.  The worksheet needs a handful of candidates, not all pairs.
    by_heading: dict[tuple[str, ...], list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        by_heading[atom.heading_path].append(atom)
    candidates: list[tuple[int, EvidenceAtom, EvidenceAtom]] = []
    seen: set[tuple[str, str]] = set()

    def append(left: EvidenceAtom, right: EvidenceAtom) -> None:
        key = tuple(sorted((left.atom_id.canonical, right.atom_id.canonical)))
        if left == right or key in seen:
            return
        seen.add(key)
        same_heading = bool(left.heading_path and left.heading_path == right.heading_path)
        score = 4 * int(same_heading) + 2 * int(left.page != right.page)
        score += int(bool(_heading_tokens(left) & _heading_tokens(right)))
        candidates.append((score, left, right))

    for heading_atoms in by_heading.values():
        ordered = sorted(heading_atoms, key=lambda atom: (atom.page, atom.atom_id.canonical))
        for left, right in zip(ordered, ordered[1:]):
            append(left, right)
        if len(ordered) > 2:
            append(ordered[0], ordered[-1])

    page_representatives: dict[int, EvidenceAtom] = {}
    for atom in atoms:
        page_representatives.setdefault(atom.page, atom)
    spread = [page_representatives[page] for page in sorted(page_representatives)]
    for left, right in zip(spread, spread[1:]):
        append(left, right)

    # Tiny synthetic documents may have no repeated heading or page.
    if not candidates:
        for left, right in zip(atoms, atoms[1:]):
            append(left, right)

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].atom_id.canonical,
            item[2].atom_id.canonical,
        )
    )
    return [(left, right) for _, left, right in candidates]


def _cross_document_pair(
    primary: Sequence[EvidenceAtom],
    others: Sequence[EvidenceAtom],
) -> tuple[EvidenceAtom, EvidenceAtom, bool] | None:
    if not primary or not others:
        return None
    best: tuple[int, str, EvidenceAtom, EvidenceAtom] | None = None
    for left in primary:
        left_tokens = _heading_tokens(left)
        for right in others:
            if left.article_id == right.article_id:
                continue
            overlap = len(left_tokens & _heading_tokens(right))
            tie = f"{left.atom_id.canonical}|{right.atom_id.canonical}"
            candidate = (overlap, tie, left, right)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
    if best is None:
        return None
    overlap, _, left, right = best
    return left, right, overlap == 0


def _slot_id(article_id: str, question_type: str, ordinal: int, atoms: Sequence[EvidenceAtom]) -> str:
    evidence = "|".join(atom.atom_id.canonical for atom in atoms) or "no-evidence"
    digest = hashlib.sha256(
        f"{article_id}|{question_type}|{ordinal}|{evidence}".encode("utf-8")
    ).hexdigest()[:16]
    return f"gold-slot-{digest}"


def prepare_gold_sampling_plan(
    structures_dir: Path,
    *,
    quota: GoldSamplingQuota = GoldSamplingQuota(),
    seed: int = 20260809,
) -> GoldSamplingPlan:
    """Sample a balanced, deterministic human annotation worksheet."""

    build = build_t0_fixed_chunks(structures_dir)
    by_article: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in build.evidence_atoms:
        by_article[atom.article_id].append(atom)

    slots: list[GoldDraftSlot] = []
    requested = Counter()
    shortfalls = Counter()

    def add_slots(
        article_id: str,
        question_type: str,
        answerable: bool,
        candidates: Sequence[tuple[tuple[EvidenceAtom, ...], str, bool]],
        count: int,
    ) -> None:
        requested[(article_id, question_type, answerable)] += count
        for ordinal, (atoms, reason, requires_confirmation) in enumerate(
            candidates[:count], start=1
        ):
            slots.append(
                GoldDraftSlot(
                    question_id=_slot_id(article_id, question_type, ordinal, atoms),
                    question_type=question_type,
                    answerable=answerable,
                    primary_article_id=article_id,
                    evidence_atoms=atoms,
                    selection_reason=reason,
                    requires_scope_confirmation=requires_confirmation,
                )
            )
        if len(candidates) < count:
            shortfalls[(article_id, question_type, answerable)] += count - len(candidates)

    ordered_by_article = {
        article_id: _atom_order(atoms, seed=seed, salt=article_id)
        for article_id, atoms in by_article.items()
    }
    for article_id in sorted(by_article):
        atoms = ordered_by_article[article_id]
        singles = [((atom,), "page_spread", False) for atom in atoms]
        add_slots(
            article_id,
            "single_evidence_fact",
            True,
            singles,
            quota.single_evidence_fact,
        )

        pairs = [
            ((left, right), "same_heading_or_nearby_evidence", False)
            for left, right in _pair_candidates(atoms)
        ]
        add_slots(
            article_id,
            "intra_document_synthesis",
            True,
            pairs,
            quota.intra_document_synthesis,
        )

        cross = _cross_document_pair(
            atoms[:64],
            [
                atom
                for other_article, other_atoms in ordered_by_article.items()
                if other_article != article_id
                for atom in other_atoms[:16]
            ],
        )
        cross_candidates = []
        if cross is not None:
            left, right, fallback = cross
            cross_candidates.append(
                (
                    (left, right),
                    "shared_heading_tokens" if not fallback else "corpus_diversity_fallback",
                    fallback,
                )
            )
        add_slots(
            article_id,
            "cross_document_synthesis",
            True,
            cross_candidates,
            quota.cross_document_synthesis,
        )

        temporal_atoms = [atom for atom in atoms if _TIME_OR_SCENARIO.search(atom.text)]
        temporal_pairs = _pair_candidates(temporal_atoms)
        temporal_candidates = [
            ((left, right), "explicit_time_or_scenario_markers", False)
            for left, right in temporal_pairs
        ]
        if len(temporal_candidates) < quota.temporal_or_scenario:
            temporal_candidates.extend(
                ((atom,), "single_explicit_time_or_scenario_marker", True)
                for atom in temporal_atoms
                if all(atom not in candidate[0] for candidate in temporal_candidates)
            )
        add_slots(
            article_id,
            "temporal_or_scenario",
            True,
            temporal_candidates,
            quota.temporal_or_scenario,
        )

        unanswerable_candidates = [
            ((), "human_authored_out_of_corpus_question", False)
            for _ in range(quota.unanswerable)
        ]
        add_slots(
            article_id,
            "unanswerable",
            False,
            unanswerable_candidates,
            quota.unanswerable,
        )

    audit = audit_gold_sampling_plan(
        slots,
        evidence_catalog=build.evidence_atoms,
        requested=requested,
        shortfalls=shortfalls,
        seed=seed,
    )
    return GoldSamplingPlan(tuple(slots), audit)


def _serialize_stratum(key: tuple[str, str, bool], count: int) -> dict[str, Any]:
    article_id, question_type, answerable = key
    return {
        "article_id": article_id,
        "question_type": question_type,
        "answerable": answerable,
        "count": count,
    }


def audit_gold_sampling_plan(
    slots: Sequence[GoldDraftSlot],
    *,
    evidence_catalog: Sequence[EvidenceAtom],
    requested: Counter[tuple[str, str, bool]] | None = None,
    shortfalls: Counter[tuple[str, str, bool]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Audit provenance and prove that no answer-bearing text was generated."""

    catalog = {atom.atom_id: atom for atom in evidence_catalog}
    ids = [slot.question_id for slot in slots]
    unknown = sorted(
        atom.atom_id.canonical
        for slot in slots
        for atom in slot.evidence_atoms
        if atom.atom_id not in catalog
    )
    content_mismatches = sorted(
        atom.atom_id.canonical
        for slot in slots
        for atom in slot.evidence_atoms
        if atom.atom_id in catalog and atom != catalog[atom.atom_id]
    )
    invalid_answerability = [
        slot.question_id
        for slot in slots
        if (slot.answerable and not slot.evidence_atoms)
        or (not slot.answerable and slot.evidence_atoms)
    ]
    strata = Counter(
        (slot.primary_article_id, slot.question_type, slot.answerable) for slot in slots
    )
    books = sorted({slot.primary_article_id for slot in slots})
    passed = (
        bool(slots)
        and len(ids) == len(set(ids))
        and not unknown
        and not content_mismatches
        and not invalid_answerability
    )
    return {
        "schema": "okfolio.rag-gold-sampling-audit.v1",
        "status": "pass" if passed else "fail",
        "gold_ready": False,
        "worksheet_state": "human_required",
        "generation_policy": {
            "llm_calls": 0,
            "auto_generated_questions": 0,
            "auto_generated_fact_claims": 0,
            "auto_generated_reference_answers": 0,
            "seed": seed,
        },
        "books": len(books),
        "book_ids": books,
        "slots": len(slots),
        "answerable_slots": sum(slot.answerable for slot in slots),
        "unanswerable_slots": sum(not slot.answerable for slot in slots),
        "unique_evidence_atoms_sampled": len(
            {atom.atom_id for slot in slots for atom in slot.evidence_atoms}
        ),
        "duplicate_question_id_count": len(ids) - len(set(ids)),
        "unknown_evidence_atom_count": len(unknown),
        "unknown_evidence_atoms_sample": unknown[:20],
        "provenance_content_mismatch_count": len(content_mismatches),
        "provenance_content_mismatch_sample": content_mismatches[:20],
        "invalid_answerability_count": len(invalid_answerability),
        "invalid_answerability_sample": invalid_answerability[:20],
        "requires_scope_confirmation": sum(
            slot.requires_scope_confirmation for slot in slots
        ),
        "strata": [
            _serialize_stratum(key, count) for key, count in sorted(strata.items())
        ],
        "requested_strata": [
            _serialize_stratum(key, count)
            for key, count in sorted((requested or Counter()).items())
        ],
        "shortfalls": [
            _serialize_stratum(key, count)
            for key, count in sorted((shortfalls or Counter()).items())
            if count
        ],
        "human_completion_required": [
            "question",
            "required_facts[].claim for answerable rows",
            "reference_answer",
            "annotation.author",
            "annotation.status",
            "human review of evidence sufficiency and question type",
        ],
    }


def write_gold_template_jsonl(
    plan: GoldSamplingPlan,
    output: Path,
    *,
    excerpt_chars: int = 800,
    overwrite: bool = False,
) -> None:
    """Write only to an explicit path; never infer a project/run directory."""

    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(slot.to_gold_template(excerpt_chars=excerpt_chars), ensure_ascii=False)
        for slot in plan.slots
    )
    output.write_text(text + ("\n" if text else ""), encoding="utf-8")
