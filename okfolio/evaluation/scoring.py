from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Literal, Mapping, Sequence

from .contracts import (
    AnswerPrediction,
    EvidenceAtomId,
    GoldQuestion,
    RetrievedUnit,
)


ProvenanceView = Literal["retrieval", "context"]


@dataclass(frozen=True)
class RetrievalScore:
    question_id: str
    provenance_view: ProvenanceView
    k: int
    returned_units: int
    returned_unique_atoms: int
    evidence_recall: float | None
    complete_evidence_set_recall: float | None
    mrr: float | None
    ndcg: float | None
    context_precision: float | None
    retrieved_atom_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnswerScore:
    question_id: str
    semantic_alignment_status: str
    gold_answerable: bool
    predicted_answerable: bool
    answerability_correct: bool
    required_fact_recall: float | None
    required_fact_precision: float | None
    all_critical_facts_present: bool | None
    contradiction: bool | None
    answer_correct: bool | None
    citation_validity: float | None
    citation_precision: float | None
    complete_evidence_set_cited: bool | None
    unknown_required_fact_ids: tuple[str, ...]
    unknown_forbidden_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class RefusalSummary:
    questions: int
    true_refusals: int
    false_refusals: int
    missed_refusals: int
    refusal_precision: float
    refusal_recall: float
    refusal_f1: float
    over_refusal_rate: float
    unsupported_answer_rate: float


def score_retrieval(
    gold: GoldQuestion,
    retrieved: Sequence[RetrievedUnit],
    *,
    k: int,
    provenance_view: ProvenanceView = "context",
    relevance_grades: Mapping[EvidenceAtomId, int] | None = None,
) -> RetrievalScore:
    """Score ranked units after expanding them to canonical evidence atoms.

    Multiple gold ``evidence_sets`` are alternatives.  Evidence Recall uses the
    best-covered set and Complete Evidence Set Recall succeeds when any one of
    those sets is fully covered.

    For rank-sensitive metrics, selected units are expanded in ranked order and
    their evidence atoms retain source order; duplicate atoms keep only their
    first occurrence.  MRR and standard graded nDCG are then computed on that
    common atom ranking.  This avoids treating a large Concept as one magic
    relevant item while a traditional chunk is scored at a finer granularity.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if provenance_view not in {"retrieval", "context"}:
        raise ValueError("provenance_view must be 'retrieval' or 'context'")
    top_units = tuple(retrieved[:k])
    unit_ids = [unit.unit_id for unit in top_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("retrieved unit ids must be unique within the ranked list")

    first_rank: dict[EvidenceAtomId, int] = {}
    ordered_unique_atoms: list[EvidenceAtomId] = []
    for rank, unit in enumerate(top_units, start=1):
        atoms = (
            unit.retrieval_evidence_atom_ids
            if provenance_view == "retrieval"
            else unit.context_evidence_atom_ids
        )
        for atom in atoms:
            if atom not in first_rank:
                first_rank[atom] = rank
                ordered_unique_atoms.append(atom)

    retrieved_atoms = set(first_rank)
    if not gold.evidence_sets:
        return RetrievalScore(
            question_id=gold.question_id,
            provenance_view=provenance_view,
            k=k,
            returned_units=len(top_units),
            returned_unique_atoms=len(retrieved_atoms),
            evidence_recall=None,
            complete_evidence_set_recall=None,
            mrr=None,
            ndcg=None,
            context_precision=None,
            retrieved_atom_ids=tuple(atom.canonical for atom in ordered_unique_atoms),
        )

    evidence_sets = tuple(set(evidence_set) for evidence_set in gold.evidence_sets)
    evidence_recall = max(
        len(retrieved_atoms & evidence_set) / len(evidence_set)
        for evidence_set in evidence_sets
    )
    complete = any(evidence_set <= retrieved_atoms for evidence_set in evidence_sets)
    gold_atoms = set().union(*evidence_sets)
    atom_positions = {
        atom: position for position, atom in enumerate(ordered_unique_atoms, start=1)
    }
    relevant_positions = [
        atom_positions[atom] for atom in gold_atoms if atom in atom_positions
    ]
    mrr = 1.0 / min(relevant_positions) if relevant_positions else 0.0

    grades = _normalise_grades(gold_atoms, relevance_grades)
    ranked_grades = [grades.get(atom, 0) for atom in ordered_unique_atoms]
    discounted_gain = sum(
        _graded_gain(grade) / math.log2(position + 1)
        for position, grade in enumerate(ranked_grades, start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[: len(ranked_grades)]
    ideal_gain = sum(
        _graded_gain(grade) / math.log2(position + 1)
        for position, grade in enumerate(ideal_grades, start=1)
    )
    ndcg = discounted_gain / ideal_gain if ideal_gain else 0.0
    relevant_retrieved = retrieved_atoms & set(grades)
    context_precision = (
        len(relevant_retrieved) / len(retrieved_atoms) if retrieved_atoms else 0.0
    )
    return RetrievalScore(
        question_id=gold.question_id,
        provenance_view=provenance_view,
        k=k,
        returned_units=len(top_units),
        returned_unique_atoms=len(retrieved_atoms),
        evidence_recall=evidence_recall,
        complete_evidence_set_recall=float(complete),
        mrr=mrr,
        ndcg=ndcg,
        context_precision=context_precision,
        retrieved_atom_ids=tuple(atom.canonical for atom in ordered_unique_atoms),
    )


def score_answer(
    gold: GoldQuestion,
    prediction: AnswerPrediction,
    *,
    evidence_universe: Iterable[EvidenceAtomId] | None = None,
    semantic_alignment_status: str = "human_reviewed",
) -> AnswerScore:
    """Deterministically score pre-aligned atomic facts and citations.

    This function never asks an LLM whether prose entails a fact.  An upstream
    human or calibrated judge must first convert prose to ``AnswerPrediction``.
    The scoring policy itself is then reproducible and provider-neutral.
    """

    if prediction.question_id != gold.question_id:
        raise ValueError("prediction question_id does not match gold question_id")
    required_by_id = {fact.fact_id: fact for fact in gold.required_facts}
    forbidden_by_id = {fact.fact_id: fact for fact in gold.forbidden_facts}
    matched_ids = set(prediction.matched_required_fact_ids)
    asserted_forbidden_ids = set(prediction.asserted_forbidden_fact_ids)
    unknown_required = tuple(sorted(matched_ids - set(required_by_id)))
    unknown_forbidden = tuple(sorted(asserted_forbidden_ids - set(forbidden_by_id)))
    matched_known = matched_ids & set(required_by_id)
    forbidden_known = asserted_forbidden_ids & set(forbidden_by_id)

    if required_by_id:
        total_weight = sum(fact.weight for fact in required_by_id.values())
        matched_weight = sum(required_by_id[fact_id].weight for fact_id in matched_known)
        required_recall = matched_weight / total_weight
        claimed_weight = matched_weight + float(len(unknown_required))
        required_precision = matched_weight / claimed_weight if claimed_weight else 0.0
    else:
        required_recall = None
        required_precision = None
    critical_ids = {
        fact.fact_id for fact in gold.required_facts if fact.critical
    }
    all_critical = critical_ids <= matched_known
    contradiction = bool(forbidden_known or unknown_forbidden)
    answerability_correct = prediction.predicted_answerable == gold.answerable

    if gold.answerable:
        answer_correct = bool(
            prediction.predicted_answerable
            and all_critical
            and not contradiction
            and not unknown_required
        )
    else:
        answer_correct = bool(
            not prediction.predicted_answerable
            and not matched_ids
            and not asserted_forbidden_ids
        )

    citation_atoms = tuple(
        citation.evidence_atom_id for citation in prediction.citations
    )
    unique_citation_atoms = set(citation_atoms)
    if evidence_universe is None:
        citation_validity = None
    elif not citation_atoms:
        citation_validity = 0.0 if gold.answerable else 1.0
    else:
        universe = set(evidence_universe)
        citation_validity = sum(atom in universe for atom in citation_atoms) / len(
            citation_atoms
        )
    if gold.answerable:
        gold_atoms = set(gold.all_gold_atoms)
        citation_precision = (
            len(unique_citation_atoms & gold_atoms) / len(unique_citation_atoms)
            if unique_citation_atoms
            else 0.0
        )
        complete_evidence_set_cited = any(
            set(evidence_set) <= unique_citation_atoms
            for evidence_set in gold.evidence_sets
        )
    else:
        citation_precision = None
        complete_evidence_set_cited = None

    return AnswerScore(
        question_id=gold.question_id,
        semantic_alignment_status=semantic_alignment_status,
        gold_answerable=gold.answerable,
        predicted_answerable=prediction.predicted_answerable,
        answerability_correct=answerability_correct,
        required_fact_recall=required_recall,
        required_fact_precision=required_precision,
        all_critical_facts_present=all_critical,
        contradiction=contradiction,
        answer_correct=answer_correct,
        citation_validity=citation_validity,
        citation_precision=citation_precision,
        complete_evidence_set_cited=complete_evidence_set_cited,
        unknown_required_fact_ids=unknown_required,
        unknown_forbidden_fact_ids=unknown_forbidden,
    )


def score_provisional_answer(
    gold: GoldQuestion,
    prediction: AnswerPrediction,
    *,
    evidence_universe: Iterable[EvidenceAtomId] | None = None,
    semantic_alignment_status: str = "provisional_structured",
) -> AnswerScore:
    """Score only fields that need no natural-language semantic judgment.

    The generator's refusal decision and deterministic citation mapping can be
    compared with Gold. Required/forbidden fact entailment cannot: claim
    candidates have not yet been aligned by a human or independent judge, so
    correctness and Joint Success remain explicitly undefined.
    """

    if prediction.question_id != gold.question_id:
        raise ValueError("prediction question_id does not match gold question_id")
    if prediction.matched_required_fact_ids or prediction.asserted_forbidden_fact_ids:
        raise ValueError(
            "provisional structured predictions must not contain Gold fact decisions"
        )

    citation_atoms = tuple(
        citation.evidence_atom_id for citation in prediction.citations
    )
    unique_citation_atoms = set(citation_atoms)
    if evidence_universe is None:
        citation_validity = None
    elif not citation_atoms:
        citation_validity = 0.0 if gold.answerable else 1.0
    else:
        universe = set(evidence_universe)
        citation_validity = sum(atom in universe for atom in citation_atoms) / len(
            citation_atoms
        )
    if gold.answerable:
        gold_atoms = set(gold.all_gold_atoms)
        citation_precision = (
            len(unique_citation_atoms & gold_atoms) / len(unique_citation_atoms)
            if unique_citation_atoms
            else 0.0
        )
        complete_evidence_set_cited = any(
            set(evidence_set) <= unique_citation_atoms
            for evidence_set in gold.evidence_sets
        )
    else:
        citation_precision = None
        complete_evidence_set_cited = None

    return AnswerScore(
        question_id=gold.question_id,
        semantic_alignment_status=semantic_alignment_status,
        gold_answerable=gold.answerable,
        predicted_answerable=prediction.predicted_answerable,
        answerability_correct=(prediction.predicted_answerable == gold.answerable),
        required_fact_recall=None,
        required_fact_precision=None,
        all_critical_facts_present=None,
        contradiction=None,
        answer_correct=None,
        citation_validity=citation_validity,
        citation_precision=citation_precision,
        complete_evidence_set_cited=complete_evidence_set_cited,
        unknown_required_fact_ids=(),
        unknown_forbidden_fact_ids=(),
    )


def joint_success(
    retrieval: RetrievalScore, answer: AnswerScore
) -> bool | None:
    """Apply the provenance-gated end-to-end success rule."""

    if retrieval.question_id != answer.question_id:
        raise ValueError("retrieval and answer scores must describe the same question")
    if answer.answer_correct is None:
        return None
    return bool(
        answer.answer_correct
        and retrieval.complete_evidence_set_recall == 1.0
        and answer.complete_evidence_set_cited is True
        and answer.citation_validity == 1.0
    )


def aggregate_refusal(scores: Sequence[AnswerScore]) -> RefusalSummary:
    """Aggregate refusal as an explicit binary classification task."""

    if not scores:
        raise ValueError("at least one answer score is required")
    true_refusals = sum(
        not score.gold_answerable and not score.predicted_answerable for score in scores
    )
    false_refusals = sum(
        score.gold_answerable and not score.predicted_answerable for score in scores
    )
    missed_refusals = sum(
        not score.gold_answerable and score.predicted_answerable for score in scores
    )
    answerable_count = sum(score.gold_answerable for score in scores)
    unanswerable_count = len(scores) - answerable_count
    precision = _safe_ratio(true_refusals, true_refusals + false_refusals)
    recall = _safe_ratio(true_refusals, true_refusals + missed_refusals)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return RefusalSummary(
        questions=len(scores),
        true_refusals=true_refusals,
        false_refusals=false_refusals,
        missed_refusals=missed_refusals,
        refusal_precision=precision,
        refusal_recall=recall,
        refusal_f1=f1,
        over_refusal_rate=_safe_ratio(false_refusals, answerable_count),
        unsupported_answer_rate=_safe_ratio(missed_refusals, unanswerable_count),
    )


def mean_defined(values: Iterable[float | None]) -> float | None:
    """Mean helper that keeps undefined metrics out of aggregate denominators."""

    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None


def _normalise_grades(
    gold_atoms: set[EvidenceAtomId],
    relevance_grades: Mapping[EvidenceAtomId, int] | None,
) -> dict[EvidenceAtomId, int]:
    if relevance_grades is None:
        return {atom: 2 for atom in gold_atoms}
    grades: dict[EvidenceAtomId, int] = {}
    for atom, grade in relevance_grades.items():
        if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
            raise ValueError("relevance grades must be non-negative integers")
        if grade:
            grades[atom] = grade
    missing = gold_atoms - set(grades)
    if missing:
        raise ValueError("relevance grades must include every gold evidence atom")
    return grades


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _graded_gain(grade: int) -> float:
    return float((2**grade) - 1)
