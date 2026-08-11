from okfolio.evaluation.contracts import (
    AnswerPrediction,
    Citation,
    EvidenceAtomId,
    FactLabel,
    GoldAnnotation,
    GoldQuestion,
    RetrievedUnit,
)
from okfolio.evaluation.scoring import (
    aggregate_refusal,
    joint_success,
    score_answer,
    score_provisional_answer,
    score_retrieval,
)


def _atom(value: str) -> EvidenceAtomId:
    return EvidenceAtomId.parse(value)


A = _atom("article-a:p003:s001")
B = _atom("article-b:p004:s002")
C = _atom("article-c:p005:b003")
Z = _atom("article-z:p006:s004")


def _gold(*, answerable: bool = True, question_id: str = "q-1") -> GoldQuestion:
    return GoldQuestion(
        question_id=question_id,
        question="测试问题",
        question_type="cross_document_synthesis",
        answerable=answerable,
        scope={},
        required_facts=(
            FactLabel("f1", "关键事实", weight=2, critical=True),
            FactLabel("f2", "补充事实", weight=1, critical=False),
        )
        if answerable
        else (),
        forbidden_facts=(FactLabel("x1", "冲突事实"),),
        evidence_sets=((A, B), (C,)) if answerable else (),
        reference_answer="参考答案" if answerable else None,
        annotation=GoldAnnotation("a", "adjudicated", "b"),
    )


def _unit(
    unit_id: str,
    *,
    retrieval_atoms: tuple[EvidenceAtomId, ...],
    context_atoms: tuple[EvidenceAtomId, ...],
) -> RetrievedUnit:
    return RetrievedUnit(
        unit_id=unit_id,
        arm="T1",
        retrieval_text=f"retrieval {unit_id}",
        context_text=f"context {unit_id}",
        retrieval_evidence_atom_ids=retrieval_atoms,
        context_evidence_atom_ids=context_atoms,
        article_ids=tuple(
            sorted({atom.article_id for atom in (*retrieval_atoms, *context_atoms)})
        ),
    )


def test_retrieval_scoring_separates_child_hit_from_parent_context():
    units = (
        _unit("u1", retrieval_atoms=(C,), context_atoms=(Z,)),
        _unit("u2", retrieval_atoms=(Z,), context_atoms=(A, B)),
    )

    child = score_retrieval(_gold(), units, k=1, provenance_view="retrieval")
    parent_at_one = score_retrieval(_gold(), units, k=1, provenance_view="context")
    parent_at_two = score_retrieval(_gold(), units, k=2, provenance_view="context")

    assert child.evidence_recall == 1.0
    assert child.complete_evidence_set_recall == 1.0
    assert child.mrr == 1.0
    assert parent_at_one.evidence_recall == 0.0
    assert parent_at_one.complete_evidence_set_recall == 0.0
    assert parent_at_one.mrr == 0.0
    assert parent_at_two.evidence_recall == 1.0
    assert parent_at_two.complete_evidence_set_recall == 1.0
    assert parent_at_two.mrr == 0.5
    assert 0.53 < parent_at_two.ndcg < 0.54
    assert parent_at_two.context_precision == 2 / 3


def test_unanswerable_retrieval_metrics_are_explicitly_undefined():
    score = score_retrieval(
        _gold(answerable=False),
        (_unit("u", retrieval_atoms=(Z,), context_atoms=(Z,)),),
        k=1,
    )

    assert score.evidence_recall is None
    assert score.complete_evidence_set_recall is None
    assert score.mrr is None
    assert score.ndcg is None


def test_answer_scoring_keeps_correctness_citations_and_joint_gate_separate():
    gold = _gold()
    prediction = AnswerPrediction(
        question_id=gold.question_id,
        predicted_answerable=True,
        matched_required_fact_ids=("f1",),
        citations=(Citation(A), Citation(B)),
    )
    answer = score_answer(gold, prediction, evidence_universe=(A, B, C, Z))
    retrieval = score_retrieval(
        gold,
        (_unit("u", retrieval_atoms=(A, B), context_atoms=(A, B)),),
        k=1,
    )

    assert answer.required_fact_recall == 2 / 3
    assert answer.required_fact_precision == 1.0
    assert answer.all_critical_facts_present is True
    assert answer.answer_correct is True
    assert answer.citation_validity == 1.0
    assert answer.citation_precision == 1.0
    assert answer.complete_evidence_set_cited is True
    assert joint_success(retrieval, answer) is True


def test_forbidden_or_unknown_fact_prevents_answer_correctness():
    score = score_answer(
        _gold(),
        AnswerPrediction(
            question_id="q-1",
            predicted_answerable=True,
            matched_required_fact_ids=("f1", "unknown"),
            asserted_forbidden_fact_ids=("x1",),
        ),
    )

    assert score.contradiction is True
    assert score.unknown_required_fact_ids == ("unknown",)
    assert score.answer_correct is False


def test_provisional_structured_answer_scores_only_deterministic_fields():
    gold = _gold()
    prediction = AnswerPrediction(
        question_id=gold.question_id,
        predicted_answerable=True,
        citations=(Citation(A), Citation(B)),
    )

    score = score_provisional_answer(
        gold, prediction, evidence_universe=(A, B, C, Z)
    )

    assert score.semantic_alignment_status == "provisional_structured"
    assert score.answerability_correct is True
    assert score.required_fact_recall is None
    assert score.required_fact_precision is None
    assert score.all_critical_facts_present is None
    assert score.contradiction is None
    assert score.answer_correct is None
    assert score.citation_validity == 1.0
    assert score.citation_precision == 1.0
    assert score.complete_evidence_set_cited is True
    retrieval = score_retrieval(
        gold,
        (_unit("u", retrieval_atoms=(A, B), context_atoms=(A, B)),),
        k=1,
    )
    assert joint_success(retrieval, score) is None


def test_refusal_summary_exposes_over_refusal_and_unsupported_answers():
    true_refusal = score_answer(
        _gold(answerable=False, question_id="u1"),
        AnswerPrediction(question_id="u1", predicted_answerable=False),
    )
    missed_refusal = score_answer(
        _gold(answerable=False, question_id="u2"),
        AnswerPrediction(question_id="u2", predicted_answerable=True),
    )
    false_refusal = score_answer(
        _gold(question_id="a1"),
        AnswerPrediction(question_id="a1", predicted_answerable=False),
    )
    summary = aggregate_refusal((true_refusal, missed_refusal, false_refusal))

    assert summary.true_refusals == 1
    assert summary.false_refusals == 1
    assert summary.missed_refusals == 1
    assert summary.refusal_precision == 0.5
    assert summary.refusal_recall == 0.5
    assert summary.refusal_f1 == 0.5
    assert summary.over_refusal_rate == 1.0
    assert summary.unsupported_answer_rate == 0.5
