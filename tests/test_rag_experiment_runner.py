from __future__ import annotations

import json
from pathlib import Path

import pytest

from okfolio.evaluation.contracts import (
    AnswerPrediction,
    Citation,
    EvidenceAtomId,
    RetrievedUnit,
)
from okfolio.evaluation.corpus import CorpusBuild, EvidenceAtom
from okfolio.evaluation.experiment_runner import (
    AlignedGenerationPipeline,
    CorpusConfig,
    ExperimentConfig,
    ExperimentStateError,
    GeneratedAnswer,
    ThreeArmExperimentRunner,
)
from okfolio.evaluation.generation import (
    AnswerContext,
    AnswerGenerationInput,
    GenerationResult,
    GenerationTiming,
    TokenUsage,
)
from okfolio.evaluation.gold import load_gold_jsonl
from okfolio.evaluation.retrieval import BackendConfig, RetrievalConfig, SearchHit


def _gold(path: Path) -> None:
    rows = [
        {
            "question_id": "q-answerable",
            "question": "policy fact",
            "question_type": "fact",
            "answerable": True,
            "scope": {},
            "required_facts": [{"fact_id": "f1", "claim": "fact one"}],
            "forbidden_facts": [],
            "evidence_sets": [["article-a:p001:b1"]],
            "reference_answer": "fact one",
            "annotation": {"author": "tester", "status": "reviewed"},
        },
        {
            "question_id": "q-refusal",
            "question": "unsupported question",
            "question_type": "unanswerable",
            "answerable": False,
            "scope": {},
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_sets": [],
            "reference_answer": None,
            "annotation": {"author": "tester", "status": "reviewed"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(tmp_path: Path, gold: Path, *, hyde_mode: str = "off") -> ExperimentConfig:
    backend = BackendConfig("fake", "fake-model", "frozen-v1")
    return ExperimentConfig(
        experiment_id="fake-three-arm",
        structures_dir=tmp_path / "structures",
        c1_run_dir=tmp_path / "agent-run",
        gold_path=gold,
        corpus=CorpusConfig(),
        retrieval=RetrievalConfig(
            bm25=backend,
            dense=backend,
            reranker=backend,
            bm25_top_k=2,
            dense_top_k=2,
            fusion_top_k=2,
            rerank_top_k=2,
            context_token_budget=100,
        ),
        hyde_mode=hyde_mode,  # type: ignore[arg-type]
        bootstrap_samples=100,
    )


def _corpus(arm: str) -> CorpusBuild:
    atoms = tuple(
        EvidenceAtom(
            atom_id=EvidenceAtomId("article-a", ordinal, "block", str(ordinal)),
            article_id="article-a",
            source_file="article-a.md",
            page=ordinal,
            block_id=f"blk-{ordinal}",
            block_type="text",
            heading_path=("section",),
            text=f"policy fact evidence {ordinal}",
            content_hash=f"hash-{ordinal}",
        )
        for ordinal in (1, 2)
    )
    units = tuple(
        RetrievedUnit(
            unit_id=f"{arm.lower()}-{ordinal}",
            arm=arm,
            retrieval_text=atom.text,
            context_text=atom.text,
            retrieval_evidence_atom_ids=(atom.atom_id,),
            context_evidence_atom_ids=(atom.atom_id,),
            article_ids=(atom.article_id,),
            metadata={"context_id": f"{arm.lower()}-context-{ordinal}"},
        )
        for ordinal, atom in enumerate(atoms, 1)
    )
    return CorpusBuild(arm, units, atoms)


class _Retriever:
    def __init__(self, corpus: CorpusBuild) -> None:
        self.corpus = corpus

    def search(self, query: str, *, limit: int):
        return tuple(
            SearchHit(unit.unit_id, float(len(self.corpus.units) - index))
            for index, unit in enumerate(self.corpus.units[:limit])
        )


class _Reranker:
    def score(self, query, candidates):
        return tuple(float(len(candidates) - index) for index, _ in enumerate(candidates))


class FakeBackend:
    def __init__(self) -> None:
        self.index_calls = 0
        self.generation_calls = 0
        self.hyde_calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prepare_index(self, corpus: CorpusBuild, index_dir: Path) -> None:
        self.index_calls += 1
        (index_dir / "fake.ready").write_text(corpus.arm, encoding="utf-8")

    def create_bm25(self, corpus, index_dir):
        return _Retriever(corpus)

    def create_dense(self, corpus, index_dir):
        return _Retriever(corpus)

    def create_reranker(self, corpus):
        return _Reranker()

    def generate(self, *, gold, arm, request):
        self.generation_calls += 1
        if gold.answerable:
            prediction = AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=True,
                matched_required_fact_ids=("f1",),
                citations=(Citation(EvidenceAtomId.parse("article-a:p001:b1")),),
            )
            text = "fact one [context, p.1]"
        else:
            prediction = AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=False,
            )
            text = "insufficient material"
        return GeneratedAnswer(
            text=text,
            prediction=prediction,
            semantic_alignment_status="human_reviewed",
        )

    def generate_hyde(self, query: str) -> str:
        self.hyde_calls += 1
        return "hypothetical expansion"


class ProvisionalFakeBackend(FakeBackend):
    def generate(self, *, gold, arm, request):
        self.generation_calls += 1
        if gold.answerable:
            prediction = AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=True,
                citations=(Citation(EvidenceAtomId.parse("article-a:p001:b1")),),
            )
            text = "fact one"
        else:
            prediction = AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=False,
            )
            text = "insufficient material"
        return GeneratedAnswer(
            text=text,
            prediction=prediction,
            semantic_alignment_status="provisional_structured",
            metadata={
                "alignment": "contract-and-provenance-only",
            },
        )


class FakeRunner(ThreeArmExperimentRunner):
    def _input_lock(self):
        return {
            "schema": "test-lock",
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint,
            "inputs": {"fixture": "frozen"},
        }

    def _build_corpora(self):
        return {arm: _corpus(arm) for arm in ("T0", "T1", "C1")}


def test_complete_runner_is_three_arm_checkpointed_and_resumable(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    backend = FakeBackend()
    runner = FakeRunner(
        config=_config(tmp_path, gold),
        output_dir=tmp_path / "run",
        backend=backend,
    )

    summary = runner.run("all")

    assert backend.index_calls == 3
    assert backend.generation_calls == 6
    assert backend.hyde_calls == 0
    assert summary["hyde_mode"] == "off"
    assert summary["metrics"]["T0"]["questions"] == 2
    assert summary["metrics"]["C1"]["answer_accuracy"] == 1.0
    assert len((tmp_path / "run/traces/retrieval.jsonl").read_text().splitlines()) == 6
    assert len((tmp_path / "run/traces/generation.jsonl").read_text().splitlines()) == 6
    assert len((tmp_path / "run/traces/scores.jsonl").read_text().splitlines()) == 6

    # A second invocation reuses every durable row and does not call providers.
    assert runner.run("all") == summary
    assert backend.index_calls == 3
    assert backend.generation_calls == 6


def test_runner_keeps_unjudged_semantic_metrics_explicitly_provisional(
    tmp_path: Path,
):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    backend = ProvisionalFakeBackend()
    runner = FakeRunner(
        config=_config(tmp_path, gold),
        output_dir=tmp_path / "run",
        backend=backend,
    )

    summary = runner.run("all")

    assert summary["metrics"]["T0"]["answer_accuracy"] is None
    assert summary["metrics"]["T0"]["joint_success_rate"] is None
    assert summary["metrics"]["T0"]["semantic_scoring"] == {
        "status": "provisional",
        "scored_rows": 0,
        "pending_rows": 2,
        "alignment_status_counts": {"provisional_structured": 2},
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "run/traces/scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["answer"]["answer_correct"] is None for row in rows)
    assert all(row["joint_success"] is None for row in rows)


def test_hyde_requires_explicit_ablation_and_is_shared_across_arms(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    backend = FakeBackend()
    runner = FakeRunner(
        config=_config(tmp_path, gold, hyde_mode="ablation"),
        output_dir=tmp_path / "run",
        backend=backend,
    )

    runner.index()
    runner.retrieve()

    # One expansion per question, never one independently generated per arm.
    assert backend.hyde_calls == 2
    rows = [json.loads(line) for line in (tmp_path / "run/traces/retrieval.jsonl").read_text().splitlines()]
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question_id"], set()).add(row["retrieval_query"])
        assert row["hyde_mode"] == "ablation"
    assert all(len(queries) == 1 for queries in by_question.values())


def test_hyde_rejects_non_ablation_experiment_modes(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)

    with pytest.raises(ValueError, match="explicit 'ablation'"):
        _config(tmp_path, gold, hyde_mode="on")


def test_readiness_rejects_incomplete_c1_before_any_write(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "one.structure.json").write_text("{}", encoding="utf-8")
    c1 = tmp_path / "agent-run"
    c1.mkdir()
    (c1 / "manifest.json").write_text('{"status":"running"}', encoding="utf-8")
    (c1 / "acceptance.json").write_text('{"status":"pass"}', encoding="utf-8")
    output = tmp_path / "must-not-exist"
    runner = ThreeArmExperimentRunner(
        config=_config(tmp_path, gold), output_dir=output
    )

    with pytest.raises(ExperimentStateError, match="expected 'complete'"):
        runner.readiness(build_corpora=False)
    assert not output.exists()


def test_readiness_reports_missing_acceptance_as_blocked_state(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "one.structure.json").write_text("{}", encoding="utf-8")
    c1 = tmp_path / "agent-run"
    c1.mkdir()
    (c1 / "manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
    output = tmp_path / "must-not-exist"
    runner = ThreeArmExperimentRunner(
        config=_config(tmp_path, gold), output_dir=output
    )

    with pytest.raises(ExperimentStateError, match="missing .*acceptance.json"):
        runner.readiness(build_corpora=False)
    assert not output.exists()


def test_readiness_is_no_write_and_does_not_require_backend(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold)
    output = tmp_path / "dry-run"
    runner = FakeRunner(config=_config(tmp_path, gold), output_dir=output)

    report = runner.readiness()

    assert report["status"] == "ready"
    assert report["writes_performed"] is False
    assert report["backend_loaded"] is False
    assert set(report["corpora"]) == {"T0", "T1", "C1"}
    assert not output.exists()


def test_aligned_pipeline_composes_generation_contract_with_visible_judge(tmp_path: Path):
    gold_path = tmp_path / "gold.jsonl"
    _gold(gold_path)
    gold = load_gold_jsonl(gold_path)[0]
    request = AnswerGenerationInput(
        question=gold.question,
        contexts=(
            AnswerContext(
                context_id="context-1",
                text="policy fact",
                page_numbers=(1,),
                evidence_ids=("article-a:p001:b1",),
            ),
        ),
    )

    class TextGenerator:
        def generate_answer(self, request, *, stream=True):
            assert stream is False
            return GenerationResult(
                text=json.dumps(
                    {
                        "schema": "okfolio.rag-answer.v1",
                        "answer": "fact one",
                        "refusal": False,
                        "refusal_reason": "",
                        "citations": [
                            {
                                "citation_id": "cite-1",
                                "context_id": "context-1",
                                "page": 1,
                            }
                        ],
                        "atomic_claim_candidates": [
                            {
                                "claim_id": "claim-1",
                                "text": "fact one",
                                "citation_ids": ["cite-1"],
                            }
                        ],
                    }
                ),
                reasoning="",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                timing=GenerationTiming(None, None, None, 25.0),
                stream_events=0,
            )

    class Aligner:
        def align(self, *, gold, arm, request, result):
            assert arm == "C1"
            assert result.text == "fact one"
            return AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=True,
                matched_required_fact_ids=("f1",),
            )

    result = AlignedGenerationPipeline(
        generator=TextGenerator(),
        aligner=Aligner(),
        semantic_alignment_status="human_reviewed",
        stream=False,
    ).generate(gold=gold, arm="C1", request=request)

    assert result.text == "fact one"
    assert tuple(str(item.evidence_atom_id) for item in result.prediction.citations) == (
        "article-a:p001:b1",
    )
    assert result.usage["total_tokens"] == 12
    assert result.timing["total_ms"] == 25.0
