from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from kmpro_wiki.evaluation.contracts import (
    AnswerPrediction,
    EvidenceAtomId,
    FactLabel,
    GoldAnnotation,
    GoldQuestion,
    RetrievedUnit,
)
from kmpro_wiki.evaluation.corpus import CorpusBuild, EvidenceAtom
from kmpro_wiki.evaluation.generation import (
    AnswerContext,
    AnswerGenerationInput,
    GenerationResult,
    GenerationTiming,
    TokenUsage,
)
from kmpro_wiki.evaluation.local_bge_backend import (
    FrozenJiebaTokenizer,
    LocalBGERAGBackend,
    LocalBackendConfigurationError,
    create_backend,
)


def _corpus(arm: str = "T0") -> CorpusBuild:
    texts = ("金融 政策", "区域 经济", "绿色 金融")
    atoms = tuple(
        EvidenceAtom(
            atom_id=EvidenceAtomId("article-a", index, "block", str(index)),
            article_id="article-a",
            source_file="article-a.md",
            page=index,
            block_id=f"block-{index}",
            block_type="text",
            heading_path=("章节",),
            text=text,
            content_hash=f"hash-{index}",
        )
        for index, text in enumerate(texts, 1)
    )
    units = tuple(
        RetrievedUnit(
            unit_id=f"{arm.lower()}-{index}",
            arm=arm,
            retrieval_text=atom.text,
            context_text=atom.text,
            retrieval_evidence_atom_ids=(atom.atom_id,),
            context_evidence_atom_ids=(atom.atom_id,),
            article_ids=(atom.article_id,),
            metadata={"context_id": f"context-{index}"},
        )
        for index, atom in enumerate(atoms, 1)
    )
    return CorpusBuild(arm, units, atoms)


class FakeJieba:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def cut(self, sentence, cut_all=False, HMM=True):
        self.calls.append((cut_all, HMM))
        return sentence.split()


class FakeDense:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((tuple(texts), dict(kwargs)))
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float("金融" in text or "policy" in text),
                    float("区域" in text),
                    float("绿色" in text),
                ]
            )
        return {"dense_vecs": vectors}


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[list[str]], dict[str, object]]] = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append((pairs, dict(kwargs)))
        return [float("金融" in document) for _, document in pairs]


class FakeGenerationClient:
    def __init__(self) -> None:
        self.answer_calls = 0

    def generate_answer(self, request, *, stream=True):
        self.answer_calls += 1
        assert stream is True
        return GenerationResult(
            text=json.dumps(
                {
                    "schema": "okfolio.rag-answer.v1",
                    "answer": "材料支持该结论。",
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
                            "text": "材料支持该结论。",
                            "citation_ids": ["cite-1"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            reasoning="",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
            timing=GenerationTiming(None, None, 12.0, 30.0),
            stream_events=3,
        )


class FakeAligner:
    def align(self, *, gold, arm, request, result):
        assert arm == "T0"
        assert result.text.startswith("材料支持")
        return AnswerPrediction(
            question_id=gold.question_id,
            # The semantic reviewer cannot override refusal or citations;
            # those come only from the structured generation contract.
            predicted_answerable=False,
            matched_required_fact_ids=("f1",),
        )


def _gold() -> GoldQuestion:
    atom = EvidenceAtomId("article-a", 1, "block", "1")
    return GoldQuestion(
        question_id="q1",
        question="金融政策是什么？",
        question_type="fact",
        answerable=True,
        scope={},
        required_facts=(FactLabel("f1", "政策事实"),),
        forbidden_facts=(),
        evidence_sets=((atom,),),
        reference_answer="政策事实",
        annotation=GoldAnnotation(author="human", status="reviewed"),
    )


def _backend(
    tmp_path: Path,
    *,
    tokenizer: FrozenJiebaTokenizer,
    dense_models: list[FakeDense],
    rerank_models: list[FakeReranker],
    generation_client=None,
    aligner=None,
) -> LocalBGERAGBackend:
    dense_dir = tmp_path / "bge-m3"
    reranker_dir = tmp_path / "reranker"
    dense_dir.mkdir(parents=True, exist_ok=True)
    reranker_dir.mkdir(parents=True, exist_ok=True)

    def load_dense(path, settings):
        assert path == dense_dir
        assert settings["device"] == "mps"
        assert settings["batch_size"] == 3
        model = FakeDense()
        dense_models.append(model)
        return model

    def load_reranker(path, settings):
        assert path == reranker_dir
        assert settings["device"] == "mps"
        assert settings["batch_size"] == 2
        model = FakeReranker()
        rerank_models.append(model)
        return model

    options = {
        "dense_model_path": str(dense_dir),
        "dense_revision": "dense-sha",
        "reranker_model_path": str(reranker_dir),
        "reranker_revision": "rerank-sha",
        "device": "mps",
        "dense_batch_size": 3,
        "reranker_batch_size": 2,
    }
    if aligner is not None:
        options["prediction_alignment_status"] = "human_reviewed"
    return LocalBGERAGBackend(
        options,
        tmp_path / "run",
        dense_loader=load_dense,
        reranker_loader=load_reranker,
        jieba_tokenizer=tokenizer,
        generation_client=generation_client,
        aligner=aligner,
        numpy_module=np,
    )


def test_index_records_frozen_jieba_policy_and_reuses_numpy_cache(tmp_path: Path):
    fake_jieba = FakeJieba()
    tokenizer = FrozenJiebaTokenizer(fake_jieba, dictionary_sha256="dict-sha")
    dense_models: list[FakeDense] = []
    backend = _backend(
        tmp_path,
        tokenizer=tokenizer,
        dense_models=dense_models,
        rerank_models=[],
    )
    corpus = _corpus()
    index_dir = tmp_path / "index"

    backend.prepare_index(corpus, index_dir)
    backend.prepare_index(corpus, index_dir)

    assert len(dense_models) == 1
    assert len(dense_models[0].calls) == 1
    assert dense_models[0].calls[0][1]["batch_size"] == 3
    matrix = np.load(index_dir / "dense_embeddings.npy")
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    bm25_metadata = json.loads(
        (index_dir / "bm25.meta.json").read_text(encoding="utf-8")
    )
    assert bm25_metadata["dictionary_sha256"] == "dict-sha"
    assert bm25_metadata["hmm"] is False

    bm25 = backend.create_bm25(corpus, index_dir)
    assert bm25.search("金融", limit=1)[0].unit_id == "t0-1"
    assert fake_jieba.calls
    assert all(call == (False, False) for call in fake_jieba.calls)


def test_dense_query_cache_is_shared_and_reranker_is_lazy_with_small_batches(
    tmp_path: Path,
):
    tokenizer = FrozenJiebaTokenizer(FakeJieba(), dictionary_sha256="dict-sha")
    dense_models: list[FakeDense] = []
    rerank_models: list[FakeReranker] = []
    backend = _backend(
        tmp_path,
        tokenizer=tokenizer,
        dense_models=dense_models,
        rerank_models=rerank_models,
    )
    corpus = _corpus()
    index_dir = tmp_path / "index"
    backend.prepare_index(corpus, index_dir)
    dense = backend.create_dense(corpus, index_dir)
    reranker = backend.create_reranker(corpus)

    assert len(rerank_models) == 0
    assert dense.search("金融 policy", limit=2)[0].unit_id == "t0-1"
    dense.search("金融 policy", limit=2)
    # One model was used for indexing and released; one online model encoded
    # the repeated query exactly once.
    assert len(dense_models) == 2
    assert len(dense_models[1].calls) == 1

    scores = reranker.score("金融", corpus.units[:2])
    assert scores == (1.0, 0.0)
    assert len(rerank_models) == 1
    assert rerank_models[0].calls[0][1] == {
        "batch_size": 2,
        "max_length": 1024,
        "normalize": True,
    }


def test_generation_uses_existing_client_and_keeps_external_alignment_explicit(
    tmp_path: Path,
):
    tokenizer = FrozenJiebaTokenizer(FakeJieba(), dictionary_sha256="dict-sha")
    client = FakeGenerationClient()
    backend = _backend(
        tmp_path,
        tokenizer=tokenizer,
        dense_models=[],
        rerank_models=[],
        generation_client=client,
        aligner=FakeAligner(),
    )
    request = AnswerGenerationInput(
        question="金融政策是什么？",
        contexts=(
            AnswerContext(
                "context-1",
                "政策事实",
                page_numbers=(1,),
                evidence_ids=("article-a:p001:b1",),
            ),
        ),
    )

    result = backend.generate(gold=_gold(), arm="T0", request=request)

    assert client.answer_calls == 1
    assert result.prediction.predicted_answerable is True
    assert result.prediction.matched_required_fact_ids == ("f1",)
    assert tuple(str(item.evidence_atom_id) for item in result.prediction.citations) == (
        "article-a:p001:b1",
    )
    assert result.metadata["alignment"] == "explicit-plugin"
    assert result.semantic_alignment_status == "human_reviewed"

    no_aligner_client = FakeGenerationClient()
    unaligned = _backend(
        tmp_path / "unaligned",
        tokenizer=tokenizer,
        dense_models=[],
        rerank_models=[],
        generation_client=no_aligner_client,
    )
    provisional = unaligned.generate(gold=_gold(), arm="T0", request=request)

    assert no_aligner_client.answer_calls == 1
    assert provisional.text == "材料支持该结论。"
    assert provisional.prediction.predicted_answerable is True
    assert provisional.prediction.matched_required_fact_ids == ()
    assert tuple(str(item.evidence_atom_id) for item in provisional.prediction.citations) == (
        "article-a:p001:b1",
    )
    assert provisional.metadata["alignment"] == "contract-and-provenance-only"
    assert (
        provisional.semantic_alignment_status
        == "provisional_structured"
    )


def test_external_aligner_requires_auditable_status_before_generation(tmp_path: Path):
    tokenizer = FrozenJiebaTokenizer(FakeJieba(), dictionary_sha256="dict-sha")
    client = FakeGenerationClient()
    backend = _backend(
        tmp_path,
        tokenizer=tokenizer,
        dense_models=[],
        rerank_models=[],
        generation_client=client,
        aligner=FakeAligner(),
    )
    del backend.options["prediction_alignment_status"]
    request = AnswerGenerationInput(
        question="金融政策是什么？",
        contexts=(
            AnswerContext(
                "context-1",
                "政策事实",
                page_numbers=(1,),
                evidence_ids=("article-a:p001:b1",),
            ),
        ),
    )

    with pytest.raises(LocalBackendConfigurationError, match="must declare"):
        backend.generate(gold=_gold(), arm="T0", request=request)
    assert client.answer_calls == 0


def test_factory_is_lazy_and_accepts_model_and_generation_settings_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dense = tmp_path / "dense"
    reranker = tmp_path / "reranker"
    dense.mkdir()
    reranker.mkdir()
    monkeypatch.setenv("RAG_BGE_M3_MODEL_PATH", str(dense))
    monkeypatch.setenv("RAG_BGE_RERANKER_MODEL_PATH", str(reranker))
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "local-openai-compatible-model")

    backend = create_backend(
        {
            "dense_model_path": "${RAG_BGE_M3_MODEL_PATH}",
            "reranker_model_path": "${RAG_BGE_RERANKER_MODEL_PATH}",
            "generation": {
                "base_url": "${RAG_LLM_BASE_URL}",
                "model": "${RAG_LLM_MODEL}",
            },
        },
        tmp_path / "run",
    )

    # Factory construction must not import/load BGE or contact generation.
    assert backend._dense_model is None
    assert backend._reranker_model is None
    assert backend._generation_client is None
    assert backend._dense_path() == dense
    assert backend._reranker_path() == reranker


def test_example_config_is_generic_and_contains_no_local_endpoint_or_secret():
    path = Path(__file__).parents[1] / "examples/rag-local-bge.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)

    assert payload["adapter"]["factory"].endswith(":create_backend")
    assert "127.0.0.1" not in serialized
    assert ".".join(("192", "168")) not in serialized
    assert "api_key" not in serialized.casefold()
