from dataclasses import replace

import pytest

from okfolio.evaluation.contracts import EvidenceAtomId, RetrievedUnit
from okfolio.evaluation.corpus import CorpusBuild, EvidenceAtom
from okfolio.evaluation.retrieval import (
    BackendConfig,
    HybridRetrievalHarness,
    RetrievalConfig,
    SearchHit,
    ThreeArmRetrievalHarness,
    reciprocal_rank_fusion,
)
from okfolio.evaluation.retrieval_adapters import (
    BgeM3EmbeddingAdapter,
    BgeRerankerAdapter,
    BM25Retriever,
    InMemoryDenseRetriever,
)


def _atom(article: str, ordinal: int) -> EvidenceAtom:
    atom_id = EvidenceAtomId(article, ordinal, "block", str(ordinal))
    return EvidenceAtom(
        atom_id=atom_id,
        article_id=article,
        source_file=f"{article}.md",
        page=ordinal,
        block_id=f"blk-{ordinal}",
        block_type="text",
        heading_path=("测试",),
        text=f"evidence {ordinal}",
        content_hash=f"h{ordinal}",
    )


def _corpus(arm: str, *, parent_duplicate: bool = False) -> CorpusBuild:
    atoms = (_atom("article-a", 1), _atom("article-a", 2), _atom("article-a", 3))
    units = []
    for ordinal, atom in enumerate(atoms, start=1):
        context = "evidence 1 shared parent" if parent_duplicate and ordinal < 3 else atom.text
        context_atoms = atoms[:2] if parent_duplicate and ordinal < 3 else (atom,)
        context_id = "parent-1" if parent_duplicate and ordinal < 3 else f"context-{ordinal}"
        units.append(
            RetrievedUnit(
                unit_id=f"{arm.lower()}-{ordinal}",
                arm=arm,
                retrieval_text=atom.text,
                context_text=context,
                retrieval_evidence_atom_ids=(atom.atom_id,),
                context_evidence_atom_ids=tuple(item.atom_id for item in context_atoms),
                article_ids=(atom.article_id,),
                metadata={"context_id": context_id},
            )
        )
    return CorpusBuild(arm, tuple(units), atoms)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = tuple(hits)
        self.requested = []

    def search(self, query: str, *, limit: int):
        self.requested.append((query, limit))
        return self.hits[:limit]


class FakeReranker:
    def __init__(self, scores):
        self.scores = dict(scores)
        self.seen = []

    def score(self, query, candidates):
        self.seen.append((query, tuple(item.unit_id for item in candidates)))
        return [self.scores[item.unit_id] for item in candidates]


def _config(**overrides) -> RetrievalConfig:
    backend = BackendConfig("test", "fake", "v1")
    values = {
        "bm25": backend,
        "dense": backend,
        "reranker": backend,
        "bm25_top_k": 2,
        "dense_top_k": 2,
        "fusion_top_k": 3,
        "rerank_top_k": 3,
        "context_token_budget": 50,
        "rrf_k": 10,
    }
    values.update(overrides)
    return RetrievalConfig(**values)


def test_rrf_uses_ranks_not_backend_score_scales_and_breaks_ties_stably():
    fused = reciprocal_rank_fusion(
        (
            (SearchHit("b", 10_000), SearchHit("a", 9_000)),
            (SearchHit("a", 0.01), SearchHit("b", 0.001)),
        ),
        limit=2,
        rrf_k=60,
    )

    assert [item.unit_id for item in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_hybrid_harness_enforces_budgets_and_emits_provenance_trace():
    corpus = _corpus("T1", parent_duplicate=True)
    bm25 = FakeRetriever([SearchHit("t1-1", 3), SearchHit("t1-3", 2)])
    dense = FakeRetriever([SearchHit("t1-2", 0.9), SearchHit("t1-3", 0.8)])
    reranker = FakeReranker({"t1-1": 0.7, "t1-2": 0.9, "t1-3": 0.8})
    harness = HybridRetrievalHarness(
        corpus=corpus,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        config=_config(),
        count_tokens=len,
    )

    result = harness.retrieve(question_id="q1", query="policy question")
    trace = result.to_dict()

    assert bm25.requested == [("policy question", 2)]
    assert dense.requested == [("policy question", 2)]
    assert reranker.seen[0][1] == ("t1-3", "t1-1", "t1-2")
    assert [item.unit_id for item in result.ranked_units] == ["t1-2", "t1-3", "t1-1"]
    # The two top child hits share one generation context and are deduplicated.
    assert [item.unit_id for item in result.context.units] == ["t1-2", "t1-3"]
    assert trace["requested_budgets"] == {
        "bm25": 2,
        "dense": 2,
        "rrf": 3,
        "reranker_input": 3,
        "reranker_output": 3,
        "context_tokens": 50,
    }
    assert trace["actual_candidates"]["reranker_scored"] == 3
    assert trace["ranked_units"][0]["retrieval_evidence_atom_ids"] == [
        "article-a:p002:b2"
    ]
    assert trace["ranked_units"][0]["context_evidence_atom_ids"] == [
        "article-a:p001:b1",
        "article-a:p002:b2",
    ]
    assert trace["context_token_count"] <= trace["context_token_budget"]


def test_harness_rejects_unknown_backend_units_and_excess_candidates():
    corpus = _corpus("T0")
    common = dict(
        corpus=corpus,
        dense=FakeRetriever([]),
        reranker=FakeReranker({}),
        config=_config(bm25_top_k=1, dense_top_k=1, fusion_top_k=1, rerank_top_k=1),
        count_tokens=len,
    )
    unknown = HybridRetrievalHarness(
        bm25=FakeRetriever([SearchHit("foreign", 1)]), **common
    )
    with pytest.raises(ValueError, match="unknown unit_id"):
        unknown.retrieve(question_id="q", query="x")

    class IgnoresLimit(FakeRetriever):
        def search(self, query, *, limit):
            return [SearchHit("t0-1", 2), SearchHit("t0-2", 1)]

    excessive = HybridRetrievalHarness(bm25=IgnoresLimit([]), **common)
    with pytest.raises(ValueError, match="fixed candidate budget"):
        excessive.retrieve(question_id="q", query="x")


def test_three_arm_harness_requires_one_identical_config():
    backend = lambda arm, config: HybridRetrievalHarness(
        corpus=_corpus(arm),
        bm25=FakeRetriever([]),
        dense=FakeRetriever([]),
        reranker=FakeReranker({}),
        config=config,
        count_tokens=len,
    )
    arms = {arm: backend(arm, _config()) for arm in ("T0", "T1", "C1")}
    harness = ThreeArmRetrievalHarness(arms)

    result = harness.retrieve(question_id="q", query="no match")

    assert tuple(result.arms) == ("T0", "T1", "C1")
    assert len({item.config_fingerprint for item in result.arms.values()}) == 1

    changed = dict(arms)
    changed["C1"] = backend("C1", _config(context_token_budget=51))
    with pytest.raises(ValueError, match="one retrieval configuration"):
        ThreeArmRetrievalHarness(changed)


def test_bm25_and_dense_adapters_rank_without_optional_dependencies():
    corpus = _corpus("T0")
    units = tuple(
        replace(unit, retrieval_text=text, context_text=text)
        for unit, text in zip(
            corpus.units,
            ("green policy", "financial policy", "regional economy"),
            strict=True,
        )
    )
    bm25 = BM25Retriever(units, tokenizer=lambda text: text.split())

    class Embedding:
        vectors = {
            "green policy": (1.0, 0.0),
            "financial policy": (0.8, 0.2),
            "regional economy": (0.0, 1.0),
            "policy": (1.0, 0.0),
        }

        def encode_documents(self, texts):
            return [self.vectors[text] for text in texts]

        def encode_queries(self, texts):
            return [self.vectors[text] for text in texts]

    dense = InMemoryDenseRetriever(units, embedding=Embedding())

    assert bm25.search("financial", limit=1)[0].unit_id == "t0-2"
    assert dense.search("policy", limit=2)[0].unit_id == "t0-1"


def test_bge_adapters_wrap_preloaded_backends_without_loading_models():
    class Encoder:
        def encode(self, texts, **kwargs):
            assert kwargs == {"batch_size": 2}
            return {"dense_vecs": [[len(text), 1] for text in texts]}

    adapter = BgeM3EmbeddingAdapter(Encoder(), batch_size=2)
    assert adapter.encode_queries(["ab"]) == ((2.0, 1.0),)

    class Ranker:
        def compute_score(self, pairs, **kwargs):
            assert kwargs == {"normalize": True}
            return [len(document) for _, document in pairs]

    reranker = BgeRerankerAdapter(Ranker(), normalize=True)
    scores = reranker.score("q", _corpus("T0").units[:2])
    assert scores == (10.0, 10.0)


def test_config_fingerprint_is_stable_across_mapping_order():
    first = replace(
        _config(),
        dense=BackendConfig("FlagEmbedding", "BAAI/bge-m3", "sha256:x", {"a": 1, "b": 2}),
    )
    second = replace(
        _config(),
        dense=BackendConfig("FlagEmbedding", "BAAI/bge-m3", "sha256:x", {"b": 2, "a": 1}),
    )
    assert first.fingerprint == second.fingerprint
