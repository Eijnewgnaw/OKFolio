"""Provider-neutral hybrid retrieval harness for controlled RAG experiments.

The harness ranks the exact :class:`~kmpro_wiki.evaluation.contracts.RetrievedUnit`
objects emitted by ``T0``, ``T1`` and ``C1`` corpus builders.  Backends return
only stable unit IDs and scores; source text and provenance always come from the
audited corpus catalog.  This prevents a framework adapter from silently
changing chunks or metadata between experiment arms.

The deterministic core intentionally does not import Haystack, FlagEmbedding,
sentence-transformers, or a vector database.  Those systems can sit behind the
small protocols below without becoming test or runtime requirements.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from .contracts import RetrievedUnit
from .corpus import Arm, ContextSelection, CorpusBuild, TokenCounter
from .corpus import select_context_by_token_budget


@dataclass(frozen=True)
class BackendConfig:
    """Serializable identity for one frozen retrieval component."""

    provider: str
    model: str
    revision: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("backend provider must be non-empty")
        if not self.model.strip():
            raise ValueError("backend model must be non-empty")
        if not self.revision.strip():
            raise ValueError("backend revision must be non-empty")
        # Fail at configuration time rather than after a long experiment.
        json.dumps(self.parameters, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class RetrievalConfig:
    """Frozen knobs shared by every experiment arm.

    Candidate limits are *requested* budgets.  A backend may return fewer hits
    only when fewer valid units exist.  The trace records requested and actual
    counts so corpus-size effects remain visible.
    """

    bm25: BackendConfig
    dense: BackendConfig
    reranker: BackendConfig
    bm25_top_k: int = 50
    dense_top_k: int = 50
    fusion_top_k: int = 50
    rerank_top_k: int = 20
    context_token_budget: int = 8_192
    rrf_k: int = 60
    rrf_bm25_weight: float = 1.0
    rrf_dense_weight: float = 1.0
    context_separator: str = "\n\n---\n\n"
    schema: str = "okfolio.rag-retrieval-config.v1"

    def __post_init__(self) -> None:
        for name in (
            "bm25_top_k",
            "dense_top_k",
            "fusion_top_k",
            "rerank_top_k",
            "context_token_budget",
            "rrf_k",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rerank_top_k > self.fusion_top_k:
            raise ValueError("rerank_top_k cannot exceed fusion_top_k")
        for name in ("rrf_bm25_weight", "rrf_dense_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.context_separator:
            raise ValueError("context_separator must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SearchHit:
    unit_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("search hit unit_id must be non-empty")
        if not math.isfinite(float(self.score)):
            raise ValueError("search hit score must be finite")


class Retriever(Protocol):
    """A sparse or dense index scoped to exactly one corpus arm."""

    def search(self, query: str, *, limit: int) -> Sequence[SearchHit]: ...


class Reranker(Protocol):
    """Assign one relevance score to each supplied candidate, in order."""

    def score(
        self, query: str, candidates: Sequence[RetrievedUnit]
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class StageHit:
    stage: str
    rank: int
    unit_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRetrievalResult:
    question_id: str
    query: str
    arm: Arm
    config_fingerprint: str
    ranked_units: tuple[RetrievedUnit, ...]
    context: ContextSelection
    stages: Mapping[str, tuple[StageHit, ...]]
    requested_budgets: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a framework-independent trace with canonical provenance."""

        def serialize_unit(unit: RetrievedUnit) -> dict[str, Any]:
            return {
                "unit_id": unit.unit_id,
                "arm": unit.arm,
                "score": unit.score,
                "article_ids": list(unit.article_ids),
                "retrieval_evidence_atom_ids": [
                    str(item) for item in unit.retrieval_evidence_atom_ids
                ],
                "context_evidence_atom_ids": [
                    str(item) for item in unit.context_evidence_atom_ids
                ],
                "context_id": unit.metadata.get("context_id"),
            }

        return {
            "schema": "okfolio.rag-retrieval-trace.v1",
            "question_id": self.question_id,
            "query": self.query,
            "arm": self.arm,
            "config_fingerprint": self.config_fingerprint,
            "requested_budgets": dict(self.requested_budgets),
            "actual_candidates": {
                name: len(hits) for name, hits in self.stages.items()
            },
            "stages": {
                name: [item.to_dict() for item in hits]
                for name, hits in self.stages.items()
            },
            "ranked_units": [serialize_unit(item) for item in self.ranked_units],
            "selected_contexts": [serialize_unit(item) for item in self.context.units],
            "context_token_count": self.context.token_count,
            "context_token_budget": self.context.token_budget,
            "skipped_unit_ids": list(self.context.skipped_unit_ids),
        }


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchHit]],
    *,
    limit: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> tuple[SearchHit, ...]:
    """Fuse rankings by RRF with deterministic unit-ID tie breaking."""

    if limit < 1 or rrf_k < 1:
        raise ValueError("limit and rrf_k must be positive")
    resolved_weights = tuple(weights or (1.0,) * len(ranked_lists))
    if len(resolved_weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists")
    if any(not math.isfinite(value) or value <= 0 for value in resolved_weights):
        raise ValueError("RRF weights must be finite and positive")

    scores: dict[str, float] = {}
    for hits, weight in zip(ranked_lists, resolved_weights, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.unit_id in seen:
                raise ValueError(f"duplicate unit in one ranking: {hit.unit_id}")
            seen.add(hit.unit_id)
            scores[hit.unit_id] = scores.get(hit.unit_id, 0.0) + weight / (
                rrf_k + rank
            )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return tuple(SearchHit(unit_id, score) for unit_id, score in ordered)


class HybridRetrievalHarness:
    """Run BM25 + dense + RRF + reranker under one frozen contract."""

    def __init__(
        self,
        *,
        corpus: CorpusBuild,
        bm25: Retriever,
        dense: Retriever,
        reranker: Reranker,
        config: RetrievalConfig,
        count_tokens: TokenCounter,
    ) -> None:
        if corpus.audit()["status"] != "pass":
            raise ValueError(f"{corpus.arm} corpus failed provenance audit")
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.config = config
        self.count_tokens = count_tokens
        self._catalog = {unit.unit_id: unit for unit in corpus.units}
        if len(self._catalog) != len(corpus.units):
            raise ValueError(f"duplicate unit_id in {corpus.arm} corpus")

    def _validate_hits(
        self, hits: Sequence[SearchHit], *, stage: str, limit: int
    ) -> tuple[SearchHit, ...]:
        if len(hits) > limit:
            raise ValueError(f"{stage} returned more than its fixed candidate budget")
        seen: set[str] = set()
        validated: list[SearchHit] = []
        for hit in hits:
            if hit.unit_id not in self._catalog:
                raise ValueError(f"{stage} returned unknown unit_id: {hit.unit_id}")
            if hit.unit_id in seen:
                raise ValueError(f"{stage} returned duplicate unit_id: {hit.unit_id}")
            seen.add(hit.unit_id)
            validated.append(hit)
        # Backends need not share score scales, but each list must be ranked.
        return tuple(
            sorted(validated, key=lambda item: (-float(item.score), item.unit_id))
        )

    @staticmethod
    def _trace(stage: str, hits: Sequence[SearchHit]) -> tuple[StageHit, ...]:
        return tuple(
            StageHit(stage, rank, hit.unit_id, float(hit.score))
            for rank, hit in enumerate(hits, start=1)
        )

    def retrieve(self, *, question_id: str, query: str) -> QueryRetrievalResult:
        if not question_id.strip() or not query.strip():
            raise ValueError("question_id and query must be non-empty")
        config = self.config
        bm25_hits = self._validate_hits(
            self.bm25.search(query, limit=config.bm25_top_k),
            stage="bm25",
            limit=config.bm25_top_k,
        )
        dense_hits = self._validate_hits(
            self.dense.search(query, limit=config.dense_top_k),
            stage="dense",
            limit=config.dense_top_k,
        )
        fused_hits = reciprocal_rank_fusion(
            (bm25_hits, dense_hits),
            limit=config.fusion_top_k,
            rrf_k=config.rrf_k,
            weights=(config.rrf_bm25_weight, config.rrf_dense_weight),
        )
        candidates = tuple(self._catalog[item.unit_id] for item in fused_hits)
        rerank_scores = tuple(float(value) for value in self.reranker.score(query, candidates))
        if len(rerank_scores) != len(candidates):
            raise ValueError("reranker must return exactly one score per candidate")
        if any(not math.isfinite(value) for value in rerank_scores):
            raise ValueError("reranker scores must be finite")
        fused_rank = {hit.unit_id: rank for rank, hit in enumerate(fused_hits, start=1)}
        all_reranked_hits = tuple(
            SearchHit(unit.unit_id, score)
            for unit, score in sorted(
                zip(candidates, rerank_scores, strict=True),
                key=lambda item: (-item[1], fused_rank[item[0].unit_id], item[0].unit_id),
            )
        )
        reranked_hits = all_reranked_hits[: config.rerank_top_k]
        ranked_units = tuple(
            replace(self._catalog[hit.unit_id], score=float(hit.score))
            for hit in reranked_hits
        )
        context = select_context_by_token_budget(
            ranked_units,
            token_budget=config.context_token_budget,
            count_tokens=self.count_tokens,
            separator=config.context_separator,
        )
        stages = {
            "bm25": self._trace("bm25", bm25_hits),
            "dense": self._trace("dense", dense_hits),
            "rrf": self._trace("rrf", fused_hits),
            "reranker_scored": self._trace("reranker_scored", all_reranked_hits),
            "reranker": self._trace("reranker", reranked_hits),
        }
        return QueryRetrievalResult(
            question_id=question_id,
            query=query,
            arm=self.corpus.arm,
            config_fingerprint=config.fingerprint,
            ranked_units=ranked_units,
            context=context,
            stages=stages,
            requested_budgets={
                "bm25": config.bm25_top_k,
                "dense": config.dense_top_k,
                "rrf": config.fusion_top_k,
                "reranker_input": config.fusion_top_k,
                "reranker_output": config.rerank_top_k,
                "context_tokens": config.context_token_budget,
            },
        )


@dataclass(frozen=True)
class ThreeArmRetrievalResult:
    question_id: str
    query: str
    config_fingerprint: str
    arms: Mapping[Arm, QueryRetrievalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "okfolio.rag-three-arm-retrieval.v1",
            "question_id": self.question_id,
            "query": self.query,
            "config_fingerprint": self.config_fingerprint,
            "arms": {arm: result.to_dict() for arm, result in self.arms.items()},
        }


class ThreeArmRetrievalHarness:
    """Enforce one configuration fingerprint across T0, T1, and C1."""

    def __init__(self, arms: Mapping[Arm, HybridRetrievalHarness]) -> None:
        required: set[Arm] = {"T0", "T1", "C1"}
        if set(arms) != required:
            raise ValueError("three-arm harness requires exactly T0, T1, and C1")
        for arm, harness in arms.items():
            if harness.corpus.arm != arm:
                raise ValueError(f"harness/corpus arm mismatch: {arm}")
        fingerprints = {harness.config.fingerprint for harness in arms.values()}
        if len(fingerprints) != 1:
            raise ValueError("all arms must share one retrieval configuration")
        self.arms = dict(arms)
        self.config_fingerprint = fingerprints.pop()

    def retrieve(self, *, question_id: str, query: str) -> ThreeArmRetrievalResult:
        results = {
            arm: self.arms[arm].retrieve(question_id=question_id, query=query)
            for arm in ("T0", "T1", "C1")
        }
        return ThreeArmRetrievalResult(
            question_id=question_id,
            query=query,
            config_fingerprint=self.config_fingerprint,
            arms=results,
        )


RetrieverFactory = Callable[[CorpusBuild], Retriever]
RerankerFactory = Callable[[CorpusBuild], Reranker]


def build_three_arm_harness(
    corpora: Mapping[Arm, CorpusBuild],
    *,
    bm25_factory: RetrieverFactory,
    dense_factory: RetrieverFactory,
    reranker_factory: RerankerFactory,
    config: RetrievalConfig,
    count_tokens: TokenCounter,
) -> ThreeArmRetrievalHarness:
    """Build isolated per-arm indices while sharing every experiment knob."""

    harnesses: dict[Arm, HybridRetrievalHarness] = {}
    for arm in ("T0", "T1", "C1"):
        if arm not in corpora:
            raise ValueError(f"missing corpus arm: {arm}")
        corpus = corpora[arm]
        harnesses[arm] = HybridRetrievalHarness(
            corpus=corpus,
            bm25=bm25_factory(corpus),
            dense=dense_factory(corpus),
            reranker=reranker_factory(corpus),
            config=config,
            count_tokens=count_tokens,
        )
    return ThreeArmRetrievalHarness(harnesses)
