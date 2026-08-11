"""Optional lightweight adapters for the retrieval evaluation harness.

Nothing in this module downloads a model.  BGE adapters receive an already
constructed local backend object, making model paths, revisions, device, and
offline policy the caller's responsibility and part of ``RetrievalConfig``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import RetrievedUnit
from .retrieval import SearchHit


Tokenizer = Callable[[str], Sequence[str]]
Vector = Sequence[float]


def whitespace_tokenizer(text: str) -> tuple[str, ...]:
    """Minimal test tokenizer; production Chinese runs must inject a frozen tokenizer."""

    return tuple(part for part in text.casefold().split() if part)


class BM25Retriever:
    """Dependency-free Okapi BM25 over immutable retrieval texts."""

    def __init__(
        self,
        units: Sequence[RetrievedUnit],
        *,
        tokenizer: Tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not units:
            raise ValueError("BM25 corpus must be non-empty")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("BM25 k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        self.units = tuple(units)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self._tokens = tuple(tuple(tokenizer(unit.retrieval_text)) for unit in units)
        if any(not tokens for tokens in self._tokens):
            raise ValueError("BM25 tokenizer produced an empty document")
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("BM25 limit must be positive")
        query_terms = tuple(self.tokenizer(query))
        total_documents = len(self.units)
        scores: list[SearchHit] = []
        for unit, tokens, frequencies in zip(
            self.units, self._tokens, self._term_frequencies, strict=True
        ):
            score = 0.0
            length_norm = 1 - self.b + self.b * len(tokens) / self._average_length
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5) /
                    (document_frequency + 0.5)
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * length_norm)
                )
            scores.append(SearchHit(unit.unit_id, score))
        return tuple(
            sorted(scores, key=lambda item: (-item.score, item.unit_id))[:limit]
        )


class BgeM3EmbeddingAdapter:
    """Adapt an existing ``BGEM3FlagModel``-like object to dense vectors.

    The wrapped backend must expose ``encode(texts, **kwargs)``.  Its result may
    be a mapping containing ``dense_vecs`` (FlagEmbedding style) or a sequence
    of vectors.  Construction/loading is deliberately outside this adapter.
    """

    def __init__(self, backend: Any, **encode_options: Any) -> None:
        if not callable(getattr(backend, "encode", None)):
            raise TypeError("BGE-M3 backend must expose encode")
        self.backend = backend
        self.encode_options = dict(encode_options)

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raw = self.backend.encode(list(texts), **self.encode_options)
        if isinstance(raw, Mapping):
            raw = raw.get("dense_vecs")
        if raw is None:
            raise ValueError("BGE-M3 backend did not return dense_vecs")
        vectors = tuple(tuple(float(value) for value in vector) for vector in raw)
        if len(vectors) != len(texts):
            raise ValueError("BGE-M3 vector count does not match text count")
        if any(not vector for vector in vectors):
            raise ValueError("BGE-M3 returned an empty vector")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError("BGE-M3 returned inconsistent vector dimensions")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("BGE-M3 returned a non-finite vector value")
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)


class InMemoryDenseRetriever:
    """Exact cosine search for probes and provider-neutral BGE-M3 integration."""

    def __init__(self, units: Sequence[RetrievedUnit], *, embedding: Any) -> None:
        if not units:
            raise ValueError("dense corpus must be non-empty")
        self.units = tuple(units)
        self.embedding = embedding
        self._vectors = tuple(
            tuple(map(float, vector))
            for vector in embedding.encode_documents(
                [unit.retrieval_text for unit in self.units]
            )
        )
        if len(self._vectors) != len(self.units):
            raise ValueError("document vector count does not match corpus size")
        dimensions = {len(vector) for vector in self._vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("document vectors must share a non-zero dimension")
        self._dimension = dimensions.pop()

    @staticmethod
    def _cosine(left: Vector, right: Vector) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("dense limit must be positive")
        encoded = self.embedding.encode_queries([query])
        if len(encoded) != 1:
            raise ValueError("query encoder must return exactly one vector")
        query_vector = tuple(map(float, encoded[0]))
        if len(query_vector) != self._dimension:
            raise ValueError("query and document vector dimensions differ")
        hits = (
            SearchHit(unit.unit_id, self._cosine(query_vector, vector))
            for unit, vector in zip(self.units, self._vectors, strict=True)
        )
        return tuple(sorted(hits, key=lambda item: (-item.score, item.unit_id))[:limit])


class BgeRerankerAdapter:
    """Adapt a preloaded BGE reranker with ``compute_score(pairs, ...)``."""

    def __init__(self, backend: Any, **score_options: Any) -> None:
        if not callable(getattr(backend, "compute_score", None)):
            raise TypeError("BGE reranker backend must expose compute_score")
        self.backend = backend
        self.score_options = dict(score_options)

    def score(
        self, query: str, candidates: Sequence[RetrievedUnit]
    ) -> tuple[float, ...]:
        if not candidates:
            return ()
        pairs = [[query, unit.retrieval_text] for unit in candidates]
        raw = self.backend.compute_score(pairs, **self.score_options)
        if isinstance(raw, (int, float)):
            raw = [raw]
        scores = tuple(float(value) for value in raw)
        if len(scores) != len(candidates):
            raise ValueError("BGE reranker score count does not match candidates")
        return scores


@dataclass(frozen=True)
class CallableReranker:
    """Small adapter useful for deterministic tests and remote rerank services."""

    scorer: Callable[[str, Sequence[RetrievedUnit]], Sequence[float]]

    def score(
        self, query: str, candidates: Sequence[RetrievedUnit]
    ) -> tuple[float, ...]:
        return tuple(float(value) for value in self.scorer(query, candidates))
