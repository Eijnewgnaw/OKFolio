"""OpenAI-compatible text generation helpers for reproducible RAG evaluation.

This module is deliberately independent from AgentWiki's compiler client.  It
accepts already-selected retrieval contexts, so retrieval experiments can keep
the context budget and evidence IDs fixed while changing only the representation
under test (for example, fixed chunks versus audited Concepts).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

from .contracts import AnswerPrediction, Citation, EvidenceAtomId


class RAGGenerationError(RuntimeError):
    """Raised when configuration or a compatible generation response is invalid."""


class RAGAnswerContractError(ValueError):
    """Raised when a generated answer violates the frozen JSON contract."""


RAG_ANSWER_SCHEMA = "okfolio.rag-answer.v1"
StructuredResponseFormat = Literal["json_schema", "json_object", "prompt_only"]


def _first(values: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class RAGGenerationConfig:
    """Provider-neutral Chat Completions configuration.

    No endpoint or credential is embedded in source code.  Role-specific RAG
    variables take precedence over the conventional ``OPENAI_*`` variables,
    which makes the same client usable with hosted OpenAI APIs, LM Studio, vLLM,
    and other OpenAI-compatible runtimes.
    """

    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 180.0
    max_output_tokens: int = 1024
    temperature: float = 0.0
    include_stream_usage: bool = True
    structured_response_format: StructuredResponseFormat = "json_schema"

    def __post_init__(self) -> None:
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        if not base_url:
            raise ValueError(
                "RAG generation base URL is required; set RAG_LLM_BASE_URL "
                "or OPENAI_BASE_URL"
            )
        if not model:
            raise ValueError(
                "RAG generation model is required; set RAG_LLM_MODEL or OPENAI_MODEL"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.structured_response_format not in {
            "json_schema",
            "json_object",
            "prompt_only",
        }:
            raise ValueError(
                "structured_response_format must be json_schema, json_object, "
                "or prompt_only"
            )
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "RAGGenerationConfig":
        values = os.environ if environ is None else environ
        return cls(
            base_url=_first(values, "RAG_LLM_BASE_URL", "OPENAI_BASE_URL"),
            model=_first(values, "RAG_LLM_MODEL", "OPENAI_MODEL"),
            api_key=_first(values, "RAG_LLM_API_KEY", "OPENAI_API_KEY"),
            timeout_seconds=float(
                _first(values, "RAG_LLM_TIMEOUT_SECONDS") or "180"
            ),
            max_output_tokens=int(
                _first(values, "RAG_LLM_MAX_OUTPUT_TOKENS") or "1024"
            ),
            temperature=float(_first(values, "RAG_LLM_TEMPERATURE") or "0"),
            include_stream_usage=(
                _first(values, "RAG_LLM_INCLUDE_STREAM_USAGE").lower()
                not in {"0", "false", "no", "off"}
            ),
            structured_response_format=(
                _first(values, "RAG_LLM_RESPONSE_FORMAT") or "json_schema"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AnswerContext:
    """One retrieved unit supplied to the answer generator.

    ``context_id`` is the stable retrieval-unit identifier used in generated
    citations. ``evidence_ids`` keep the representation-independent evidence
    atoms available to downstream evaluation.
    """

    context_id: str
    text: str
    title: str = ""
    source_id: str = ""
    page_numbers: tuple[int, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.context_id.strip():
            raise ValueError("context_id must not be empty")
        if not self.text.strip():
            raise ValueError("context text must not be empty")
        if any(page < 1 for page in self.page_numbers):
            raise ValueError("page numbers must be positive")


@dataclass(frozen=True)
class AnswerGenerationInput:
    """Stable input contract shared by every RAG experiment arm."""

    question: str
    contexts: tuple[AnswerContext, ...]
    language: str = "zh-CN"
    require_citations: bool = True
    refuse_when_unsupported: bool = True
    additional_instructions: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.contexts:
            raise ValueError("at least one answer context is required")
        context_ids = [item.context_id for item in self.contexts]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("answer context IDs must be unique")


@dataclass(frozen=True)
class StructuredCitation:
    """A model citation resolved to canonical evidence atoms by deterministic code."""

    citation_id: str
    context_id: str
    page: int
    evidence_atom_ids: tuple[str, ...]


@dataclass(frozen=True)
class AtomicClaimCandidate:
    """A claim proposed by the answer model, not a correctness judgment."""

    claim_id: str
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class StructuredAnswer:
    """Strict provider-neutral answer envelope.

    Atomic claim candidates are intentionally not aligned to Gold fact IDs.
    That semantic decision remains the responsibility of a human reviewer or
    an independently calibrated judge.
    """

    answer: str
    refusal: bool
    refusal_reason: str
    citations: tuple[StructuredCitation, ...]
    atomic_claim_candidates: tuple[AtomicClaimCandidate, ...]
    schema: str = RAG_ANSWER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationTiming:
    """Client-observed generation timing in milliseconds.

    ``ttft_ms`` is the first user-visible content latency.  Reasoning-capable
    runtimes can emit hidden reasoning before visible content, so
    ``first_generation_ms`` is retained separately.
    """

    first_event_ms: float | None
    first_generation_ms: float | None
    ttft_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class GenerationResult:
    text: str
    reasoning: str
    finish_reason: str | None
    usage: TokenUsage
    timing: GenerationTiming
    stream_events: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structured_answer_json_schema() -> dict[str, Any]:
    """Return the exact schema sent to compatible providers.

    The schema deliberately avoids provider-specific extensions and keywords
    such as ``uniqueItems`` that are not implemented by every local runtime.
    Duplicate identifiers are rejected by :func:`parse_structured_answer`.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "answer",
            "refusal",
            "refusal_reason",
            "citations",
            "atomic_claim_candidates",
        ],
        "properties": {
            "schema": {"type": "string", "const": RAG_ANSWER_SCHEMA},
            "answer": {"type": "string", "minLength": 1},
            "refusal": {"type": "boolean"},
            "refusal_reason": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["citation_id", "context_id", "page"],
                    "properties": {
                        "citation_id": {"type": "string", "minLength": 1},
                        "context_id": {"type": "string", "minLength": 1},
                        "page": {"type": "integer", "minimum": 1},
                    },
                },
            },
            "atomic_claim_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "text", "citation_ids"],
                    "properties": {
                        "claim_id": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1},
                        "citation_ids": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        },
    }


def structured_answer_response_format(
    mode: StructuredResponseFormat,
) -> dict[str, Any] | None:
    if mode == "prompt_only":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "okfolio_rag_answer_v1",
            "strict": True,
            "schema": structured_answer_json_schema(),
        },
    }


def _exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, location: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise RAGAnswerContractError(
            f"{location} must contain exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _non_empty_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RAGAnswerContractError(f"{location} must be a non-empty string")
    return value.strip()


def _object_list(value: Any, *, location: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise RAGAnswerContractError(f"{location} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise RAGAnswerContractError(f"{location} entries must be objects")
    return value


def parse_structured_answer(
    raw: str, request: AnswerGenerationInput
) -> StructuredAnswer:
    """Parse and deterministically map one model answer to evidence atoms.

    This function never reads Gold facts and never decides whether a natural-
    language claim is correct. It validates the wire contract and maps only a
    cited ``context_id + page`` pair to atoms already present in that context.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise RAGAnswerContractError("generated answer must be non-empty JSON")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RAGAnswerContractError("generated answer must be one JSON object") from error
    if not isinstance(payload, Mapping):
        raise RAGAnswerContractError("generated answer must be one JSON object")
    _exact_keys(
        payload,
        {
            "schema",
            "answer",
            "refusal",
            "refusal_reason",
            "citations",
            "atomic_claim_candidates",
        },
        location="answer",
    )
    if payload["schema"] != RAG_ANSWER_SCHEMA:
        raise RAGAnswerContractError(
            f"answer.schema must equal {RAG_ANSWER_SCHEMA!r}"
        )
    answer_text = _non_empty_string(payload["answer"], location="answer.answer")
    refusal = payload["refusal"]
    if not isinstance(refusal, bool):
        raise RAGAnswerContractError("answer.refusal must be a boolean")
    refusal_reason_value = payload["refusal_reason"]
    if not isinstance(refusal_reason_value, str):
        raise RAGAnswerContractError("answer.refusal_reason must be a string")
    refusal_reason = refusal_reason_value.strip()

    contexts = {item.context_id: item for item in request.contexts}
    parsed_citations: list[StructuredCitation] = []
    citation_ids: set[str] = set()
    context_pages: set[tuple[str, int]] = set()
    for index, item in enumerate(
        _object_list(payload["citations"], location="answer.citations")
    ):
        location = f"answer.citations[{index}]"
        _exact_keys(
            item, {"citation_id", "context_id", "page"}, location=location
        )
        citation_id = _non_empty_string(
            item["citation_id"], location=f"{location}.citation_id"
        )
        context_id = _non_empty_string(
            item["context_id"], location=f"{location}.context_id"
        )
        page = item["page"]
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise RAGAnswerContractError(f"{location}.page must be a positive integer")
        if citation_id in citation_ids:
            raise RAGAnswerContractError(f"duplicate citation_id: {citation_id}")
        citation_ids.add(citation_id)
        if (context_id, page) in context_pages:
            raise RAGAnswerContractError(
                f"duplicate context/page citation: {context_id}, p.{page}"
            )
        context_pages.add((context_id, page))
        context = contexts.get(context_id)
        if context is None:
            raise RAGAnswerContractError(
                f"{location} references unknown context: {context_id}"
            )
        if page not in context.page_numbers:
            raise RAGAnswerContractError(
                f"{location} page {page} is not a supplied context page for {context_id}"
            )
        evidence_atoms: list[str] = []
        for evidence_id in context.evidence_ids:
            try:
                atom = EvidenceAtomId.parse(evidence_id)
            except (TypeError, ValueError) as error:
                raise RAGAnswerContractError(
                    f"context {context_id} contains an invalid evidence atom id"
                ) from error
            if atom.page == page:
                evidence_atoms.append(atom.canonical)
        if not evidence_atoms:
            raise RAGAnswerContractError(
                f"{location} cannot map {context_id}, p.{page} to an evidence atom"
            )
        parsed_citations.append(
            StructuredCitation(
                citation_id=citation_id,
                context_id=context_id,
                page=page,
                evidence_atom_ids=tuple(dict.fromkeys(evidence_atoms)),
            )
        )

    claims: list[AtomicClaimCandidate] = []
    claim_ids: set[str] = set()
    used_citations: set[str] = set()
    for index, item in enumerate(
        _object_list(
            payload["atomic_claim_candidates"],
            location="answer.atomic_claim_candidates",
        )
    ):
        location = f"answer.atomic_claim_candidates[{index}]"
        _exact_keys(item, {"claim_id", "text", "citation_ids"}, location=location)
        claim_id = _non_empty_string(item["claim_id"], location=f"{location}.claim_id")
        claim_text = _non_empty_string(item["text"], location=f"{location}.text")
        if claim_id in claim_ids:
            raise RAGAnswerContractError(f"duplicate claim_id: {claim_id}")
        claim_ids.add(claim_id)
        raw_ids = item["citation_ids"]
        if not isinstance(raw_ids, list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_ids
        ):
            raise RAGAnswerContractError(
                f"{location}.citation_ids must be an array of non-empty strings"
            )
        claim_citations = tuple(value.strip() for value in raw_ids)
        if len(claim_citations) != len(set(claim_citations)):
            raise RAGAnswerContractError(f"{location} contains duplicate citation IDs")
        unknown = set(claim_citations) - citation_ids
        if unknown:
            raise RAGAnswerContractError(
                f"{location} references unknown citation IDs: {sorted(unknown)}"
            )
        if request.require_citations and not claim_citations:
            raise RAGAnswerContractError(f"{location} must cite supporting evidence")
        used_citations.update(claim_citations)
        claims.append(
            AtomicClaimCandidate(
                claim_id=claim_id,
                text=claim_text,
                citation_ids=claim_citations,
            )
        )

    if refusal:
        if not refusal_reason:
            raise RAGAnswerContractError("a refusal requires refusal_reason")
        if parsed_citations or claims:
            raise RAGAnswerContractError(
                "a refusal must not contain citations or atomic claim candidates"
            )
    else:
        if refusal_reason:
            raise RAGAnswerContractError(
                "a non-refusal must use an empty refusal_reason"
            )
        if not claims:
            raise RAGAnswerContractError(
                "a non-refusal requires atomic claim candidates"
            )
        if request.require_citations and not parsed_citations:
            raise RAGAnswerContractError("a non-refusal requires citations")
        unused = citation_ids - used_citations
        if unused:
            raise RAGAnswerContractError(
                f"citations not used by any atomic claim candidate: {sorted(unused)}"
            )

    return StructuredAnswer(
        answer=answer_text,
        refusal=refusal,
        refusal_reason=refusal_reason,
        citations=tuple(parsed_citations),
        atomic_claim_candidates=tuple(claims),
    )


def structured_answer_prediction(
    question_id: str, answer: StructuredAnswer
) -> AnswerPrediction:
    """Project contract fields to a provisional scorer input.

    Only answerability and deterministically mapped citations are populated.
    Gold fact IDs remain empty until a separate semantic reviewer supplies
    them.
    """

    if not question_id.strip():
        raise ValueError("question_id must be non-empty")
    atoms = tuple(
        EvidenceAtomId.parse(value)
        for citation in answer.citations
        for value in citation.evidence_atom_ids
    )
    atoms = tuple(dict.fromkeys(atoms))
    return AnswerPrediction(
        question_id=question_id,
        predicted_answerable=not answer.refusal,
        citations=tuple(Citation(atom) for atom in atoms),
    )


def render_hyde_messages(query: str, *, language: str = "zh-CN") -> list[dict[str, str]]:
    """Render a hypothetical-document request without claiming it is evidence."""

    if not query.strip():
        raise ValueError("HyDE query must not be empty")
    return [
        {
            "role": "system",
            "content": (
                "You create a hypothetical passage only for query expansion. "
                "Write a concise passage that a relevant source document might contain. "
                "Do not mention HyDE, retrieval, uncertainty, or citations. Do not treat "
                "the passage as verified evidence. Output only the passage."
            ),
        },
        {
            "role": "user",
            "content": f"Language: {language}\nQuery: {query.strip()}",
        },
    ]


def render_answer_messages(request: AnswerGenerationInput) -> list[dict[str, str]]:
    """Render one frozen, structured-answer prompt for every experiment arm."""

    rendered_contexts: list[str] = []
    for context in request.contexts:
        metadata: list[str] = [f"id={context.context_id}"]
        if context.title:
            metadata.append(f"title={context.title}")
        if context.source_id:
            metadata.append(f"source={context.source_id}")
        if context.page_numbers:
            metadata.append(
                "pages=" + ",".join(str(page) for page in context.page_numbers)
            )
        rendered_contexts.append(
            f"[CONTEXT {'; '.join(metadata)}]\n{context.text.strip()}\n[/CONTEXT]"
        )

    rules = [
        "Answer only from the supplied contexts; never use unsupported facts.",
        f"Write the answer in {request.language}.",
        (
            "Return exactly one JSON object matching schema "
            f"{RAG_ANSWER_SCHEMA}; output no Markdown fences or extra prose."
        ),
        (
            "Split a non-refusal answer into concise atomic_claim_candidates. "
            "These are claim candidates only, not judgments against a reference answer."
        ),
        (
            "Each citation must use a supplied context_id and one of that context's "
            "listed page numbers. Give citations stable IDs such as cite-1, then "
            "refer to those IDs from atomic_claim_candidates."
        ),
    ]
    if request.refuse_when_unsupported:
        rules.append(
            "If the contexts do not support an answer, explicitly say that the "
            "provided material is insufficient."
        )
    if request.require_citations:
        rules.append(
            "Every non-refusal atomic claim candidate must cite at least one citation."
        )
    if request.additional_instructions.strip():
        rules.append(request.additional_instructions.strip())

    context_block = "\n\n".join(rendered_contexts)
    return [
        {
            "role": "system",
            "content": (
                "\n".join(f"- {rule}" for rule in rules)
                + "\n\nJSON Schema:\n"
                + json.dumps(
                    structured_answer_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{request.question.strip()}\n\n"
                f"Contexts:\n{context_block}"
            ),
        },
    ]


def _usage(payload: Mapping[str, Any]) -> TokenUsage:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return TokenUsage()

    def optional_int(name: str) -> int | None:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return TokenUsage(
        prompt_tokens=optional_int("prompt_tokens"),
        completion_tokens=optional_int("completion_tokens"),
        total_tokens=optional_int("total_tokens"),
    )


class OpenAICompatibleRAGClient:
    """Small Chat Completions client with optional streaming measurements."""

    def __init__(
        self,
        config: RAGGenerationConfig,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._client = client
        self._clock = clock

    def generate_hyde(
        self, query: str, *, language: str = "zh-CN", stream: bool = False
    ) -> GenerationResult:
        return self.complete(
            render_hyde_messages(query, language=language), stream=stream
        )

    def generate_answer(
        self, request: AnswerGenerationInput, *, stream: bool = True
    ) -> GenerationResult:
        return self.complete(
            render_answer_messages(request),
            stream=stream,
            response_format=structured_answer_response_format(
                self.config.structured_response_format
            ),
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        stream: bool,
        response_format: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        if not messages:
            raise ValueError("messages must not be empty")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "stream": stream,
        }
        if stream and self.config.include_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        if self._client is not None:
            return self._request(self._client, payload, stream=stream)
        with httpx.Client(
            timeout=self.config.timeout_seconds, trust_env=False
        ) as client:
            return self._request(client, payload, stream=stream)

    def _request(
        self, client: httpx.Client, payload: dict[str, Any], *, stream: bool
    ) -> GenerationResult:
        url = f"{self.config.base_url}/chat/completions"
        headers = (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )
        if stream:
            return self._request_stream(client, url, headers, payload)
        started = self._clock()
        try:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RAGGenerationError(
                f"generation request failed: {type(error).__name__}"
            ) from error
        total_ms = (self._clock() - started) * 1000.0
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(text, str) or not isinstance(reasoning, str):
            raise RAGGenerationError("generation response contains invalid text fields")
        if not text.strip():
            raise RAGGenerationError("generation response contains no visible content")
        return GenerationResult(
            text=text,
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason"),
            usage=_usage(data),
            timing=GenerationTiming(
                first_event_ms=None,
                first_generation_ms=None,
                ttft_ms=None,
                total_ms=total_ms,
            ),
            stream_events=0,
        )

    def _request_stream(
        self,
        client: httpx.Client,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
    ) -> GenerationResult:
        started = self._clock()
        first_event_ms: float | None = None
        first_generation_ms: float | None = None
        ttft_ms: float | None = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = TokenUsage()
        finish_reason: str | None = None
        stream_events = 0
        try:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    encoded = line[5:].strip()
                    if not encoded or encoded == "[DONE]":
                        continue
                    try:
                        data = json.loads(encoded)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    elapsed_ms = (self._clock() - started) * 1000.0
                    stream_events += 1
                    if first_event_ms is None:
                        first_event_ms = elapsed_ms
                    usage_candidate = _usage(data)
                    if usage_candidate != TokenUsage():
                        usage = usage_candidate
                    choices = data.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        if first_generation_ms is None:
                            first_generation_ms = elapsed_ms
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                        if first_generation_ms is None:
                            first_generation_ms = elapsed_ms
                        if ttft_ms is None:
                            ttft_ms = elapsed_ms
        except httpx.HTTPError as error:
            raise RAGGenerationError(
                f"streaming generation failed: {type(error).__name__}"
            ) from error
        total_ms = (self._clock() - started) * 1000.0
        text = "".join(text_parts)
        if not text.strip():
            raise RAGGenerationError("streaming generation produced no visible content")
        return GenerationResult(
            text=text,
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            timing=GenerationTiming(
                first_event_ms=first_event_ms,
                first_generation_ms=first_generation_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
            ),
            stream_events=stream_events,
        )


__all__ = [
    "AtomicClaimCandidate",
    "AnswerContext",
    "AnswerGenerationInput",
    "GenerationResult",
    "GenerationTiming",
    "OpenAICompatibleRAGClient",
    "RAGGenerationConfig",
    "RAGAnswerContractError",
    "RAGGenerationError",
    "RAG_ANSWER_SCHEMA",
    "StructuredAnswer",
    "StructuredCitation",
    "TokenUsage",
    "parse_structured_answer",
    "render_answer_messages",
    "render_hyde_messages",
    "structured_answer_json_schema",
    "structured_answer_prediction",
    "structured_answer_response_format",
]
