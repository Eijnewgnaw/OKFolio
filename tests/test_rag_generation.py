from __future__ import annotations

import json

import httpx
import pytest

from okfolio.evaluation.generation import (
    AtomicClaimCandidate,
    AnswerContext,
    AnswerGenerationInput,
    OpenAICompatibleRAGClient,
    RAGGenerationConfig,
    StructuredAnswer,
    parse_structured_answer,
    render_answer_messages,
)


def test_config_uses_role_specific_env_and_hides_key_from_repr():
    config = RAGGenerationConfig.from_env(
        {
            "RAG_LLM_BASE_URL": "https://compatible.example/v1/",
            "RAG_LLM_MODEL": "rag-model",
            "RAG_LLM_API_KEY": "secret-value",
            "OPENAI_BASE_URL": "https://ignored.example/v1",
            "OPENAI_MODEL": "ignored-model",
        }
    )

    assert config.base_url == "https://compatible.example/v1"
    assert config.model == "rag-model"
    assert config.api_key == "secret-value"
    assert "secret-value" not in repr(config)


def test_config_requires_explicit_endpoint_and_model():
    with pytest.raises(ValueError, match="base URL"):
        RAGGenerationConfig.from_env({})
    with pytest.raises(ValueError, match="model"):
        RAGGenerationConfig(base_url="https://compatible.example/v1", model="")


def test_answer_contract_renders_stable_context_and_page_citation_rules():
    request = AnswerGenerationInput(
        question="补贴标准是多少？",
        contexts=(
            AnswerContext(
                context_id="unit-7",
                source_id="article-a",
                title="政策报告",
                page_numbers=(12, 13),
                evidence_ids=("article-a:p12:s3",),
                text="补贴标准为符合条件投资额的百分之五十。",
            ),
        ),
    )

    messages = render_answer_messages(request)

    assert messages[0]["role"] == "system"
    assert "okfolio.rag-answer.v1" in messages[0]["content"]
    assert "atomic_claim_candidates" in messages[0]["content"]
    assert "id=unit-7" in messages[1]["content"]
    assert "pages=12,13" in messages[1]["content"]
    assert "article-a:p12:s3" not in messages[1]["content"]


def test_non_streaming_hyde_uses_fake_transport_and_collects_usage():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "假设性相关段落"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 6,
                    "total_tokens": 26,
                },
            },
        )

    clock_values = iter([2.0, 2.125])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OpenAICompatibleRAGClient(
        RAGGenerationConfig(
            base_url="https://compatible.example/v1",
            model="test-model",
            api_key="test-key",
        ),
        client=client,
        clock=lambda: next(clock_values),
    )

    result = generator.generate_hyde("什么是绿色金融？")

    assert result.text == "假设性相关段落"
    assert result.usage.total_tokens == 26
    assert result.timing.total_ms == pytest.approx(125.0)
    assert result.timing.ttft_ms is None
    assert captured["url"] == "https://compatible.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["stream"] is False  # type: ignore[index]
    assert "response_format" not in captured["payload"]  # type: ignore[operator]
    client.close()


def test_streaming_answer_measures_visible_ttft_separately_from_reasoning():
    captured: dict[str, object] = {}
    stream_body = "\n".join(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"分析"}}]}',
            'data: {"choices":[{"delta":{"content":"结论"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            (
                'data: {"choices":[],"usage":{"prompt_tokens":30,'
                '"completion_tokens":8,"total_tokens":38}}'
            ),
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body.encode(),
        )

    clock_values = iter([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OpenAICompatibleRAGClient(
        RAGGenerationConfig(
            base_url="https://compatible.example/v1",
            model="test-model",
        ),
        client=client,
        clock=lambda: next(clock_values),
    )
    request = AnswerGenerationInput(
        question="结论是什么？",
        contexts=(AnswerContext(context_id="unit-1", text="上下文证据。"),),
    )

    result = generator.generate_answer(request)

    assert result.text == "结论"
    assert result.reasoning == "分析"
    assert result.finish_reason == "stop"
    assert result.stream_events == 4
    assert result.usage.total_tokens == 38
    assert result.timing.first_event_ms == pytest.approx(100.0)
    assert result.timing.first_generation_ms == pytest.approx(100.0)
    assert result.timing.ttft_ms == pytest.approx(200.0)
    assert result.timing.total_ms == pytest.approx(500.0)
    assert captured["authorization"] is None
    assert captured["payload"]["stream"] is True  # type: ignore[index]
    assert captured["payload"]["stream_options"] == {  # type: ignore[index]
        "include_usage": True
    }
    assert captured["payload"]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    client.close()


def test_answer_contract_rejects_duplicate_context_ids():
    with pytest.raises(ValueError, match="unique"):
        AnswerGenerationInput(
            question="问题",
            contexts=(
                AnswerContext(context_id="same", text="一"),
                AnswerContext(context_id="same", text="二"),
            ),
        )


def test_structured_answer_contract_maps_context_page_to_canonical_atoms():
    request = AnswerGenerationInput(
        question="补贴标准是多少？",
        contexts=(
            AnswerContext(
                context_id="unit-7",
                text="补贴标准为百分之五十。",
                page_numbers=(12, 13),
                evidence_ids=(
                    "article-a:p012:b3",
                    "article-a:p012:b4",
                    "article-a:p013:b1",
                ),
            ),
        ),
    )
    raw = json.dumps(
        {
            "schema": "okfolio.rag-answer.v1",
            "answer": "补贴标准为百分之五十。",
            "refusal": False,
            "refusal_reason": "",
            "citations": [
                {"citation_id": "cite-1", "context_id": "unit-7", "page": 12}
            ],
            "atomic_claim_candidates": [
                {
                    "claim_id": "claim-1",
                    "text": "补贴标准为百分之五十。",
                    "citation_ids": ["cite-1"],
                }
            ],
        },
        ensure_ascii=False,
    )

    answer = parse_structured_answer(raw, request)

    assert isinstance(answer, StructuredAnswer)
    assert answer.atomic_claim_candidates == (
        AtomicClaimCandidate(
            claim_id="claim-1",
            text="补贴标准为百分之五十。",
            citation_ids=("cite-1",),
        ),
    )
    assert answer.citations[0].evidence_atom_ids == (
        "article-a:p012:b3",
        "article-a:p012:b4",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["citations"][0].update(page=99), "supplied context page"),
        (
            lambda row: row["atomic_claim_candidates"][0].update(
                citation_ids=["missing"]
            ),
            "unknown citation",
        ),
        (lambda row: row.update(extra="not allowed"), "exactly"),
    ],
)
def test_structured_answer_contract_fails_closed(mutate, message):
    request = AnswerGenerationInput(
        question="问题",
        contexts=(
            AnswerContext(
                context_id="context-1",
                text="证据",
                page_numbers=(1,),
                evidence_ids=("article-a:p001:b1",),
            ),
        ),
    )
    row = {
        "schema": "okfolio.rag-answer.v1",
        "answer": "结论。",
        "refusal": False,
        "refusal_reason": "",
        "citations": [
            {"citation_id": "cite-1", "context_id": "context-1", "page": 1}
        ],
        "atomic_claim_candidates": [
            {"claim_id": "claim-1", "text": "结论。", "citation_ids": ["cite-1"]}
        ],
    }
    mutate(row)

    with pytest.raises(ValueError, match=message):
        parse_structured_answer(json.dumps(row, ensure_ascii=False), request)


def test_structured_refusal_has_no_claims_or_citations():
    request = AnswerGenerationInput(
        question="材料没有答案的问题",
        contexts=(
            AnswerContext(
                context_id="context-1",
                text="无关材料",
                page_numbers=(1,),
                evidence_ids=("article-a:p001:b1",),
            ),
        ),
    )
    raw = json.dumps(
        {
            "schema": "okfolio.rag-answer.v1",
            "answer": "现有材料不足以回答该问题。",
            "refusal": True,
            "refusal_reason": "上下文未提供所需事实。",
            "citations": [],
            "atomic_claim_candidates": [],
        },
        ensure_ascii=False,
    )

    parsed = parse_structured_answer(raw, request)

    assert parsed.refusal is True
    assert parsed.citations == ()
    assert parsed.atomic_claim_candidates == ()
