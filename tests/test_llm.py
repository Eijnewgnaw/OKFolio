import json

import httpx
import pytest

from kmpro_wiki.agentwiki.llm import (
    LLMClient,
    LLMError,
    LLMOutputTruncated,
    render_compile_prompt,
    render_enrich_prompt,
)


def test_compile_prompt_replaces_inputs_but_preserves_slug_instruction():
    template = "{title}|{source_file}|{content}|{slug}"

    rendered = render_compile_prompt(template, "标题", "a.md", "正文")

    assert rendered == "标题|a.md|正文|{slug}"


def test_enrich_prompt_replaces_content_and_index():
    assert render_enrich_prompt("{content}\n{index}", "概念", "索引") == "概念\n索引"


def test_client_reads_chat_completion_content(httpx_mock):
    httpx_mock.add_response(
        url="http://llm/v1/chat/completions",
        json={"choices": [{"message": {"content": "RESULT"}}]},
    )
    client = LLMClient("http://llm/v1/", "secret", "model", retry_delay=0)

    assert client.complete("prompt") == "RESULT"
    request = httpx_mock.get_request()
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.read().decode("utf-8").find('"temperature":0') >= 0


def test_client_omits_authorization_for_authless_local_runtime(httpx_mock):
    httpx_mock.add_response(
        url="http://llm/v1/chat/completions",
        json={"choices": [{"message": {"content": "RESULT"}}]},
    )
    client = LLMClient("http://llm/v1", "", "model", retry_delay=0)

    assert client.complete("prompt") == "RESULT"
    assert "Authorization" not in httpx_mock.get_request().headers


def test_client_sends_json_object_with_schema_in_prompt(httpx_mock):
    httpx_mock.add_response(
        url="http://llm/v1/chat/completions",
        json={"choices": [{"message": {"content": '{"ok":true}'}}]},
    )
    client = LLMClient("http://llm/v1", "secret", "model", retry_delay=0)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    assert client.complete(
        "prompt", json_schema_name="ok_response", json_schema=schema
    ) == '{"ok":true}'
    payload = json.loads(httpx_mock.get_request().content)
    assert payload["response_format"] == {"type": "json_object"}
    assert "强制结构化输出约束" in payload["messages"][0]["content"]
    assert '"required":["ok"]' in payload["messages"][0]["content"]


def test_client_can_use_strict_json_schema_response_format(httpx_mock):
    httpx_mock.add_response(
        url="http://llm/v1/chat/completions",
        json={"choices": [{"message": {"content": '{"ok":true}'}}]},
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    client = LLMClient(
        "http://llm/v1",
        "secret",
        "model",
        retry_delay=0,
        response_format="json_schema",
    )

    assert client.complete(
        "prompt", json_schema_name="ok_response", json_schema=schema
    ) == '{"ok":true}'
    payload = json.loads(httpx_mock.get_request().content)
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "ok_response", "strict": True, "schema": schema},
    }


def test_client_retries_server_error(httpx_mock):
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": "recovered"}}]}
    )
    client = LLMClient(
        "http://llm/v1", "secret", "model", max_attempts=2, retry_delay=0
    )

    assert client.complete("prompt") == "recovered"
    assert len(httpx_mock.get_requests()) == 2


def test_client_rejects_empty_content(httpx_mock):
    httpx_mock.add_response(json={"choices": [{"message": {"content": ""}}]})
    client = LLMClient("http://llm/v1", "secret", "model", retry_delay=0)

    with pytest.raises(LLMError, match="empty"):
        client.complete("prompt")


def test_client_reports_timeout_retry_without_sensitive_content(httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("slow model"))
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": "recovered"}}]}
    )
    events: list[str] = []
    client = LLMClient(
        "http://llm/v1",
        "secret-key",
        "model",
        max_attempts=2,
        retry_delay=0,
        on_event=events.append,
    )

    assert client.complete("sensitive prompt") == "recovered"
    assert events[0] == "llm.retry attempt=1/2 reason=ReadTimeout"
    assert "secret-key" not in "\n".join(events)
    assert "sensitive prompt" not in "\n".join(events)


def test_client_accepts_compat_reasoning_content_without_vendor_fields(httpx_mock):
    httpx_mock.add_response(
        url="http://llm/v1/chat/completions",
        json={
            "choices": [
                {"message": {"content": None, "reasoning_content": "RESULT"}}
            ]
        },
    )
    client = LLMClient(
        "http://llm/v1",
        "secret",
        "model",
        enable_thinking=False,
        max_tokens=32768,
        retry_delay=0,
    )

    assert client.complete("prompt") == "RESULT"
    request = httpx_mock.get_request()
    payload = json.loads(request.content)
    assert "chat_template_kwargs" not in payload
    assert payload["max_tokens"] == 32768


def test_client_can_send_optional_chat_template_kwargs(httpx_mock):
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": "RESULT"}}]},
    )
    client = LLMClient(
        "http://llm/v1",
        "secret",
        "model",
        retry_delay=0,
        send_chat_template_kwargs=True,
        enable_thinking=False,
    )

    assert client.complete("prompt") == "RESULT"
    payload = json.loads(httpx_mock.get_request().content)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_client_reports_reasoning_and_usage_metrics(httpx_mock, monkeypatch):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "reasoning_content": "RESULT"},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }
    )
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr("kmpro_wiki.agentwiki.llm.time.monotonic", lambda: next(ticks))
    events: list[str] = []
    client = LLMClient(
        "http://llm/v1",
        "secret",
        "model",
        enable_thinking=False,
        on_event=events.append,
        retry_delay=0,
    )

    assert client.complete("提示词") == "RESULT"
    assert events == [
        "llm.done schema=none elapsed_ms=250 thinking=false format=json_object "
        "prompt_chars=3 content_chars=0 reasoning_chars=6 selected=reasoning_content "
        "finish=stop prompt_tokens=12 completion_tokens=3 total_tokens=15"
    ]


@pytest.mark.parametrize(
    ("finish_key", "finish_reason"),
    [
        ("finish_reason", "length"),
        ("finish_reason", "max_tokens"),
        ("finishReason", "maxPredictedTokensReached"),
    ],
)
def test_client_rejects_truncated_output_before_returning_partial_content(
    httpx_mock,
    monkeypatch,
    finish_key,
    finish_reason,
):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    finish_key: finish_reason,
                    "message": {"content": '{"unfinished":'},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8191,
                "total_tokens": 8203,
            },
        }
    )
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(
        "kmpro_wiki.agentwiki.llm.time.monotonic", lambda: next(ticks)
    )
    events: list[str] = []
    client = LLMClient(
        "http://llm/v1",
        "secret",
        "model",
        max_tokens=8192,
        max_attempts=3,
        on_event=events.append,
        retry_delay=0,
    )

    with pytest.raises(LLMOutputTruncated) as raised:
        client.complete("提示词")

    assert raised.value.finish_reason == finish_reason
    assert raised.value.prompt_tokens == 12
    assert raised.value.completion_tokens == 8191
    assert raised.value.total_tokens == 8203
    assert len(httpx_mock.get_requests()) == 1
    assert events == [
        "llm.done schema=none elapsed_ms=250 thinking=false "
        "format=json_object prompt_chars=3 content_chars=14 reasoning_chars=0 "
        f"selected=content finish={finish_reason} prompt_tokens=12 "
        "completion_tokens=8191 total_tokens=8203"
    ]
