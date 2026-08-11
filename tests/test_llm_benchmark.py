import pytest

from kmpro_wiki.evaluation.llm_benchmark import (
    _StreamAccumulator,
    _percentile,
    _sse_payloads,
)


def test_sse_parser_ignores_noise_and_done_marker():
    payloads = list(
        _sse_payloads(
            [
                "event: message",
                'data: {"choices":[{"delta":{"content":"知"}}]}',
                "data: not-json",
                "data: [DONE]",
            ]
        )
    )

    assert payloads == [{"choices": [{"delta": {"content": "知"}}]}]


def test_stream_accumulator_distinguishes_reasoning_from_visible_ttft():
    accumulator = _StreamAccumulator()
    accumulator.observe(
        {"choices": [{"delta": {"reasoning_content": "先分析"}}]}, 120.0
    )
    accumulator.observe(
        {"choices": [{"delta": {"content": "答案"}}]}, 260.0
    )
    accumulator.observe(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
        300.0,
    )

    result = accumulator.finish(520.0)

    assert result.first_event_ms == 120.0
    assert result.first_generation_ms == 120.0
    assert result.first_content_ms == 260.0
    assert result.reasoning_chars == 3
    assert result.content_chars == 2
    assert result.completion_tokens_per_second == pytest.approx(10.0)
    assert result.content_chars_per_second == pytest.approx(2 / 0.26)


def test_percentile_uses_linear_interpolation():
    assert _percentile([10.0, 20.0, 30.0], 0.50) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 0.95) == pytest.approx(29.0)
    assert _percentile([], 0.95) is None
