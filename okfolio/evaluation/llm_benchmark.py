from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class StreamMeasurement:
    """Client-observed timing for one Chat Completions stream.

    ``first_generation_ms`` includes either visible content or model reasoning.
    ``first_content_ms`` is the user-visible first-token latency. Keeping both
    avoids hiding the latency of reasoning models behind an ambiguous TTFT.
    """

    first_event_ms: float | None
    first_generation_ms: float | None
    first_content_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    content_chars: int
    reasoning_chars: int
    stream_events: int
    completion_tokens_per_second: float | None
    content_chars_per_second: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _StreamAccumulator:
    def __init__(self) -> None:
        self.first_event_ms: float | None = None
        self.first_generation_ms: float | None = None
        self.first_content_ms: float | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.content_chars = 0
        self.reasoning_chars = 0
        self.stream_events = 0

    def observe(self, payload: dict[str, Any], elapsed_ms: float) -> None:
        self.stream_events += 1
        if self.first_event_ms is None:
            self.first_event_ms = elapsed_ms

        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(content, str) and content:
                    self.content_chars += len(content)
                    if self.first_content_ms is None:
                        self.first_content_ms = elapsed_ms
                    if self.first_generation_ms is None:
                        self.first_generation_ms = elapsed_ms
                if isinstance(reasoning, str) and reasoning:
                    self.reasoning_chars += len(reasoning)
                    if self.first_generation_ms is None:
                        self.first_generation_ms = elapsed_ms

        usage = payload.get("usage") or {}
        if isinstance(usage, dict):
            self.prompt_tokens = _optional_int(
                usage.get("prompt_tokens"), self.prompt_tokens
            )
            self.completion_tokens = _optional_int(
                usage.get("completion_tokens"), self.completion_tokens
            )
            self.total_tokens = _optional_int(
                usage.get("total_tokens"), self.total_tokens
            )

    def finish(self, total_ms: float) -> StreamMeasurement:
        generation_start = self.first_generation_ms
        generation_seconds = (
            max((total_ms - generation_start) / 1000.0, 1e-9)
            if generation_start is not None
            else None
        )
        token_rate = (
            self.completion_tokens / generation_seconds
            if self.completion_tokens is not None and generation_seconds is not None
            else None
        )
        content_seconds = (
            max((total_ms - self.first_content_ms) / 1000.0, 1e-9)
            if self.first_content_ms is not None
            else None
        )
        char_rate = (
            self.content_chars / content_seconds
            if content_seconds is not None
            else None
        )
        return StreamMeasurement(
            first_event_ms=self.first_event_ms,
            first_generation_ms=self.first_generation_ms,
            first_content_ms=self.first_content_ms,
            total_ms=total_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            content_chars=self.content_chars,
            reasoning_chars=self.reasoning_chars,
            stream_events=self.stream_events,
            completion_tokens_per_second=token_rate,
            content_chars_per_second=char_rate,
        )


def _optional_int(value: Any, previous: int | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else previous


def _sse_payloads(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def measure_chat_stream(
    client: httpx.Client,
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    include_usage: bool,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamMeasurement:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = clock()
    accumulator = _StreamAccumulator()
    with client.stream(
        "POST",
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
    ) as response:
        response.raise_for_status()
        for item in _sse_payloads(response.iter_lines()):
            accumulator.observe(item, (clock() - started) * 1000.0)
    return accumulator.finish((clock() - started) * 1000.0)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _series_summary(
    measurements: list[StreamMeasurement], field: str
) -> dict[str, float | None]:
    values = [
        float(value)
        for item in measurements
        if (value := getattr(item, field)) is not None
    ]
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def run_stream_benchmark(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    requests: int = 3,
    warmup: int = 1,
    max_tokens: int = 128,
    timeout: float = 180.0,
    include_usage: bool = False,
) -> dict[str, Any]:
    if requests < 1:
        raise ValueError("requests must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    measurements: list[StreamMeasurement] = []
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for _ in range(warmup):
            measure_chat_stream(
                client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                include_usage=include_usage,
            )
        for _ in range(requests):
            measurements.append(
                measure_chat_stream(
                    client,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    include_usage=include_usage,
                )
            )

    return {
        "schema": "okfolio.openai-stream-benchmark.v1",
        "model": model,
        "warmup_requests": warmup,
        "measured_requests": requests,
        "include_usage": include_usage,
        "measurements": [item.to_dict() for item in measurements],
        "summary": {
            field: _series_summary(measurements, field)
            for field in (
                "first_event_ms",
                "first_generation_ms",
                "first_content_ms",
                "total_ms",
                "completion_tokens_per_second",
                "content_chars_per_second",
            )
        },
    }
