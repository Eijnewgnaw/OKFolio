from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping


@dataclass(frozen=True)
class PairedBootstrapResult:
    pairs: int
    samples: int
    confidence: float
    baseline_mean: float
    candidate_mean: float
    delta: float
    lower: float
    upper: float
    seed: int


def paired_bootstrap_delta(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Percentile CI for the paired mean delta ``candidate - baseline``."""

    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "paired score ids differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    if not baseline:
        raise ValueError("at least one paired score is required")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    ids = sorted(baseline)
    deltas = []
    for question_id in ids:
        left = float(baseline[question_id])
        right = float(candidate[question_id])
        if not (math.isfinite(left) and math.isfinite(right)):
            raise ValueError("paired scores must be finite")
        deltas.append(right - left)

    rng = random.Random(seed)
    count = len(deltas)
    bootstrap_means = sorted(
        fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrapResult(
        pairs=count,
        samples=samples,
        confidence=confidence,
        baseline_mean=fmean(float(baseline[question_id]) for question_id in ids),
        candidate_mean=fmean(float(candidate[question_id]) for question_id in ids),
        delta=fmean(deltas),
        lower=_percentile(bootstrap_means, tail),
        upper=_percentile(bootstrap_means, 1.0 - tail),
        seed=seed,
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction

