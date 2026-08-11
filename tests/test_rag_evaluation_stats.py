import pytest

from okfolio.evaluation.stats import paired_bootstrap_delta


def test_paired_bootstrap_is_reproducible_and_preserves_pairing():
    baseline = {"q1": 0.2, "q2": 0.5, "q3": 0.7}
    candidate = {"q1": 0.4, "q2": 0.7, "q3": 0.9}

    result = paired_bootstrap_delta(
        baseline,
        candidate,
        samples=500,
        confidence=0.95,
        seed=42,
    )

    assert result.pairs == 3
    assert result.delta == pytest.approx(0.2)
    assert result.lower == pytest.approx(0.2)
    assert result.upper == pytest.approx(0.2)


def test_paired_bootstrap_rejects_unpaired_or_non_finite_scores():
    with pytest.raises(ValueError, match="ids differ"):
        paired_bootstrap_delta({"q1": 0.0}, {"q2": 1.0})
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_delta({"q1": 0.0}, {"q1": float("nan")})

