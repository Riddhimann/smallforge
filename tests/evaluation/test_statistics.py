from __future__ import annotations

import pytest

from smallforge.evaluation import (
    bootstrap_mean_interval,
    paired_bootstrap_difference,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate_and_stays_bounded() -> None:
    low, high = wilson_interval(8, 10)

    assert 0.0 <= low < 0.8 < high <= 1.0
    assert low == pytest.approx(0.4902, abs=0.001)
    assert high == pytest.approx(0.9433, abs=0.001)


def test_bootstrap_interval_is_seeded_and_contains_mean() -> None:
    first = bootstrap_mean_interval([1, 2, 3, 4], iterations=500, seed=42)
    second = bootstrap_mean_interval([1, 2, 3, 4], iterations=500, seed=42)

    assert first == second
    assert first[0] <= 2.5 <= first[1]


def test_paired_bootstrap_preserves_task_pairing() -> None:
    low, high = paired_bootstrap_difference(
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        iterations=1000,
        seed=7,
    )

    assert low >= 0.0
    assert low <= 0.5 <= high


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_intervals_reject_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 1, confidence)
