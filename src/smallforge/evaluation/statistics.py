"""Small, reproducible statistical helpers used by evaluation reports."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from statistics import NormalDist, fmean


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def wilson_interval(
    successes: float, sample_size: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``successes`` accepts a float so mean binary metrics can be converted back to a
    count without unnecessary rounding. It must still fall within ``[0, n]``.
    """

    _validate_confidence(confidence)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 <= successes <= sample_size:
        raise ValueError("successes must be between 0 and sample_size")
    proportion = successes / sample_size
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / sample_size
    centre = (proportion + z_squared / (2.0 * sample_size)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_size
            + z_squared / (4.0 * sample_size * sample_size)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    location = probability * (len(sorted_values) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return float(sorted_values[lower])
    weight = location - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bootstrap_mean_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    *,
    iterations: int = 4000,
    seed: int = 1729,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval for a sample mean."""

    _validate_confidence(confidence)
    sample = tuple(float(value) for value in values)
    if not sample:
        raise ValueError("At least one value is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if len(sample) == 1:
        return sample[0], sample[0]
    generator = random.Random(seed)
    size = len(sample)
    estimates = sorted(
        fmean(sample[generator.randrange(size)] for _ in range(size)) for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def paired_bootstrap_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    *,
    iterations: int = 4000,
    seed: int = 1729,
) -> tuple[float, float]:
    """CI for the paired mean difference ``candidate - baseline``."""

    _validate_confidence(confidence)
    if len(baseline) != len(candidate):
        raise ValueError("Paired samples must have the same length")
    if not baseline:
        raise ValueError("At least one paired observation is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    differences = tuple(
        float(new) - float(old) for old, new in zip(baseline, candidate, strict=False)
    )
    if len(differences) == 1:
        return differences[0], differences[0]
    generator = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        fmean(differences[generator.randrange(size)] for _ in range(size))
        for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)
