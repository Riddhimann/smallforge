"""Run scoring and paired ablation comparisons."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from statistics import fmean
from typing import Any

from .contracts import (
    AblationComparison,
    CandidateResult,
    EvaluationRun,
    MetricDelta,
    MetricSummary,
    TaskScore,
    VariantComparison,
)
from .loader import BenchmarkSuite
from .metrics import score_task
from .statistics import bootstrap_mean_interval, paired_bootstrap_difference, wilson_interval

METRIC_FIELDS = (
    "task_accuracy",
    "schema_validity",
    "tool_selection_accuracy",
    "tool_argument_accuracy",
    "groundedness",
    "citation_precision",
    "citation_recall",
    "abstention_accuracy",
    "verification_accuracy",
    "retry_count",
    "latency_ms",
)

BINARY_METRICS = {
    "task_accuracy",
    "schema_validity",
    "abstention_accuracy",
    "verification_accuracy",
}


def aggregate_scores(
    scores: Sequence[TaskScore],
    *,
    confidence: float = 0.95,
    bootstrap_iterations: int = 4000,
    seed: int = 1729,
) -> dict[str, MetricSummary]:
    """Macro-average task metrics, excluding non-applicable ``None`` values."""

    if not scores:
        raise ValueError("Cannot aggregate an empty score sequence")
    summaries: dict[str, MetricSummary] = {}
    for offset, metric in enumerate(METRIC_FIELDS):
        values = [float(value) for score in scores if (value := getattr(score, metric)) is not None]
        if not values:
            continue
        mean = fmean(values)
        if metric in BINARY_METRICS and all(value in {0.0, 1.0} for value in values):
            low, high = wilson_interval(sum(values), len(values), confidence)
            method = "Wilson score"
        else:
            low, high = bootstrap_mean_interval(
                values,
                confidence,
                iterations=bootstrap_iterations,
                seed=seed + offset,
            )
            method = "bootstrap percentile"
        summaries[metric] = MetricSummary(
            metric=metric,
            value=mean,
            sample_size=len(values),
            ci_low=low,
            ci_high=high,
            confidence=confidence,
            interval_method=method,
        )
    return summaries


def _coerce_candidate(task_id: str, value: Any) -> CandidateResult:
    if isinstance(value, CandidateResult):
        return value
    if not isinstance(value, Mapping):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        else:
            raise ValueError("Candidate output must be a mapping or CandidateResult")
    data = dict(value)
    data.setdefault("task_id", task_id)
    return CandidateResult.from_mapping(data)


def evaluate_predictions(
    suite: BenchmarkSuite,
    predictions: Iterable[Mapping[str, Any] | CandidateResult] | Mapping[str, Any],
    *,
    system_name: str,
    variant: str,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    synthetic: bool = False,
    strict: bool = True,
    confidence: float = 0.95,
    bootstrap_iterations: int = 4000,
) -> EvaluationRun:
    """Score previously captured predictions against a benchmark suite.

    ``predictions`` may be a task-id keyed mapping or an iterable of result
    mappings. Missing tasks are represented as failed outputs when ``strict`` is
    false; unexpected IDs always raise to prevent accidental cross-suite scoring.
    """

    if isinstance(predictions, Mapping):
        normalized: dict[str, CandidateResult] = {
            str(task_id): _coerce_candidate(str(task_id), value)
            for task_id, value in predictions.items()
        }
    else:
        normalized = {}
        for raw in predictions:
            if isinstance(raw, CandidateResult):
                result = raw
            elif isinstance(raw, Mapping):
                result = CandidateResult.from_mapping(raw)
            else:
                raise ValueError("Every prediction must be a mapping or CandidateResult")
            if result.task_id in normalized:
                raise ValueError(f"Duplicate prediction for task {result.task_id!r}")
            normalized[result.task_id] = result
    task_ids = {task.id for task in suite.tasks}
    unexpected = sorted(set(normalized) - task_ids)
    if unexpected:
        raise ValueError("Predictions contain unknown task IDs: " + ", ".join(unexpected))
    missing = sorted(task_ids - set(normalized))
    if missing and strict:
        raise ValueError("Predictions are missing task IDs: " + ", ".join(missing))
    scores: list[TaskScore] = []
    for task in suite.tasks:
        result = normalized.get(task.id)
        if result is None:
            result = CandidateResult(task_id=task.id, error="No prediction was provided")
        scores.append(score_task(task, result))
    score_tuple = tuple(scores)
    run_metadata = {"benchmark_data_synthetic": suite.synthetic, **dict(metadata or {})}
    return EvaluationRun(
        run_id=run_id or str(uuid.uuid4()),
        benchmark_name=suite.name,
        benchmark_version=suite.version,
        system_name=system_name,
        variant=variant,
        scores=score_tuple,
        aggregates=aggregate_scores(
            score_tuple,
            confidence=confidence,
            bootstrap_iterations=bootstrap_iterations,
        ),
        metadata=run_metadata,
        synthetic=synthetic,
    )


def evaluate_callable(
    suite: BenchmarkSuite,
    candidate: Callable[[Mapping[str, Any]], Mapping[str, Any] | CandidateResult],
    *,
    system_name: str,
    variant: str,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    synthetic: bool = False,
    confidence: float = 0.95,
    bootstrap_iterations: int = 4000,
) -> EvaluationRun:
    """Execute a synchronous candidate once per task and capture wall latency."""

    predictions: list[CandidateResult] = []
    for task in suite.tasks:
        started = time.perf_counter()
        try:
            result = _coerce_candidate(task.id, candidate(task.to_payload()))
        except Exception as exc:  # benchmark runners must retain per-task failures
            result = CandidateResult(task_id=task.id, error=f"{type(exc).__name__}: {exc}")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.latency_ms <= 0.0:
            result = replace(result, latency_ms=elapsed_ms)
        predictions.append(result)
    return evaluate_predictions(
        suite,
        predictions,
        system_name=system_name,
        variant=variant,
        run_id=run_id,
        metadata=metadata,
        synthetic=synthetic,
        confidence=confidence,
        bootstrap_iterations=bootstrap_iterations,
    )


def compare_runs(
    baseline: EvaluationRun,
    candidates: EvaluationRun | Sequence[EvaluationRun],
    *,
    metrics: Sequence[str] = METRIC_FIELDS,
    confidence: float = 0.95,
    bootstrap_iterations: int = 4000,
    seed: int = 1729,
    notes: Sequence[str] = (),
) -> AblationComparison:
    """Create paired candidate-minus-baseline comparisons on common task IDs."""

    candidate_runs = (candidates,) if isinstance(candidates, EvaluationRun) else tuple(candidates)
    if not candidate_runs:
        raise ValueError("At least one candidate run is required")
    baseline_by_id = {score.task_id: score for score in baseline.scores}
    comparisons: list[VariantComparison] = []
    for run_index, run in enumerate(candidate_runs):
        if (run.benchmark_name, run.benchmark_version) != (
            baseline.benchmark_name,
            baseline.benchmark_version,
        ):
            raise ValueError("All runs must use the same benchmark name and version")
        candidate_by_id = {score.task_id: score for score in run.scores}
        common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
        if not common_ids:
            raise ValueError(f"Variant {run.variant!r} has no tasks in common with the baseline")
        deltas: dict[str, MetricDelta] = {}
        for metric_index, metric in enumerate(metrics):
            if metric not in METRIC_FIELDS:
                raise ValueError(f"Unknown comparison metric {metric!r}")
            pairs = [
                (
                    getattr(baseline_by_id[task_id], metric),
                    getattr(candidate_by_id[task_id], metric),
                )
                for task_id in common_ids
            ]
            applicable = [
                (float(old), float(new))
                for old, new in pairs
                if old is not None and new is not None
            ]
            if not applicable:
                continue
            old_values, new_values = zip(*applicable, strict=False)
            old_mean, new_mean = fmean(old_values), fmean(new_values)
            difference = new_mean - old_mean
            low, high = paired_bootstrap_difference(
                old_values,
                new_values,
                confidence,
                iterations=bootstrap_iterations,
                seed=seed + run_index * len(metrics) + metric_index,
            )
            relative = None if old_mean == 0.0 else difference / abs(old_mean) * 100.0
            deltas[metric] = MetricDelta(
                metric=metric,
                baseline_value=old_mean,
                candidate_value=new_mean,
                absolute_delta=difference,
                relative_delta_percent=relative,
                sample_size=len(applicable),
                ci_low=low,
                ci_high=high,
                confidence=confidence,
            )
        comparisons.append(
            VariantComparison(
                variant=run.variant,
                common_task_count=len(common_ids),
                deltas=deltas,
            )
        )
    return AblationComparison(
        benchmark_name=baseline.benchmark_name,
        benchmark_version=baseline.benchmark_version,
        baseline_variant=baseline.variant,
        comparisons=tuple(comparisons),
        synthetic=baseline.synthetic or any(run.synthetic for run in candidate_runs),
        notes=tuple(notes),
    )
