from __future__ import annotations

import json

import pytest

from smallforge.evaluation import (
    compare_runs,
    evaluate_callable,
    evaluate_predictions,
    load_benchmark,
    load_run,
    render_ablation_markdown,
    render_markdown_report,
    write_ablation_report,
    write_report,
)


def _synthetic_predictions(correct: bool, latency_ms: float):
    suite = load_benchmark("v1")
    predictions = []
    for task in suite.tasks:
        match = task.expected["match"]
        kind = match["kind"]
        value = match.get("value")
        item = {
            "task_id": task.id,
            "answer": value if correct else "synthetic wrong answer",
            "latency_ms": latency_ms,
            "retry_count": 1 if correct else 0,
        }
        if kind.startswith("json_"):
            item["structured_output"] = value if correct else {"bad": True}
        if kind == "abstain":
            item["abstained"] = correct
        if kind == "verification":
            item["verification"] = value if correct else "pass"
        predictions.append(item)
    return predictions


def test_evaluate_predictions_requires_complete_suite_by_default() -> None:
    suite = load_benchmark("v1")

    with pytest.raises(ValueError, match="missing task IDs"):
        evaluate_predictions(
            suite,
            [{"task_id": "arith_001", "answer": 1090}],
            system_name="synthetic-fixture",
            variant="partial",
            synthetic=True,
            bootstrap_iterations=100,
        )


def test_allow_missing_records_failures_instead_of_hiding_them() -> None:
    suite = load_benchmark("v1")
    run = evaluate_predictions(
        suite,
        [{"task_id": "arith_001", "answer": 1090}],
        system_name="synthetic-fixture",
        variant="partial",
        synthetic=True,
        strict=False,
        bootstrap_iterations=100,
    )

    assert len(run.scores) == 20
    assert sum(score.error is not None for score in run.scores) == 19
    assert run.metadata["benchmark_data_synthetic"] is True


def test_evaluate_callable_captures_exceptions_and_latency() -> None:
    suite = load_benchmark("v1")

    def candidate(payload):
        if payload["task_id"] == "arith_001":
            raise RuntimeError("synthetic failure")
        return {"answer": None}

    run = evaluate_callable(
        suite,
        candidate,
        system_name="synthetic-callable",
        variant="exception-demo",
        synthetic=True,
        bootstrap_iterations=100,
    )

    first = run.scores[0]
    assert first.error == "RuntimeError: synthetic failure"
    assert first.latency_ms > 0


def test_paired_ablation_reports_improvement_and_cost_tradeoff() -> None:
    suite = load_benchmark("v1")
    baseline = evaluate_predictions(
        suite,
        _synthetic_predictions(False, 100.0),
        system_name="synthetic-fixture",
        variant="standalone-synthetic",
        synthetic=True,
        bootstrap_iterations=100,
    )
    harnessed = evaluate_predictions(
        suite,
        _synthetic_predictions(True, 150.0),
        system_name="synthetic-fixture",
        variant="harnessed-synthetic",
        synthetic=True,
        bootstrap_iterations=100,
    )

    comparison = compare_runs(
        baseline,
        harnessed,
        metrics=("task_accuracy", "latency_ms", "retry_count"),
        bootstrap_iterations=200,
    )

    deltas = comparison.comparisons[0].deltas
    assert deltas["task_accuracy"].absolute_delta > 0
    assert deltas["latency_ms"].absolute_delta == 50.0
    assert deltas["retry_count"].absolute_delta == 1.0
    assert comparison.synthetic is True


def test_json_and_markdown_reports_round_trip_and_label_synthetic_outputs(tmp_path) -> None:
    suite = load_benchmark("v1")
    baseline = evaluate_predictions(
        suite,
        _synthetic_predictions(False, 100.0),
        system_name="synthetic-fixture",
        variant="standalone-synthetic",
        synthetic=True,
        bootstrap_iterations=100,
    )
    harnessed = evaluate_predictions(
        suite,
        _synthetic_predictions(True, 150.0),
        system_name="synthetic-fixture",
        variant="harnessed-synthetic",
        synthetic=True,
        bootstrap_iterations=100,
    )
    comparison = compare_runs(
        baseline,
        harnessed,
        metrics=("task_accuracy", "latency_ms"),
        bootstrap_iterations=100,
    )

    json_path, markdown_path = write_report(baseline, tmp_path, stem="baseline")
    comparison_json, comparison_markdown = write_ablation_report(comparison, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["synthetic"] is True
    assert "Synthetic outputs" in markdown_path.read_text(encoding="utf-8")
    assert load_run(json_path).scores == baseline.scores
    assert json.loads(comparison_json.read_text(encoding="utf-8"))["synthetic"] is True
    assert "Synthetic outputs" in comparison_markdown.read_text(encoding="utf-8")
    assert "candidate outputs in this run are fixtures" in render_markdown_report(baseline).lower()
    assert "confidence interval" in render_ablation_markdown(comparison).lower()
