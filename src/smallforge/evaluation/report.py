"""Machine-readable and human-readable SmallForge evaluation reports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from .contracts import (
    AblationComparison,
    EvaluationRun,
    MetricSummary,
    TaskScore,
)

HIGHER_IS_BETTER = {
    "task_accuracy",
    "schema_validity",
    "tool_selection_accuracy",
    "tool_argument_accuracy",
    "groundedness",
    "citation_precision",
    "citation_recall",
    "abstention_accuracy",
    "verification_accuracy",
}
LOWER_IS_BETTER = {"retry_count", "latency_ms"}


def _format_metric(name: str, value: float) -> str:
    if name == "latency_ms":
        return f"{value:,.1f} ms"
    if name == "retry_count":
        return f"{value:.2f}"
    return f"{value * 100.0:.1f}%"


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return normalized or "evaluation"


def render_markdown_report(run: EvaluationRun) -> str:
    """Render a complete evaluation run without implying production validity."""

    lines = [
        f"# SmallForge evaluation: {run.variant}",
        "",
        "> **Scope:** This benchmark uses synthetic task data. Scores measure this versioned",
        "> suite only and do not establish real-world, production, safety, or model quality.",
    ]
    if run.synthetic:
        lines.extend(
            [
                "> **Synthetic outputs:** Candidate outputs in this run are fixtures, not model results.",
            ]
        )
    lines.extend(
        [
            "",
            "## Run information",
            "",
            f"- Run ID: `{run.run_id}`",
            f"- System: `{run.system_name}`",
            f"- Variant: `{run.variant}`",
            f"- Benchmark: `{run.benchmark_name}` `{run.benchmark_version}`",
            f"- Created (UTC): `{run.created_at}`",
            f"- Evaluated tasks: `{len(run.scores)}`",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Estimate | Confidence interval | n | Method | Direction |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for name, summary in run.aggregates.items():
        direction = "higher is better" if name in HIGHER_IS_BETTER else "lower is better"
        interval = (
            f"{_format_metric(name, summary.ci_low)} – {_format_metric(name, summary.ci_high)}"
        )
        lines.append(
            f"| `{name}` | {_format_metric(name, summary.value)} | {interval} | "
            f"{summary.sample_size} | {summary.interval_method} | {direction} |"
        )

    categories = sorted({score.category for score in run.scores})
    lines.extend(
        [
            "",
            "## Accuracy by category",
            "",
            "| Category | Correct | Tasks | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in categories:
        values = [score.task_accuracy for score in run.scores if score.category == category]
        lines.append(
            f"| {_escape_table(category)} | {sum(values):.0f} | {len(values)} | {fmean(values) * 100.0:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Task traces",
            "",
            "| Task | Category | Accuracy | Schema | Tool | Args | Grounded | Retries | Latency | Error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    def optional(value: float | None) -> str:
        return "—" if value is None else f"{value * 100.0:.0f}%"

    for score in run.scores:
        lines.append(
            f"| `{_escape_table(score.task_id)}` | {_escape_table(score.category)} | "
            f"{score.task_accuracy * 100.0:.0f}% | {optional(score.schema_validity)} | "
            f"{optional(score.tool_selection_accuracy)} | {optional(score.tool_argument_accuracy)} | "
            f"{optional(score.groundedness)} | {score.retry_count} | {score.latency_ms:,.1f} ms | "
            f"{_escape_table(score.error or '')} |"
        )

    diagnostic_scores = [score for score in run.scores if score.diagnostics]
    if diagnostic_scores:
        lines.extend(["", "## Diagnostics", ""])
        for score in diagnostic_scores:
            lines.append(
                f"- `{score.task_id}`: "
                + "; ".join(_escape_table(item) for item in score.diagnostics)
            )

    lines.extend(
        [
            "",
            "## Metric interpretation",
            "",
            "- `task_accuracy`: deterministic match against the task's declared answer rule.",
            "- `schema_validity`: validation against the task's dependency-free JSON Schema subset.",
            "- `tool_selection_accuracy`: multiset F1 over expected and emitted tool names.",
            "- `tool_argument_accuracy`: expected argument/rule matches after same-name call pairing.",
            "- `groundedness`: citation provenance and claim-to-evidence attachment, not semantic entailment.",
            "- `retry_count` and `latency_ms`: observable execution cost; lower is better.",
            "",
        ]
    )
    return "\n".join(lines)


def render_ablation_markdown(comparison: AblationComparison) -> str:
    lines = [
        "# SmallForge ablation comparison",
        "",
        "> **Scope:** Results apply only to the named benchmark version. A confidence interval",
        "> crossing zero does not demonstrate a reliable improvement on this suite.",
    ]
    if comparison.synthetic:
        lines.append(
            "> **Synthetic outputs:** At least one compared run uses fixture outputs, not model results."
        )
    lines.extend(
        [
            "",
            f"- Benchmark: `{comparison.benchmark_name}` `{comparison.benchmark_version}`",
            f"- Baseline: `{comparison.baseline_variant}`",
        ]
    )
    for item in comparison.comparisons:
        lines.extend(
            [
                "",
                f"## {item.variant} versus {comparison.baseline_variant}",
                "",
                f"Paired common tasks: `{item.common_task_count}`",
                "",
                "| Metric | Baseline | Candidate | Delta | Delta CI | Direction |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for name, delta in item.deltas.items():
            direction = "higher is better" if name in HIGHER_IS_BETTER else "lower is better"
            lines.append(
                f"| `{name}` | {_format_metric(name, delta.baseline_value)} | "
                f"{_format_metric(name, delta.candidate_value)} | "
                f"{_format_metric(name, delta.absolute_delta)} | "
                f"{_format_metric(name, delta.ci_low)} – {_format_metric(name, delta.ci_high)} | "
                f"{direction} |"
            )
    if comparison.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in comparison.notes)
    lines.append("")
    return "\n".join(lines)


def write_report(
    run: EvaluationRun, output_dir: str | Path, *, stem: str | None = None
) -> tuple[Path, Path]:
    """Write UTF-8 JSON and Markdown reports, returning both paths."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    name = _slug(stem or f"{run.variant}-{run.run_id}")
    json_path, markdown_path = target / f"{name}.json", target / f"{name}.md"
    json_path.write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown_report(run), encoding="utf-8")
    return json_path, markdown_path


def write_ablation_report(
    comparison: AblationComparison, output_dir: str | Path, *, stem: str = "ablation-comparison"
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    name = _slug(stem)
    json_path, markdown_path = target / f"{name}.json", target / f"{name}.md"
    json_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_ablation_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def load_run(path: str | Path) -> EvaluationRun:
    """Restore a run written by :func:`write_report` for later comparison."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Evaluation run JSON must be an object")
    aggregates = {
        str(name): MetricSummary(**summary) for name, summary in value.get("aggregates", {}).items()
    }
    scores = tuple(
        TaskScore(
            **{
                **score,
                "diagnostics": tuple(score.get("diagnostics", ())),
            }
        )
        for score in value.get("scores", ())
    )
    return EvaluationRun(
        run_id=str(value["run_id"]),
        benchmark_name=str(value["benchmark_name"]),
        benchmark_version=str(value["benchmark_version"]),
        system_name=str(value["system_name"]),
        variant=str(value["variant"]),
        scores=scores,
        aggregates=aggregates,
        metadata=dict(value.get("metadata", {})),
        created_at=str(value["created_at"]),
        synthetic=bool(value.get("synthetic", False)),
    )
