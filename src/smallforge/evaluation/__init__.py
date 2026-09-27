"""Public evaluation API for the SmallForge benchmark harness."""

from .contracts import (
    AblationComparison,
    CandidateResult,
    Claim,
    EvaluationRun,
    MetricDelta,
    MetricSummary,
    TaskScore,
    ToolCall,
    VariantComparison,
)
from .evaluator import (
    METRIC_FIELDS,
    aggregate_scores,
    compare_runs,
    evaluate_callable,
    evaluate_predictions,
)
from .loader import (
    BenchmarkFormatError,
    BenchmarkSuite,
    BenchmarkTask,
    available_versions,
    benchmark_registry,
    load_benchmark,
)
from .metrics import is_read_only_sql, score_task, validate_json_schema
from .report import (
    load_run,
    render_ablation_markdown,
    render_markdown_report,
    write_ablation_report,
    write_report,
)
from .statistics import bootstrap_mean_interval, paired_bootstrap_difference, wilson_interval

__all__ = [
    "AblationComparison",
    "BenchmarkFormatError",
    "BenchmarkSuite",
    "BenchmarkTask",
    "CandidateResult",
    "Claim",
    "EvaluationRun",
    "METRIC_FIELDS",
    "MetricDelta",
    "MetricSummary",
    "TaskScore",
    "ToolCall",
    "VariantComparison",
    "aggregate_scores",
    "available_versions",
    "benchmark_registry",
    "bootstrap_mean_interval",
    "compare_runs",
    "evaluate_callable",
    "evaluate_predictions",
    "is_read_only_sql",
    "load_benchmark",
    "load_run",
    "paired_bootstrap_difference",
    "render_ablation_markdown",
    "render_markdown_report",
    "score_task",
    "validate_json_schema",
    "wilson_interval",
    "write_ablation_report",
    "write_report",
]
