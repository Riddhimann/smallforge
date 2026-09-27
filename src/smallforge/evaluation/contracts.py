"""Dependency-free contracts for SmallForge benchmark evaluation.

The evaluation package deliberately does not import the runtime harness.  A model,
agent, or fixture only needs to return a mapping accepted by
``CandidateResult.from_mapping``.  This keeps benchmark results comparable across
orchestrators and lets the harness evolve without changing historical task files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool invocation emitted by a candidate system."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | ToolCall) -> ToolCall:
        if isinstance(value, cls):
            return value
        name = value.get("name") or value.get("tool") or value.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every tool call requires a non-empty name")
        arguments = value.get("arguments", value.get("args", {}))
        if not isinstance(arguments, Mapping):
            raise ValueError(f"Tool call arguments for {name!r} must be an object")
        return cls(name=name.strip(), arguments=dict(arguments))


@dataclass(frozen=True)
class Claim:
    """A generated claim and the evidence identifiers attached to it."""

    text: str
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Claim) -> Claim:
        if isinstance(value, cls):
            return value
        evidence = value.get("evidence_ids", value.get("citations", ()))
        return cls(
            text=str(value.get("text", "")),
            evidence_ids=tuple(str(item) for item in evidence or ()),
        )


@dataclass(frozen=True)
class CandidateResult:
    """Framework-neutral output produced for one benchmark task.

    ``answer`` is the human-facing answer. ``structured_output`` is used for tasks
    that request JSON.  The remaining fields expose observable harness behaviour;
    none are inferred from hidden chain-of-thought.
    """

    task_id: str
    answer: Any = None
    structured_output: Any = None
    tool_calls: tuple[ToolCall, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    abstained: bool = False
    verification: str | None = None
    retry_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | CandidateResult) -> CandidateResult:
        if isinstance(value, cls):
            return value
        task_id = value.get("task_id") or value.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Candidate result requires a non-empty task_id")
        raw_calls = value.get("tool_calls", ()) or ()
        raw_evidence = value.get("evidence_ids", value.get("citations", ())) or ()
        raw_claims = value.get("claims", ()) or ()
        retry_count = value.get("retry_count", value.get("retries", 0))
        latency_ms = value.get("latency_ms", value.get("elapsed_ms", 0.0))
        if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative number")
        metadata = value.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            task_id=task_id.strip(),
            answer=value.get("answer"),
            structured_output=value.get("structured_output", value.get("output")),
            tool_calls=tuple(ToolCall.from_value(item) for item in raw_calls),
            evidence_ids=tuple(str(item) for item in raw_evidence),
            claims=tuple(Claim.from_value(item) for item in raw_claims),
            abstained=bool(value.get("abstained", value.get("refused", False))),
            verification=(
                None
                if value.get("verification") is None
                else str(value.get("verification")).lower()
            ),
            retry_count=retry_count,
            latency_ms=float(latency_ms),
            error=None if value.get("error") is None else str(value.get("error")),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskScore:
    """Metrics for one task. ``None`` means the metric is not applicable."""

    task_id: str
    category: str
    task_accuracy: float
    schema_validity: float | None
    tool_selection_accuracy: float | None
    tool_argument_accuracy: float | None
    groundedness: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float | None
    verification_accuracy: float | None
    retry_count: int
    latency_ms: float
    error: str | None = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    """An aggregate estimate with a confidence interval."""

    metric: str
    value: float
    sample_size: int
    ci_low: float
    ci_high: float
    confidence: float
    interval_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRun:
    """A scored run for one named system variant."""

    run_id: str
    benchmark_name: str
    benchmark_version: str
    system_name: str
    variant: str
    scores: tuple[TaskScore, ...]
    aggregates: Mapping[str, MetricSummary]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "system_name": self.system_name,
            "variant": self.variant,
            "created_at": self.created_at,
            "synthetic": self.synthetic,
            "metadata": dict(self.metadata),
            "aggregates": {name: item.to_dict() for name, item in self.aggregates.items()},
            "scores": [score.to_dict() for score in self.scores],
        }


@dataclass(frozen=True)
class MetricDelta:
    """Paired candidate-minus-baseline change for one metric."""

    metric: str
    baseline_value: float
    candidate_value: float
    absolute_delta: float
    relative_delta_percent: float | None
    sample_size: int
    ci_low: float
    ci_high: float
    confidence: float
    interval_method: str = "paired bootstrap percentile"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VariantComparison:
    variant: str
    common_task_count: int
    deltas: Mapping[str, MetricDelta]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "common_task_count": self.common_task_count,
            "deltas": {name: item.to_dict() for name, item in self.deltas.items()},
        }


@dataclass(frozen=True)
class AblationComparison:
    """One baseline compared with one or more harness variants."""

    benchmark_name: str
    benchmark_version: str
    baseline_variant: str
    comparisons: tuple[VariantComparison, ...]
    synthetic: bool = False
    notes: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "baseline_variant": self.baseline_variant,
            "synthetic": self.synthetic,
            "notes": list(self.notes),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }
