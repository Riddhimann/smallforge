"""Shared, transport-safe schemas for SmallForge.

The core deliberately keeps model I/O, tool I/O and run traces explicit.  That
makes a baseline run and a harnessed run comparable without relying on hidden
global state or provider-specific response objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base class that rejects silently misspelled fields."""

    model_config = ConfigDict(extra="forbid")


class RunMode(StrEnum):
    BASELINE = "baseline"
    HARNESS = "harness"


class EventStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content cannot be blank")
        return value


class ToolDefinition(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=1_000)
    parameters: dict[str, Any]


class ModelRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    response_schema: dict[str, Any] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1_024, ge=1, le=32_768)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelResponse(StrictModel):
    text: str
    model: str
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    provider_request_id: str | None = None


class PlanStep(StrictModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    purpose: str = Field(min_length=1, max_length=500)
    tool: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool", mode="before")
    @classmethod
    def normalize_quoted_tool_name(cls, value: Any) -> Any:
        # Some constrained small models emit a JSON string whose content still
        # includes one matching quote pair (for example, "'calculator'").
        if isinstance(value, str):
            cleaned = value.strip()
            if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
                return cleaned[1:-1].strip()
        return value


class ExecutionPlan(StrictModel):
    rationale: str = Field(default="", max_length=1_000)
    steps: list[PlanStep] = Field(min_length=1, max_length=8)
    answer_instruction: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def unique_step_ids(self) -> ExecutionPlan:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        return self


class ToolCall(StrictModel):
    step_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Evidence(StrictModel):
    source_id: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=20_000)
    score: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    step_id: str
    tool_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def error_fields_match_status(self) -> ToolResult:
        if self.success and (self.error_code or self.error_message):
            raise ValueError("successful tool results cannot contain errors")
        if not self.success and not self.error_code:
            raise ValueError("failed tool results require an error_code")
        return self


class AnswerEnvelope(StrictModel):
    answer: str = Field(min_length=1, max_length=30_000)
    citations: list[str] = Field(default_factory=list, max_length=64)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("citations")
    @classmethod
    def citations_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("citations must be unique")
        return value


class VerificationDecision(StrictModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list, max_length=32)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=32)
    correction_hint: str | None = Field(default=None, max_length=2_000)


class ValidationIssue(StrictModel):
    stage: str
    message: str
    path: str | None = None


class TraceEvent(StrictModel):
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: str
    status: EventStatus
    message: str
    duration_ms: float | None = Field(default=None, ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class RunTrace(StrictModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    mode: RunMode
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)

    def add(
        self,
        stage: str,
        status: EventStatus,
        message: str,
        *,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                sequence=len(self.events),
                stage=stage,
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def finish(self) -> None:
        self.completed_at = datetime.now(UTC)


class RunMetrics(StrictModel):
    latency_ms: float = Field(default=0.0, ge=0.0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    schema_valid: bool = True
    verification_passed: bool | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunResult(StrictModel):
    run_id: str
    mode: RunMode
    task: str
    answer: AnswerEnvelope | None
    success: bool
    model: str
    plan: ExecutionPlan | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verification: VerificationDecision | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    trace: RunTrace
    error: str | None = None

    @model_validator(mode="after")
    def success_requires_answer(self) -> RunResult:
        if self.success and self.answer is None:
            raise ValueError("successful runs require an answer")
        return self
