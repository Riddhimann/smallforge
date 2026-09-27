from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExperimentMode = Literal[
    "baseline",
    "structured",
    "tools",
    "retrieval",
    "verified",
    "full",
]


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=120)
    model_id: str = Field(min_length=1, max_length=300)
    benchmark_id: str = Field(min_length=1, max_length=120)
    modes: list[ExperimentMode] = Field(default_factory=lambda: ["baseline", "full"])
    max_tasks: int | None = Field(default=None, ge=1, le=200)

    @field_validator("modes")
    @classmethod
    def modes_must_be_unique(cls, value: list[ExperimentMode]) -> list[ExperimentMode]:
        if not value:
            raise ValueError("Select at least one experiment mode")
        if len(value) != len(set(value)):
            raise ValueError("Experiment modes must be unique")
        return value


class ModelDescriptor(BaseModel):
    id: str
    name: str
    provider: str
    available: bool
    details: dict[str, object] = Field(default_factory=dict)


class BenchmarkDescriptor(BaseModel):
    id: str
    name: str
    description: str
    task_count: int
    categories: list[str]
    provenance: Literal["curated", "synthetic_fixture"]
