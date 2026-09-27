"""Structured-output parsing and Pydantic validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from smallforge.schemas import ValidationIssue

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(slots=True)
class StructuredOutputError(ValueError):
    stage: str
    issues: list[ValidationIssue]

    def __str__(self) -> str:
        return "; ".join(issue.message for issue in self.issues)


@dataclass(slots=True)
class ValidatedOutput(Generic[SchemaT]):
    value: SchemaT
    normalized_json: str


def validate_structured_output(
    text: str,
    schema: type[SchemaT],
    *,
    stage: str,
) -> ValidatedOutput[SchemaT]:
    """Extract one JSON object and validate it against ``schema``.

    Small models frequently wrap otherwise valid JSON in a short explanation or
    a Markdown fence. We tolerate that presentation noise, but never repair or
    invent fields: semantic validation remains strict.
    """

    try:
        raw = _first_json_object(text)
    except ValueError as exc:
        raise StructuredOutputError(
            stage,
            [ValidationIssue(stage=stage, message=str(exc), path=None)],
        ) from exc
    try:
        value = schema.model_validate(raw)
    except ValidationError as exc:
        issues = [
            ValidationIssue(
                stage=stage,
                path=".".join(str(part) for part in error["loc"]) or None,
                message=error["msg"],
            )
            for error in exc.errors(include_url=False, include_input=False)
        ]
        raise StructuredOutputError(stage, issues) from exc
    return ValidatedOutput(
        value=value,
        normalized_json=json.dumps(value.model_dump(mode="json"), sort_keys=True),
    )


def _first_json_object(text: str) -> dict[str, object]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("model returned an empty response")
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(candidate[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("response did not contain a valid JSON object")
