"""Deterministic metrics for SmallForge task outputs.

These metrics intentionally measure observable outcomes.  In particular,
``groundedness`` verifies citation provenance against benchmark labels; it is not
presented as semantic entailment.  A future NLI or human judge can be added as a
separate metric without changing these definitions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import CandidateResult, TaskScore, ToolCall
from .loader import BenchmarkTask

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _normalized_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.casefold().split())


def _numbers_equal(actual: Any, expected: Any, tolerance: float) -> bool:
    def parse(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, (int, float)):
            return float(value)
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
        if matches:
            # Worked answers typically state intermediate values first and
            # their declared final answer last.
            return float(matches[-1].replace(",", ""))
        words = re.findall(r"[A-Za-z]+", str(value).lower())
        word_values = [NUMBER_WORDS[word] for word in words if word in NUMBER_WORDS]
        if word_values:
            return float(word_values[-1])
        raise ValueError

    try:
        return math.isclose(parse(actual), parse(expected), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return False


def _json_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _json_subset(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) >= len(expected)
            and all(
                _json_subset(found, wanted) for found, wanted in zip(actual, expected, strict=False)
            )
        )
    if (
        isinstance(expected, float)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isclose(float(actual), expected, abs_tol=1e-9, rel_tol=1e-9)
    return actual == expected


def score_answer(task: BenchmarkTask, result: CandidateResult) -> float:
    rule = task.expected["match"]
    kind = str(rule.get("kind", "exact")).casefold()
    expected = rule.get("value")
    actual = result.answer
    if kind in {"json_exact", "json_subset"}:
        actual = result.structured_output if result.structured_output is not None else result.answer
    if kind == "exact":
        return float(actual == expected)
    if kind == "normalized_text":
        return float(_normalized_text(actual) == _normalized_text(expected))
    if kind == "contains_all":
        haystack = _normalized_text(actual)
        needles = rule.get("values", expected if isinstance(expected, list) else ())
        return float(bool(needles) and all(_normalized_text(item) in haystack for item in needles))
    if kind == "numeric":
        return float(_numbers_equal(actual, expected, float(rule.get("tolerance", 1e-9))))
    if kind == "json_exact":
        return float(actual == expected)
    if kind == "json_subset":
        return float(_json_subset(actual, expected))
    if kind == "unordered_list":
        return float(
            isinstance(actual, list)
            and Counter(map(str, actual)) == Counter(map(str, expected or ()))
        )
    if kind == "regex":
        return float(
            bool(re.search(str(rule.get("pattern", "")), str(actual or ""), re.IGNORECASE))
        )
    if kind == "abstain":
        return float(result.abstained)
    if kind == "verification":
        return float(_normalized_text(result.verification) == _normalized_text(expected))
    raise ValueError(f"Unsupported answer match kind {kind!r} in task {task.id}")


def validate_json_schema(value: Any, schema: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Validate the dependency-free JSON Schema subset used by the benchmark."""

    errors: list[str] = []

    def check(instance: Any, rule: Mapping[str, Any], path: str) -> None:
        expected_type = rule.get("type")
        type_options = [expected_type] if isinstance(expected_type, str) else expected_type
        if type_options:
            predicates = {
                "object": lambda item: isinstance(item, Mapping),
                "array": lambda item: isinstance(item, list),
                "string": lambda item: isinstance(item, str),
                "number": lambda item: isinstance(item, (int, float))
                and not isinstance(item, bool),
                "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
                "boolean": lambda item: isinstance(item, bool),
                "null": lambda item: item is None,
            }
            unknown = [item for item in type_options if item not in predicates]
            if unknown:
                errors.append(f"{path}: unsupported schema type {unknown[0]!r}")
                return
            if not any(predicates[item](instance) for item in type_options):
                errors.append(f"{path}: expected {' or '.join(type_options)}")
                return
        if "enum" in rule and instance not in rule["enum"]:
            errors.append(f"{path}: value is not in enum")
        if isinstance(instance, Mapping):
            for required in rule.get("required", ()):
                if required not in instance:
                    errors.append(f"{path}.{required}: required property is missing")
            properties = rule.get("properties", {})
            if isinstance(properties, Mapping):
                for key, child_rule in properties.items():
                    if key in instance and isinstance(child_rule, Mapping):
                        check(instance[key], child_rule, f"{path}.{key}")
            if rule.get("additionalProperties") is False and isinstance(properties, Mapping):
                for key in instance:
                    if key not in properties:
                        errors.append(f"{path}.{key}: additional property is not allowed")
        if isinstance(instance, list):
            if "minItems" in rule and len(instance) < int(rule["minItems"]):
                errors.append(f"{path}: expected at least {rule['minItems']} items")
            if "maxItems" in rule and len(instance) > int(rule["maxItems"]):
                errors.append(f"{path}: expected at most {rule['maxItems']} items")
            child_rule = rule.get("items")
            if isinstance(child_rule, Mapping):
                for index, child in enumerate(instance):
                    check(child, child_rule, f"{path}[{index}]")
        if isinstance(instance, str):
            if "minLength" in rule and len(instance) < int(rule["minLength"]):
                errors.append(f"{path}: string is shorter than {rule['minLength']}")
            if "pattern" in rule and not re.search(str(rule["pattern"]), instance):
                errors.append(f"{path}: string does not match pattern")
        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if "minimum" in rule and instance < rule["minimum"]:
                errors.append(f"{path}: value is below minimum")
            if "maximum" in rule and instance > rule["maximum"]:
                errors.append(f"{path}: value is above maximum")

    check(value, schema, "$")
    return not errors, tuple(errors)


def schema_validity(
    task: BenchmarkTask, result: CandidateResult
) -> tuple[float | None, tuple[str, ...]]:
    schema = task.expected.get("structured_schema")
    if not isinstance(schema, Mapping):
        return None, ()
    valid, errors = validate_json_schema(result.structured_output, schema)
    return float(valid), errors


def tool_selection_accuracy(
    expected_calls: Sequence[Mapping[str, Any]], actual_calls: Sequence[ToolCall]
) -> float:
    """Multiset F1 over expected and actual tool names."""

    expected = Counter(str(call.get("name", "")) for call in expected_calls)
    actual = Counter(call.name for call in actual_calls)
    if not expected and not actual:
        return 1.0
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values()) if actual else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|grant|revoke|vacuum)\b",
    re.IGNORECASE,
)


def is_read_only_sql(query: str, *, required_tokens: Iterable[str] = ()) -> bool:
    """Conservative textual guard used for benchmark argument validation.

    Production SQL execution must still enforce database-level read-only access.
    """

    without_comments = re.sub(r"/\*.*?\*/|--[^\r\n]*", " ", query, flags=re.DOTALL).strip()
    if not re.match(r"^(select|with)\b", without_comments, re.IGNORECASE):
        return False
    if _SQL_FORBIDDEN.search(without_comments):
        return False
    statements = [item.strip() for item in without_comments.split(";") if item.strip()]
    if len(statements) != 1:
        return False
    lowered = without_comments.casefold()
    return all(str(token).casefold() in lowered for token in required_tokens)


def _value_matches_rule(actual: Any, expected: Any, rule: Mapping[str, Any]) -> bool:
    kind = str(rule.get("kind", "exact")).casefold()
    if kind == "exact":
        return actual == rule.get("value", expected)
    if kind == "normalized_text":
        return _normalized_text(actual) == _normalized_text(rule.get("value", expected))
    if kind == "numeric":
        return _numbers_equal(
            actual, rule.get("value", expected), float(rule.get("tolerance", 1e-9))
        )
    if kind == "one_of":
        return actual in rule.get("values", ())
    if kind == "contains_all":
        return all(
            _normalized_text(token) in _normalized_text(actual) for token in rule.get("values", ())
        )
    if kind == "readonly_sql":
        return isinstance(actual, str) and is_read_only_sql(
            actual, required_tokens=rule.get("required_tokens", ())
        )
    if kind == "json_subset":
        return _json_subset(actual, rule.get("value", expected))
    raise ValueError(f"Unsupported tool argument rule {kind!r}")


def _arguments_score(expected_call: Mapping[str, Any], actual: ToolCall) -> float:
    expected_arguments = expected_call.get("arguments", {}) or {}
    argument_rules = expected_call.get("argument_rules", {}) or {}
    match_mode = str(expected_call.get("argument_match", "exact"))
    if not isinstance(expected_arguments, Mapping) or not isinstance(argument_rules, Mapping):
        raise ValueError("Expected tool arguments and argument_rules must be objects")
    keys = tuple(dict.fromkeys([*expected_arguments.keys(), *argument_rules.keys()]))
    if not keys:
        return 1.0
    correct = 0
    for key in keys:
        if key not in actual.arguments:
            continue
        rule = argument_rules.get(key, {"kind": "exact"})
        if not isinstance(rule, Mapping):
            raise ValueError(f"Argument rule for {key!r} must be an object")
        if _value_matches_rule(actual.arguments[key], expected_arguments.get(key), rule):
            correct += 1
    score = correct / len(keys)
    if match_mode == "exact" and set(actual.arguments) != set(keys):
        extra_count = len(set(actual.arguments) - set(keys))
        score *= len(keys) / (len(keys) + extra_count)
    elif match_mode not in {"exact", "subset"}:
        raise ValueError(f"Unsupported argument_match mode {match_mode!r}")
    return score


def tool_argument_accuracy(
    expected_calls: Sequence[Mapping[str, Any]], actual_calls: Sequence[ToolCall]
) -> float:
    """Greedily pair same-name calls, then average expected argument validity."""

    if not expected_calls:
        return 1.0 if not actual_calls else 0.0
    remaining = set(range(len(actual_calls)))
    scores: list[float] = []
    for expected in expected_calls:
        candidates = [
            index for index in remaining if actual_calls[index].name == expected.get("name")
        ]
        if not candidates:
            scores.append(0.0)
            continue
        ranked = [(index, _arguments_score(expected, actual_calls[index])) for index in candidates]
        index, score = max(ranked, key=lambda pair: pair[1])
        remaining.remove(index)
        scores.append(score)
    return sum(scores) / len(scores)


def grounding_metrics(
    task: BenchmarkTask, result: CandidateResult
) -> tuple[float | None, float | None, float | None, tuple[str, ...]]:
    grounding = task.expected.get("grounding")
    if not isinstance(grounding, Mapping):
        return None, None, None, ()
    required = {str(item) for item in grounding.get("required_ids", ())}
    allowed = {str(item) for item in grounding.get("allowed_ids", ())} or required
    cited = set(result.evidence_ids)
    for claim in result.claims:
        cited.update(claim.evidence_ids)
    diagnostics: list[str] = []
    if grounding.get("expect_no_citations"):
        correct = not cited
        if cited:
            diagnostics.append(
                "Expected no citations, but candidate cited: " + ", ".join(sorted(cited))
            )
        return float(correct), float(correct), None, tuple(diagnostics)
    valid = cited & allowed
    precision = len(valid) / len(cited) if cited else 0.0
    recall = len(cited & required) / len(required) if required else 1.0
    provenance_f1 = (
        0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    )
    if result.claims:
        supported_claims = sum(
            bool(claim.evidence_ids) and set(claim.evidence_ids).issubset(allowed)
            for claim in result.claims
        )
        claim_rate = supported_claims / len(result.claims)
        groundedness = (claim_rate + provenance_f1) / 2.0
    else:
        groundedness = provenance_f1
    unknown = cited - allowed
    missing = required - cited
    if unknown:
        diagnostics.append("Unknown or disallowed evidence IDs: " + ", ".join(sorted(unknown)))
    if missing:
        diagnostics.append("Missing required evidence IDs: " + ", ".join(sorted(missing)))
    return groundedness, precision, recall, tuple(diagnostics)


def score_task(task: BenchmarkTask, result: CandidateResult) -> TaskScore:
    """Score one candidate result against one benchmark task."""

    if result.task_id != task.id:
        raise ValueError(f"Result task_id {result.task_id!r} does not match task {task.id!r}")
    diagnostics: list[str] = []
    try:
        answer_accuracy = score_answer(task, result)
    except (TypeError, ValueError) as exc:
        answer_accuracy = 0.0
        diagnostics.append(str(exc))
    schema_score, schema_errors = schema_validity(task, result)
    diagnostics.extend(schema_errors)
    expected_calls_value = task.expected.get("tool_calls")
    if expected_calls_value is None:
        selection_score = None
        argument_score = None
    else:
        expected_calls = tuple(expected_calls_value)
        try:
            selection_score = tool_selection_accuracy(expected_calls, result.tool_calls)
            argument_score = tool_argument_accuracy(expected_calls, result.tool_calls)
        except (TypeError, ValueError) as exc:
            selection_score = 0.0
            argument_score = 0.0
            diagnostics.append(str(exc))
    groundedness, citation_precision, citation_recall, grounding_errors = grounding_metrics(
        task, result
    )
    diagnostics.extend(grounding_errors)
    expected_abstention = task.expected.get("abstain")
    abstention = (
        None
        if expected_abstention is None
        else float(result.abstained == bool(expected_abstention))
    )
    expected_verification = task.expected.get("verification")
    verification = (
        None
        if expected_verification is None
        else float(_normalized_text(result.verification) == _normalized_text(expected_verification))
    )
    if result.error:
        diagnostics.append(f"Candidate error: {result.error}")
    return TaskScore(
        task_id=task.id,
        category=task.category,
        task_accuracy=answer_accuracy,
        schema_validity=schema_score,
        tool_selection_accuracy=selection_score,
        tool_argument_accuracy=argument_score,
        groundedness=groundedness,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        abstention_accuracy=abstention,
        verification_accuracy=verification,
        retry_count=result.retry_count,
        latency_ms=result.latency_ms,
        error=result.error,
        diagnostics=tuple(diagnostics),
    )
