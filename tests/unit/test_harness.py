from __future__ import annotations

import asyncio
import json
import re

from smallforge.harness import AgentHarness, BaselineRunner
from smallforge.models import ScriptedTestAdapter
from smallforge.schemas import ModelRequest
from smallforge.tools import CalculatorTool, ToolRegistry


def test_same_test_model_improves_when_harness_uses_tool_and_retry() -> None:
    """Mechanism test only; real benchmark claims must use Ollama/provider adapters."""

    def model_behavior(request: ModelRequest) -> str:
        stage = request.metadata["stage"]
        if stage == "baseline":
            return "390"
        if stage == "planner":
            return json.dumps(
                {
                    "rationale": "Use exact arithmetic.",
                    "steps": [
                        {
                            "id": "calculate",
                            "purpose": "Compute the requested expression.",
                            "tool": "calculator",
                            "arguments": {"expression": "(17 * 23) + 11"},
                        }
                    ],
                    "answer_instruction": "Report the exact calculated value.",
                }
            )
        if stage == "synthesis" and request.metadata["attempt"] == 0:
            return json.dumps(
                {"answer": "The value is 999.", "citations": ["invented"], "confidence": 0.9}
            )
        evidence_id = re.search(r"calculator:[a-f0-9]{12}", request.messages[1].content).group(0)  # type: ignore[union-attr]
        return json.dumps(
            {"answer": "The value is 402.", "citations": [evidence_id], "confidence": 0.99}
        )

    model = ScriptedTestAdapter(response_factory=model_behavior, model_name="same-small-test-model")
    baseline = asyncio.run(BaselineRunner(model).run("What is (17 * 23) + 11?"))
    harnessed = asyncio.run(
        AgentHarness(model, ToolRegistry([CalculatorTool()]), max_retries=2).run(
            "What is (17 * 23) + 11?"
        )
    )

    assert baseline.model == harnessed.model == "same-small-test-model"
    assert baseline.answer is not None and baseline.answer.answer == "390"
    assert harnessed.success is True
    assert harnessed.answer is not None and "402" in harnessed.answer.answer
    assert harnessed.verification is not None and harnessed.verification.passed is True
    assert harnessed.metrics.retries == 1
    assert harnessed.metrics.model_calls == 3  # plan + rejected synthesis + corrected synthesis
    assert harnessed.metrics.tool_calls == 1
    assert [event.sequence for event in harnessed.trace.events] == list(
        range(len(harnessed.trace.events))
    )


def test_retry_budget_is_strictly_bounded_for_malformed_answers() -> None:
    def model_behavior(request: ModelRequest) -> str:
        if request.metadata["stage"] == "planner":
            return json.dumps(
                {
                    "rationale": "Calculate.",
                    "steps": [
                        {
                            "id": "one",
                            "purpose": "calculate",
                            "tool": "calculator",
                            "arguments": {"expression": "2 + 2"},
                        }
                    ],
                    "answer_instruction": "Give the result.",
                }
            )
        return "not-json"

    model = ScriptedTestAdapter(response_factory=model_behavior)
    result = asyncio.run(
        AgentHarness(model, ToolRegistry([CalculatorTool()]), max_retries=1).run("Compute 2 + 2")
    )

    assert result.success is False
    assert result.metrics.model_calls == 3  # one plan plus exactly two synthesis attempts
    assert result.metrics.retries == 1
    assert result.answer is None
    assert result.error is not None and "retry budget" in result.error.lower()


def test_invalid_plan_is_repaired_once_before_execution() -> None:
    planner_calls = 0

    def model_behavior(request: ModelRequest) -> str:
        nonlocal planner_calls
        if request.metadata["stage"] == "planner":
            planner_calls += 1
            tool = "not_registered" if planner_calls == 1 else "calculator"
            return json.dumps(
                {
                    "rationale": "Calculate.",
                    "steps": [
                        {
                            "id": "one",
                            "purpose": "calculate",
                            "tool": tool,
                            "arguments": {"expression": "2 + 2"},
                        }
                    ],
                    "answer_instruction": "Give the result.",
                }
            )
        evidence_id = re.search(r"calculator:[a-f0-9]{12}", request.messages[1].content).group(0)  # type: ignore[union-attr]
        return json.dumps(
            {"answer": "The value is 4.", "citations": [evidence_id], "confidence": 1.0}
        )

    model = ScriptedTestAdapter(response_factory=model_behavior)
    result = asyncio.run(
        AgentHarness(
            model,
            ToolRegistry([CalculatorTool()]),
            max_retries=0,
            planner_retries=1,
        ).run("Compute 2 + 2")
    )

    assert result.success is True
    assert planner_calls == 2
    assert result.metrics.retries == 1
    assert any("not registered" in issue.message for issue in result.validation_issues)


def test_policy_keeps_safe_steps_when_small_model_invents_extra_tools() -> None:
    def model_behavior(request: ModelRequest) -> str:
        if request.metadata["stage"] == "planner":
            return json.dumps(
                {
                    "rationale": "Calculate, then over-plan.",
                    "steps": [
                        {
                            "id": "calculate",
                            "purpose": "calculate",
                            "tool": "calculator",
                            "arguments": {"expression": "5 + 5"},
                        },
                        {
                            "id": "invented",
                            "purpose": "unnecessary processing",
                            "tool": "format_magic",
                            "arguments": {},
                        },
                    ],
                    "answer_instruction": "Give the result.",
                }
            )
        evidence_id = re.search(r"calculator:[a-f0-9]{12}", request.messages[1].content).group(0)  # type: ignore[union-attr]
        return json.dumps(
            {"answer": "The value is 10.", "citations": [evidence_id], "confidence": 1.0}
        )

    result = asyncio.run(
        AgentHarness(
            ScriptedTestAdapter(response_factory=model_behavior),
            ToolRegistry([CalculatorTool()]),
            max_retries=0,
        ).run("Compute 5 + 5")
    )

    assert result.success is True
    assert result.metrics.tool_calls == 1
    assert any(issue.stage == "planning_policy" for issue in result.validation_issues)
    assert any(event.stage == "planning_policy" for event in result.trace.events)
