from __future__ import annotations

import pytest

from smallforge.harness import EvidenceVerifier, StructuredOutputError, validate_structured_output
from smallforge.schemas import AnswerEnvelope, Evidence, PlanStep, ToolResult


def test_plan_step_normalizes_one_redundant_quote_pair_in_tool_name() -> None:
    step = PlanStep(
        id="one",
        purpose="calculate",
        tool="'calculator'",
        arguments={"expression": "2 + 2"},
    )

    assert step.tool == "calculator"


def test_structured_validation_handles_fence_but_rejects_wrong_schema() -> None:
    valid = validate_structured_output(
        '```json\n{"answer":"42","citations":["calc"],"confidence":0.9}\n```',
        AnswerEnvelope,
        stage="answer",
    )
    assert valid.value.answer == "42"

    with pytest.raises(StructuredOutputError) as captured:
        validate_structured_output(
            '{"answer":"42","confidence":"very sure","unexpected":true}',
            AnswerEnvelope,
            stage="answer",
        )
    assert captured.value.issues


def test_verifier_accepts_supported_number_and_rejects_invented_number() -> None:
    evidence = [Evidence(source_id="calc", content="17 * 23 + 11 = 402", score=1.0)]
    tool_results = [
        ToolResult(
            step_id="one",
            tool_name="calculator",
            success=True,
            output={"result": 402},
            evidence=evidence,
        )
    ]
    verifier = EvidenceVerifier()

    supported = verifier.verify(
        AnswerEnvelope(answer="The computed value is 402.", citations=["calc"], confidence=0.95),
        evidence,
        tool_results,
    )
    invented = verifier.verify(
        AnswerEnvelope(answer="The computed value is 999.", citations=["calc"], confidence=0.95),
        evidence,
        tool_results,
    )

    assert supported.passed is True
    assert invented.passed is False
    assert "unsupported number: 999" in invented.unsupported_claims


def test_verifier_rejects_unknown_citation() -> None:
    decision = EvidenceVerifier().verify(
        AnswerEnvelope(answer="A supported-looking claim.", citations=["made-up"], confidence=0.8),
        [Evidence(source_id="actual", content="A supported claim.")],
        [],
    )

    assert decision.passed is False
    assert any("unknown citation" in claim for claim in decision.unsupported_claims)


def test_verifier_matches_numeric_digits_to_number_words_in_evidence() -> None:
    evidence = [Evidence(source_id="policy", content="The maximum is five unused days.")]
    decision = EvidenceVerifier().verify(
        AnswerEnvelope(answer="The maximum is 5 days.", citations=["policy"]),
        evidence,
        [
            ToolResult(
                step_id="retrieve",
                tool_name="retrieve",
                success=True,
                output={},
                evidence=evidence,
            )
        ],
    )

    assert decision.passed is True
