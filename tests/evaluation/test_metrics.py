from __future__ import annotations

from smallforge.evaluation import (
    CandidateResult,
    ToolCall,
    is_read_only_sql,
    load_benchmark,
    score_task,
)
from smallforge.evaluation.metrics import (
    tool_argument_accuracy,
    tool_selection_accuracy,
    validate_json_schema,
)


def _task(task_id: str):
    return load_benchmark("v1").by_id()[task_id]


def test_arithmetic_answer_and_tool_arguments_are_scored_independently() -> None:
    result = CandidateResult(
        task_id="arith_001",
        answer="1,090 cases",
        tool_calls=(ToolCall("calculator", {"expression": "1250 * 0.872"}),),
        latency_ms=12.0,
    )

    score = score_task(_task("arith_001"), result)

    assert score.task_accuracy == 1.0
    assert score.tool_selection_accuracy == 1.0
    assert score.tool_argument_accuracy == 1.0
    assert score.schema_validity is None


def test_numeric_answer_scores_the_declared_final_value_after_working() -> None:
    result = CandidateResult(
        task_id="arith_001",
        answer="1,250 * 0.872 = 1,090. Therefore the answer is 1,090 cases.",
    )

    assert score_task(_task("arith_001"), result).task_accuracy == 1.0


def test_numeric_answer_accepts_an_unambiguous_number_word() -> None:
    result = CandidateResult(task_id="ret_001", answer="five")

    assert score_task(_task("ret_001"), result).task_accuracy == 1.0


def test_schema_metric_rejects_extra_and_wrong_typed_fields() -> None:
    result = CandidateResult(
        task_id="ext_001",
        structured_output={
            "invoice_id": "INV-2048",
            "currency": "USD",
            "total": "1842.75",
            "due_date": "2026-09-14",
            "invented": True,
        },
        tool_calls=(
            ToolCall("extract_structured", {"document_id": "DOC-INV-1", "schema_name": "invoice"}),
        ),
    )

    score = score_task(_task("ext_001"), result)

    assert score.task_accuracy == 0.0
    assert score.schema_validity == 0.0
    assert any("expected number" in message for message in score.diagnostics)
    assert any("additional property" in message for message in score.diagnostics)


def test_json_schema_subset_supports_nullable_values() -> None:
    schema = _task("ext_004").expected["structured_schema"]
    valid, errors = validate_json_schema(
        {
            "finding_id": "F-77",
            "rating": "High",
            "owner": None,
            "due_date": "2026-12-01",
            "open_actions": 4,
        },
        schema,
    )

    assert valid is True
    assert errors == ()


def test_sql_guard_accepts_one_select_and_rejects_mutation_or_multiple_statements() -> None:
    assert is_read_only_sql(
        "SELECT SUM(amount) FROM transactions", required_tokens=["sum", "transactions"]
    )
    assert is_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not is_read_only_sql("DELETE FROM transactions")
    assert not is_read_only_sql("SELECT 1; DROP TABLE transactions")
    assert not is_read_only_sql("SELECT 1; SELECT 2")


def test_sql_task_scores_semantically_valid_query_without_exact_string_match() -> None:
    result = CandidateResult(
        task_id="sql_001",
        answer=200,
        tool_calls=(
            ToolCall(
                "sql_query",
                {
                    "query": (
                        "SELECT SUM(amount) FROM transactions "
                        "WHERE status = 'settled' AND region = 'East'"
                    )
                },
            ),
        ),
    )

    score = score_task(_task("sql_001"), result)

    assert score.task_accuracy == 1.0
    assert score.tool_selection_accuracy == 1.0
    assert score.tool_argument_accuracy == 1.0


def test_tool_selection_is_multiset_f1_and_arguments_are_separate() -> None:
    expected = ({"name": "retrieve"}, {"name": "calculator"})
    actual = (ToolCall("retrieve", {}), ToolCall("retrieve", {}))

    assert tool_selection_accuracy(expected, actual) == 0.5
    assert tool_argument_accuracy(expected, actual) == 0.5


def test_grounding_penalizes_unknown_and_missing_evidence() -> None:
    result = CandidateResult(
        task_id="ret_004",
        answer=1.4,
        evidence_ids=("FIN-Q3", "MADE-UP"),
    )

    score = score_task(_task("ret_004"), result)

    assert score.citation_precision == 0.5
    assert score.citation_recall == 0.5
    assert score.groundedness == 0.5
    assert any("MADE-UP" in message for message in score.diagnostics)
    assert any("FIN-Q2" in message for message in score.diagnostics)


def test_unanswerable_task_rewards_abstention_without_fake_citation() -> None:
    result = CandidateResult(
        task_id="unans_001",
        abstained=True,
        verification="insufficient",
        tool_calls=(ToolCall("retrieve", {"query": "northstar employees"}),),
    )

    score = score_task(_task("unans_001"), result)

    assert score.task_accuracy == 1.0
    assert score.abstention_accuracy == 1.0
    assert score.verification_accuracy == 1.0
    assert score.groundedness == 1.0


def test_verification_is_not_conflated_with_citation_presence() -> None:
    wrong_verdict = CandidateResult(
        task_id="verify_001",
        verification="pass",
        evidence_ids=("CEDAR-1",),
        tool_calls=(
            ToolCall(
                "verify_claim",
                {
                    "claim": "Project Cedar reduced financial losses by 18%.",
                    "evidence_ids": ["CEDAR-1"],
                },
            ),
        ),
    )

    score = score_task(_task("verify_001"), wrong_verdict)

    assert score.groundedness == 1.0
    assert score.verification_accuracy == 0.0
    assert score.task_accuracy == 0.0
