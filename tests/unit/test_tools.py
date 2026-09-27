from __future__ import annotations

import asyncio

from smallforge.schemas import ToolCall
from smallforge.tools import (
    CalculatorTool,
    LexicalRetrievalTool,
    ReadOnlySQLiteTool,
    RetrievalDocument,
    ToolRegistry,
)


def execute(registry: ToolRegistry, call: ToolCall):
    return asyncio.run(registry.execute(call))


def test_calculator_evaluates_math_without_eval() -> None:
    registry = ToolRegistry([CalculatorTool()])
    result = execute(
        registry,
        ToolCall(step_id="calc", name="calculator", arguments={"expression": "(17 * 23) + 11"}),
    )

    assert result.success is True
    assert result.output["result"] == 402
    assert result.evidence[0].content.endswith("= 402")


def test_calculator_blocks_code_execution_and_resource_abuse() -> None:
    registry = ToolRegistry([CalculatorTool()])
    injection = execute(
        registry,
        ToolCall(
            step_id="inject",
            name="calculator",
            arguments={"expression": '__import__("os").system("whoami")'},
        ),
    )
    huge_power = execute(
        registry,
        ToolCall(step_id="power", name="calculator", arguments={"expression": "2 ** 999999"}),
    )

    assert injection.success is False
    assert injection.error_code == "unsupported_expression"
    assert huge_power.success is False
    assert huge_power.error_code == "unsafe_power"


def test_sql_is_parameterized_bounded_and_read_only() -> None:
    sql = ReadOnlySQLiteTool(
        {
            "sales": [
                {"region": "east", "amount": 10},
                {"region": "west", "amount": 20},
                {"region": "east", "amount": 15},
            ]
        },
        max_rows=10,
    )
    registry = ToolRegistry([sql])
    selected = execute(
        registry,
        ToolCall(
            step_id="query",
            name="sql_query",
            arguments={
                "query": "SELECT SUM(amount) AS total FROM sales WHERE region = :region",
                "parameters": {"region": "east"},
            },
        ),
    )
    write = execute(
        registry,
        ToolCall(step_id="write", name="sql_query", arguments={"query": "DELETE FROM sales"}),
    )
    pragma = execute(
        registry,
        ToolCall(
            step_id="pragma", name="sql_query", arguments={"query": "PRAGMA table_info(sales)"}
        ),
    )

    assert selected.success is True
    assert selected.output["rows"] == [{"total": 25}]
    assert write.error_code == "read_only_violation"
    assert pragma.error_code == "read_only_violation"
    sql.close()


def test_lexical_retrieval_ranks_matching_document_and_returns_no_false_match() -> None:
    tool = LexicalRetrievalTool(
        [
            RetrievalDocument(
                source_id="python", text="Python uses readable syntax for data work."
            ),
            RetrievalDocument(source_id="sql", text="SQL aggregation uses GROUP BY and SUM."),
            RetrievalDocument(
                source_id="rag", text="Retrieval augmented generation grounds an LLM."
            ),
        ]
    )
    registry = ToolRegistry([tool])
    result = execute(
        registry,
        ToolCall(
            step_id="retrieve",
            name="retrieve",
            arguments={"query": "How does SQL aggregation use SUM?", "top_k": 2},
        ),
    )
    none = execute(
        registry,
        ToolCall(
            step_id="none",
            name="retrieve",
            arguments={"query": "quokka marsupial", "top_k": 2},
        ),
    )

    assert result.success is True
    assert result.evidence[0].source_id == "sql"
    assert none.output["match_count"] == 0
