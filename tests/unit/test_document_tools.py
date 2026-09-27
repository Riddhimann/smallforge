import asyncio

from smallforge.schemas import ToolCall
from smallforge.tools import (
    ClaimVerificationTool,
    StructuredExtractionTool,
    ToolRegistry,
)


def test_extraction_tool_enforces_document_and_schema_allowlists() -> None:
    registry = ToolRegistry(
        [
            StructuredExtractionTool(
                [{"source_id": "doc-1", "text": "Invoice INV-7 total 12.50"}],
                {"invoice": {"type": "object", "required": ["invoice_id"]}},
            )
        ]
    )
    accepted = asyncio.run(
        registry.execute(
            ToolCall(
                step_id="one",
                name="extract_structured",
                arguments={"document_id": "doc-1", "schema_name": "invoice"},
            )
        )
    )
    rejected = asyncio.run(
        registry.execute(
            ToolCall(
                step_id="two",
                name="extract_structured",
                arguments={"document_id": "missing", "schema_name": "invoice"},
            )
        )
    )

    assert accepted.success
    assert accepted.evidence[0].source_id == "doc-1"
    assert not rejected.success
    assert rejected.error_code == "unknown_document"


def test_claim_tool_returns_only_requested_allowlisted_evidence() -> None:
    registry = ToolRegistry(
        [
            ClaimVerificationTool(
                [
                    {"source_id": "a", "text": "Deadline is May 1."},
                    {"source_id": "b", "text": "Deadline is May 8."},
                ]
            )
        ]
    )
    result = asyncio.run(
        registry.execute(
            ToolCall(
                step_id="verify",
                name="verify_claim",
                arguments={"claim": "Deadline is May 1.", "evidence_ids": ["a", "b"]},
            )
        )
    )

    assert result.success
    assert [item.source_id for item in result.evidence] == ["a", "b"]
