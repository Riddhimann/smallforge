"""Task-local document access tools used by extraction and claim checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smallforge.schemas import Evidence
from smallforge.tools.base import BaseTool, ToolExecutionError, ToolPayload
from smallforge.tools.retrieval import RetrievalDocument


class StructuredExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=256)
    schema_name: str = Field(min_length=1, max_length=100)


class StructuredExtractionTool(BaseTool):
    """Expose one allowlisted document and its output schema to synthesis.

    The language model still performs semantic extraction. The tool enforces the
    document/schema boundary and makes the source available as citable evidence.
    """

    name = "extract_structured"
    description = (
        "Load one supplied document and its named JSON output schema for typed extraction."
    )
    input_model = StructuredExtractionInput

    def __init__(
        self,
        documents: Iterable[RetrievalDocument | dict[str, object]],
        schemas: Mapping[str, Mapping[str, Any]],
    ) -> None:
        parsed = [RetrievalDocument.model_validate(item) for item in documents]
        self._documents = {document.source_id: document for document in parsed}
        if len(self._documents) != len(parsed):
            raise ValueError("document source_ids must be unique")
        self._schemas = {str(name): dict(schema) for name, schema in schemas.items()}
        if not self._schemas:
            raise ValueError("at least one extraction schema is required")

    async def run(self, arguments: BaseModel) -> ToolPayload:
        parsed = StructuredExtractionInput.model_validate(arguments)
        document = self._documents.get(parsed.document_id)
        if document is None:
            raise ToolExecutionError("unknown_document", "The requested document is not available.")
        schema = self._schemas.get(parsed.schema_name)
        if schema is None:
            raise ToolExecutionError(
                "unknown_schema", "The requested output schema is not available."
            )
        return ToolPayload(
            output={
                "document_id": document.source_id,
                "document_text": document.text,
                "schema_name": parsed.schema_name,
                "output_schema": schema,
            },
            evidence=[
                Evidence(
                    source_id=document.source_id,
                    content=document.text,
                    score=1.0,
                    metadata={**document.metadata, "kind": "extraction_source"},
                )
            ],
        )


class ClaimVerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=5_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class ClaimVerificationTool(BaseTool):
    """Fetch a bounded set of evidence for an explicit claim.

    The tool does not manufacture a verdict. It returns authoritative task-local
    evidence, leaving the observable synthesis step to label pass/fail/insufficient.
    """

    name = "verify_claim"
    description = (
        "Inspect a claim against selected supplied evidence. Returns the evidence; "
        "the final answer must label pass, fail, or insufficient."
    )
    input_model = ClaimVerificationInput

    def __init__(self, documents: Iterable[RetrievalDocument | dict[str, object]]) -> None:
        parsed = [RetrievalDocument.model_validate(item) for item in documents]
        self._documents = {document.source_id: document for document in parsed}
        if len(self._documents) != len(parsed):
            raise ValueError("document source_ids must be unique")

    async def run(self, arguments: BaseModel) -> ToolPayload:
        parsed = ClaimVerificationInput.model_validate(arguments)
        if len(parsed.evidence_ids) != len(set(parsed.evidence_ids)):
            raise ToolExecutionError("duplicate_evidence", "Evidence identifiers must be unique.")
        unknown = [item for item in parsed.evidence_ids if item not in self._documents]
        if unknown:
            raise ToolExecutionError(
                "unknown_evidence",
                "One or more requested evidence identifiers are not available.",
            )
        evidence = [
            Evidence(
                source_id=source_id,
                content=self._documents[source_id].text,
                score=1.0,
                metadata={**self._documents[source_id].metadata, "kind": "claim_evidence"},
            )
            for source_id in parsed.evidence_ids
        ]
        return ToolPayload(
            output={
                "claim": parsed.claim,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "instruction": "Return pass, fail, or insufficient using only this evidence.",
            },
            evidence=evidence,
        )
