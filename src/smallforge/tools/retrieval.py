"""Small, dependency-free BM25 lexical retrieval for harness experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from smallforge.schemas import Evidence
from smallforge.tools.base import BaseTool, ToolPayload

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


class RetrievalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=50_000)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_has_terms(cls, value: str) -> str:
        if not TOKEN_PATTERN.search(value):
            raise ValueError("query must contain at least one searchable token")
        return value


class LexicalRetrievalTool(BaseTool):
    name = "retrieve"
    description = "Retrieve the most relevant local evidence with BM25 lexical scoring."
    input_model = RetrievalInput

    def __init__(
        self,
        documents: Iterable[RetrievalDocument | dict[str, object]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = [RetrievalDocument.model_validate(item) for item in documents]
        source_ids = [document.source_id for document in self.documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("retrieval source_ids must be unique")
        self.k1 = k1
        self.b = b
        self._tokens = [self._tokenize(document.text) for document in self.documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for terms in self._term_frequencies:
            self._document_frequency.update(terms.keys())
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 1.0
        )

    async def run(self, arguments: BaseModel) -> ToolPayload:
        parsed = RetrievalInput.model_validate(arguments)
        query_terms = self._tokenize(parsed.query)
        scored: list[tuple[float, int]] = []
        corpus_size = len(self.documents)
        for index, frequencies in enumerate(self._term_frequencies):
            score = 0.0
            doc_length = len(self._tokens[index])
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.documents[item[1]].source_id))

        evidence = [
            Evidence(
                source_id=self.documents[index].source_id,
                content=self.documents[index].text,
                score=round(score, 6),
                metadata={**self.documents[index].metadata, "rank": rank},
            )
            for rank, (score, index) in enumerate(scored[: parsed.top_k], start=1)
        ]
        return ToolPayload(
            output={
                "query": parsed.query,
                "matches": [item.model_dump() for item in evidence],
                "match_count": len(evidence),
            },
            evidence=evidence,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
