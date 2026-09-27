"""Internal model-call accounting shared by harness stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from smallforge.models import ModelAdapter
from smallforge.schemas import ModelRequest, ModelResponse


@dataclass(slots=True)
class ModelCallLedger:
    model: ModelAdapter
    responses: list[ModelResponse] = field(default_factory=list)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.model.generate(request)
        self.responses.append(response)
        return response

    @property
    def prompt_tokens(self) -> int:
        return sum(response.usage.prompt_tokens for response in self.responses)

    @property
    def completion_tokens(self) -> int:
        return sum(response.usage.completion_tokens for response in self.responses)
