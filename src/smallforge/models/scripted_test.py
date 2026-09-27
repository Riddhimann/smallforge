"""Deterministic adapter for unit tests only.

This adapter is deliberately named and documented as a test double.  It must
not be presented as experimental evidence about a real language model.
"""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeAlias

from smallforge.models.base import ModelProviderError
from smallforge.schemas import ModelRequest, ModelResponse, TokenUsage

ScriptValue: TypeAlias = str | ModelResponse | Exception
ResponseFactory: TypeAlias = Callable[[ModelRequest], ScriptValue | Awaitable[ScriptValue]]


class ScriptedTestAdapter:
    """A queue/callback-driven fake; never use it for benchmark claims."""

    def __init__(
        self,
        responses: Iterable[ScriptValue] | None = None,
        *,
        response_factory: ResponseFactory | None = None,
        model_name: str = "scripted-test-double",
    ) -> None:
        if responses is not None and response_factory is not None:
            raise ValueError("provide responses or response_factory, not both")
        self._responses = deque(responses or [])
        self._factory = response_factory
        self._model_name = model_name
        self.requests: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if self._factory is not None:
            value = self._factory(request)
            if inspect.isawaitable(value):
                value = await value
        elif self._responses:
            value = self._responses.popleft()
        else:
            raise ModelProviderError("scripted test adapter has no response remaining")

        if isinstance(value, Exception):
            raise value
        if isinstance(value, ModelResponse):
            return value
        return ModelResponse(
            text=value,
            model=self._model_name,
            usage=TokenUsage(),
        )
