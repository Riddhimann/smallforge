"""Provider-neutral model interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from smallforge.schemas import ModelRequest, ModelResponse


class ModelError(RuntimeError):
    """Base error raised by model adapters."""


class ModelConfigurationError(ModelError):
    """The adapter was configured with invalid or missing settings."""


class ModelProviderError(ModelError):
    """The provider rejected a request or returned an invalid response."""


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal interface shared by baseline and harnessed execution."""

    @property
    def model_name(self) -> str:
        """Stable model identifier used in experiment records."""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response for a validated request."""
