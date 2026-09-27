"""Language-model adapters."""

from .base import ModelAdapter, ModelConfigurationError, ModelError, ModelProviderError
from .ollama import OllamaAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .scripted_test import ScriptedTestAdapter

__all__ = [
    "ModelAdapter",
    "ModelConfigurationError",
    "ModelError",
    "ModelProviderError",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "ScriptedTestAdapter",
]
