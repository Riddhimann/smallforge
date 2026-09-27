from __future__ import annotations

from typing import Any

import httpx

from smallforge.api.contracts import ModelDescriptor
from smallforge.config import Settings
from smallforge.models import ModelAdapter, OllamaAdapter, OpenAICompatibleAdapter


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def list_models(self) -> list[ModelDescriptor]:
        items = await self._ollama_models()
        if self.settings.openai_base_url and self.settings.openai_model:
            items.append(
                ModelDescriptor(
                    id=f"openai-compatible:{self.settings.openai_model}",
                    name=self.settings.openai_model,
                    provider="OpenAI-compatible",
                    available=bool(self.settings.openai_api_key),
                    details={"base_url": self.settings.openai_base_url},
                )
            )
        return items

    async def _ollama_models(self) -> list[ModelDescriptor]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.settings.ollama_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return []

        descriptors: list[ModelDescriptor] = []
        for raw in models:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or raw.get("model") or "").strip()
            if not name:
                continue
            details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
            family = str(details.get("family") or "").lower()
            # Ollama also lists embedding-only encoders in /api/tags. They cannot
            # satisfy chat requests, so keep the experiment selector generation-only.
            if family in {"bert", "nomic-bert"}:
                continue
            descriptors.append(
                ModelDescriptor(
                    id=f"ollama:{name}",
                    name=name,
                    provider="Ollama (local)",
                    available=True,
                    details={
                        "size_bytes": raw.get("size"),
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level"),
                        "family": details.get("family"),
                        "modified_at": raw.get("modified_at"),
                    },
                )
            )
        return descriptors

    async def get_adapter(self, model_id: str) -> ModelAdapter:
        if model_id.startswith("ollama:"):
            name = model_id.removeprefix("ollama:")
            available = {model.id for model in await self._ollama_models()}
            if model_id not in available:
                raise ValueError(f"Ollama model is not installed: {name}")
            return OllamaAdapter(name, base_url=self.settings.ollama_url)

        if model_id.startswith("openai-compatible:"):
            name = model_id.removeprefix("openai-compatible:")
            if not self.settings.openai_base_url or not self.settings.openai_api_key:
                raise ValueError("OpenAI-compatible provider is not configured")
            return OpenAICompatibleAdapter(
                name,
                base_url=self.settings.openai_base_url,
                api_key=self.settings.openai_api_key,
            )

        raise ValueError(f"Unknown model id: {model_id}")

    async def health_details(self) -> dict[str, Any]:
        models = await self.list_models()
        return {
            "ollama_reachable": any(model.provider == "Ollama (local)" for model in models),
            "available_model_count": sum(model.available for model in models),
        }
