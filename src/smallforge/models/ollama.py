"""Async adapter for Ollama's local chat API."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from smallforge.models.base import ModelProviderError
from smallforge.schemas import ModelRequest, ModelResponse, TokenUsage


class OllamaAdapter:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        self._model = model.strip()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.response_schema is not None:
            payload["format"] = _ollama_compatible_schema(request.response_schema)

        started = perf_counter()
        try:
            response = await self._client.post("api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message.content was not text")
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip().replace("\n", " ")[:500]
            raise ModelProviderError(
                f"Ollama request failed for model {self._model!r}: "
                f"HTTP {exc.response.status_code}{': ' + detail if detail else ''}"
            ) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                f"Ollama request failed for model {self._model!r}: {type(exc).__name__}"
            ) from exc

        return ModelResponse(
            text=content,
            model=str(body.get("model") or self._model),
            latency_ms=(perf_counter() - started) * 1_000,
            usage=TokenUsage(
                prompt_tokens=max(0, int(body.get("prompt_eval_count") or 0)),
                completion_tokens=max(0, int(body.get("eval_count") or 0)),
            ),
            finish_reason=str(body.get("done_reason")) if body.get("done_reason") else None,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaAdapter:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _ollama_compatible_schema(value: Any) -> Any:
    """Remove annotation and large-bound keywords before grammar compilation.

    Some llama.cpp/Ollama runner combinations compile bounds such as a 30,000
    character `maxLength` into a prohibitively large grammar. Full Pydantic
    validation still runs after generation, so provider-side simplification does
    not weaken the application contract.
    """

    omitted = {
        "title",
        "description",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "pattern",
    }
    if isinstance(value, dict):
        return {
            key: _ollama_compatible_schema(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_ollama_compatible_schema(item) for item in value]
    return value
