"""Adapter for OpenAI-compatible chat-completions endpoints.

It intentionally depends only on httpx so the core can talk to local gateways
(vLLM, llama.cpp, LiteLLM) without pulling in a provider SDK.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from smallforge.models.base import ModelProviderError
from smallforge.schemas import ModelRequest, ModelResponse, TokenUsage


class OpenAICompatibleAdapter:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        if not base_url.strip() and client is None:
            raise ValueError("base_url cannot be blank")
        self._model = model.strip()
        self._owns_client = client is None
        headers = dict(extra_headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "smallforge_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        started = perf_counter()
        try:
            response = await self._client.post(
                "chat/completions", json=payload, headers=self._headers or None
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("choices[0].message.content was not text")
            usage = body.get("usage") or {}
        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError(
                f"OpenAI-compatible request failed for model {self._model!r}: {type(exc).__name__}"
            ) from exc

        return ModelResponse(
            text=content,
            model=str(body.get("model") or self._model),
            latency_ms=(perf_counter() - started) * 1_000,
            usage=TokenUsage(
                prompt_tokens=max(0, int(usage.get("prompt_tokens") or 0)),
                completion_tokens=max(0, int(usage.get("completion_tokens") or 0)),
            ),
            finish_reason=(
                str(choice.get("finish_reason")) if choice.get("finish_reason") else None
            ),
            provider_request_id=str(body.get("id")) if body.get("id") else None,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleAdapter:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
