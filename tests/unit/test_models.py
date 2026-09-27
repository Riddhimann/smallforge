from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from smallforge.models import (
    ModelProviderError,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ScriptedTestAdapter,
)
from smallforge.schemas import ChatMessage, ModelRequest


def request() -> ModelRequest:
    return ModelRequest(
        messages=[ChatMessage(role="user", content="hello")],
        response_schema={
            "type": "object",
            "title": "Large annotated schema",
            "properties": {"answer": {"type": "string", "maxLength": 30_000}},
        },
        seed=42,
    )


def test_ollama_adapter_maps_schema_seed_and_usage() -> None:
    seen: dict[str, object] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["path"] = http_request.url.path
        seen["payload"] = json.loads(http_request.content)
        return httpx.Response(
            200,
            json={
                "model": "tiny:latest",
                "message": {"role": "assistant", "content": '{"answer":"hi"}'},
                "prompt_eval_count": 12,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test/"
    )
    adapter = OllamaAdapter("tiny:latest", client=client)
    response = asyncio.run(adapter.generate(request()))
    asyncio.run(client.aclose())

    assert seen["path"] == "/api/chat"
    assert seen["payload"]["options"]["seed"] == 42  # type: ignore[index]
    assert seen["payload"]["format"]["type"] == "object"  # type: ignore[index]
    assert "title" not in seen["payload"]["format"]  # type: ignore[operator,index]
    assert "maxLength" not in seen["payload"]["format"]["properties"]["answer"]  # type: ignore[index]
    assert response.usage.total_tokens == 17


def test_openai_compatible_adapter_preserves_v1_path_and_structured_format() -> None:
    seen: dict[str, object] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["path"] = http_request.url.path
        seen["authorization"] = http_request.headers.get("authorization")
        seen["payload"] = json.loads(http_request.content)
        return httpx.Response(
            200,
            json={
                "id": "req-1",
                "model": "small-instruct",
                "choices": [{"message": {"content": '{"answer":"hi"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://local.test/v1/"
    )
    adapter = OpenAICompatibleAdapter(
        "small-instruct", base_url="https://ignored.test/v1", api_key="secret", client=client
    )
    response = asyncio.run(adapter.generate(request()))
    asyncio.run(client.aclose())

    assert seen["path"] == "/v1/chat/completions"
    assert seen["authorization"] == "Bearer secret"
    assert seen["payload"]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert seen["payload"]["seed"] == 42  # type: ignore[index]
    assert response.provider_request_id == "req-1"


def test_adapter_sanitizes_malformed_provider_response() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []})),
        base_url="https://local.test/v1/",
    )
    adapter = OpenAICompatibleAdapter("small", base_url="https://local.test/v1", client=client)
    with pytest.raises(ModelProviderError, match="IndexError"):
        asyncio.run(adapter.generate(request()))
    asyncio.run(client.aclose())


def test_scripted_adapter_is_deterministic_test_double() -> None:
    adapter = ScriptedTestAdapter(["first", "second"])
    first = asyncio.run(adapter.generate(request()))
    second = asyncio.run(adapter.generate(request()))

    assert [first.text, second.text] == ["first", "second"]
    assert len(adapter.requests) == 2
