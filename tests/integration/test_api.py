from pathlib import Path

import httpx
import pytest

from smallforge.api.app import create_app
from smallforge.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        ollama_url="http://127.0.0.1:1",
        default_model="ollama:not-installed",
        openai_base_url=None,
        openai_api_key=None,
        openai_model=None,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        max_concurrent_experiments=1,
    )


@pytest.mark.asyncio
async def test_api_lists_versioned_benchmark_and_empty_experiment_registry(
    tmp_path: Path,
) -> None:
    transport = httpx.ASGITransport(app=create_app(_settings(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        benchmarks = await client.get("/api/benchmarks")
        experiments = await client.get("/api/experiments")

    assert benchmarks.status_code == 200
    assert benchmarks.json()["items"][0]["id"] == "v1"
    assert benchmarks.json()["items"][0]["provenance"] == "synthetic_fixture"
    assert experiments.status_code == 200
    assert experiments.json() == {"items": []}


@pytest.mark.asyncio
async def test_api_returns_structured_not_found_response(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(_settings(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/experiments/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment was not found."
    assert response.headers["x-request-id"]

