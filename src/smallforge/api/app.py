from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from smallforge import __version__
from smallforge.api.contracts import BenchmarkDescriptor, ExperimentRequest
from smallforge.config import Settings
from smallforge.evaluation import benchmark_registry
from smallforge.experiments import ExperimentRunner, ExperimentStore
from smallforge.models.registry import ModelRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    configured.ensure_directories()
    store = ExperimentStore(configured.data_dir / "experiments.db")
    models = ModelRegistry(configured)
    runner = ExperimentRunner(
        store,
        models,
        configured.results_dir,
        max_concurrent=configured.max_concurrent_experiments,
    )

    application = FastAPI(
        title="SmallForge Agent Reliability Lab",
        description="Run provenance-labelled, paired small-model harness experiments.",
        version=__version__,
    )
    application.state.settings = configured
    application.state.store = store
    application.state.models = models
    application.state.runner = runner
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5174", "http://localhost:5174"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    @application.middleware("http")
    async def request_id(request: Request, call_next) -> Response:
        identifier = request.headers.get("x-request-id") or uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = identifier
        return response

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        details = await models.health_details()
        return {
            "status": "ok" if details["available_model_count"] else "degraded",
            "version": __version__,
            "active_experiments": runner.active_count,
            "storage": {"status": "ok", "backend": "sqlite", "path": str(store.database_path)},
            **details,
        }

    @application.get("/api/models")
    async def list_models() -> dict[str, object]:
        items = await models.list_models()
        items.sort(key=lambda item: (item.id != configured.default_model, item.name.lower()))
        return {"items": [item.model_dump(mode="json") for item in items]}

    @application.get("/api/benchmarks")
    async def list_benchmarks() -> dict[str, object]:
        return {"items": [item.model_dump(mode="json") for item in _benchmarks()]}

    @application.get("/api/experiments")
    async def list_experiments(limit: int = 50) -> dict[str, object]:
        return {"items": store.list(limit)}

    @application.post(
        "/api/experiments",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_experiment(payload: ExperimentRequest) -> dict[str, object]:
        model_by_id = {item.id: item for item in await models.list_models()}
        selected = model_by_id.get(payload.model_id)
        if selected is None or not selected.available:
            raise HTTPException(status_code=422, detail="Select an available generation model.")
        benchmark_ids = {item.id for item in _benchmarks()}
        if payload.benchmark_id not in benchmark_ids:
            raise HTTPException(status_code=422, detail="Unknown benchmark version.")
        return runner.submit(payload)

    @application.get("/api/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict[str, object]:
        record = store.get(experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        return record

    frontend = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend.is_dir():
        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:

        @application.get("/")
        async def index() -> dict[str, str]:
            return {
                "name": "SmallForge Agent Reliability Lab",
                "status": "API ready; run the Vite frontend on port 5174.",
                "docs": "/docs",
            }

    return application


def _benchmarks() -> list[BenchmarkDescriptor]:
    descriptors = []
    for item in benchmark_registry():
        categories = item.get("categories", {})
        category_names = (
            list(categories) if isinstance(categories, dict) else list(categories or [])
        )
        descriptors.append(
            BenchmarkDescriptor(
                id=str(item["directory"]),
                name=str(item["name"]),
                description=str(item["description"]),
                task_count=int(item["task_count"]),
                categories=category_names,
                provenance="synthetic_fixture" if item["synthetic"] else "curated",
            )
        )
    return descriptors


app = create_app()
