from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from smallforge.api.contracts import ExperimentRequest
from smallforge.config import Settings
from smallforge.evaluation import benchmark_registry
from smallforge.experiments import ExperimentRunner, ExperimentStore
from smallforge.models.registry import ModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smallforge",
        description="SmallForge small-model reliability experiments",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="Check providers, models, storage, and benchmarks")

    serve = subcommands.add_parser("serve", help="Start the API and built dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    run = subcommands.add_parser("run", help="Run an experiment synchronously")
    run.add_argument("--name", default="CLI paired experiment")
    run.add_argument("--model", default=None, help="Model ID returned by `smallforge doctor`")
    run.add_argument("--benchmark", default="v1")
    run.add_argument("--modes", default="baseline,full")
    run.add_argument("--max-tasks", type=int, default=None)
    return parser


async def _doctor(settings: Settings) -> int:
    settings.ensure_directories()
    registry = ModelRegistry(settings)
    models = await registry.list_models()
    print("SmallForge doctor")
    print(f"  data:       {settings.data_dir}")
    print(f"  results:    {settings.results_dir}")
    print(
        f"  benchmarks: {', '.join(item['directory'] for item in benchmark_registry()) or 'none'}"
    )
    if not models:
        print("  models:     none (start Ollama or configure an OpenAI-compatible endpoint)")
        return 1
    print("  models:")
    for model in models:
        marker = "*" if model.id == settings.default_model else "-"
        size = model.details.get("parameter_size") if isinstance(model.details, dict) else None
        print(f"    {marker} {model.id} ({size or 'size unknown'})")
    return 0


async def _run(settings: Settings, args: argparse.Namespace) -> int:
    settings.ensure_directories()
    request = ExperimentRequest(
        name=args.name,
        model_id=args.model or settings.default_model,
        benchmark_id=args.benchmark,
        modes=[part.strip() for part in args.modes.split(",") if part.strip()],
        max_tasks=args.max_tasks,
    )
    runner = ExperimentRunner(
        ExperimentStore(settings.data_dir / "experiments.db"),
        ModelRegistry(settings),
        settings.results_dir,
        max_concurrent=settings.max_concurrent_experiments,
    )
    record = await runner.run_now(request)
    if record["status"] != "completed":
        print(record["error"])
        return 1
    print(json.dumps(record["results"]["summary"], indent=2))
    print(f"Reports written to {settings.results_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "doctor":
        return asyncio.run(_doctor(settings))
    if args.command == "run":
        return asyncio.run(_run(settings, args))
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "smallforge.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
