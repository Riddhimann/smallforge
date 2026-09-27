"""CLI for scoring captured outputs and comparing ablation runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evaluator import compare_runs, evaluate_predictions
from .loader import benchmark_registry, load_benchmark
from .report import load_run, write_ablation_report, write_report


def _load_predictions(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        results: list[Mapping[str, Any]] = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Prediction line {line_number} must be a JSON object")
            results.append(value)
        return results
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and "predictions" in value:
        value = value["predictions"]
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("Prediction JSON must be an array or an object with a predictions array")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SmallForge candidate outputs")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List installed benchmark versions")
    score = commands.add_parser("score", help="Score a JSON/JSONL predictions file")
    score.add_argument("--benchmark", default="v1")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--system-name", required=True)
    score.add_argument("--variant", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument(
        "--synthetic-fixture-results",
        action="store_true",
        help="Mark outputs as synthetic fixtures rather than model results",
    )
    score.add_argument("--allow-missing", action="store_true")
    compare = commands.add_parser(
        "compare", help="Compare candidate run JSON files with a baseline"
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, action="append", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print(json.dumps(benchmark_registry(), indent=2))
        return 0
    if args.command == "score":
        suite = load_benchmark(args.benchmark)
        run = evaluate_predictions(
            suite,
            _load_predictions(args.predictions),
            system_name=args.system_name,
            variant=args.variant,
            synthetic=args.synthetic_fixture_results,
            strict=not args.allow_missing,
        )
        json_path, markdown_path = write_report(run, args.output_dir)
        print(json.dumps({"json_report": str(json_path), "markdown_report": str(markdown_path)}))
        return 0
    baseline = load_run(args.baseline)
    candidates = [load_run(path) for path in args.candidate]
    comparison = compare_runs(baseline, candidates)
    json_path, markdown_path = write_ablation_report(comparison, args.output_dir)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
