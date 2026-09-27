from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from smallforge.api.contracts import ExperimentRequest
from smallforge.evaluation import (
    CandidateResult,
    Claim,
    ToolCall,
    compare_runs,
    evaluate_predictions,
    load_benchmark,
    write_ablation_report,
    write_report,
)
from smallforge.evaluation.loader import BenchmarkSuite, BenchmarkTask
from smallforge.harness import AgentHarness, BaselineRunner
from smallforge.models import ModelAdapter
from smallforge.models.registry import ModelRegistry
from smallforge.schemas import ChatMessage, ModelRequest, RunResult
from smallforge.tools import (
    CalculatorTool,
    ClaimVerificationTool,
    LexicalRetrievalTool,
    ReadOnlySQLiteTool,
    RetrievalDocument,
    StructuredExtractionTool,
    ToolRegistry,
)

from .store import ExperimentStore


class DirectAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=30_000)
    citations: list[str] = Field(default_factory=list, max_length=64)
    abstained: bool = False
    verification: Literal["pass", "fail", "insufficient"] | None = None


@dataclass(slots=True)
class TaskObservation:
    candidate: CandidateResult
    record: dict[str, Any]


class ExperimentRunner:
    """Runs immutable experiment requests and persists progress/results."""

    def __init__(
        self,
        store: ExperimentStore,
        model_registry: ModelRegistry,
        results_dir: Path,
        *,
        max_concurrent: int = 1,
    ) -> None:
        self.store = store
        self.model_registry = model_registry
        self.results_dir = results_dir
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    def submit(self, request: ExperimentRequest) -> dict[str, Any]:
        record = self.store.create(request.model_dump(mode="json"))
        task = asyncio.create_task(self._execute_safely(record["id"], request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def run_now(self, request: ExperimentRequest) -> dict[str, Any]:
        record = self.store.create(request.model_dump(mode="json"))
        await self._execute_safely(record["id"], request)
        completed = self.store.get(record["id"])
        assert completed is not None
        return completed

    async def _execute_safely(self, experiment_id: str, request: ExperimentRequest) -> None:
        async with self._semaphore:
            self.store.mark_running(experiment_id)
            try:
                results = await self._execute(experiment_id, request)
                self.store.mark_completed(experiment_id, results)
            except Exception as exc:
                self.store.mark_failed(
                    experiment_id,
                    f"Experiment failed safely ({type(exc).__name__}): {str(exc)[:1000]}",
                )

    async def _execute(self, experiment_id: str, request: ExperimentRequest) -> dict[str, Any]:
        original_suite = load_benchmark(request.benchmark_id)
        tasks = (
            original_suite.tasks[: request.max_tasks] if request.max_tasks else original_suite.tasks
        )
        suite = replace(original_suite, tasks=tuple(tasks))
        adapter = await self.model_registry.get_adapter(request.model_id)
        total = len(tasks) * len(request.modes)
        completed = 0
        runs = []
        records: list[dict[str, Any]] = []
        try:
            for mode in request.modes:
                predictions: list[CandidateResult] = []
                mode_records: list[dict[str, Any]] = []
                for task in tasks:
                    self.store.set_progress(
                        experiment_id,
                        {
                            "completed": completed,
                            "total": total,
                            "percent": round(100 * completed / max(total, 1), 1),
                            "current_task": task.id,
                            "current_mode": mode,
                        },
                    )
                    observation = await self._run_task(adapter, task, mode)
                    predictions.append(observation.candidate)
                    mode_records.append(observation.record)
                    completed += 1

                run = evaluate_predictions(
                    suite,
                    predictions,
                    system_name=request.model_id,
                    variant=mode,
                    run_id=f"{experiment_id}-{mode}",
                    metadata={
                        "inference_provenance": "real_model",
                        "benchmark_provenance": (
                            "synthetic" if suite.synthetic else "curated_real_world"
                        ),
                        "seed": 42,
                    },
                    synthetic=False,
                )
                score_by_task = {score.task_id: score for score in run.scores}
                for record in mode_records:
                    record["score"] = score_by_task[record["task_id"]].to_dict()
                records.extend(mode_records)
                runs.append(run)
                write_report(run, self.results_dir, stem=f"{experiment_id}-{mode}")

            self.store.set_progress(
                experiment_id,
                {"completed": total, "total": total, "percent": 100, "current_task": None},
            )
            return self._result_payload(experiment_id, request, suite, runs, records)
        finally:
            close = getattr(adapter, "aclose", None)
            if callable(close):
                await close()

    async def _run_task(
        self,
        adapter: ModelAdapter,
        task: BenchmarkTask,
        mode: str,
    ) -> TaskObservation:
        if mode == "baseline":
            rendered = _render_task(task, include_resources=False)
            result = await BaselineRunner(
                adapter,
                system_prompt=(
                    "Complete the benchmark task directly. Use only facts in the task and supplied "
                    "resources. Do not claim to call tools. Follow the requested answer format."
                ),
            ).run(rendered)
            return _from_harness_result(task, mode, result)

        rendered = _render_task(task, include_resources=True)
        if mode == "structured":
            return await _run_structured(adapter, task, rendered)

        allowed = _allowed_tools_for_mode(task, mode)
        registry, closeables = _tool_registry(task, allowed)
        if not registry.names:
            return await _run_structured(adapter, task, rendered, displayed_mode=mode)
        try:
            if mode in {"tools", "retrieval"}:
                max_retries, threshold = 0, 0.0
            elif mode == "verified":
                max_retries, threshold = 0, 0.75
            else:
                max_retries, threshold = 2, 0.75
            result = await AgentHarness(
                adapter,
                registry,
                max_steps=4,
                max_retries=max_retries,
                verification_threshold=threshold,
                seed=42,
            ).run(rendered)
            return _from_harness_result(task, mode, result)
        finally:
            for closeable in closeables:
                closeable.close()

    def _result_payload(
        self,
        experiment_id: str,
        request: ExperimentRequest,
        suite: BenchmarkSuite,
        runs: list,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_variant = {run.variant: run for run in runs}
        baseline = by_variant.get("baseline")
        candidates = [run for run in runs if run.variant != "baseline"]
        comparison = None
        flat_comparisons: list[dict[str, Any]] = []
        if baseline is not None and candidates:
            comparison = compare_runs(
                baseline,
                candidates,
                notes=(
                    "Same model, seed, benchmark tasks, and host were used for paired variants.",
                    "The benchmark data are synthetic; inference was performed by the configured real model.",
                ),
            )
            write_ablation_report(
                comparison,
                self.results_dir,
                stem=f"{experiment_id}-comparison",
            )
            for variant in comparison.comparisons:
                for metric, delta in variant.deltas.items():
                    flat_comparisons.append(
                        {
                            "key": metric,
                            "metric": metric,
                            "label": f"{_metric_label(metric)} · {_mode_label(variant.variant)}",
                            "variant": variant.variant,
                            "direction": "lower"
                            if metric in {"latency_ms", "retry_count"}
                            else "higher",
                            **delta.to_dict(),
                        }
                    )

        baseline_accuracy = _aggregate_value(baseline, "task_accuracy")
        best = max(
            candidates,
            key=lambda item: _aggregate_value(item, "task_accuracy") or -math.inf,
            default=None,
        )
        best_accuracy = _aggregate_value(best, "task_accuracy")
        ablations = []
        for run in runs:
            accuracy = _aggregate_value(run, "task_accuracy") or 0.0
            ablations.append(
                {
                    "id": run.variant,
                    "mode": run.variant,
                    "label": _mode_label(run.variant),
                    "score": accuracy,
                    "success_rate": accuracy,
                    "delta": (
                        accuracy - baseline_accuracy if baseline_accuracy is not None else 0.0
                    ),
                    "tasks": len(run.scores),
                    "latency_ms": _aggregate_value(run, "latency_ms"),
                }
            )

        payload = {
            "summary": {
                "model_id": request.model_id,
                "benchmark": suite.name,
                "benchmark_version": suite.version,
                "task_count": len(suite.tasks),
                "variant_count": len(runs),
                "baseline_accuracy": baseline_accuracy,
                "best_variant": best.variant if best else None,
                "best_accuracy": best_accuracy,
                "accuracy_delta": (
                    best_accuracy - baseline_accuracy
                    if best_accuracy is not None and baseline_accuracy is not None
                    else None
                ),
            },
            "comparisons": flat_comparisons,
            "ablations": ablations,
            "task_results": records,
            "runs": {run.variant: run.to_dict() for run in runs},
            "ablation": comparison.to_dict() if comparison else None,
            "provenance": {
                "inference": "real_model",
                "benchmark": "synthetic" if suite.synthetic else "curated_real_world",
                "model_id": request.model_id,
                "seed": 42,
                "report_stem": experiment_id,
            },
            "result_kind": "real_model",
            "synthetic": False,
        }
        return payload


async def _run_structured(
    adapter: ModelAdapter,
    task: BenchmarkTask,
    rendered: str,
    *,
    displayed_mode: str = "structured",
) -> TaskObservation:
    extraction_schema = task.expected.get("structured_schema")
    response_schema = extraction_schema or DirectAnswer.model_json_schema()
    started = asyncio.get_running_loop().time()
    try:
        response = await adapter.generate(
            ModelRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "Return exactly one JSON object matching the supplied schema. Use only "
                            "the task resources. Do not fabricate missing facts."
                        ),
                    ),
                    ChatMessage(role="user", content=rendered),
                ],
                response_schema=dict(response_schema),
                temperature=0.0,
                max_tokens=1_500,
                seed=42,
                metadata={"stage": "structured", "task_id": task.id},
            )
        )
        parsed = _first_json_value(response.text)
        if extraction_schema:
            answer_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            structured_output = parsed
            citations: list[str] = []
            abstained = False
            verification = None
        else:
            envelope = DirectAnswer.model_validate(parsed)
            answer_text = envelope.answer
            structured_output = None
            citations = envelope.citations
            abstained = envelope.abstained or _looks_like_abstention(answer_text)
            verification = envelope.verification or _verification_label(answer_text)
        elapsed = (asyncio.get_running_loop().time() - started) * 1_000
        candidate = CandidateResult(
            task_id=task.id,
            answer=answer_text,
            structured_output=structured_output,
            evidence_ids=tuple(citations),
            claims=(Claim(text=answer_text, evidence_ids=tuple(citations)),) if citations else (),
            abstained=abstained,
            verification=verification,
            latency_ms=elapsed,
            metadata={
                "model": response.model,
                "model_calls": 1,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "schema_valid": True,
            },
        )
        record = {
            "task_id": task.id,
            "task": task.prompt,
            "category": task.category,
            "mode": displayed_mode,
            "success": True,
            "answer": answer_text,
            "evidence": list(citations),
            "verification": verification,
            "metrics": dict(candidate.metadata),
            "trace": [
                {
                    "step": 0,
                    "stage": "structured_generation",
                    "status": "succeeded",
                    "message": "Provider-native JSON schema output validated.",
                    "duration_ms": elapsed,
                }
            ],
            "retries": 0,
            "error": None,
        }
        return TaskObservation(candidate, record)
    except Exception as exc:
        elapsed = (asyncio.get_running_loop().time() - started) * 1_000
        safe_error = f"Structured generation failed ({type(exc).__name__})."
        return TaskObservation(
            CandidateResult(task_id=task.id, latency_ms=elapsed, error=safe_error),
            {
                "task_id": task.id,
                "task": task.prompt,
                "category": task.category,
                "mode": displayed_mode,
                "success": False,
                "answer": "",
                "evidence": [],
                "verification": None,
                "metrics": {"latency_ms": elapsed, "schema_valid": False},
                "trace": [
                    {
                        "step": 0,
                        "stage": "structured_generation",
                        "status": "failed",
                        "message": safe_error,
                        "duration_ms": elapsed,
                    }
                ],
                "retries": 0,
                "error": safe_error,
            },
        )


def _from_harness_result(
    task: BenchmarkTask,
    displayed_mode: str,
    result: RunResult,
) -> TaskObservation:
    answer_text = result.answer.answer if result.answer else ""
    citations = tuple(result.answer.citations) if result.answer else ()
    structured_output = _maybe_json(answer_text) if task.expected.get("structured_schema") else None
    semantic_verification = _verification_label(answer_text)
    tool_calls = tuple(
        ToolCall(name=step.tool, arguments=step.arguments)
        for step in (result.plan.steps if result.plan else [])
    )
    candidate = CandidateResult(
        task_id=task.id,
        answer=answer_text,
        structured_output=structured_output,
        tool_calls=tool_calls,
        evidence_ids=citations,
        claims=(Claim(text=answer_text, evidence_ids=citations),) if citations else (),
        abstained=_looks_like_abstention(answer_text),
        verification=semantic_verification,
        retry_count=result.metrics.retries,
        latency_ms=result.metrics.latency_ms,
        error=result.error,
        metadata={
            "model": result.model,
            "run_success": result.success,
            **result.metrics.model_dump(mode="json"),
        },
    )
    record = {
        "task_id": task.id,
        "task": task.prompt,
        "category": task.category,
        "mode": displayed_mode,
        "success": result.success,
        "answer": answer_text,
        "evidence": [item.model_dump(mode="json") for item in result.evidence],
        "verification": result.verification.model_dump(mode="json")
        if result.verification
        else semantic_verification,
        "metrics": result.metrics.model_dump(mode="json"),
        "trace": result.trace.model_dump(mode="json")["events"],
        "tool_calls": [
            {"name": call.name, "arguments": dict(call.arguments)} for call in tool_calls
        ],
        "retries": result.metrics.retries,
        "error": result.error,
    }
    return TaskObservation(candidate, record)


def _tool_registry(
    task: BenchmarkTask,
    allowed_names: set[str],
) -> tuple[ToolRegistry, list[ReadOnlySQLiteTool]]:
    tools = []
    closeables: list[ReadOnlySQLiteTool] = []
    documents = _documents(task.resources.get("documents", []))
    if "calculator" in allowed_names:
        tools.append(CalculatorTool())
    if "sql_query" in allowed_names:
        tables = _tables(task.resources)
        if tables:
            sql = ReadOnlySQLiteTool(tables)
            tools.append(sql)
            closeables.append(sql)
    if "retrieve" in allowed_names and documents:
        tools.append(LexicalRetrievalTool(documents))
    if "extract_structured" in allowed_names and documents:
        schema = task.expected.get("structured_schema")
        schema_name = _extraction_schema_name(task)
        if isinstance(schema, Mapping) and schema_name:
            tools.append(StructuredExtractionTool(documents, {schema_name: schema}))
    if "verify_claim" in allowed_names and documents:
        tools.append(ClaimVerificationTool(documents))
    return ToolRegistry(tools), closeables


def _allowed_tools_for_mode(task: BenchmarkTask, mode: str) -> set[str]:
    advertised = {str(item.get("name")) for item in task.available_tools}
    if mode == "tools":
        return advertised & {"calculator", "sql_query", "extract_structured"}
    if mode == "retrieval":
        return advertised & {"retrieve", "calculator"}
    return advertised


def _documents(raw: object) -> list[RetrievalDocument]:
    if not isinstance(raw, list):
        return []
    documents = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        source_id = str(item.get("id") or item.get("source_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if source_id and text:
            documents.append(
                RetrievalDocument(
                    source_id=source_id,
                    text=text,
                    metadata={"title": str(item.get("title") or source_id)},
                )
            )
    return documents


def _tables(resources: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    database = resources.get("database")
    if not isinstance(database, Mapping) or not isinstance(database.get("tables"), Mapping):
        return {}
    converted: dict[str, list[dict[str, Any]]] = {}
    for table_name, definition in database["tables"].items():
        if not isinstance(definition, Mapping):
            continue
        columns = definition.get("columns")
        rows = definition.get("rows")
        if not isinstance(columns, Mapping) or not isinstance(rows, list):
            continue
        names = list(columns)
        converted[str(table_name)] = [
            dict(zip(names, row, strict=True))
            for row in rows
            if isinstance(row, list) and len(row) == len(names)
        ]
    return {name: rows for name, rows in converted.items() if rows}


def _extraction_schema_name(task: BenchmarkTask) -> str | None:
    calls = task.expected.get("tool_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        if isinstance(call, Mapping) and call.get("name") == "extract_structured":
            arguments = call.get("arguments")
            if isinstance(arguments, Mapping) and isinstance(arguments.get("schema_name"), str):
                return arguments["schema_name"]
    return None


def _render_task(task: BenchmarkTask, *, include_resources: bool = True) -> str:
    payload: dict[str, Any] = {
        "task_id": task.id,
        "category": task.category,
        "instruction": task.prompt,
    }
    if include_resources:
        payload["resources"] = task.resources
    else:
        payload["resource_access"] = (
            "No retrieval or tool access. Answer only from the instruction and prior model knowledge."
        )
    schema = task.expected.get("structured_schema")
    if isinstance(schema, Mapping):
        payload["required_output_schema"] = schema
        payload["schema_name"] = _extraction_schema_name(task)
        payload["format_instruction"] = (
            "The final answer text must be a compact JSON object matching required_output_schema."
        )
    elif task.category == "verification":
        payload["format_instruction"] = (
            "Begin the final answer with exactly pass, fail, or insufficient; cite inspected source IDs."
        )
    elif task.category == "unanswerable":
        payload["format_instruction"] = (
            "If the resources do not establish the fact, explicitly state insufficient information."
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("response did not contain JSON")


def _maybe_json(text: str) -> Any:
    try:
        return _first_json_value(text)
    except ValueError:
        return None


def _looks_like_abstention(text: str) -> bool:
    normalized = text.lower()
    phrases = (
        "insufficient information",
        "insufficient evidence",
        "cannot determine",
        "can't determine",
        "not specified",
        "not establish",
        "unable to determine",
        "unknown from",
    )
    return any(phrase in normalized for phrase in phrases)


def _verification_label(text: str) -> str | None:
    normalized = re.sub(r"^[^a-z]+", "", text.lower().strip())
    if normalized.startswith("insufficient"):
        return "insufficient"
    if normalized.startswith("pass"):
        return "pass"
    if normalized.startswith("fail") or normalized.startswith("false"):
        return "fail"
    return None


def _aggregate_value(run: Any, metric: str) -> float | None:
    if run is None:
        return None
    summary = run.aggregates.get(metric)
    return float(summary.value) if summary else None


def _metric_label(metric: str) -> str:
    return {
        "task_accuracy": "Task accuracy",
        "schema_validity": "Schema validity",
        "tool_selection_accuracy": "Tool selection",
        "tool_argument_accuracy": "Tool arguments",
        "groundedness": "Groundedness",
        "citation_precision": "Citation precision",
        "citation_recall": "Citation recall",
        "abstention_accuracy": "Safe abstention",
        "verification_accuracy": "Verification accuracy",
        "retry_count": "Retry count",
        "latency_ms": "Latency",
    }.get(metric, metric.replace("_", " ").title())


def _mode_label(mode: str) -> str:
    return {
        "baseline": "Direct baseline",
        "structured": "Structured output",
        "tools": "Tool execution",
        "retrieval": "Retrieval grounded",
        "verified": "Verified response",
        "full": "Full harness",
    }.get(mode, mode.replace("_", " ").title())
