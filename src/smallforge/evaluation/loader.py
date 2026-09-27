"""Load and validate immutable, versioned SmallForge benchmark suites."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BenchmarkFormatError(ValueError):
    """Raised when benchmark data violates the public task contract."""


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    version: str
    category: str
    prompt: str
    expected: Mapping[str, Any]
    available_tools: tuple[Mapping[str, Any], ...] = ()
    resources: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    difficulty: str = "medium"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, line_number: int | None = None
    ) -> BenchmarkTask:
        where = f" on line {line_number}" if line_number is not None else ""
        required = ("id", "version", "category", "prompt", "expected")
        missing = [key for key in required if key not in value]
        if missing:
            raise BenchmarkFormatError(f"Task{where} is missing fields: {', '.join(missing)}")
        for key in ("id", "version", "category", "prompt"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise BenchmarkFormatError(f"Task field {key!r}{where} must be a non-empty string")
        if not isinstance(value["expected"], Mapping):
            raise BenchmarkFormatError(f"Task expected{where} must be an object")
        tools = value.get("available_tools", []) or []
        if not isinstance(tools, list) or any(not isinstance(tool, Mapping) for tool in tools):
            raise BenchmarkFormatError(f"Task available_tools{where} must be an array of objects")
        resources = value.get("resources", {}) or {}
        metadata = value.get("metadata", {}) or {}
        if not isinstance(resources, Mapping) or not isinstance(metadata, Mapping):
            raise BenchmarkFormatError(f"Task resources and metadata{where} must be objects")
        expected = dict(value["expected"])
        if "match" not in expected or not isinstance(expected["match"], Mapping):
            raise BenchmarkFormatError(f"Task expected.match{where} must be an object")
        if not isinstance(expected["match"].get("kind"), str):
            raise BenchmarkFormatError(f"Task expected.match.kind{where} must be a string")
        return cls(
            id=value["id"].strip(),
            version=value["version"].strip(),
            category=value["category"].strip(),
            prompt=value["prompt"].strip(),
            expected=expected,
            available_tools=tuple(dict(tool) for tool in tools),
            resources=dict(resources),
            tags=tuple(str(tag) for tag in value.get("tags", ()) or ()),
            difficulty=str(value.get("difficulty", "medium")),
            metadata=dict(metadata),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the public input passed to a candidate system (never expected labels)."""

        return {
            "task_id": self.id,
            "benchmark_version": self.version,
            "category": self.category,
            "prompt": self.prompt,
            "available_tools": [dict(tool) for tool in self.available_tools],
            "resources": dict(self.resources),
            "tags": list(self.tags),
            "difficulty": self.difficulty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    version: str
    description: str
    tasks: tuple[BenchmarkTask, ...]
    synthetic: bool
    manifest: Mapping[str, Any]
    source_dir: Path

    def by_id(self) -> dict[str, BenchmarkTask]:
        return {task.id: task for task in self.tasks}

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({task.category for task in self.tasks}))


def default_benchmark_root() -> Path:
    project_data = Path(__file__).resolve().parents[3] / "benchmarks"
    if project_data.is_dir():
        return project_data
    return Path(__file__).resolve().parents[1] / "benchmark_data"


def available_versions(root: str | Path | None = None) -> tuple[str, ...]:
    base = Path(root) if root is not None else default_benchmark_root()
    if not base.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in base.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    )


def benchmark_registry(root: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    """List lightweight manifest metadata without loading task labels."""

    base = Path(root) if root is not None else default_benchmark_root()
    entries: list[dict[str, Any]] = []
    for directory_name in available_versions(base):
        manifest_path = base / directory_name / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkFormatError(
                f"Invalid benchmark manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise BenchmarkFormatError(f"Benchmark manifest must be an object: {manifest_path}")
        entries.append(
            {
                "directory": directory_name,
                "name": manifest.get("name"),
                "version": manifest.get("version"),
                "description": manifest.get("description"),
                "task_count": manifest.get("task_count"),
                "categories": manifest.get("categories", {}),
                "synthetic": bool(manifest.get("synthetic", False)),
            }
        )
    return tuple(entries)


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkFormatError(
                        f"Invalid JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise BenchmarkFormatError(
                        f"Expected a JSON object in {path} line {line_number}"
                    )
                yield line_number, value
    except OSError as exc:
        raise BenchmarkFormatError(f"Could not read benchmark tasks at {path}: {exc}") from exc


def load_benchmark(
    version_or_path: str | Path = "v1", *, root: str | Path | None = None
) -> BenchmarkSuite:
    """Load a suite by directory name (``v1``) or explicit directory path."""

    supplied = Path(version_or_path)
    if supplied.is_dir():
        suite_dir = supplied.resolve()
    else:
        base = Path(root) if root is not None else default_benchmark_root()
        suite_dir = (base / supplied).resolve()
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BenchmarkFormatError(f"Benchmark manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkFormatError(f"Invalid benchmark manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise BenchmarkFormatError("Benchmark manifest must be a JSON object")
    required = ("name", "version", "description", "tasks_file", "task_count", "synthetic")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise BenchmarkFormatError(f"Manifest is missing fields: {', '.join(missing)}")
    tasks_path = suite_dir / str(manifest["tasks_file"])
    tasks = tuple(
        BenchmarkTask.from_dict(value, line_number=line) for line, value in _read_jsonl(tasks_path)
    )
    ids = [task.id for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise BenchmarkFormatError(f"Duplicate task IDs: {', '.join(duplicates)}")
    if len(tasks) != manifest["task_count"]:
        raise BenchmarkFormatError(
            f"Manifest declares {manifest['task_count']} tasks, but {len(tasks)} were loaded"
        )
    mismatched = [task.id for task in tasks if task.version != manifest["version"]]
    if mismatched:
        raise BenchmarkFormatError(
            "Task benchmark versions do not match the manifest: " + ", ".join(mismatched)
        )
    declared_categories = manifest.get("categories")
    if isinstance(declared_categories, Mapping):
        actual: dict[str, int] = {}
        for task in tasks:
            actual[task.category] = actual.get(task.category, 0) + 1
        expected_counts = {str(key): int(value) for key, value in declared_categories.items()}
        if actual != expected_counts:
            raise BenchmarkFormatError(
                f"Manifest category counts {expected_counts!r} do not match tasks {actual!r}"
            )
    return BenchmarkSuite(
        name=str(manifest["name"]),
        version=str(manifest["version"]),
        description=str(manifest["description"]),
        tasks=tasks,
        synthetic=bool(manifest["synthetic"]),
        manifest=dict(manifest),
        source_dir=suite_dir,
    )
