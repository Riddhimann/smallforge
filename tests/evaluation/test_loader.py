from __future__ import annotations

import json

import pytest

from smallforge.evaluation import (
    BenchmarkFormatError,
    available_versions,
    benchmark_registry,
    load_benchmark,
)


def test_default_suite_is_versioned_complete_and_synthetic() -> None:
    suite = load_benchmark("v1")

    assert suite.name == "smallforge-core-synthetic"
    assert suite.version == "1.0.0"
    assert suite.synthetic is True
    assert len(suite.tasks) == 20
    assert suite.categories() == (
        "arithmetic",
        "read_only_sql",
        "retrieval",
        "structured_extraction",
        "unanswerable",
        "verification",
    )
    assert len({task.id for task in suite.tasks}) == len(suite.tasks)


def test_candidate_payload_never_leaks_expected_labels() -> None:
    task = load_benchmark("v1").tasks[0]
    payload = task.to_payload()

    assert payload["task_id"] == task.id
    assert "expected" not in payload
    assert "prompt" in payload
    assert "available_tools" in payload


def test_registry_lists_manifest_without_loading_labels() -> None:
    versions = available_versions()
    registry = benchmark_registry()

    assert "v1" in versions
    assert registry[0]["directory"] == "v1"
    assert registry[0]["task_count"] == 20
    assert registry[0]["synthetic"] is True


def test_loader_rejects_manifest_count_mismatch(tmp_path) -> None:
    suite_dir = tmp_path / "broken"
    suite_dir.mkdir()
    manifest = {
        "name": "synthetic-broken",
        "version": "1.0.0",
        "description": "Synthetic test fixture.",
        "tasks_file": "tasks.jsonl",
        "task_count": 2,
        "synthetic": True,
    }
    (suite_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    task = {
        "id": "only_one",
        "version": "1.0.0",
        "category": "arithmetic",
        "prompt": "Synthetic prompt",
        "expected": {"match": {"kind": "numeric", "value": 1}},
    }
    (suite_dir / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkFormatError, match="declares 2 tasks"):
        load_benchmark(suite_dir)
