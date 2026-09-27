from pathlib import Path

from smallforge.experiments import ExperimentStore


def test_experiment_lifecycle(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "runs.db")
    created = store.create(
        {
            "name": "smoke",
            "model_id": "test:scripted",
            "benchmark_id": "core-v1",
            "modes": ["baseline", "full"],
        }
    )

    assert created["status"] == "pending"
    store.mark_running(created["id"])
    assert store.get(created["id"])["status"] == "running"  # type: ignore[index]
    store.set_progress(created["id"], {"completed": 1, "total": 4, "percent": 25})
    assert store.get(created["id"])["progress"]["percent"] == 25  # type: ignore[index]

    store.mark_completed(created["id"], {"summary": {"accuracy": 1.0}})
    completed = store.get(created["id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["results"]["summary"]["accuracy"] == 1.0
    assert store.list()[0]["id"] == created["id"]
