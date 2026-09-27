from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ExperimentStore:
    """Small, durable experiment registry.

    SQLite makes local runs inspectable and restart-safe without adding an external
    service. Each record stores its immutable request and versioned result payload.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress_json TEXT
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "progress_json" not in columns:
                connection.execute("ALTER TABLE experiments ADD COLUMN progress_json TEXT")

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        experiment_id = uuid4().hex[:12]
        created_at = utc_now()
        name = str(request.get("name") or f"Experiment {created_at[:19]}")
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments
                    (id, name, status, request_json, created_at)
                VALUES (?, ?, 'pending', ?, ?)
                """,
                (experiment_id, name, json.dumps(request, sort_keys=True), created_at),
            )
        record = self.get(experiment_id)
        assert record is not None
        return record

    def mark_running(self, experiment_id: str) -> None:
        self._update(
            experiment_id,
            status="running",
            started_at=utc_now(),
            error=None,
        )

    def mark_completed(self, experiment_id: str, result: dict[str, Any]) -> None:
        self._update(
            experiment_id,
            status="completed",
            result_json=json.dumps(result, sort_keys=True),
            completed_at=utc_now(),
            error=None,
        )

    def mark_failed(self, experiment_id: str, error: str) -> None:
        self._update(
            experiment_id,
            status="failed",
            error=error[:4000],
            completed_at=utc_now(),
        )

    def set_progress(self, experiment_id: str, progress: dict[str, Any]) -> None:
        self._update(experiment_id, progress_json=json.dumps(progress, sort_keys=True))

    def _update(self, experiment_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "result_json",
            "error",
            "started_at",
            "completed_at",
            "progress_json",
        }
        if not values or not set(values).issubset(allowed):
            raise ValueError("Unsupported experiment update")
        assignments = ", ".join(f"{column} = ?" for column in values)
        parameters = [*values.values(), experiment_id]
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE experiments SET {assignments} WHERE id = ?",  # noqa: S608
                parameters,
            )
            if cursor.rowcount != 1:
                raise KeyError(experiment_id)

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return self._serialise(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 200))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [self._serialise(row) for row in rows]

    @staticmethod
    def _serialise(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "results": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
        }
