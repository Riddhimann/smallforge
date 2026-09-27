"""A read-only, resource-bounded in-memory SQLite tool."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from smallforge.schemas import Evidence
from smallforge.tools.base import BaseTool, ToolExecutionError, ToolPayload

SqlScalar: TypeAlias = str | int | float | None
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class SqlQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=5_000)
    parameters: dict[str, SqlScalar] = Field(default_factory=dict)


class ReadOnlySQLiteTool(BaseTool):
    name = "sql_query"
    description = (
        "Run one read-only SELECT/WITH query against the experiment's in-memory SQLite data."
    )
    input_model = SqlQueryInput

    def __init__(
        self,
        tables: Mapping[str, Sequence[Mapping[str, SqlScalar]]],
        *,
        max_rows: int = 100,
        max_vm_steps: int = 200_000,
    ) -> None:
        if not 1 <= max_rows <= 1_000:
            raise ValueError("max_rows must be between 1 and 1000")
        self.max_rows = max_rows
        self.max_vm_steps = max_vm_steps
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._seed(tables)
        self._connection.execute("PRAGMA query_only = ON")
        self._connection.set_authorizer(self._authorize)

    async def run(self, arguments: BaseModel) -> ToolPayload:
        parsed = SqlQueryInput.model_validate(arguments)
        query = parsed.query.strip()
        if "--" in query or "/*" in query or "*/" in query:
            raise ToolExecutionError("unsafe_sql", "SQL comments are not allowed.")
        first_token = query.split(None, 1)[0].upper()
        if first_token not in {"SELECT", "WITH"}:
            raise ToolExecutionError(
                "read_only_violation", "Only SELECT or WITH queries are allowed."
            )
        if ";" in query.rstrip(";"):
            raise ToolExecutionError("multiple_statements", "Only one SQL statement is allowed.")

        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1_000
            return int(steps > self.max_vm_steps)

        try:
            async with self._lock:
                self._connection.set_progress_handler(progress, 1_000)
                try:
                    cursor = self._connection.execute(query, parsed.parameters)
                    raw_rows = cursor.fetchmany(self.max_rows + 1)
                    columns = [item[0] for item in (cursor.description or [])]
                finally:
                    self._connection.set_progress_handler(None, 0)
        except sqlite3.DatabaseError as exc:
            message = str(exc).lower()
            if "not authorized" in message or "readonly" in message:
                code = "read_only_violation"
                safe = "The query attempted an operation outside the read-only policy."
            elif "interrupted" in message:
                code = "query_budget_exceeded"
                safe = "The query exceeded its execution budget."
            else:
                code = "invalid_sql"
                safe = "The SQL query could not be executed."
            raise ToolExecutionError(code, safe) from exc

        truncated = len(raw_rows) > self.max_rows
        rows = [
            {column: self._safe_cell(row[column]) for column in columns}
            for row in raw_rows[: self.max_rows]
        ]
        output = {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}
        serialized = json.dumps(output, ensure_ascii=False, sort_keys=True)
        return ToolPayload(
            output=output,
            evidence=[
                Evidence(
                    source_id=("sql:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]),
                    content=serialized[:19_000],
                    score=1.0,
                    metadata={"kind": "sql_result", "truncated": truncated},
                )
            ],
        )

    def close(self) -> None:
        self._connection.close()

    def _seed(self, tables: Mapping[str, Sequence[Mapping[str, SqlScalar]]]) -> None:
        for table_name, rows in tables.items():
            self._validate_identifier(table_name)
            if not rows:
                raise ValueError(f"table {table_name!r} must contain at least one row")
            columns = list(rows[0])
            if not columns:
                raise ValueError(f"table {table_name!r} must contain at least one column")
            for column in columns:
                self._validate_identifier(column)
            if any(set(row) != set(columns) for row in rows):
                raise ValueError(f"all rows in table {table_name!r} must share the same columns")
            definitions = ", ".join(
                f'"{column}" {self._sqlite_type([row[column] for row in rows])}'
                for column in columns
            )
            self._connection.execute(f'CREATE TABLE "{table_name}" ({definitions})')
            placeholders = ", ".join("?" for _ in columns)
            quoted_columns = ", ".join(f'"{column}"' for column in columns)
            self._connection.executemany(
                f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
                [[row[column] for column in columns] for row in rows],
            )
        self._connection.commit()

    @staticmethod
    def _validate_identifier(value: str) -> None:
        if not IDENTIFIER.fullmatch(value):
            raise ValueError(f"unsafe SQLite identifier: {value!r}")

    @staticmethod
    def _sqlite_type(values: Sequence[SqlScalar]) -> str:
        non_null = [value for value in values if value is not None]
        if non_null and all(
            isinstance(value, int) and not isinstance(value, bool) for value in non_null
        ):
            return "INTEGER"
        if non_null and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null
        ):
            return "REAL"
        return "TEXT"

    @staticmethod
    def _safe_cell(value: object) -> SqlScalar:
        if value is None or isinstance(value, (int, float)):
            return value
        return str(value)[:2_000]

    @staticmethod
    def _authorize(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
        denied_names = (
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_INDEX",
            "SQLITE_CREATE_TEMP_TABLE",
            "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_CREATE_TEMP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TEMP_INDEX",
            "SQLITE_DROP_TEMP_TABLE",
            "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_DROP_TEMP_VIEW",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
            "SQLITE_ALTER_TABLE",
            "SQLITE_REINDEX",
            "SQLITE_ANALYZE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_PRAGMA",
            "SQLITE_TRANSACTION",
            "SQLITE_SAVEPOINT",
        )
        denied = {getattr(sqlite3, name, -999_999) for name in denied_names}
        if action in denied:
            return sqlite3.SQLITE_DENY
        if action == getattr(sqlite3, "SQLITE_FUNCTION", -1):
            function_name = (arg2 or arg1 or "").lower()
            if function_name in {"load_extension", "readfile", "writefile"}:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
