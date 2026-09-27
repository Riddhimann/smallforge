from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | None, fallback: str) -> Path:
    path = Path(value or fallback)
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_url: str
    default_model: str
    openai_base_url: str | None
    openai_api_key: str | None
    openai_model: str | None
    data_dir: Path
    results_dir: Path
    max_concurrent_experiments: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(_project_root() / ".env", override=False)
        return cls(
            ollama_url=os.getenv("SMALLFORGE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            default_model=os.getenv(
                "SMALLFORGE_DEFAULT_MODEL",
                "ollama:qwen2.5:3b",
            ),
            openai_base_url=os.getenv("SMALLFORGE_OPENAI_BASE_URL") or None,
            openai_api_key=os.getenv("SMALLFORGE_OPENAI_API_KEY") or None,
            openai_model=os.getenv("SMALLFORGE_OPENAI_MODEL") or None,
            data_dir=_resolve_path(os.getenv("SMALLFORGE_DATA_DIR"), "data"),
            results_dir=_resolve_path(os.getenv("SMALLFORGE_RESULTS_DIR"), "results"),
            max_concurrent_experiments=max(
                1, int(os.getenv("SMALLFORGE_MAX_CONCURRENT_EXPERIMENTS", "1"))
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
