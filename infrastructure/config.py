from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AppMode(str, Enum):
    LOCAL = "local"
    DEMO = "demo"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Filesystem locations used by concrete infrastructure adapters.

    Local mode reads a single raw MIMIC-IV hosp-module database
    (``data/mimic_hosp.duckdb``). There is no separate vocabulary file since
    that schema has no standardized concept table. Demo mode is unrelated
    to real data and keeps its own synthetic OMOP-shaped pair of databases.
    """

    project_root: Path
    clinical_database: Path
    vocabulary_database: Path | None
    data_directory: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        *,
        mode: AppMode | str = AppMode.LOCAL,
    ) -> "ProjectPaths":
        root = Path(project_root).expanduser().resolve()
        data = root / "data"
        selected_mode = AppMode(mode)
        if selected_mode is AppMode.CUSTOM:
            raise ValueError("Custom mode requires explicit adapter database paths")
        if selected_mode is AppMode.DEMO:
            demo_directory = data / "demo"
            return cls(
                project_root=root,
                clinical_database=demo_directory / "clinical.duckdb",
                vocabulary_database=demo_directory / "vocabulary.duckdb",
                data_directory=data,
            )
        return cls(
            project_root=root,
            clinical_database=data / "mimic_hosp.duckdb",
            vocabulary_database=None,
            data_directory=data,
        )

    def validate_databases(self) -> None:
        missing = [
            path
            for path in (
                self.clinical_database,
                self.vocabulary_database,
            )
            if path is not None and not path.exists()
        ]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"required database files do not exist: {joined}"
            )


@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    paths: ProjectPaths
    mode: AppMode = AppMode.LOCAL
    adapter_directory: Path | None = None
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_think: bool = False
    ollama_preload: bool = True
    ollama_timeout_seconds: float = 300.0
    semantic_recovery_enabled: bool = True
    semantic_recovery_confidence: float = 0.90
    terminology_expansion_enabled: bool = True
    terminology_expansion_model: str = "medgemma1.5:4b"
    duckdb_threads: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AppMode(self.mode))
        if self.mode is AppMode.CUSTOM and self.adapter_directory is None:
            raise ValueError("Custom mode requires adapter_directory")
        if not 0.0 <= self.semantic_recovery_confidence <= 1.0:
            raise ValueError(
                "semantic_recovery_confidence must be between 0 and 1"
            )
