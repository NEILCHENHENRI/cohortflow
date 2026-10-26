from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    producer_node_id: str | None = None
    producer_operator: str | None = None
    execution_version: int = 1
    source_artifact_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class Artifact:
    artifact_id: str
    artifact_type: str
    value: Any
    provenance: ArtifactProvenance = field(
        default_factory=ArtifactProvenance
    )
    schema: dict[str, Any] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    semantic_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id cannot be empty")
        if not self.artifact_type.strip():
            raise ValueError("artifact_type cannot be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.semantic_hash is None:
            self.semantic_hash = self.compute_semantic_hash()

    def compute_semantic_hash(self) -> str:
        payload = {
            "artifact_type": self.artifact_type,
            "value": self.value,
            "schema": self.schema,
            "metadata": self.metadata,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
