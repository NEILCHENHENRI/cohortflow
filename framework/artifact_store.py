from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from framework.artifact import Artifact


class ArtifactStore(ABC):
    @abstractmethod
    def put(self, artifact: Artifact) -> None:
        ...

    @abstractmethod
    def get(self, artifact_id: str) -> Artifact:
        ...

    @abstractmethod
    def contains(self, artifact_id: str) -> bool:
        ...

    @abstractmethod
    def delete(self, artifact_id: str) -> None:
        ...

    @abstractmethod
    def ids(self) -> tuple[str, ...]:
        ...

    def put_many(self, artifacts: Iterable[Artifact]) -> None:
        for artifact in artifacts:
            self.put(artifact)


class InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def put(self, artifact: Artifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact: {artifact_id}") from exc

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def delete(self, artifact_id: str) -> None:
        self._artifacts.pop(artifact_id, None)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))

    def snapshot(self) -> dict[str, Artifact]:
        return dict(self._artifacts)
