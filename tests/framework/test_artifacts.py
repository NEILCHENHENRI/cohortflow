from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore


def test_artifact_hash_is_stable():
    a = Artifact("x", "number", {"value": 1})
    b = Artifact("y", "number", {"value": 1})
    assert a.semantic_hash == b.semantic_hash


def test_in_memory_artifact_store():
    store = InMemoryArtifactStore()
    artifact = Artifact("x", "number", 1)
    store.put(artifact)

    assert store.contains("x")
    assert store.get("x").value == 1
    assert store.ids() == ("x",)

    store.delete("x")
    assert not store.contains("x")
