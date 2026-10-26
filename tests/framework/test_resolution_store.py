from framework.resolution import (
    InMemoryResolutionStore, ResolutionKey, ResolutionRecord,
)

def test_resolution_store_roundtrip():
    store = InMemoryResolutionStore()
    key = ResolutionKey("omop_concept", "Condition:hypertension")
    record = ResolutionRecord(
        key, "123", "Hypertension stage 1", {"concept_id": 123}, "human"
    )
    store.put(record)
    assert store.get(key) == record
