from __future__ import annotations

from agents.cohort.adapters.llm_inspector import AdapterLLMInspector
import pytest


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stage_label"] == "Adapter field inspection":
            return {
                "fields": [
                    {
                        "field_id": "labs.value",
                        "description": "numeric laboratory result",
                        "numeric": True,
                        "datetime": False,
                    }
                ]
            }
        return {"sql_expression": 'TRY_CAST("value" AS DOUBLE)'}


def test_llm_inspector_asks_only_for_description_numeric_and_datetime():
    fake = FakeLLM()
    inspector = AdapterLLMInspector(fake)
    result = inspector.describe_fields([
        {"field_id": "labs.value", "table": "labs", "column": "value", "storage_type": "object"}
    ])
    assert result["labs.value"]["numeric"] is True
    schema = fake.calls[0]["schema"]
    properties = schema["properties"]["fields"]["items"]["properties"]
    assert set(properties) == {"field_id", "description", "numeric", "datetime"}


def test_conversion_is_a_separate_sample_based_llm_call():
    fake = FakeLLM()
    inspector = AdapterLLMInspector(fake)
    expression = inspector.propose_conversion(column="value", samples=["1.2", "3.4"], target_type="DOUBLE")
    assert expression == 'TRY_CAST("value" AS DOUBLE)'
    assert fake.calls[-1]["stage_label"] == "Adapter DOUBLE normalization"


@pytest.mark.parametrize("fields", [
    [],
    [{"field_id": "unknown", "description": "Value", "numeric": True, "datetime": False}],
    [{"field_id": "labs.value", "description": "Value", "numeric": "false", "datetime": False}],
    [{"field_id": "labs.value", "description": "Value", "numeric": True, "datetime": False}] * 2,
])
def test_inspector_rejects_incomplete_or_ill_typed_model_metadata(fields):
    class Response:
        def generate_structured(self, **kwargs):
            return {"fields": fields}
    with pytest.raises(ValueError):
        AdapterLLMInspector(Response()).describe_fields([{"field_id": "labs.value"}])
