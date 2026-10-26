from __future__ import annotations

from prompts.adapter import ADAPTER_FIELD_SYSTEM, ADAPTER_CONVERSION_SYSTEM

from typing import Any
import json

from framework.agent import StructuredLLM


_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field_id", "description", "numeric", "datetime"],
                "properties": {
                    "field_id": {"type": "string"},
                    "description": {"type": "string"},
                    "numeric": {"type": "boolean"},
                    "datetime": {"type": "boolean"},
                },
            },
        }
    },
}

_CONVERSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sql_expression"],
    "properties": {"sql_expression": {"type": "string"}},
}




class AdapterLLMInspector:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def describe_fields(self, fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        payload = self._llm.generate_structured(
            system_prompt=ADAPTER_FIELD_SYSTEM,
            user_prompt=json.dumps({"fields": fields}, ensure_ascii=False),
            schema=_FIELD_SCHEMA,
            stage_label="Adapter field inspection",
        )
        result: dict[str, dict[str, Any]] = {}
        expected = {item["field_id"] for item in fields}
        if not isinstance(payload, dict) or not isinstance(payload.get("fields"), list):
            raise ValueError("field inspection must return a fields array")
        for item in payload.get("fields", []):
            if not isinstance(item, dict) or not isinstance(item.get("field_id"), str):
                raise ValueError("field inspection entries require a field_id")
            field_id = item["field_id"]
            if field_id not in expected or field_id in result:
                raise ValueError(f"unknown or duplicate field_id: {field_id}")
            if not isinstance(item.get("description"), str) or not item["description"].strip():
                raise ValueError(f"field description must be non-empty: {field_id}")
            if type(item.get("numeric")) is not bool or type(item.get("datetime")) is not bool:
                raise ValueError(f"field type flags must be booleans: {field_id}")
            result[field_id] = item
        if set(result) != expected:
            raise ValueError("field inspection omitted requested fields")
        return result

    def propose_conversion(
        self,
        *,
        column: str,
        samples: list[str],
        target_type: str,
    ) -> str:
        payload = self._llm.generate_structured(
            system_prompt=ADAPTER_CONVERSION_SYSTEM,
            user_prompt=json.dumps({
                "column": column,
                "target_type": target_type,
                "sample_values": samples[:20],
            }),
            schema=_CONVERSION_SCHEMA,
            stage_label=f"Adapter {target_type} normalization",
        )
        expression = payload.get("sql_expression") if isinstance(payload, dict) else None
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("LLM returned an empty normalization expression")
        return expression.strip()
