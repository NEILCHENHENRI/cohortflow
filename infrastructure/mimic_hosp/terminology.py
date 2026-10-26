from __future__ import annotations

from prompts.terminology import MEDICAL_TERMINOLOGY_EXPANSION_SYSTEM

from dataclasses import dataclass
from typing import Any

from framework.agent import ProgressCallback, StructuredLLM


_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["canonical_form", "equivalent_terms"],
    "properties": {
        "canonical_form": {"type": "string"},
        "equivalent_terms": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
    },
}



@dataclass(slots=True)
class LLMMedicalTerminologyExpander:
    llm: StructuredLLM

    def expand(
        self, term: str, *, domain: str, on_progress: ProgressCallback | None = None,
    ) -> tuple[str, ...]:
        payload = self.llm.generate_structured(
            system_prompt=MEDICAL_TERMINOLOGY_EXPANSION_SYSTEM,
            user_prompt=f"DOMAIN: {domain}\nTERM: {term}",
            schema=_EXPANSION_SCHEMA,
            on_progress=on_progress,
            stage_label="Medical terminology expansion",
        )
        values: list[str] = []
        canonical = payload.get("canonical_form")
        if isinstance(canonical, str) and canonical.strip():
            values.append(canonical.strip())
        raw = payload.get("equivalent_terms", [])
        if isinstance(raw, list):
            values.extend(x.strip() for x in raw if isinstance(x, str) and x.strip())
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = " ".join(value.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= 8:
                break
        return tuple(result)
