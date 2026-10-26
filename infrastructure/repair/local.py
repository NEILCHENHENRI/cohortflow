from __future__ import annotations

from prompts.repair import REPAIR_ANALYSIS_SYSTEM, REPAIR_PATCH_SYSTEM

import json
import math
from typing import Any

from framework.repair.models import (
    FailureAction,
    FailureAssessment,
    RepairContext,
    RepairProposal,
    RepairScope,
)

_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [item.value for item in FailureAction],
        },
        "scope": {
            "type": "string",
            "enum": [item.value for item in RepairScope],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "diagnosis": {"type": "string"},
        "rationale": {"type": "string"},
        "likely_files": {"type": "array", "items": {"type": "string"}},
        "likely_symbols": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "action", "scope", "confidence", "diagnosis", "rationale",
        "likely_files", "likely_symbols",
    ],
    "additionalProperties": False,
}

_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "unified_diff": {"type": "string"},
        "tests_to_run": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnosis", "confidence", "summary", "unified_diff", "tests_to_run"],
    "additionalProperties": False,
}




class LocalRepairBackend:
    name = "local"

    def __init__(self, llm) -> None:
        self.llm = llm

    def analyze(self, context: RepairContext) -> FailureAssessment:
        payload = self.llm.generate_structured(
            system_prompt=REPAIR_ANALYSIS_SYSTEM,
            user_prompt=_render_context(context),
            schema=_ANALYSIS_SCHEMA,
            stage_label="Local failure analysis",
        )
        return _assessment(payload)

    def propose(
        self,
        context: RepairContext,
        assessment: FailureAssessment,
    ) -> RepairProposal:
        payload = self.llm.generate_structured(
            system_prompt=REPAIR_PATCH_SYSTEM,
            user_prompt=(
                _render_context(context)
                + "\n\nFAILURE ASSESSMENT:\n"
                + json.dumps({
                    "diagnosis": assessment.diagnosis,
                    "rationale": assessment.rationale,
                    "scope": assessment.scope.value,
                    "likely_files": assessment.likely_files,
                    "likely_symbols": assessment.likely_symbols,
                }, ensure_ascii=False)
            ),
            schema=_REPAIR_SCHEMA,
            stage_label="Local repair patch",
        )
        return _proposal(self.name, payload)


def _render_context(context: RepairContext) -> str:
    failure = context.failure
    parts = [
        "Unchanged task/operation contract: " + json.dumps(context.task_context, default=str),
        f"Origin: {failure.origin}",
        f"Exception: {failure.exception_type}: {failure.message}",
        f"Node: {failure.node_id or 'n/a'}",
        f"Operator: {failure.operator_name or 'n/a'}",
        f"HTTP status: {failure.status_code if failure.status_code is not None else 'n/a'}",
        "Traceback:\n" + failure.traceback[-14000:],
    ]
    if failure.metadata:
        parts.append("Runtime metadata:\n" + json.dumps(failure.metadata, ensure_ascii=False, default=str)[:6000])
    for path, text in context.source_context:
        parts.append(f"SOURCE {path}:\n{text}")
    for path, text in context.test_context:
        parts.append(f"TEST {path}:\n{text}")
    if context.prior_attempt:
        parts.append("PRIOR REPAIR ATTEMPT / VALIDATION:\n" + context.prior_attempt[-12000:])
    return "\n\n".join(parts)


def _assessment(payload: dict[str, Any]) -> FailureAssessment:
    if not isinstance(payload, dict):
        raise ValueError("failure assessment must be an object")
    return FailureAssessment(
        action=FailureAction(str(payload.get("action") or FailureAction.STOP.value)),
        scope=RepairScope(str(payload.get("scope") or RepairScope.UNKNOWN.value)),
        confidence=_confidence(payload),
        diagnosis=_text(payload, "diagnosis"),
        rationale=_text(payload, "rationale"),
        likely_files=_strings(payload, "likely_files"),
        likely_symbols=_strings(payload, "likely_symbols"),
    )


def _proposal(backend: str, payload: dict[str, Any]) -> RepairProposal:
    if not isinstance(payload, dict):
        raise ValueError("repair proposal must be an object")
    return RepairProposal(
        backend=backend,
        diagnosis=_text(payload, "diagnosis"),
        confidence=_confidence(payload),
        summary=_text(payload, "summary"),
        unified_diff=_text(payload, "unified_diff"),
        tests_to_run=_strings(payload, "tests_to_run"),
        requires_human_review=True,
    )


def _confidence(payload: dict[str, Any]) -> float:
    value = payload.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("repair confidence must be a finite number between 0 and 1")
    return float(value)


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"repair {key} must be text")
    return value


def _strings(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"repair {key} must be an array of non-empty strings")
    return tuple(values)
