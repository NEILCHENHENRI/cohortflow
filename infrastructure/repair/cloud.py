from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib import request, error

from framework.repair.models import FailureAssessment, RepairContext, RepairProposal
from infrastructure.repair.local import (
    _ANALYSIS_SCHEMA,
    _REPAIR_SCHEMA,
    _assessment,
    _proposal,
    _render_context,
)
from prompts.repair import REPAIR_ANALYSIS_SYSTEM, REPAIR_PATCH_SYSTEM


class CommercialRepairError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CommercialRepairConfig:
    provider: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 120.0


class CommercialRepairBackend:
    name = "cloud"

    def __init__(self, config: CommercialRepairConfig) -> None:
        provider = config.provider.casefold().strip()
        if provider != "openai":
            raise ValueError("cloud repair provider must be 'openai'")
        if not config.api_key.strip():
            raise ValueError("api_key cannot be empty")
        if not config.model.strip():
            raise ValueError("cloud repair model cannot be empty")
        self.config = config
        self.provider = provider
        self._opener = request.build_opener(request.ProxyHandler({}))

    def analyze(self, context: RepairContext) -> FailureAssessment:
        payload = self._structured(
            system=REPAIR_ANALYSIS_SYSTEM,
            user=_render_context(context),
            schema=_ANALYSIS_SCHEMA,
        )
        return _assessment(payload)

    def propose(
        self,
        context: RepairContext,
        assessment: FailureAssessment,
    ) -> RepairProposal:
        payload = self._structured(
            system=REPAIR_PATCH_SYSTEM,
            user=(
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
        )
        proposal = _proposal(f"cloud:{self.provider}", payload)
        return RepairProposal(
            backend=proposal.backend,
            diagnosis=proposal.diagnosis,
            confidence=proposal.confidence,
            summary=proposal.summary,
            unified_diff=proposal.unified_diff,
            tests_to_run=proposal.tests_to_run,
            requires_human_review=True,
            metadata={"provider": self.provider, "model": self.config.model},
        )

    def _structured(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        suffix = "\n\nReturn JSON only matching this schema:\n" + json.dumps(schema)
        body = {
            "model": self.config.model,
            "store": False,
            "input": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user + suffix},
            ],
        }
        data = self._post(
            "https://api.openai.com/v1/responses",
            body,
            {"Authorization": f"Bearer {self.config.api_key}"},
        )
        text = data.get("output_text")
        if not isinstance(text, str):
            chunks: list[str] = []
            for item in data.get("output", []):
                if isinstance(item, dict):
                    for content in item.get("content", []):
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            chunks.append(content["text"])
            text = "".join(chunks)
        return _decode_object(text)


    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req = request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            api_exc = CommercialRepairError(f"{self.provider} HTTP {exc.code}: {detail[:600]}")
            api_exc.status_code = exc.code
            raise api_exc from exc
        except (error.URLError, TimeoutError) as exc:
            raise CommercialRepairError(f"Could not reach {self.provider}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CommercialRepairError(f"{self.provider} response was not a JSON object")
        return payload


def _decode_object(text: str | None) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise CommercialRepairError("repair model returned no text")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CommercialRepairError("repair model did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise CommercialRepairError("repair model JSON must be an object")
    return payload
