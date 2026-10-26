from __future__ import annotations

from typing import Any

from agents.cohort.schema import validate_requirement_payload
from agents.cohort.structured import SUPPORTED_DOMAINS
from framework.errors import DomainValidationError
from framework.logic import PredicateText, parse_boolean_expression, prune_predicates
from framework.manual_edit import (
    StructuredManualEditProvider,
    ManualEditChoice,
    ManualEditField,
    ManualEditForm,
    ManualEditIssue,
    ManualEditResult,
    ManualEditSection,
    ManualFieldType,
)
from framework.task import Objective, Requirement, TaskDelta, TaskSpec


class CohortManualEditProvider(StructuredManualEditProvider):
    def __init__(self, dataset=None):
        self.dataset = dataset

    empty_logic_message = "Eligibility logic cannot be empty."

    def _validate_payload(self, kind: str, payload: dict[str, Any]) -> None:
        if self.dataset is not None:
            self.dataset.validate_requirement(kind, payload)
        else:
            validate_requirement_payload(kind, payload)

    def _predicate_text(self, requirement: Requirement) -> str:
        term = requirement.payload.get("term")
        return term.strip() if isinstance(term, str) and term.strip() else requirement.requirement_id

    def build_form(
        self,
        task: TaskSpec,
        *,
        version: int,
    ) -> ManualEditForm:
        sections: list[ManualEditSection] = [
            ManualEditSection(
                section_id="cohort",
                title="Cohort definition",
                description=(
                    "Edit the high-impact fields that determine cohort meaning."
                ),
                fields=(
                    ManualEditField(
                        "objective.description",
                        "Objective",
                        ManualFieldType.TEXT,
                        task.objective.description,
                        required=True,
                    ),
                    ManualEditField(
                        "logic.expression",
                        "Eligibility logic",
                        ManualFieldType.TEXT,
                        (
                            task.logic.render_boolean()
                            if task.logic is not None
                            else " AND ".join(
                                item.requirement_id for item in task.requirements
                            )
                        ),
                        required=True,
                        help_text=(
                            "Use requirement IDs with AND, OR, NOT, and "
                            "parentheses, e.g. adult AND (diabetes OR hypertension)."
                        ),
                    ),
                ),
            )
        ]

        for requirement in task.requirements:
            fields = [
                ManualEditField(
                    f"requirements.{requirement.requirement_id}.requirement_id",
                    "Requirement ID",
                    ManualFieldType.TEXT,
                    requirement.requirement_id,
                    editable=False,
                ),
                ManualEditField(
                    f"requirements.{requirement.requirement_id}.kind",
                    "Kind",
                    ManualFieldType.SELECT,
                    requirement.kind,
                    required=True,
                    choices=tuple(
                        ManualEditChoice(kind, kind.title())
                        for kind in (self.dataset.domains() if self.dataset else SUPPORTED_DOMAINS)
                    ),
                ),
                ManualEditField(
                    f"requirements.{requirement.requirement_id}.polarity",
                    "Polarity",
                    ManualFieldType.SELECT,
                    requirement.polarity,
                    required=True,
                    choices=(
                        ManualEditChoice("include", "Include"),
                        ManualEditChoice("exclude", "Exclude"),
                    ),
                ),
                ManualEditField(
                    f"requirements.{requirement.requirement_id}.required",
                    "Required",
                    ManualFieldType.BOOLEAN,
                    requirement.required,
                ),
            ]
            fields.extend(
                self._payload_fields(requirement)
            )
            sections.append(
                ManualEditSection(
                    section_id=f"requirement:{requirement.requirement_id}",
                    title=self._requirement_title(requirement),
                    fields=tuple(fields),
                    description=(
                        "Concept terms are validated during execution. "
                        "Ambiguous terms will still trigger confirmation."
                    ),
                )
            )

        return ManualEditForm(
            form_id=f"cohort-manual-edit-v{version}",
            title="Manual cohort recovery",
            description=(
                "Edits are staged separately. They are validated and shown as "
                "a change before execution."
            ),
            task_version=version,
            sections=tuple(sections),
        )


    def _payload_fields(
        self,
        requirement: Requirement,
    ) -> list[ManualEditField]:
        prefix = f"requirements.{requirement.requirement_id}.payload"
        payload = requirement.payload
        fields: list[ManualEditField] = []
        if requirement.kind == "condition":
            fields.append(
                ManualEditField(
                    f"{prefix}.term",
                    "Clinical condition or symptom term",
                    ManualFieldType.TEXT,
                    payload.get("term", ""),
                    required=True,
                )
            )
        elif requirement.kind == "medication":
            fields.extend((
                ManualEditField(
                    f"{prefix}.term",
                    "Medication term",
                    ManualFieldType.TEXT,
                    payload.get("term", ""),
                    required=True,
                ),
                ManualEditField(
                    f"{prefix}.exposure_mode",
                    "Exposure mode",
                    ManualFieldType.SELECT,
                    payload.get("exposure_mode", "any_recorded"),
                    required=True,
                    choices=(
                        ManualEditChoice(
                            "any_recorded",
                            "Any recorded exposure",
                        ),
                        ManualEditChoice(
                            "active_at_index",
                            "Active at index date",
                        ),
                    ),
                ),
            ))
        else:
            fields.append(
                ManualEditField(
                    f"{prefix}.json",
                    "Payload",
                    ManualFieldType.JSON,
                    payload,
                    required=True,
                    help_text=(
                        "Advanced structured payload. Keep the canonical keys."
                    ),
                )
            )
        fields.append(
            ManualEditField(
                f"requirements.{requirement.requirement_id}.remove",
                "Remove this requirement",
                ManualFieldType.BOOLEAN,
                False,
            )
        )
        return fields

    def _payload_from_submission(
        self,
        current: Requirement,
        kind: str,
        submitted: dict[str, Any],
        prefix: str,
        issues: list[ManualEditIssue],
    ) -> dict[str, Any] | None:
        payload_prefix = f"{prefix}.payload"
        if kind == "condition":
            return {
                "term": str(
                    submitted.get(
                        f"{payload_prefix}.term",
                        current.payload.get("term", ""),
                    )
                ).strip()
            }
        if kind == "medication":
            return {
                "term": str(
                    submitted.get(
                        f"{payload_prefix}.term",
                        current.payload.get("term", ""),
                    )
                ).strip(),
                "exposure_mode": str(
                    submitted.get(
                        f"{payload_prefix}.exposure_mode",
                        current.payload.get(
                            "exposure_mode",
                            "any_recorded",
                        ),
                    )
                ),
            }
        raw = submitted.get(f"{payload_prefix}.json", current.payload)
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            import json
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(
                    ManualEditIssue(
                        f"{payload_prefix}.json",
                        f"Invalid JSON: {exc}",
                    )
                )
                return None
            if not isinstance(value, dict):
                issues.append(
                    ManualEditIssue(
                        f"{payload_prefix}.json",
                        "Payload JSON must be an object.",
                    )
                )
                return None
            return value
        issues.append(
            ManualEditIssue(
                f"{payload_prefix}.json",
                "Payload must be a JSON object.",
            )
        )
        return None

    @staticmethod
    def _requirement_title(requirement: Requirement) -> str:
        term = requirement.payload.get("term")
        return str(term or requirement.requirement_id)
