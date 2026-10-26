from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from framework.errors import DomainValidationError
from framework.logic import PredicateText, parse_boolean_expression, prune_predicates
from framework.task import Objective, Requirement, TaskDelta, TaskSpec


FALLBACK_COMMANDS = frozenset({
    "/fallback",
    "/sos",
    "/incorrigible",
    "/unsalvageable",
})


def is_fallback_command(message: str) -> bool:
    cleaned = message.strip()
    if not cleaned:
        return False
    first = cleaned.split(maxsplit=1)[0].casefold()
    return first in FALLBACK_COMMANDS


class ManualFieldType(str, Enum):
    TEXT = "text"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    SELECT = "select"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ManualEditChoice:
    value: str
    label: str


@dataclass(frozen=True, slots=True)
class ManualEditField:
    field_id: str
    label: str
    field_type: ManualFieldType
    value: Any
    required: bool = False
    choices: tuple[ManualEditChoice, ...] = ()
    help_text: str | None = None
    editable: bool = True


@dataclass(frozen=True, slots=True)
class ManualEditSection:
    section_id: str
    title: str
    fields: tuple[ManualEditField, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class ManualEditForm:
    form_id: str
    title: str
    description: str
    task_version: int
    sections: tuple[ManualEditSection, ...]


@dataclass(frozen=True, slots=True)
class ManualEditIssue:
    field_id: str
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class ManualEditResult:
    task_delta: TaskDelta | None
    issues: tuple[ManualEditIssue, ...] = ()
    rationale: str = "Human manual fallback edit."

    @property
    def valid(self) -> bool:
        return self.task_delta is not None and not any(
            issue.severity == "error" for issue in self.issues
        )


@runtime_checkable
class ManualEditProvider(Protocol):
    def build_manual_edit_form(
        self,
        task: TaskSpec,
        *,
        version: int,
    ) -> ManualEditForm:
        ...

    def compile_manual_edit(
        self,
        original_task: TaskSpec,
        submitted: dict[str, Any],
    ) -> ManualEditResult:
        ...


class StructuredManualEditProvider(ABC):
    """Compile form edits transactionally while domains own fields and payloads."""

    empty_logic_message = "Task logic cannot be empty."

    @abstractmethod
    def build_form(self, task: TaskSpec, *, version: int) -> ManualEditForm: ...

    @abstractmethod
    def _validate_payload(self, kind: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def _payload_from_submission(
        self, current: Requirement, kind: str, submitted: dict[str, Any],
        prefix: str, issues: list[ManualEditIssue],
    ) -> dict[str, Any] | None: ...

    def _predicate_text(self, requirement: Requirement) -> str:
        return requirement.requirement_id

    def compile(
        self,
        original_task: TaskSpec,
        submitted: dict[str, Any],
    ) -> ManualEditResult:
        issues: list[ManualEditIssue] = []
        objective_text = str(
            submitted.get(
                "objective.description",
                original_task.objective.description,
            )
        ).strip()
        if not objective_text:
            issues.append(
                ManualEditIssue(
                    "objective.description",
                    "Objective cannot be empty.",
                )
            )

        replacements: list[Requirement] = []
        removals: list[str] = []
        for current in original_task.requirements:
            prefix = f"requirements.{current.requirement_id}"
            remove = bool(submitted.get(f"{prefix}.remove", False))
            if remove:
                removals.append(current.requirement_id)
                continue

            kind = str(
                submitted.get(f"{prefix}.kind", current.kind)
            ).strip()
            polarity = str(
                submitted.get(f"{prefix}.polarity", current.polarity)
            ).strip()
            required = bool(
                submitted.get(f"{prefix}.required", current.required)
            )
            payload = self._payload_from_submission(
                current,
                kind,
                submitted,
                prefix,
                issues,
            )
            if payload is None:
                continue
            try:
                self._validate_payload(kind, payload)
                replacements.append(
                    Requirement(
                        requirement_id=current.requirement_id,
                        kind=kind,
                        payload=payload,
                        polarity=polarity,
                        required=required,
                    )
                )
            except (ValueError, DomainValidationError) as exc:
                issues.append(
                    ManualEditIssue(prefix, str(exc))
                )

        if len(removals) == len(original_task.requirements):
            issues.append(
                ManualEditIssue(
                    "requirements",
                    "At least one requirement must remain.",
                )
            )

        if issues:
            return ManualEditResult(
                task_delta=None,
                issues=tuple(issues),
            )

        remaining_by_id = {
            item.requirement_id: item
            for item in replacements
            if item.requirement_id not in removals
        }
        original_logic_text = (
            original_task.logic.render_boolean()
            if original_task.logic is not None
            else " AND ".join(remaining_by_id)
        )
        logic_text = str(
            submitted.get("logic.expression", original_logic_text)
        ).strip()
        # Removing a requirement through the fallback form should not force the
        # user to manually edit the Boolean expression as a second step. If the
        # expression itself was left untouched, prune removed leaves first.
        if removals and logic_text == original_logic_text and original_task.logic is not None:
            pruned = prune_predicates(original_task.logic, removals)
            logic_text = "" if pruned is None else pruned.render_boolean()
        if not logic_text:
            issues.append(
                ManualEditIssue("logic.expression", self.empty_logic_message)
            )
            return ManualEditResult(task_delta=None, issues=tuple(issues))

        predicate_texts = []
        old_leaf_text = {
            leaf.predicate_id: leaf.text or leaf.predicate_id or ""
            for leaf in original_task.logic.leaves()
        } if original_task.logic is not None else {}
        for requirement_id, requirement in remaining_by_id.items():
            text = old_leaf_text.get(requirement_id) or self._predicate_text(requirement)
            predicate_texts.append(PredicateText(requirement_id, text))
        try:
            revised_logic = parse_boolean_expression(logic_text, predicate_texts)
        except (TypeError, ValueError) as exc:
            return ManualEditResult(
                task_delta=None,
                issues=(ManualEditIssue("logic.expression", str(exc)),),
            )

        delta = TaskDelta(
            objective=Objective(
                kind=original_task.objective.kind,
                description=objective_text,
            ),
            replace_requirements=tuple(replacements),
            remove_requirement_ids=tuple(removals),
            logic=revised_logic,
            rationale="Human manual fallback edit.",
        )
        return ManualEditResult(
            task_delta=delta,
            rationale="Human manual fallback edit.",
        )
