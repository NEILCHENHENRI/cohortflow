from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from framework.logic import (
    LogicExpression,
    append_conjuncts,
    prune_predicates,
    validate_unique_predicates,
)


@dataclass(frozen=True, slots=True)
class Objective:
    kind: str
    description: str

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("objective kind cannot be empty")
        if not self.description.strip():
            raise ValueError("objective description cannot be empty")


@dataclass(frozen=True, slots=True)
class Deliverable:
    deliverable_id: str
    kind: str
    description: str = ""
    required: bool = True
    format: str | None = None

    def __post_init__(self) -> None:
        if not self.deliverable_id.strip():
            raise ValueError("deliverable_id cannot be empty")
        if not self.kind.strip():
            raise ValueError("deliverable kind cannot be empty")


@dataclass(frozen=True, slots=True)
class Requirement:
    requirement_id: str
    kind: str
    payload: dict[str, Any]
    polarity: str = "include"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.requirement_id.strip():
            raise ValueError("requirement_id cannot be empty")
        if not self.kind.strip():
            raise ValueError("requirement kind cannot be empty")
        if self.polarity not in {"include", "exclude"}:
            raise ValueError("polarity must be 'include' or 'exclude'")


@dataclass(frozen=True, slots=True)
class Assumption:
    assumption_id: str
    statement: str
    status: str = "unresolved"

    def __post_init__(self) -> None:
        if not self.assumption_id.strip():
            raise ValueError("assumption_id cannot be empty")
        if not self.statement.strip():
            raise ValueError("assumption statement cannot be empty")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    domain: str
    objective: Objective
    deliverables: tuple[Deliverable, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    logic: LogicExpression | None = None
    context: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[Assumption, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    original_request: str = ""

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("task domain cannot be empty")
        _ensure_unique(
            (item.deliverable_id for item in self.deliverables),
            "deliverable_id",
        )
        _ensure_unique(
            (item.requirement_id for item in self.requirements),
            "requirement_id",
        )
        _ensure_unique(
            (item.assumption_id for item in self.assumptions),
            "assumption_id",
        )
        if self.logic is not None:
            validate_unique_predicates(self.logic)
            requirement_ids = {
                item.requirement_id for item in self.requirements
            }
            unknown = set(self.logic.predicate_ids()) - requirement_ids
            if unknown:
                raise ValueError(
                    "logic references unknown requirement IDs: "
                    + ", ".join(sorted(unknown))
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskDelta:
    acts: tuple[Any, ...] = ()
    objective: Objective | None = None
    add_deliverables: tuple[Deliverable, ...] = ()
    remove_deliverable_ids: tuple[str, ...] = ()
    add_requirements: tuple[Requirement, ...] = ()
    remove_requirement_ids: tuple[str, ...] = ()
    replace_requirements: tuple[Requirement, ...] = ()
    logic: LogicExpression | None = None
    context_updates: dict[str, Any] = field(default_factory=dict)
    context_removals: tuple[str, ...] = ()
    metadata_updates: dict[str, Any] = field(default_factory=dict)
    metadata_removals: tuple[str, ...] = ()
    assumption_updates: tuple[Assumption, ...] = ()
    remove_assumption_ids: tuple[str, ...] = ()
    rationale: str = ""


def apply_task_delta(task: TaskSpec, delta: TaskDelta) -> TaskSpec:
    if delta.acts:
        from framework.task_delta import apply_delta_acts
        task = apply_delta_acts(task, delta.acts)
        delta = replace(delta, acts=())

    deliverables = {
        item.deliverable_id: item for item in task.deliverables
    }
    for item_id in delta.remove_deliverable_ids:
        deliverables.pop(item_id, None)
    for item in delta.add_deliverables:
        deliverables[item.deliverable_id] = item

    requirements = {
        item.requirement_id: item for item in task.requirements
    }
    for item_id in delta.remove_requirement_ids:
        requirements.pop(item_id, None)
    for item in delta.replace_requirements:
        requirements[item.requirement_id] = item
    for item in delta.add_requirements:
        requirements[item.requirement_id] = item

    assumptions = {
        item.assumption_id: item for item in task.assumptions
    }
    for item_id in delta.remove_assumption_ids:
        assumptions.pop(item_id, None)
    for item in delta.assumption_updates:
        assumptions[item.assumption_id] = item

    context = dict(task.context)
    for key in delta.context_removals:
        context.pop(key, None)
    context.update(delta.context_updates)

    metadata = dict(task.metadata)
    for key in delta.metadata_removals:
        metadata.pop(key, None)
    metadata.update(delta.metadata_updates)

    logic = delta.logic
    if logic is None and task.logic is not None:
        logic = prune_predicates(task.logic, delta.remove_requirement_ids)
        existing_logic_ids = set() if logic is None else set(logic.predicate_ids())
        new_conjuncts: list[LogicExpression] = []
        for requirement in (*delta.add_requirements, *delta.replace_requirements):
            if requirement.requirement_id in existing_logic_ids:
                continue
            leaf = LogicExpression.predicate(
                requirement.requirement_id,
                _requirement_display_text(requirement),
            )
            if requirement.polarity == "exclude":
                leaf = LogicExpression.not_(leaf)
            new_conjuncts.append(leaf)
        logic = append_conjuncts(logic, new_conjuncts)

    return TaskSpec(
        domain=task.domain,
        objective=delta.objective or task.objective,
        deliverables=tuple(deliverables.values()),
        requirements=tuple(requirements.values()),
        logic=logic,
        context=context,
        assumptions=tuple(assumptions.values()),
        metadata=metadata,
        original_request=task.original_request,
    )



def _requirement_display_text(requirement: Requirement) -> str:
    term = requirement.payload.get("term")
    if isinstance(term, str) and term.strip():
        return term.strip()
    return requirement.requirement_id.replace("_", " ")


def _ensure_unique(values: object, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value}")
        seen.add(value)
