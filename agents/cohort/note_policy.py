from __future__ import annotations

"""Deterministic helpers for executing a structured cohort without note criteria."""

from dataclasses import replace
from typing import Any

from framework.logic import assign_logic_node_ids, prune_predicates
from framework.task import TaskSpec


def without_note_requirements(task: TaskSpec) -> TaskSpec:
    """Return *task* with note predicates and their structural references removed.

    This is intentionally a structural operation over an already-parsed TaskSpec.
    It never reparses or reinterprets the user's request.  Removing every criterion
    would turn the cohort into an unconstrained population, so that case is rejected.
    """

    note_ids = {
        requirement.requirement_id
        for requirement in task.requirements
        if requirement.kind == "note"
    }
    if not note_ids:
        return task

    requirements = tuple(
        requirement
        for requirement in task.requirements
        if requirement.requirement_id not in note_ids
    )
    if not requirements:
        raise ValueError(
            "Cannot proceed without notes because every cohort requirement is note-based."
        )

    logic = task.logic
    if logic is not None:
        logic = assign_logic_node_ids(prune_predicates(logic, note_ids))
    if logic is None:
        raise ValueError(
            "Cannot proceed without notes because removing note criteria leaves no cohort logic."
        )

    context = dict(task.context)
    raw_relations = context.get("relations", ())
    if isinstance(raw_relations, (list, tuple)):
        context["relations"] = [
            cleaned
            for relation in raw_relations
            if (cleaned := _prune_relation(relation, note_ids)) is not None
        ]

    metadata = dict(task.metadata)
    metadata["notes_skipped"] = True
    metadata["include_notes"] = False

    return replace(
        task,
        requirements=requirements,
        logic=logic,
        context=context,
        metadata=metadata,
    )


def _prune_relation(relation: Any, removed_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(relation, dict):
        return None
    result = dict(relation)

    # Pairwise relations become meaningless if either endpoint is removed.
    for key in ("left_requirement_id", "right_requirement_id", "left", "right"):
        value = result.get(key)
        if isinstance(value, str) and value in removed_ids:
            return None

    if "members" in result:
        raw_members = result.get("members")
        if not isinstance(raw_members, list):
            return None
        members = [
            cleaned
            for member in raw_members
            if (cleaned := _prune_relation_member(member, removed_ids)) is not None
        ]
        # SAME_ENCOUNTER/SAME_ADMISSION with fewer than two surviving members
        # carries no useful relational constraint.
        if len(members) < 2:
            return None
        result["members"] = members

    for key in ("requirements", "requirement_ids"):
        values = result.get(key)
        if isinstance(values, list):
            filtered = [value for value in values if value not in removed_ids]
            if not filtered:
                return None
            result[key] = filtered

    return result


def _prune_relation_member(member: Any, removed_ids: set[str]) -> Any | None:
    if isinstance(member, str):
        return None if member in removed_ids else member
    if not isinstance(member, dict):
        return member

    item = dict(member)
    requirement_id = item.get("requirement_id")
    if isinstance(requirement_id, str):
        return None if requirement_id in removed_ids else item

    requirement_ids = item.get("requirement_ids")
    if isinstance(requirement_ids, list):
        filtered = [value for value in requirement_ids if value not in removed_ids]
        if not filtered:
            return None
        if len(filtered) == 1:
            # A one-child OR/AND relation member is just that requirement.
            return filtered[0]
        item["requirement_ids"] = filtered
        return item

    return item
