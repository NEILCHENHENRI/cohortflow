from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, TypeAlias

from framework.errors import DomainValidationError
from framework.logic import LogicExpression, assign_logic_node_ids, prune_predicates
from framework.task import Requirement, TaskSpec


ATTACHMENT_MODES = {"and_child", "or_child", "relation_member", "replace_left", "replace_right"}


@dataclass(frozen=True, slots=True)
class AttachmentSpec:
    target_ref: str
    mode: str

    def __post_init__(self) -> None:
        if not self.target_ref.strip():
            raise ValueError("attachment target_ref cannot be empty")
        if self.mode not in ATTACHMENT_MODES:
            raise ValueError(f"unsupported attachment mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class AddPredicateAct:
    requirement: Requirement
    attachments: tuple[AttachmentSpec, ...]
    # Natural-language predicate wording from the semantic correction pass.
    # This is presentation/provenance text, not part of structured identity.
    display_text: str | None = None


@dataclass(frozen=True, slots=True)
class RemovePredicateAct:
    target_ref: str


@dataclass(frozen=True, slots=True)
class ReplacePredicateAct:
    target_ref: str
    requirement: Requirement
    # Preserve the correction LLM's pre-compilation wording for the logic leaf.
    display_text: str | None = None


@dataclass(frozen=True, slots=True)
class UpdatePredicateFieldsAct:
    target_ref: str
    changes: dict[str, Any]
    remove_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SetPredicateNegationAct:
    target_ref: str
    excluded: bool


@dataclass(frozen=True, slots=True)
class ChangeGroupOperatorAct:
    target_ref: str
    operator: str


@dataclass(frozen=True, slots=True)
class ReplaceLogicSubtreeAct:
    target_ref: str
    expression: LogicExpression


@dataclass(frozen=True, slots=True)
class MovePredicateAct:
    target_ref: str
    attachment: AttachmentSpec


@dataclass(frozen=True, slots=True)
class AddRelationAct:
    relation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RemoveRelationAct:
    target_ref: str


@dataclass(frozen=True, slots=True)
class UpdateRelationAct:
    target_ref: str
    changes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UpdateAggregationAct:
    target_ref: str
    changes: dict[str, Any]
    remove_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SetResolutionAct:
    target_ref: str
    resolution: dict[str, Any]


DeltaAct: TypeAlias = (
    AddPredicateAct | RemovePredicateAct | ReplacePredicateAct |
    UpdatePredicateFieldsAct | SetPredicateNegationAct |
    ChangeGroupOperatorAct | ReplaceLogicSubtreeAct | MovePredicateAct |
    AddRelationAct | RemoveRelationAct | UpdateRelationAct |
    UpdateAggregationAct | SetResolutionAct
)


def normalize_relations(raw: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    next_index = 1
    for item in raw:
        relation = deepcopy(dict(item))
        rid = str(relation.get("relation_id", "")).strip()
        if not rid:
            while f"r{next_index}" in used:
                next_index += 1
            rid = f"r{next_index}"
            next_index += 1
        if rid in used:
            raise ValueError(f"duplicate relation_id: {rid}")
        used.add(rid)
        relation["relation_id"] = rid
        result.append(relation)
    return result


def apply_delta_acts(task: TaskSpec, acts: Iterable[DeltaAct]) -> TaskSpec:
    requirements = {r.requirement_id: r for r in task.requirements}
    logic = assign_logic_node_ids(task.logic) if task.logic is not None else None
    context = deepcopy(task.context)
    pred_to_id, id_to_pred = _ensure_predicate_refs(context, requirements, logic)
    relations = normalize_relations(context.get("relations", ()))
    context["relations"] = relations

    for act in tuple(acts):
        if isinstance(act, AddPredicateAct):
            new_ref = act.requirement.requirement_id
            if new_ref in pred_to_id or new_ref in requirements:
                raise ValueError(f"predicate ref already exists: {new_ref}")
            rid = _new_requirement_id(act.requirement, requirements)
            requirement = replace(act.requirement, requirement_id=rid)
            requirements[rid] = requirement
            pred_to_id[new_ref] = rid
            id_to_pred[rid] = new_ref
            bool_attachments = [a for a in act.attachments if a.mode in {"and_child", "or_child"}]
            if len(bool_attachments) != 1:
                raise ValueError("new predicates require exactly one Boolean attachment")
            logic = _attach_leaf(
                logic, requirement, bool_attachments[0], pred_to_id,
                display_text=act.display_text,
            )
            for attachment in act.attachments:
                if attachment is bool_attachments[0]:
                    continue
                relations = _apply_relation_attachment(relations, rid, attachment)

        elif isinstance(act, RemovePredicateAct):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            _require_predicate(requirements, rid)
            requirements.pop(rid)
            logic = prune_predicates(logic, (rid,)) if logic is not None else None
            relations = _remove_predicate_from_relations(relations, rid)
            _remove_resolution(context, rid)
            ref = id_to_pred.pop(rid, None)
            if ref:
                pred_to_id.pop(ref, None)

        elif isinstance(act, ReplacePredicateAct):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            _require_predicate(requirements, rid)
            replacement = replace(act.requirement, requirement_id=rid)
            requirements[rid] = replacement
            logic = _replace_leaf_text(
                logic, rid, _clean_display_text(act.display_text) or _display_text(replacement)
            )
            _remove_resolution(context, rid)

        elif isinstance(act, (UpdatePredicateFieldsAct, UpdateAggregationAct)):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            old = _require_predicate(requirements, rid)
            payload = dict(old.payload)
            for key in act.remove_fields:
                payload.pop(key, None)
            payload.update(act.changes)
            requirements[rid] = replace(old, payload=payload)
            _remove_resolution(context, rid)

        elif isinstance(act, SetPredicateNegationAct):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            old = _require_predicate(requirements, rid)
            requirements[rid] = replace(old, polarity="exclude" if act.excluded else "include")
            logic = _set_negation(logic, rid, act.excluded)

        elif isinstance(act, ChangeGroupOperatorAct):
            if act.operator not in {"and", "or"}:
                raise ValueError("group operator must be and/or")
            logic = _change_group_operator(logic, act.target_ref, act.operator)

        elif isinstance(act, ReplaceLogicSubtreeAct):
            logic = _replace_group(logic, act.target_ref, act.expression.remap_predicate_ids(pred_to_id))

        elif isinstance(act, MovePredicateAct):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            requirement = _require_predicate(requirements, rid)
            # Moving a predicate is structural only. Keep its existing surface
            # wording instead of regenerating it from the structured payload
            # (which is opaque for attributes such as age).
            display_text = _logic_leaf_text(logic, rid)
            logic = prune_predicates(logic, (rid,)) if logic is not None else None
            logic = _attach_leaf(
                logic, requirement, act.attachment, pred_to_id,
                display_text=display_text,
            )

        elif isinstance(act, AddRelationAct):
            relation = _translate_relation_refs(deepcopy(act.relation), pred_to_id, requirements)
            if not relation.get("relation_id"):
                relation["relation_id"] = _next_ref("r", [r["relation_id"] for r in relations])
            if any(r["relation_id"] == relation["relation_id"] for r in relations):
                raise ValueError(f"relation already exists: {relation['relation_id']}")
            relations.append(relation)

        elif isinstance(act, RemoveRelationAct):
            before = len(relations)
            relations = [r for r in relations if r["relation_id"] != act.target_ref]
            if len(relations) == before:
                raise ValueError(f"unknown relation ref: {act.target_ref}")

        elif isinstance(act, UpdateRelationAct):
            relation = _relation_by_id(relations, act.target_ref)
            relation.update(_translate_relation_refs(deepcopy(act.changes), pred_to_id, requirements))
            relation["relation_id"] = act.target_ref

        elif isinstance(act, SetResolutionAct):
            rid = _resolve_pred_ref(act.target_ref, pred_to_id, requirements)
            overrides = dict(context.get("resolution_overrides", {}))
            overrides[rid] = deepcopy(act.resolution)
            context["resolution_overrides"] = overrides

        else:  # pragma: no cover - defensive for future act types
            raise TypeError(f"unsupported delta act: {type(act).__name__}")

    if not requirements:
        raise ValueError("correction cannot remove every cohort predicate")
    if logic is None:
        raise ValueError("correction produced empty cohort logic")

    context["relations"] = _validate_relation_refs(relations, set(requirements))
    structure_refs = dict(context.get("structure_refs", {}))
    structure_refs["predicates"] = dict(pred_to_id)
    context["structure_refs"] = structure_refs
    result = TaskSpec(
        domain=task.domain,
        objective=task.objective,
        deliverables=task.deliverables,
        requirements=tuple(requirements.values()),
        logic=assign_logic_node_ids(logic),
        context=context,
        assumptions=task.assumptions,
        metadata=task.metadata,
        original_request=task.original_request,
    )
    _detect_simple_contradictions(result)
    return result



def _ensure_predicate_refs(
    context: dict[str, Any], requirements: dict[str, Requirement], logic: LogicExpression | None,
) -> tuple[dict[str, str], dict[str, str]]:
    raw = context.get("structure_refs", {})
    existing = raw.get("predicates", {}) if isinstance(raw, dict) else {}
    pred_to_id = {
        str(ref): str(rid) for ref, rid in dict(existing).items()
        if str(rid) in requirements
    }
    id_to_pred = {rid: ref for ref, rid in pred_to_id.items()}
    ordered = list(logic.predicate_ids()) if logic is not None else list(requirements)
    next_index = 1
    for rid in ordered:
        if rid in id_to_pred or rid not in requirements:
            continue
        while f"p{next_index}" in pred_to_id:
            next_index += 1
        ref = f"p{next_index}"
        next_index += 1
        pred_to_id[ref] = rid
        id_to_pred[rid] = ref
    for rid in requirements:
        if rid in id_to_pred:
            continue
        while f"p{next_index}" in pred_to_id:
            next_index += 1
        ref = f"p{next_index}"
        next_index += 1
        pred_to_id[ref] = rid
        id_to_pred[rid] = ref
    return pred_to_id, id_to_pred


def _resolve_pred_ref(ref: str, pred_to_id: dict[str, str], requirements: dict[str, Requirement]) -> str:
    if ref in pred_to_id:
        return pred_to_id[ref]
    if ref in requirements:
        return ref
    raise ValueError(f"unknown predicate ref: {ref}")


def _new_requirement_id(requirement: Requirement, requirements: dict[str, Requirement]) -> str:
    if requirement.requirement_id not in requirements and not re.fullmatch(r"p\d+", requirement.requirement_id):
        return requirement.requirement_id
    payload = json.dumps(
        {"kind": requirement.kind, "payload": requirement.payload, "polarity": requirement.polarity},
        sort_keys=True, default=str,
    )
    stem = re.sub(r"[^a-z0-9]+", "_", str(requirement.payload.get("term") or requirement.kind).casefold()).strip("_")[:24] or "criterion"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    base = f"{stem}_{digest}"
    candidate = base
    suffix = 2
    while candidate in requirements:
        candidate = f"{base}_{suffix}"; suffix += 1
    return candidate


def _translate_relation_refs(value: Any, pred_to_id: dict[str, str], requirements: dict[str, Requirement]) -> Any:
    if isinstance(value, list):
        return [_translate_relation_refs(x, pred_to_id, requirements) for x in value]
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    for key in ("left_requirement_id", "right_requirement_id"):
        if key in result and isinstance(result[key], str):
            result[key] = _resolve_pred_ref(result[key], pred_to_id, requirements)
    if "members" in result and isinstance(result["members"], list):
        members = []
        for member in result["members"]:
            if isinstance(member, str):
                members.append(_resolve_pred_ref(member, pred_to_id, requirements))
            elif isinstance(member, dict):
                m = deepcopy(member)
                if m.get("kind") == "predicate" and isinstance(m.get("requirement_id"), str):
                    m["requirement_id"] = _resolve_pred_ref(m["requirement_id"], pred_to_id, requirements)
                if m.get("kind") == "or" and isinstance(m.get("requirement_ids"), list):
                    m["requirement_ids"] = [_resolve_pred_ref(x, pred_to_id, requirements) for x in m["requirement_ids"]]
                members.append(m)
            else:
                members.append(member)
        result["members"] = members
    return result


def _require_predicate(requirements: dict[str, Requirement], ref: str) -> Requirement:
    try:
        return requirements[ref]
    except KeyError as exc:
        raise ValueError(f"unknown predicate ref: {ref}") from exc


def _clean_display_text(value: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _logic_leaf_text(logic: LogicExpression | None, rid: str) -> str | None:
    if logic is None:
        return None
    for leaf in logic.leaves():
        if leaf.predicate_id == rid:
            return _clean_display_text(leaf.text)
    return None


def _display_text(requirement: Requirement) -> str:
    for key in ("term", "claim"):
        value = requirement.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return requirement.requirement_id.replace("_", " ")


def _replace_leaf_text(expr: LogicExpression | None, rid: str, text: str) -> LogicExpression | None:
    if expr is None:
        return None
    if expr.kind == "predicate":
        return LogicExpression.predicate(rid, text) if expr.predicate_id == rid else expr
    return replace(expr, children=tuple(_replace_leaf_text(c, rid, text) for c in expr.children))


def _set_negation(expr: LogicExpression | None, rid: str, excluded: bool) -> LogicExpression | None:
    if expr is None:
        return None
    if expr.kind == "not" and expr.children[0].kind == "predicate" and expr.children[0].predicate_id == rid:
        return expr if excluded else expr.children[0]
    if expr.kind == "predicate" and expr.predicate_id == rid:
        if not excluded:
            return expr
        return LogicExpression.not_(expr, node_id=None)
    return replace(expr, children=tuple(_set_negation(c, rid, excluded) for c in expr.children))


def _attach_leaf(
    logic: LogicExpression | None, requirement: Requirement, attachment: AttachmentSpec,
    pred_to_id: dict[str, str] | None = None, *, display_text: str | None = None,
) -> LogicExpression:
    if attachment.mode not in {"and_child", "or_child"}:
        raise ValueError("Boolean attachment must use and_child/or_child")
    leaf = LogicExpression.predicate(
        requirement.requirement_id,
        _clean_display_text(display_text) or _display_text(requirement),
    )
    if requirement.polarity == "exclude":
        leaf = LogicExpression.not_(leaf)
    operator = "and" if attachment.mode == "and_child" else "or"
    if logic is None:
        return leaf
    logic = assign_logic_node_ids(logic)
    if attachment.target_ref == "root":
        return _attach_to_expression(logic, leaf, operator)
    if re.fullmatch(r"p\d+", attachment.target_ref):
        if pred_to_id is None or attachment.target_ref not in pred_to_id:
            raise ValueError(f"unknown predicate ref: {attachment.target_ref}")
        return _attach_to_predicate_id(logic, pred_to_id[attachment.target_ref], leaf, operator)
    return _attach_to_group(logic, attachment.target_ref, leaf, operator)


def _attach_to_expression(target: LogicExpression, leaf: LogicExpression, operator: str) -> LogicExpression:
    """Combine target X with a new leaf N as (X OP N).

    Apply only associative flattening for the same Boolean operator:
      (A OR B) OR C  -> A OR B OR C
      (A AND B) AND C -> A AND B AND C

    No distributive, absorption, De Morgan, or other symbolic rewrites are done.
    Refs and unrelated cohort structure stay untouched.
    """
    if target.kind == operator:
        return replace(target, children=(*target.children, leaf))
    return LogicExpression(operator, node_id=None, children=(target, leaf))



def _attach_to_predicate_id(
    expr: LogicExpression, target_id: str, leaf: LogicExpression, operator: str,
) -> LogicExpression:
    if expr.kind == "predicate":
        if expr.predicate_id != target_id:
            raise ValueError(f"unknown predicate anchor: {target_id}")
        return LogicExpression(operator, node_id=None, children=(expr, leaf))
    changed = False
    children = []
    for child in expr.children:
        if target_id in child.predicate_ids():
            child = _attach_to_predicate_id(child, target_id, leaf, operator)
            changed = True
        children.append(child)
    if not changed:
        raise ValueError(f"unknown predicate anchor: {target_id}")
    return replace(expr, children=tuple(children))

def _attach_to_group(expr: LogicExpression, target_ref: str, leaf: LogicExpression, operator: str) -> LogicExpression:
    if expr.node_id == target_ref:
        return _attach_to_expression(expr, leaf, operator)
    if expr.kind == "predicate":
        raise ValueError(f"unknown logic group ref: {target_ref}")
    if not _contains_group(expr, target_ref):
        raise ValueError(f"unknown logic group ref: {target_ref}")
    children = tuple(
        _attach_to_group(child, target_ref, leaf, operator)
        if _contains_group(child, target_ref) else child
        for child in expr.children
    )
    return replace(expr, children=children)


def _contains_group(expr: LogicExpression, ref: str) -> bool:
    return expr.node_id == ref or any(_contains_group(c, ref) for c in expr.children)


def _change_group_operator(expr: LogicExpression | None, ref: str, operator: str) -> LogicExpression:
    if expr is None:
        raise ValueError("task has no logic")
    if expr.node_id == ref:
        if expr.kind not in {"and", "or"}:
            raise ValueError(f"logic ref {ref} is not an AND/OR group")
        return replace(expr, kind=operator)
    if expr.kind == "predicate":
        raise ValueError(f"unknown logic group ref: {ref}")
    found = False
    children = []
    for child in expr.children:
        if _contains_group(child, ref):
            child = _change_group_operator(child, ref, operator)
            found = True
        children.append(child)
    if not found:
        raise ValueError(f"unknown logic group ref: {ref}")
    return replace(expr, children=tuple(children))


def _replace_group(expr: LogicExpression | None, ref: str, replacement: LogicExpression) -> LogicExpression:
    if expr is None:
        raise ValueError("task has no logic")
    if expr.node_id == ref:
        if replacement.kind != "predicate":
            replacement = replace(replacement, node_id=ref)
        return replacement
    if expr.kind == "predicate":
        raise ValueError(f"unknown logic group ref: {ref}")
    changed = False
    children = []
    for child in expr.children:
        if _contains_group(child, ref):
            child = _replace_group(child, ref, replacement)
            changed = True
        children.append(child)
    if not changed:
        raise ValueError(f"unknown logic group ref: {ref}")
    return replace(expr, children=tuple(children))


def _apply_relation_attachment(relations: list[dict[str, Any]], rid: str, attachment: AttachmentSpec) -> list[dict[str, Any]]:
    relation = _relation_by_id(relations, attachment.target_ref)
    if attachment.mode == "relation_member":
        if str(relation.get("relation", "")).casefold() != "same_admission":
            raise ValueError("relation_member requires same_admission relation")
        members = list(relation.get("members", []))
        members.append(rid)
        relation["members"] = members
    elif attachment.mode in {"replace_left", "replace_right"}:
        key = "left_requirement_id" if attachment.mode == "replace_left" else "right_requirement_id"
        relation[key] = rid
    else:
        raise ValueError("unsupported relation attachment")
    return relations


def _relation_by_id(relations: list[dict[str, Any]], ref: str) -> dict[str, Any]:
    for relation in relations:
        if relation.get("relation_id") == ref:
            return relation
    raise ValueError(f"unknown relation ref: {ref}")


def _remove_predicate_from_relations(relations: list[dict[str, Any]], rid: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for relation in relations:
        kind = str(relation.get("relation", "")).casefold()
        if kind in {"before", "after"}:
            if rid in {relation.get("left_requirement_id"), relation.get("right_requirement_id")}:
                continue
            result.append(relation)
            continue
        if kind == "same_admission":
            new_members = []
            for member in relation.get("members", []):
                if isinstance(member, str):
                    if member != rid:
                        new_members.append(member)
                elif isinstance(member, dict) and member.get("kind") == "or":
                    ids = [x for x in member.get("requirement_ids", []) if x != rid]
                    if len(ids) >= 2:
                        new_members.append({**member, "requirement_ids": ids})
                    elif len(ids) == 1:
                        new_members.append(ids[0])
                elif isinstance(member, dict) and member.get("kind") == "predicate":
                    if member.get("requirement_id") != rid:
                        new_members.append(member)
                else:
                    new_members.append(member)
            if len(new_members) >= 2:
                relation["members"] = new_members
                result.append(relation)
            continue
        result.append(relation)
    return result


def _validate_relation_refs(relations: list[dict[str, Any]], known: set[str]) -> list[dict[str, Any]]:
    normalized = normalize_relations(relations)
    validated = list(relation_constraints({"relations": normalized}))
    for relation in validated:
        refs: set[str] = set()
        if relation["relation"] in {"before", "after"}:
            refs |= {relation["left_requirement_id"], relation["right_requirement_id"]}
        else:
            for member in relation["members"]:
                if member["kind"] == "predicate":
                    refs.add(member["requirement_id"])
                else:
                    refs.update(member["requirement_ids"])
        missing = refs - known
        if missing:
            raise ValueError("relation references unknown predicate refs: " + ", ".join(sorted(missing)))
    return normalized


def _remove_resolution(context: dict[str, Any], ref: str) -> None:
    overrides = dict(context.get("resolution_overrides", {}))
    overrides.pop(ref, None)
    if overrides:
        context["resolution_overrides"] = overrides
    else:
        context.pop("resolution_overrides", None)


def _next_ref(prefix: str, refs: Iterable[str]) -> str:
    used = set(refs)
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _detect_simple_contradictions(task: TaskSpec) -> None:
    # Conservative numeric interval check only for requirements that clearly
    # target the same semantic scalar. More complex clinical contradictions
    # deliberately remain outside deterministic cleanup.
    groups: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for req in task.requirements:
        payload = req.payload
        op = payload.get("operator")
        value = payload.get("value")
        if op not in {">", ">=", "<", "<=", "="} or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        identity = str(payload.get("term") or payload.get("attribute") or "")
        if not identity:
            continue
        groups.setdefault((req.kind, identity.casefold(), req.polarity), []).append((op, float(value)))
    for key, constraints in groups.items():
        lower = float("-inf")
        upper = float("inf")
        for op, value in constraints:
            if op in {">", ">="}:
                lower = max(lower, value)
            elif op in {"<", "<="}:
                upper = min(upper, value)
            else:
                lower = max(lower, value); upper = min(upper, value)
        if lower > upper:
            raise ValueError(f"correction creates contradictory numeric constraints for {key[1]}")

# Public read-only helper for correction/presentation layers.
def predicate_ref_maps(task: TaskSpec) -> tuple[dict[str, str], dict[str, str]]:
    logic = assign_logic_node_ids(task.logic) if task.logic is not None else None
    return _ensure_predicate_refs(deepcopy(task.context), {r.requirement_id: r for r in task.requirements}, logic)


# Shared relation contract. The same_admission spelling is retained for stored
# tasks and existing adapters. New vocabulary requires a separate behavior change.
RELATION_KINDS = ("before", "after", "same_admission")


def relation_constraints(task_context: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return validated cross-requirement relation constraints from TaskSpec context.

    Relations are backend-independent execution semantics.  They stay outside
    atomic requirements so the same leaves can be reused when only relationship
    structure changes.
    """
    raw = task_context.get("relations", ())
    if raw in (None, ()):
        return ()
    if not isinstance(raw, (list, tuple)):
        raise DomainValidationError("context.relations must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise DomainValidationError("each relation must be an object")
        kind = str(item.get("relation", "")).casefold().strip()
        if kind in {"before", "after"}:
            allowed = {
                "relation_id", "relation", "left_requirement_id", "right_requirement_id",
                "max_gap_hours", "same_admission",
            }
            unknown = set(item) - allowed
            if unknown:
                raise DomainValidationError(
                    "unsupported event relation fields: " + ", ".join(sorted(unknown))
                )
            left = str(item.get("left_requirement_id", "")).strip()
            right = str(item.get("right_requirement_id", "")).strip()
            if not left or not right:
                raise DomainValidationError("event relation endpoints must be non-empty")
            gap = item.get("max_gap_hours")
            if gap is not None and (isinstance(gap, bool) or not isinstance(gap, int) or gap < 0):
                raise DomainValidationError("max_gap_hours must be a non-negative integer")
            result.append({
                "relation_id": str(item.get("relation_id") or f"r{index}"),
                "relation": kind,
                "left_requirement_id": left,
                "right_requirement_id": right,
                "max_gap_hours": gap,
                "same_admission": bool(item.get("same_admission", False)),
            })
            continue
        if kind == "same_admission":
            allowed = {"relation_id", "relation", "members"}
            unknown = set(item) - allowed
            if unknown:
                raise DomainValidationError(
                    "unsupported same_admission fields: " + ", ".join(sorted(unknown))
                )
            members = item.get("members")
            if not isinstance(members, list) or len(members) < 2:
                raise DomainValidationError("same_admission requires at least two members")
            normalized: list[dict[str, Any]] = []
            for member in members:
                if isinstance(member, str):
                    normalized.append({"kind": "predicate", "requirement_id": member})
                    continue
                if not isinstance(member, dict):
                    raise DomainValidationError("same_admission member must be requirement ID or expression")
                mkind = str(member.get("kind", "")).casefold()
                if mkind == "predicate":
                    rid = str(member.get("requirement_id", "")).strip()
                    if not rid:
                        raise DomainValidationError("same_admission predicate requires requirement_id")
                    normalized.append({"kind": "predicate", "requirement_id": rid})
                elif mkind == "or":
                    ids = member.get("requirement_ids")
                    if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(x, str) and x.strip() for x in ids):
                        raise DomainValidationError("same_admission OR requires at least two requirement_ids")
                    normalized.append({"kind": "or", "requirement_ids": [x.strip() for x in ids]})
                else:
                    raise DomainValidationError("same_admission supports predicate and OR members")
            result.append({"relation_id": str(item.get("relation_id") or f"r{index}"), "relation": "same_admission", "members": normalized})
            continue
        raise DomainValidationError(f"unsupported relation kind: {kind!r}")
    return tuple(result)
