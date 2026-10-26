from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from typing import Any, Protocol, runtime_checkable

from framework.operator import (
    Interruptibility,
    OperatorContext,
    OperatorRegistry,
    OperatorRole,
    OperatorSpec,
    SideEffectLevel,
)
from framework.logic import LogicExpression
from framework.task import Requirement, TaskSpec


@runtime_checkable
class ConceptResolver(Protocol):
    def resolve_condition(self, term: str) -> Any:
        ...

    def resolve_procedure(self, term: str) -> Any:
        ...

    def resolve_lab(self, term: str) -> Any:
        ...

    def resolve_medication(self, term: str) -> Any:
        ...


@runtime_checkable
class NoteMatcher(Protocol):
    think: bool

    def patients_with_note(
        self,
        payload: dict[str, Any],
        *,
        candidate_subject_ids: set[int] | None = None,
    ) -> set[int]:
        ...

    def admissions_with_note(self, payload: dict[str, Any]) -> set[tuple[int, int]]:
        ...


class CohortRepository(Protocol):
    source_label: str

    def all_patients(self) -> set[int]:
        ...

    def patients_with_condition(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        ...

    def patients_with_procedure(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        ...

    def patients_with_lab(self, concept: Any, payload: dict[str, Any]) -> set[int]:
        ...

    def patients_with_medication(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        ...

    def patients_with_patient_attribute(self, payload: dict[str, Any]) -> set[int]:
        ...

    def patients_with_admission(self, payload: dict[str, Any]) -> set[int]:
        ...

    def patients_with_note(self, payload: dict[str, Any]) -> set[int]:
        ...

    def admissions_with_note(self, payload: dict[str, Any]) -> set[tuple[int, int]]:
        ...



@dataclass(slots=True)
class CohortOperatorServices:
    resolver: ConceptResolver
    repository: CohortRepository
    note_matcher: NoteMatcher | None = None


def register_cohort_operators(
    services: CohortOperatorServices,
) -> OperatorRegistry:
    registry = OperatorRegistry()

    registry.register(
        OperatorSpec(
            name="cohort_universe",
            role=OperatorRole.RETRIEVE,
            description="Retrieve the patient universe used to evaluate NOT.",
            input_types=(),
            output_types=("patient_set",),
            interruptibility=Interruptibility.CANCELLABLE,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
        ),
        _cohort_universe(services),
    )

    registry.register(
        OperatorSpec(
            name="evaluate_requirement",
            role=OperatorRole.RETRIEVE,
            description=(
                "Evaluate one structured cohort requirement and return the "
                "matching patient IDs."
            ),
            input_types=("requirement",),
            output_types=("patient_set",),
            interruptibility=Interruptibility.CANCELLABLE,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
            metadata={"may_use_llm": True},
        ),
        _evaluate_requirement(services),
    )

    registry.register(
        OperatorSpec(
            name="evaluate_relation",
            role=OperatorRole.RETRIEVE,
            description="Evaluate a cross-requirement temporal or encounter relation.",
            input_types=("requirement",),
            output_types=("patient_set",),
            interruptibility=Interruptibility.CANCELLABLE,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
            metadata={"may_use_llm": True},
        ),
        _evaluate_relation(services),
    )

    registry.register(
        OperatorSpec(
            name="note_candidate_filter",
            role=OperatorRole.AGGREGATE,
            description="Evaluate structured prerequisites for a note requirement.",
            input_types=("patient_set",),
            output_types=("patient_set",),
            interruptibility=Interruptibility.ATOMIC,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
        ),
        _note_candidate_filter,
    )

    registry.register(
        OperatorSpec(
            name="aggregate_cohort",
            role=OperatorRole.AGGREGATE,
            description=(
                "Combine inclusion patient sets using AND/OR and subtract exclusions."
            ),
            input_types=("patient_set",),
            output_types=("patient_set",),
            interruptibility=Interruptibility.ATOMIC,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
        ),
        _aggregate_cohort,
    )

    registry.register(
        OperatorSpec(
            name="validate_cohort",
            role=OperatorRole.VALIDATE,
            description=(
                "Validate basic consistency properties of the materialized "
                "cohort."
            ),
            input_types=("task_spec", "patient_set"),
            output_types=("validation_report",),
            interruptibility=Interruptibility.ATOMIC,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
        ),
        _validate_cohort,
    )

    registry.register(
        OperatorSpec(
            name="summarize_cohort",
            role=OperatorRole.TRANSFORM,
            description=(
                "Create a concise structured summary of the cohort result."
            ),
            input_types=(
                "task_spec",
                "patient_set",
                "validation_report",
            ),
            output_types=("cohort_summary",),
            interruptibility=Interruptibility.ATOMIC,
            side_effects=SideEffectLevel.NONE,
            cacheable=True,
            retryable=True,
        ),
        _summarize_cohort,
    )
    return registry


def _cohort_universe(services: CohortOperatorServices):
    def execute(
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: OperatorContext,
    ) -> dict[str, Any]:
        del inputs, config
        context.raise_if_cancelled()
        with context.measure("repository_query"):
            patients = services.repository.all_patients()
        with context.measure("materialize_patient_set"):
            patient_set = set(patients)
        return {"all_patient_ids": patient_set}

    return execute


def _requirement_progress_context(requirement: Requirement, services: CohortOperatorServices) -> dict[str, Any]:
    category = {
        "condition": "condition",
        "procedure": "procedure",
        "lab": "lab",
        "medication": "drug",
        "patient": "patient",
        "admission": "admission",
        "note": "note",
    }.get(requirement.kind, requirement.kind)
    return {
        "requirement_id": requirement.requirement_id,
        "category": category,
        "database": getattr(services.repository, "source_label", type(services.repository).__name__),
        "term": requirement.payload.get("term"),
    }


def _concept_progress_payload(requirement: Requirement, concept: Any, services: CohortOperatorServices) -> str:
    payload = _requirement_progress_context(requirement, services)
    payload.update({
        "resolved_to": getattr(concept, "concept_name", None),
        "concept_id": getattr(concept, "concept_id", None),
        "vocabulary": getattr(concept, "vocabulary_id", None),
        "resolution_source": getattr(concept, "resolution_source", None),
    })
    return json.dumps(payload, ensure_ascii=False, default=str)


def _call_concept_resolver(resolver_method, term: str, context: OperatorContext):
    """Pass progress through when a resolver supports it without breaking older adapters."""
    try:
        parameters = inspect.signature(resolver_method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "on_progress" in parameters:
        return resolver_method(term, on_progress=context.emit_progress)
    return resolver_method(term)


def _evaluate_requirement(services: CohortOperatorServices):
    def execute(
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: OperatorContext,
    ) -> dict[str, Any]:
        context.raise_if_cancelled()
        requirement = next(
            (value for value in inputs.values() if isinstance(value, Requirement)),
            None,
        )
        if not isinstance(requirement, Requirement):
            raise TypeError("evaluate_requirement expects a Requirement")

        payload = requirement.payload
        context.emit_progress(
            "requirement_context",
            json.dumps(
                _requirement_progress_context(requirement, services),
                ensure_ascii=False,
                default=str,
            ),
        )
        concept = None
        note_decisions: dict[int, dict[str, Any]] | None = None
        if requirement.kind.startswith("dataset:") and hasattr(services.repository, "evaluate_requirement"):
            with context.measure("repository_query"):
                patients = services.repository.evaluate_requirement(requirement.kind, payload)
        elif requirement.kind in {"condition", "procedure", "lab", "medication"}:
            resolver_method = getattr(services.resolver, {
                "condition": "resolve_condition",
                "procedure": "resolve_procedure",
                "lab": "resolve_lab",
                "medication": "resolve_medication",
            }[requirement.kind])
            with context.measure("concept_resolution"):
                concept = _call_concept_resolver(
                    resolver_method,
                    _required_text(payload, "term"),
                    context,
                )
            context.emit_progress(
                "concept_resolved",
                _concept_progress_payload(requirement, concept, services),
            )
            with context.measure("repository_query"):
                if requirement.kind == "condition":
                    patients = services.repository.patients_with_condition(concept, payload)
                elif requirement.kind == "procedure":
                    patients = services.repository.patients_with_procedure(concept, payload)
                elif requirement.kind == "lab":
                    patients = services.repository.patients_with_lab(concept, payload)
                else:
                    patients = services.repository.patients_with_medication(concept, payload)
        elif requirement.kind == "patient":
            with context.measure("repository_query"):
                patients = services.repository.patients_with_patient_attribute(payload)
        elif requirement.kind == "admission":
            with context.measure("repository_query"):
                patients = services.repository.patients_with_admission(payload)
        elif requirement.kind == "note":
            candidate_artifact = config.get("candidate_artifact")
            candidate_subject_ids = (
                None if not candidate_artifact else set(inputs[candidate_artifact])
            )
            if services.note_matcher is not None:
                method = services.note_matcher.patients_with_note
                try:
                    parameters = inspect.signature(method).parameters
                except (TypeError, ValueError):
                    parameters = {}
                if "candidate_subject_ids" in parameters:
                    patients = method(
                        payload,
                        candidate_subject_ids=candidate_subject_ids,
                    )
                else:
                    patients = method(payload)
                    if candidate_subject_ids is not None:
                        patients = set(patients) & candidate_subject_ids
                provenance_method = getattr(services.note_matcher, "decision_artifacts", None)
                if callable(provenance_method):
                    note_decisions = provenance_method(
                        payload,
                        candidate_subject_ids=candidate_subject_ids,
                    )
            else:
                # Fallback retained for demo/test repositories that do not configure
                # the semantic note pipeline. Real MIMIC local mode supplies a
                # HybridNoteMatcher.
                with context.measure("repository_query"):
                    patients = services.repository.patients_with_note(payload)
        else:
            raise ValueError(
                f"unsupported requirement kind: {requirement.kind}"
            )

        context.raise_if_cancelled()
        with context.measure("materialize_patient_set"):
            patient_set = set(patients)
        applied_fact_overrides = []
        for override in config.get("fact_overrides", ()):
            patient_id = int(override["patient_id"])
            present = bool(override.get("present", False))
            if present:
                patient_set.add(patient_id)
            else:
                patient_set.discard(patient_id)
            applied_fact_overrides.append(dict(override))
        evidence = {
            "requirement_id": requirement.requirement_id,
            "kind": requirement.kind,
            "polarity": requirement.polarity,
            "required": requirement.required,
            "payload": dict(requirement.payload),
            "patient_count": len(patient_set),
            "patient_ids": sorted(patient_set),
            "fact_overrides": applied_fact_overrides,
            "retrieval_scope": dict(config.get("retrieval_scope", {})),
        }
        if requirement.kind == "note" and note_decisions is not None:
            evidence["note_adjudications"] = [
                note_decisions[pid]
                for pid in sorted(patient_set)
                if pid in note_decisions
            ]
        if requirement.kind in {"condition", "procedure", "lab", "medication"}:
            entries = getattr(concept, "entries", ())
            evidence["resolved_concept"] = {
                "concept_id": getattr(concept, "concept_id", None),
                "concept_name": getattr(concept, "concept_name", None),
                "domain_id": getattr(concept, "domain_id", None),
                "vocabulary_id": getattr(concept, "vocabulary_id", None),
                "resolution_source": getattr(
                    concept,
                    "resolution_source",
                    None,
                ),
                "entries": [
                    {
                        "code": entry.code,
                        "display_name": entry.display_name,
                        "vocabulary_id": entry.vocabulary_id,
                        "source_tables": list(
                            getattr(entry, "source_tables", ())
                        ),
                    }
                    for entry in entries
                ],
            }
        result = {
            config["output_artifact"]: patient_set,
        }
        evidence_artifact = config.get("evidence_artifact")
        if evidence_artifact:
            result[evidence_artifact] = evidence
        return result

    return execute


def _note_candidate_filter(
    inputs: dict[str, Any],
    config: dict[str, Any],
    context: OperatorContext,
) -> dict[str, Any]:
    context.raise_if_cancelled()
    logic = LogicExpression.from_dict(config["logic"])
    artifact_by_requirement = dict(config["artifact_by_requirement"])
    requirement_sets = {
        requirement_id: set(inputs[artifact_id])
        for requirement_id, artifact_id in artifact_by_requirement.items()
    }
    universe_artifact = config.get("universe_artifact")
    universe = None if universe_artifact is None else set(inputs[universe_artifact])
    return {
        config["output_artifact"]: _evaluate_logic(logic, requirement_sets, universe)
    }


def _resolve_requirement_concept(requirement: Requirement, services: CohortOperatorServices):
    if requirement.kind not in {"condition", "procedure", "lab", "medication"}:
        return None
    method = getattr(services.resolver, {
        "condition": "resolve_condition",
        "procedure": "resolve_procedure",
        "lab": "resolve_lab",
        "medication": "resolve_medication",
    }[requirement.kind])
    return method(_required_text(requirement.payload, "term"))


def _evaluate_relation(services: CohortOperatorServices):
    def execute(
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: OperatorContext,
    ) -> dict[str, Any]:
        context.raise_if_cancelled()
        requirements = {
            value.requirement_id: value
            for value in inputs.values()
            if isinstance(value, Requirement)
        }
        relation = dict(config["relation"])
        kind = relation["relation"]

        if kind in {"before", "after"}:
            left = requirements[relation["left_requirement_id"]]
            right = requirements[relation["right_requirement_id"]]
            with context.measure("concept_resolution"):
                left_concept = _resolve_requirement_concept(left, services)
                right_concept = _resolve_requirement_concept(right, services)
            with context.measure("repository_query"):
                patients = services.repository.patients_satisfying_relation(
                    relation=kind,
                    left_kind=left.kind,
                    left_payload=left.payload,
                    right_kind=right.kind,
                    right_payload=right.payload,
                    left_concept=left_concept,
                    right_concept=right_concept,
                    max_gap_hours=relation.get("max_gap_hours"),
                    same_admission=bool(relation.get("same_admission", False)),
                )
        elif kind == "same_admission":
            pair_sets: dict[str, set[tuple[int, int]]] = {}
            for requirement_id in config.get("requirement_ids", ()): 
                requirement = requirements[requirement_id]
                if requirement.kind == "note" and services.note_matcher is not None:
                    pairs = services.note_matcher.admissions_with_note(requirement.payload)
                else:
                    with context.measure("concept_resolution"):
                        concept = _resolve_requirement_concept(requirement, services)
                    with context.measure("repository_query"):
                        pairs = services.repository.admissions_matching_requirement(
                            requirement.kind, requirement.payload, concept
                        )
                pair_sets[requirement_id] = set(pairs)

            member_sets: list[set[tuple[int, int]]] = []
            for member in relation["members"]:
                if member["kind"] == "predicate":
                    member_sets.append(pair_sets[member["requirement_id"]])
                else:
                    combined: set[tuple[int, int]] = set()
                    for requirement_id in member["requirement_ids"]:
                        combined |= pair_sets[requirement_id]
                    member_sets.append(combined)
            common = set.intersection(*member_sets) if member_sets else set()
            patients = {subject_id for subject_id, _ in common}
        else:
            raise ValueError(f"unsupported relation: {kind!r}")

        return {config["output_artifact"]: set(patients)}

    return execute


def _aggregate_cohort(
    inputs: dict[str, Any],
    config: dict[str, Any],
    context: OperatorContext,
) -> dict[str, Any]:
    context.raise_if_cancelled()
    with context.measure("boolean_aggregation"):
        return _aggregate_logic_cohort(inputs, config)


def _aggregate_logic_cohort(
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    logic = LogicExpression.from_dict(config["logic"])
    artifact_by_requirement = dict(config.get("artifact_by_requirement", {}))
    requirement_sets = {
        requirement_id: set(inputs[artifact_id])
        for requirement_id, artifact_id in artifact_by_requirement.items()
    }
    universe_artifact = config.get("universe_artifact")
    universe = (
        None
        if universe_artifact is None
        else set(inputs[universe_artifact])
    )
    result = _evaluate_logic(logic, requirement_sets, universe)
    relation_artifacts = tuple(config.get("relation_artifacts", ()))
    relation_sets = [set(inputs[artifact_id]) for artifact_id in relation_artifacts]
    for relation_set in relation_sets:
        result &= relation_set
    overrides = dict(config.get("individual_overrides", {}))
    force_include = {int(x) for x in overrides.get("include", ())}
    force_exclude = {int(x) for x in overrides.get("exclude", ())}
    result |= force_include
    result -= force_exclude

    all_patients = set().union(*requirement_sets.values()) if requirement_sets else set()
    all_patients |= force_include | force_exclude
    if universe is not None:
        all_patients |= universe
    membership: dict[int, dict[str, Any]] = {}
    for patient_id in sorted(all_patients):
        matched = [
            requirement_id
            for requirement_id, patient_ids in requirement_sets.items()
            if patient_id in patient_ids
        ]
        membership[patient_id] = {
            "matched_include_requirements": matched,
            "matched_exclude_requirements": [],
            "included": patient_id in result,
        }

    return {
        "cohort_patient_ids": result,
        "cohort_evidence": {
            "logic": logic.to_dict(),
            "logic_text": logic.render(),
            "requirement_patient_ids": {
                requirement_id: sorted(patient_ids)
                for requirement_id, patient_ids in requirement_sets.items()
            },
            "input_counts": {
                requirement_id: len(patient_ids)
                for requirement_id, patient_ids in requirement_sets.items()
            },
            "relation_counts": {
                artifact_id: len(set(inputs[artifact_id]))
                for artifact_id in relation_artifacts
            },
            "result_count": len(result),
            "result_patient_ids": sorted(result),
            "individual_overrides": {
                "include": sorted(force_include),
                "exclude": sorted(force_exclude),
            },
            "patient_membership": membership,
        },
    }


def _evaluate_logic(
    expression: LogicExpression,
    requirement_sets: dict[str, set[int]],
    universe: set[int] | None,
) -> set[int]:
    if expression.kind == "predicate":
        predicate_id = expression.predicate_id or ""
        try:
            return set(requirement_sets[predicate_id])
        except KeyError as exc:
            raise KeyError(
                f"logic references missing patient set: {predicate_id}"
            ) from exc
    if expression.kind == "not":
        if universe is None:
            raise ValueError("NOT requires a patient universe")
        return set(universe) - _evaluate_logic(
            expression.children[0], requirement_sets, universe
        )
    child_sets = [
        _evaluate_logic(child, requirement_sets, universe)
        for child in expression.children
    ]
    if expression.kind == "and":
        return set.intersection(*child_sets)
    if expression.kind == "or":
        return set.union(*child_sets)
    raise ValueError(f"unsupported logic kind: {expression.kind}")


def _validate_cohort(
    inputs: dict[str, Any],
    config: dict[str, Any],
    context: OperatorContext,
) -> dict[str, Any]:
    del config
    context.raise_if_cancelled()
    task = inputs["task_spec"]
    patients = set(inputs["cohort_patient_ids"])

    if not isinstance(task, TaskSpec):
        raise TypeError("validate_cohort expects a TaskSpec")

    warnings: list[str] = []
    if not patients:
        warnings.append("The resulting cohort is empty.")
    if any(
        requirement.polarity == "exclude"
        for requirement in task.requirements
    ) and not any(
        requirement.polarity == "include"
        for requirement in task.requirements
    ):
        warnings.append(
            "The task contains exclusions but no inclusion universe."
        )

    return {
        "cohort_validation": {
            "valid": not any(
                warning.startswith("The task contains exclusions")
                for warning in warnings
            ),
            "warnings": warnings,
            "cohort_size": len(patients),
        }
    }


def _summarize_cohort(
    inputs: dict[str, Any],
    config: dict[str, Any],
    context: OperatorContext,
) -> dict[str, Any]:
    del config
    context.raise_if_cancelled()
    task: TaskSpec = inputs["task_spec"]
    patients = set(inputs["cohort_patient_ids"])
    validation = inputs["cohort_validation"]

    return {
        "cohort_summary": {
            "objective": task.objective.description,
            "cohort_size": len(patients),
            "sample_patient_ids": sorted(patients)[:10],
            "requirement_count": len(task.requirements),
            "logic": None if task.logic is None else task.logic.to_dict(),
            "logic_text": None if task.logic is None else task.logic.render(),
            "validation": validation,
        }
    }


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"payload field {key!r} must be non-empty text")
    return value.strip()


