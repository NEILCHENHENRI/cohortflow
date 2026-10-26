from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from infrastructure.omop.capabilities import DataCapabilities

from agents.cohort.structured import SUPPORTED_DOMAINS, relation_constraints
from framework.graph import (
    ExecutionGraph,
    ExecutionNode,
    NodeKind,
    NodeStatus,
)
from framework.logic import LogicExpression
from framework.task import Requirement, TaskSpec


def stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class CohortPlanner:
    capabilities: DataCapabilities | None = None
    supported_kinds: tuple[str, ...] = SUPPORTED_DOMAINS

    """Deterministically compiles a cohort TaskSpec into an execution DAG."""

    def build_graph(self, task: TaskSpec) -> ExecutionGraph:
        self._validate_task(task)
        graph = ExecutionGraph()

        graph.add_node(
            ExecutionNode(
                node_id="input:task",
                kind=NodeKind.INPUT,
                outputs=("task_spec",),
                config={"task_hash": self._task_semantic_key(task)},
                status=NodeStatus.COMPLETED,
            )
        )

        requirement_nodes: dict[str, str] = {}
        patient_artifacts: dict[str, str] = {}
        ordered_requirements = sorted(
            task.requirements, key=lambda item: item.requirement_id
        )
        # Structured requirements run first. Note branches are added only after
        # their structured co-requirements exist so they can consume a narrowed
        # candidate set rather than search the whole note corpus.
        for requirement in (r for r in ordered_requirements if r.kind != "note"):
            query_node, patient_artifact = self._add_requirement_branch(
                graph,
                requirement,
                task=task,
            )
            requirement_nodes[requirement.requirement_id] = query_node
            patient_artifacts[requirement.requirement_id] = patient_artifact

        requirement_by_id = {r.requirement_id: r for r in ordered_requirements}
        for requirement in (r for r in ordered_requirements if r.kind == "note"):
            guard = self._structured_guard_for_note(
                task.logic,
                requirement.requirement_id,
                requirement_by_id,
            )
            candidate_node = None
            candidate_artifact = None
            if guard is not None:
                candidate_node, candidate_artifact = self._add_note_candidate_filter(
                    graph,
                    requirement,
                    guard,
                    requirement_nodes,
                    patient_artifacts,
                )
            query_node, patient_artifact = self._add_requirement_branch(
                graph,
                requirement,
                task=task,
                candidate_node=candidate_node,
                candidate_artifact=candidate_artifact,
            )
            requirement_nodes[requirement.requirement_id] = query_node
            patient_artifacts[requirement.requirement_id] = patient_artifact

        relation_nodes, relation_artifacts = self._add_relation_constraints(
            graph, task, requirement_nodes
        )

        self._add_logic_aggregate(
            graph,
            task.logic,
            requirement_nodes,
            patient_artifacts,
            relation_nodes=relation_nodes,
            relation_artifacts=relation_artifacts,
            individual_overrides=task.context.get("individual_overrides", {}),
        )

        graph.add_node(
            ExecutionNode(
                node_id="validate:cohort",
                kind=NodeKind.COMPUTATION,
                operator_name="validate_cohort",
                inputs=("task_spec", "cohort_patient_ids"),
                outputs=("cohort_validation",),
            ),
            depends_on=("input:task", "aggregate:cohort"),
        )

        graph.add_node(
            ExecutionNode(
                node_id="deliver:summary",
                kind=NodeKind.COMPUTATION,
                operator_name="summarize_cohort",
                inputs=(
                    "task_spec",
                    "cohort_patient_ids",
                    "cohort_validation",
                ),
                outputs=("cohort_summary",),
            ),
            depends_on=(
                "input:task",
                "aggregate:cohort",
                "validate:cohort",
            ),
        )
        return graph

    def build_input_values(self, task: TaskSpec) -> dict[str, Any]:
        values: dict[str, Any] = {"task_spec": task}
        for requirement in task.requirements:
            values[self.requirement_artifact_id(requirement)] = requirement
        return values

    def _add_requirement_branch(
        self,
        graph: ExecutionGraph,
        requirement: Requirement,
        *,
        task: TaskSpec,
        candidate_node: str | None = None,
        candidate_artifact: str | None = None,
    ) -> tuple[str, str]:
        key = stable_key(requirement.requirement_id)
        requirement_artifact = self.requirement_artifact_id(requirement)
        patient_artifact = f"patients:{key}"
        evidence_artifact = f"evidence:{key}"
        input_node = f"input:requirement:{key}"
        query_node = f"evaluate:requirement:{key}"

        graph.add_node(
            ExecutionNode(
                node_id=input_node,
                kind=NodeKind.INPUT,
                outputs=(requirement_artifact,),
                config={
                    "requirement_id": requirement.requirement_id,
                    "kind": requirement.kind,
                    "payload": requirement.payload,
                    "polarity": requirement.polarity,
                    "required": requirement.required,
                },
                status=NodeStatus.COMPLETED,
            )
        )
        graph.add_node(
            ExecutionNode(
                node_id=query_node,
                kind=NodeKind.COMPUTATION,
                operator_name="evaluate_requirement",
                inputs=(requirement_artifact,),
                outputs=(patient_artifact, evidence_artifact),
                config={
                    "output_artifact": patient_artifact,
                    "evidence_artifact": evidence_artifact,
                    "requirement_id": requirement.requirement_id,
                    "fact_overrides": [
                        dict(item)
                        for item in task.context.get("fact_overrides", ())
                        if item.get("requirement_id") == requirement.requirement_id
                    ],
                    "retrieval_scope": dict(
                        task.context.get("retrieval_scope", {})
                    ),
                    "requirement_kind": requirement.kind,
                    "candidate_artifact": candidate_artifact,
                },
            ),
            depends_on=tuple(
                node for node in (input_node, candidate_node) if node is not None
            ),
        )
        if candidate_artifact is not None:
            graph.nodes[query_node].inputs = (
                requirement_artifact,
                candidate_artifact,
            )
        return query_node, patient_artifact

    def _add_note_candidate_filter(
        self,
        graph: ExecutionGraph,
        note_requirement: Requirement,
        guard: LogicExpression,
        requirement_nodes: dict[str, str],
        patient_artifacts: dict[str, str],
    ) -> tuple[str, str]:
        key = stable_key(note_requirement.requirement_id)
        node_id = f"note:candidates:{key}"
        output = f"note_candidates:{key}"
        predicate_ids = guard.predicate_ids()
        dependencies = [requirement_nodes[rid] for rid in predicate_ids]
        inputs = [patient_artifacts[rid] for rid in predicate_ids]
        universe_artifact = None
        if guard.contains_not():
            universe_node = "cohort:universe"
            if universe_node not in graph.nodes:
                graph.add_node(
                    ExecutionNode(
                        node_id=universe_node,
                        kind=NodeKind.COMPUTATION,
                        operator_name="cohort_universe",
                        outputs=("all_patient_ids",),
                    )
                )
            dependencies.append(universe_node)
            inputs.append("all_patient_ids")
            universe_artifact = "all_patient_ids"
        graph.add_node(
            ExecutionNode(
                node_id=node_id,
                kind=NodeKind.COMPUTATION,
                operator_name="note_candidate_filter",
                inputs=tuple(inputs),
                outputs=(output,),
                config={
                    "logic": guard.to_dict(),
                    "artifact_by_requirement": {
                        rid: patient_artifacts[rid] for rid in predicate_ids
                    },
                    "universe_artifact": universe_artifact,
                    "output_artifact": output,
                },
            ),
            depends_on=tuple(dict.fromkeys(dependencies)),
        )
        return node_id, output

    @classmethod
    def _structured_guard_for_note(
        cls,
        logic: LogicExpression,
        target_id: str,
        requirements: dict[str, Requirement],
    ) -> LogicExpression | None:
        """Return structured conditions that must hold for target note to matter.

        Note leaves are treated as unknown/possibly true. Along the target path,
        only AND siblings impose prerequisites. OR siblings do not. Within a
        sibling subtree we retain the strongest safe structured prerequisite.
        """

        def abstract(node: LogicExpression) -> LogicExpression | None:
            if node.kind == "predicate":
                rid = node.predicate_id or ""
                requirement = requirements.get(rid)
                return None if requirement is None or requirement.kind == "note" else node
            if node.kind == "not":
                child = abstract(node.children[0])
                # NOT over an unknown note-dependent expression cannot safely
                # narrow candidates. Preserve only fully structured NOT.
                if child is None or any(
                    requirements.get(rid) is not None
                    and requirements[rid].kind == "note"
                    for rid in node.children[0].predicate_ids()
                ):
                    return None
                return LogicExpression.not_(child)
            children = [abstract(child) for child in node.children]
            if node.kind == "and":
                kept = [child for child in children if child is not None]
                if not kept:
                    return None
                return kept[0] if len(kept) == 1 else LogicExpression.and_(*kept)
            # OR can only constrain when every branch has a structured guard.
            if any(child is None for child in children):
                return None
            kept = [child for child in children if child is not None]
            return kept[0] if len(kept) == 1 else LogicExpression.or_(*kept)

        guards: list[LogicExpression] = []

        def walk(node: LogicExpression) -> bool:
            if node.kind == "predicate":
                return node.predicate_id == target_id
            for index, child in enumerate(node.children):
                if walk(child):
                    if node.kind == "and":
                        for sibling_index, sibling in enumerate(node.children):
                            if sibling_index == index:
                                continue
                            guard = abstract(sibling)
                            if guard is not None:
                                guards.append(guard)
                    return True
            return False

        if not walk(logic) or not guards:
            return None
        return guards[0] if len(guards) == 1 else LogicExpression.and_(*guards)

    def _add_relation_constraints(
        self,
        graph: ExecutionGraph,
        task: TaskSpec,
        requirement_nodes: dict[str, str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        constraints = relation_constraints(task.context)
        if not constraints:
            return (), ()
        requirements = {item.requirement_id: item for item in task.requirements}
        nodes: list[str] = []
        artifacts: list[str] = []
        for index, relation in enumerate(constraints, start=1):
            referenced: list[str] = []
            if relation["relation"] in {"before", "after"}:
                referenced = [
                    relation["left_requirement_id"],
                    relation["right_requirement_id"],
                ]
            else:
                for member in relation["members"]:
                    if member["kind"] == "predicate":
                        referenced.append(member["requirement_id"])
                    else:
                        referenced.extend(member["requirement_ids"])
            unknown = sorted(set(referenced) - set(requirements))
            if unknown:
                raise ValueError(
                    "relation references unknown requirement IDs: " + ", ".join(unknown)
                )
            relation_ref = relation.get("relation_id") or f"r{index}"
            node_id = f"relation:{relation_ref}:{relation['relation']}"
            artifact_id = f"relation_patients:{relation_ref}"
            input_artifacts = tuple(
                self.requirement_artifact_id(requirements[rid])
                for rid in dict.fromkeys(referenced)
            )
            graph.add_node(
                ExecutionNode(
                    node_id=node_id,
                    kind=NodeKind.COMPUTATION,
                    operator_name="evaluate_relation",
                    inputs=input_artifacts,
                    outputs=(artifact_id,),
                    config={
                        "relation": relation,
                        "requirement_ids": list(dict.fromkeys(referenced)),
                        "output_artifact": artifact_id,
                    },
                ),
                depends_on=tuple(
                    requirement_nodes[rid] for rid in dict.fromkeys(referenced)
                ),
            )
            nodes.append(node_id)
            artifacts.append(artifact_id)
        return tuple(nodes), tuple(artifacts)

    def _add_logic_aggregate(
        self,
        graph: ExecutionGraph,
        logic: LogicExpression,
        requirement_nodes: dict[str, str],
        patient_artifacts: dict[str, str],
        *,
        relation_nodes: tuple[str, ...] = (),
        relation_artifacts: tuple[str, ...] = (),
        individual_overrides: dict[str, Any] | None = None,
    ) -> None:
        dependencies = [
            requirement_nodes[predicate_id]
            for predicate_id in logic.predicate_ids()
        ]
        inputs = [
            patient_artifacts[predicate_id]
            for predicate_id in logic.predicate_ids()
        ]
        dependencies.extend(relation_nodes)
        inputs.extend(relation_artifacts)
        if logic.contains_not():
            if "cohort:universe" not in graph.nodes:
                graph.add_node(
                    ExecutionNode(
                        node_id="cohort:universe",
                        kind=NodeKind.COMPUTATION,
                        operator_name="cohort_universe",
                        outputs=("all_patient_ids",),
                    )
                )
            dependencies.append("cohort:universe")
            inputs.append("all_patient_ids")

        graph.add_node(
            ExecutionNode(
                node_id="aggregate:cohort",
                kind=NodeKind.COMPUTATION,
                operator_name="aggregate_cohort",
                inputs=tuple(inputs),
                outputs=("cohort_patient_ids", "cohort_evidence"),
                config={
                    "logic": logic.to_dict(),
                    "artifact_by_requirement": patient_artifacts,
                    "universe_artifact": (
                        "all_patient_ids" if logic.contains_not() else None
                    ),
                    "individual_overrides": dict(individual_overrides or {}),
                    "relation_artifacts": list(relation_artifacts),
                },
            ),
            depends_on=tuple(dict.fromkeys(dependencies)),
        )

    @staticmethod
    def requirement_artifact_id(requirement: Requirement) -> str:
        return f"requirement:{stable_key(requirement.requirement_id)}"

    @staticmethod
    def _task_semantic_key(task: TaskSpec) -> str:
        relevant = (
            task.objective,
            task.deliverables,
            task.requirements,
            task.logic,
            task.assumptions,
            task.context,
            task.metadata,
        )
        return stable_key(repr(relevant))

    def _validate_task(self, task: TaskSpec) -> None:
        if task.domain != "cohort_curation":
            raise ValueError("CohortPlanner requires cohort_curation domain")
        unsupported = {
            requirement.kind
            for requirement in task.requirements
            if requirement.kind not in self.supported_kinds
        }
        if unsupported:
            raise ValueError(
                f"unsupported cohort requirement kinds: {sorted(unsupported)}"
            )
        if self.capabilities is not None:
            for requirement in task.requirements:
                self.capabilities.validate_requirement(
                    requirement.kind,
                    requirement.payload,
                )
        constraints = relation_constraints(task.context)
        known_ids = {item.requirement_id for item in task.requirements}
        for relation in constraints:
            if relation["relation"] in {"before", "after"}:
                referenced = {relation["left_requirement_id"], relation["right_requirement_id"]}
            else:
                referenced = set()
                for member in relation["members"]:
                    if member["kind"] == "predicate":
                        referenced.add(member["requirement_id"])
                    else:
                        referenced.update(member["requirement_ids"])
            unknown_rel = referenced - known_ids
            if unknown_rel:
                raise ValueError("relation references unknown requirement IDs: " + ", ".join(sorted(unknown_rel)))
        if task.logic is None:
            raise ValueError("cohort task must have canonical Boolean logic")
        if not task.requirements:
            raise ValueError("cohort task must contain at least one requirement")
