from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from framework.graph import ExecutionGraph
from framework.task import TaskDelta, TaskSpec, apply_task_delta


class MutationKind(str, Enum):
    UPDATE_VALUE = "update_value"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    ADD_ENTITY = "add_entity"
    REMOVE_ENTITY = "remove_entity"
    REQUEST_INFORMATION = "request_information"
    INVALIDATE_ARTIFACT = "invalidate_artifact"
    OVERRIDE_RESULT = "override_result"
    APPROVE_RESULT = "approve_result"


class CorrectionFallbackRequired(RuntimeError):
    """Robust correction could not be interpreted safely.

    The current task remains untouched. UI/controller code may offer the user
    an explicit Full Regeneration fallback rather than invoking it silently.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason or message


class CorrectionMode(str, Enum):
    ROBUST_DELTA = "robust_delta"
    FULL_REGENERATION = "full_regeneration"


class CorrectionScope(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CorrectionTarget:
    target_type: str
    target_id: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactMutation:
    kind: MutationKind
    target: CorrectionTarget
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorrectionIntent:
    user_text: str
    summary: str
    desired_effect: str
    scope: CorrectionScope
    mutations: tuple[ArtifactMutation, ...]
    confidence: float
    requires_clarification: bool = False
    clarification_question: str | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("correction confidence must be between 0 and 1")
        if self.requires_clarification and not self.clarification_question:
            raise ValueError(
                "clarification_question is required when clarification is needed"
            )


@dataclass(frozen=True, slots=True)
class CorrectionAnalysis:
    intent: CorrectionIntent
    task_delta: TaskDelta


@dataclass(frozen=True, slots=True)
class CorrectionPlan:
    intent: CorrectionIntent
    task_delta: TaskDelta
    changed_nodes: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    recompute_nodes: tuple[str, ...]
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    recompute_boundary: tuple[str, ...]

    @property
    def is_local(self) -> bool:
        return self.intent.scope is CorrectionScope.LOCAL


class CorrectionPlanner:
    """Localizes a semantic task correction onto the execution graph."""

    def plan(
        self,
        *,
        current_task: TaskSpec,
        current_graph: ExecutionGraph,
        analysis: CorrectionAnalysis,
        build_graph,
    ) -> CorrectionPlan:
        proposed_task = apply_task_delta(current_task, analysis.task_delta)
        proposed_graph = build_graph(proposed_task)

        old_ids = set(current_graph.nodes)
        new_ids = set(proposed_graph.nodes)
        added = new_ids - old_ids
        removed = old_ids - new_ids

        changed = {
            node_id
            for node_id in old_ids & new_ids
            if (
                current_graph.nodes[node_id].semantic_fingerprint
                != proposed_graph.nodes[node_id].semantic_fingerprint
            )
        }

        affected = set(changed) | set(added)
        if changed or added:
            affected.update(proposed_graph.descendants(changed | added))

        # Removed nodes may have surviving descendants whose inputs/configuration
        # changed in the proposed graph. Include matching surviving descendants.
        if removed:
            old_descendants = current_graph.descendants(removed)
            affected.update(old_descendants & new_ids)

        recompute = affected - added
        boundary = {
            node_id
            for node_id in affected
            if not (proposed_graph.parents(node_id) & affected)
        }

        return CorrectionPlan(
            intent=analysis.intent,
            task_delta=analysis.task_delta,
            changed_nodes=tuple(sorted(changed)),
            affected_nodes=tuple(sorted(affected)),
            recompute_nodes=tuple(sorted(recompute)),
            added_nodes=tuple(sorted(added)),
            removed_nodes=tuple(sorted(removed)),
            recompute_boundary=tuple(sorted(boundary)),
        )
