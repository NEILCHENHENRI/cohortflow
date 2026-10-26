from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any
import threading

from framework.agent import DomainAgent
from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore
from framework.graph import ExecutionGraph, NodeStatus
from framework.correction import (
    CorrectionAnalysis,
    CorrectionPlan,
    CorrectionPlanner,
)
from framework.reconcile import ReuseReport, reconcile_graphs
from framework.resolution import PendingResolution, ResolutionRequired
from framework.pending_interaction import (
    ContinuationKind,
    PendingInteraction,
    PendingInteractionKind,
)
from framework.execution_backend import ExecutionBackend
from framework.runtime import ExecutionMode, GraphRuntime, RuntimeResult
from framework.langgraph_runtime import LangGraphRuntime
from framework.failure import FailureEvent
from framework.task import TaskDelta, TaskSpec, apply_task_delta
from framework.timing import measure
from framework.manual_edit import ManualEditForm, ManualEditResult


class SessionStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    STAGED = "staged"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    FAILED = "failed"


@dataclass(slots=True)
class StagedExecution:
    base_version: int
    task: TaskSpec
    graph: ExecutionGraph
    artifacts: InMemoryArtifactStore
    reuse_report: ReuseReport
    rationale: str = ""
    status: StageStatus = StageStatus.STAGED
    correction_plan: CorrectionPlan | None = None


@dataclass(slots=True)
class AgentState:
    task: TaskSpec
    graph: ExecutionGraph
    version: int = 1
    status: SessionStatus = SessionStatus.READY
    pending_interaction: PendingInteraction | None = None
    staged: StagedExecution | None = None
    last_reuse_report: ReuseReport | None = None
    last_correction_plan: CorrectionPlan | None = None

    @property
    def paused(self) -> bool:
        return self.status is SessionStatus.WAITING_FOR_HUMAN

    @property
    def cancelled(self) -> bool:
        return self.status is SessionStatus.CANCELLED

    @property
    def pending_resolution(self) -> PendingResolution | None:
        pending = self.pending_interaction
        if (
            pending is None
            or pending.kind is not PendingInteractionKind.SEMANTIC_RESOLUTION
        ):
            return None
        return pending.resolution

    @pending_resolution.setter
    def pending_resolution(self, value: PendingResolution | None) -> None:
        if value is None:
            if (
                self.pending_interaction is not None
                and self.pending_interaction.kind
                is PendingInteractionKind.SEMANTIC_RESOLUTION
            ):
                self.pending_interaction = None
            return
        continuation = (
            ContinuationKind.STAGED_EXECUTION
            if self.staged is not None
            else ContinuationKind.INITIAL_EXECUTION
        )
        self.pending_interaction = PendingInteraction.semantic_resolution(
            value,
            continuation=continuation,
        )

    def snapshot(self, artifact_ids: tuple[str, ...]) -> dict[str, Any]:
        report = self.last_reuse_report
        return {
            "task": self.task.to_dict(),
            "version": self.version,
            "status": self.status.value,
            "paused": self.paused,
            "cancelled": self.cancelled,
            "pending_interaction": (
                None
                if self.pending_interaction is None
                else {
                    "kind": self.pending_interaction.kind.value,
                    "continuation": self.pending_interaction.continuation.value,
                    "question": self.pending_interaction.question,
                    "resolution_id": (
                        None
                        if self.pending_interaction.resolution is None
                        else self.pending_interaction.resolution.resolution_id
                    ),
                    "original_message":
                        self.pending_interaction.original_message,
                }
            ),
            "last_correction_plan": None if self.last_correction_plan is None else {
                "scope": self.last_correction_plan.intent.scope.value,
                "affected_nodes": list(self.last_correction_plan.affected_nodes),
                "recompute_boundary": list(self.last_correction_plan.recompute_boundary),
            },
            "staged": None if self.staged is None else {
                "base_version": self.staged.base_version,
                "status": self.staged.status.value,
                "rationale": self.staged.rationale,
            },
            "graph": self.graph.to_dict(),
            "artifact_ids": list(artifact_ids),
            "last_reuse_report": None if report is None else {
                "reused_nodes": list(report.reused_nodes),
                "invalidated_nodes": list(report.invalidated_nodes),
                "added_nodes": list(report.added_nodes),
                "removed_nodes": list(report.removed_nodes),
            },
        }


@dataclass(slots=True)
class SteerableSession:
    """Generic lifecycle for every domain agent."""

    domain_agent: DomainAgent
    max_workers: int = 1
    state: AgentState | None = field(init=False, default=None)
    pending_initial_interaction: PendingInteraction | None = field(init=False, default=None)
    artifacts: InMemoryArtifactStore = field(
        init=False, default_factory=InMemoryArtifactStore
    )
    # Session-lifetime cache keyed by a node plus its full upstream semantic
    # lineage. It survives stopped/superseded graph versions, allowing
    # valid work from an unfinished run to be reused if that branch reappears.
    artifact_cache: dict[str, tuple[Artifact, ...]] = field(
        init=False, default_factory=dict
    )
    runtime: GraphRuntime = field(init=False)
    execution_backend: ExecutionBackend = field(
        init=False, default=ExecutionBackend.NATIVE
    )
    correction_planner: CorrectionPlanner = field(
        init=False, default_factory=CorrectionPlanner
    )
    repair_manager: Any = field(init=False, default=None, repr=False)
    last_failure: FailureEvent | None = field(init=False, default=None)
    last_repair_proposal: object | None = field(init=False, default=None)
    last_repair_outcome: object | None = field(init=False, default=None)
    _yield_event: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )
    _yield_reason: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        initial_mode = (
            ExecutionMode.PARALLEL
            if self.max_workers > 1
            else ExecutionMode.SEQUENTIAL
        )
        self.runtime = self._build_runtime(
            ExecutionBackend.NATIVE,
            max_workers=self.max_workers,
            execution_mode=initial_mode,
        )

    def _build_runtime(
        self,
        backend: ExecutionBackend,
        *,
        max_workers: int,
        execution_mode: ExecutionMode,
    ) -> GraphRuntime:
        registry = self.domain_agent.operator_registry()
        runtime_type = (
            GraphRuntime
            if backend is ExecutionBackend.NATIVE
            else LangGraphRuntime
        )
        return runtime_type(
            registry,
            max_workers=max_workers,
            execution_mode=execution_mode,
            recovery_handler=self._recover_operation,
            recovery_enabled=self._repair_enabled,
        )

    def _repair_enabled(self):
        from framework.repair.models import RepairMode
        return self.repair_manager is not None and self.repair_manager.policy.mode is not RepairMode.OFF

    def repair_failure(self, failure: FailureEvent):
        """Diagnose failures outside an operator, without automatic replay/retry."""
        if not self._repair_enabled():
            return self.repair_manager.repair(failure) if self.repair_manager is not None else None
        return self._recover_operation(failure, None, None, False)

    def _recover_operation(self, failure, node, retry, safe_to_retry):
        from copy import deepcopy
        from framework.repair.models import RecoveryHooks
        def task_context():
            state = self.state
            return {
                "version": state.version if state else None,
                "task": state.task.to_dict() if state else None,
                "staged_task": state.staged.task.to_dict() if state and state.staged else None,
                "node": node.semantic_fingerprint if node else None,
                "execution_version": node.execution_version if node else None,
            }
        def snapshot():
            return hashlib.sha256(json.dumps(task_context(), sort_keys=True, default=str).encode()).hexdigest()
        context = deepcopy(task_context())
        self.last_failure = failure
        hooks = RecoveryHooks(
            operation_key=f"{id(self)}:{node.node_id if node else failure.origin}", snapshot=snapshot,
            retry=retry, safe_to_retry=safe_to_retry,
            assessment=getattr(self.domain_agent, "assess_failure", lambda _: None)(failure),
            replay=getattr(self.domain_agent, "repair_replay", lambda *_: None)(failure, deepcopy(context)),
            task_context=context,
        )
        outcome = self.repair_manager.repair(failure, hooks=hooks)
        self.last_repair_outcome = outcome
        self.last_repair_proposal = outcome.proposal
        return outcome

    def request_initial_parse_clarification(
        self, *, original_message: str, question: str, rationale: str
    ) -> PendingInteraction:
        pending = PendingInteraction.initial_parse_clarification(
            original_message=original_message, question=question, rationale=rationale
        )
        self.pending_initial_interaction = pending
        return pending

    def clear_initial_parse_clarification(self) -> None:
        self.pending_initial_interaction = None

    def initialize(
        self,
        task: TaskSpec,
        *,
        on_progress=None,
    ) -> AgentState:
        graph = self.domain_agent.run_module(
            "session.build_graph",
            lambda: self.domain_agent.build_graph(task),
            on_progress=on_progress,
        )
        self.artifacts = InMemoryArtifactStore()
        self.domain_agent.run_module(
            "session.put_inputs",
            lambda: self._put_inputs(task, self.artifacts),
            on_progress=on_progress,
        )
        self.pending_initial_interaction = None
        self.state = AgentState(
            task=task,
            graph=graph,
        )
        return self.state

    def execute(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        state.status = SessionStatus.RUNNING
        try:
            result = self.domain_agent.run_module(
                "execution.runtime",
                lambda: self.runtime.run(
                    state.graph,
                    self.artifacts,
                    execution_version=state.version,
                    on_progress=on_progress,
                ),
                on_progress=on_progress,
            )
        except ResolutionRequired as exc:
            self._archive_completed(state.graph, self.artifacts)
            resolution = exc.pending(state.version)
            state.pending_interaction = PendingInteraction.semantic_resolution(
                resolution,
                continuation=ContinuationKind.INITIAL_EXECUTION,
            )
            state.status = SessionStatus.WAITING_FOR_HUMAN
            raise
        except Exception as exc:
            self._archive_completed(state.graph, self.artifacts)
            state.status = SessionStatus.FAILED
            failed = next((node for node in state.graph.nodes.values() if node.failure is not None), None)
            self.last_failure = (
                failed.failure if failed is not None else FailureEvent.capture(exc, origin="execution.runtime")
            )
            raise
        self._archive_completed(state.graph, self.artifacts)
        state.status = (
            SessionStatus.CANCELLED if result.stopped else SessionStatus.COMPLETED
        )
        return result

    @staticmethod
    def _defer_note_requirement(node) -> bool:
        return (
            node.operator_name == "evaluate_requirement"
            and node.config.get("requirement_kind") == "note"
        )

    def execute_until_note_confirmation(self, on_progress=None) -> RuntimeResult:
        """Run every currently available non-note node, then pause before notes."""
        state = self._require_state()
        state.status = SessionStatus.RUNNING
        try:
            result = self.domain_agent.run_module(
                "execution.note_preflight",
                lambda: self.runtime.run(
                    state.graph,
                    self.artifacts,
                    execution_version=state.version,
                    on_progress=on_progress,
                    defer_node=self._defer_note_requirement,
                ),
                on_progress=on_progress,
            )
        except ResolutionRequired as exc:
            self._archive_completed(state.graph, self.artifacts)
            resolution = exc.pending(state.version)
            state.pending_interaction = PendingInteraction.semantic_resolution(
                resolution,
                continuation=ContinuationKind.INITIAL_NOTE_PREFLIGHT,
            )
            state.status = SessionStatus.WAITING_FOR_HUMAN
            raise
        except Exception as exc:
            self._archive_completed(state.graph, self.artifacts)
            state.status = SessionStatus.FAILED
            failed = next(
                (node for node in state.graph.nodes.values() if node.failure is not None),
                None,
            )
            self.last_failure = (
                failed.failure
                if failed is not None
                else FailureEvent.capture(exc, origin="execution.note_preflight")
            )
            raise
        self._archive_completed(state.graph, self.artifacts)
        state.status = SessionStatus.CANCELLED if result.stopped else SessionStatus.READY
        return result


    def analyze_correction(
        self,
        message: str,
        on_progress=None,
    ) -> CorrectionAnalysis:
        state = self._require_state()
        # A staged task is already the newest materialized semantic state, even
        # while its graph is still executing. Later corrections must never target
        # the older committed TaskSpec.
        base_task = state.staged.task if state.staged is not None else state.task
        return self.domain_agent.run_module(
            "correction.interpret",
            lambda: self.domain_agent.interpret_correction(
                base_task,
                message,
                on_progress=on_progress,
            ),
            on_progress=on_progress,
        )

    def plan_correction(
        self,
        analysis: CorrectionAnalysis,
        on_progress=None,
    ) -> CorrectionPlan:
        state = self._require_state()
        use_staged = state.staged is not None and state.status is SessionStatus.CANCELLED
        base_task = state.staged.task if use_staged else state.task
        base_graph = state.staged.graph if use_staged else state.graph
        return self.domain_agent.run_module(
            "correction.plan",
            lambda: self.correction_planner.plan(
                current_task=base_task,
                current_graph=base_graph,
                analysis=analysis,
                build_graph=self.domain_agent.build_graph,
            ),
            on_progress=on_progress,
        )

    def stage_correction(
        self,
        analysis: CorrectionAnalysis,
        plan: CorrectionPlan | None = None,
        *,
        on_progress=None,
    ) -> ReuseReport:
        plan = plan or self.plan_correction(analysis, on_progress=on_progress)
        report = self.stage(
            analysis.task_delta,
            on_progress=on_progress,
        )
        staged = self._require_staged()
        staged.correction_plan = plan
        return report

    def request_correction_clarification(
        self,
        *,
        original_message: str,
        question: str,
        analysis: CorrectionAnalysis | None = None,
        rationale: str | None = None,
    ) -> None:
        state = self._require_state()
        state.pending_interaction = PendingInteraction.correction_clarification(
            original_message=original_message,
            question=question,
            analysis=analysis,
            rationale=rationale,
        )
        state.status = SessionStatus.WAITING_FOR_HUMAN

    def stage(
        self,
        delta: TaskDelta,
        *,
        on_progress=None,
    ) -> ReuseReport:
        state = self._require_state()
        replacing_stopped_stage = (
            state.staged is not None and state.status is SessionStatus.CANCELLED
        )
        if state.staged is not None and not replacing_stopped_stage:
            raise RuntimeError("a task update is already staged")
        base_task = state.staged.task if replacing_stopped_stage else state.task
        base_graph = state.staged.graph if replacing_stopped_stage else state.graph
        base_artifacts = (
            state.staged.artifacts if replacing_stopped_stage else self.artifacts
        )
        task = self.domain_agent.run_module(
            "correction.apply_delta",
            lambda: apply_task_delta(base_task, delta),
            on_progress=on_progress,
        )
        graph = self.domain_agent.run_module(
            "correction.build_graph",
            lambda: self.domain_agent.build_graph(task),
            on_progress=on_progress,
        )
        artifacts, report = self.domain_agent.run_module(
            "correction.reconcile_graphs",
            lambda: reconcile_graphs(base_graph, graph, base_artifacts),
            on_progress=on_progress,
        )
        self.domain_agent.run_module(
            "correction.put_inputs",
            lambda: self._put_inputs(task, artifacts),
            on_progress=on_progress,
        )
        cached_reuse = self.domain_agent.run_module(
            "correction.restore_artifact_cache",
            lambda: self._restore_cached(graph, artifacts),
            on_progress=on_progress,
        )
        if cached_reuse:
            report = ReuseReport(
                reused_nodes=tuple(sorted(set(report.reused_nodes) | set(cached_reuse))),
                invalidated_nodes=tuple(
                    node_id for node_id in report.invalidated_nodes
                    if node_id not in set(cached_reuse)
                ),
                added_nodes=tuple(
                    node_id for node_id in report.added_nodes
                    if node_id not in set(cached_reuse)
                ),
                removed_nodes=report.removed_nodes,
            )
        state.staged = StagedExecution(
            base_version=state.version,
            task=task,
            graph=graph,
            artifacts=artifacts,
            reuse_report=report,
            rationale=delta.rationale,
        )
        return report

    def execute_staged(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        staged = self._require_staged()
        staged.status = StageStatus.RUNNING
        state.status = SessionStatus.RUNNING
        try:
            result = self.domain_agent.run_module(
                "execution.runtime",
                lambda: self.runtime.run(
                    staged.graph,
                    staged.artifacts,
                    execution_version=state.version + 1,
                    on_progress=on_progress,
                ),
                on_progress=on_progress,
            )
        except ResolutionRequired as exc:
            self._archive_completed(staged.graph, staged.artifacts)
            resolution = exc.pending(state.version + 1)
            state.pending_interaction = PendingInteraction.semantic_resolution(
                resolution,
                continuation=ContinuationKind.STAGED_EXECUTION,
            )
            state.status = SessionStatus.WAITING_FOR_HUMAN
            staged.status = StageStatus.WAITING_FOR_HUMAN
            raise
        except Exception:
            self._archive_completed(staged.graph, staged.artifacts)
            state.status = SessionStatus.FAILED
            staged.status = StageStatus.FAILED
            raise
        self._archive_completed(staged.graph, staged.artifacts)
        if result.stopped:
            state.status = SessionStatus.CANCELLED
            return result
        self._commit_staged(result)
        return result

    def execute_staged_until_note_confirmation(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        staged = self._require_staged()
        staged.status = StageStatus.RUNNING
        state.status = SessionStatus.RUNNING
        try:
            result = self.domain_agent.run_module(
                "execution.note_preflight",
                lambda: self.runtime.run(
                    staged.graph,
                    staged.artifacts,
                    execution_version=state.version + 1,
                    on_progress=on_progress,
                    defer_node=self._defer_note_requirement,
                ),
                on_progress=on_progress,
            )
        except ResolutionRequired as exc:
            self._archive_completed(staged.graph, staged.artifacts)
            resolution = exc.pending(state.version + 1)
            state.pending_interaction = PendingInteraction.semantic_resolution(
                resolution,
                continuation=ContinuationKind.STAGED_NOTE_PREFLIGHT,
            )
            state.status = SessionStatus.WAITING_FOR_HUMAN
            staged.status = StageStatus.WAITING_FOR_HUMAN
            raise
        except Exception:
            self._archive_completed(staged.graph, staged.artifacts)
            state.status = SessionStatus.FAILED
            staged.status = StageStatus.FAILED
            raise
        self._archive_completed(staged.graph, staged.artifacts)
        state.status = SessionStatus.CANCELLED if result.stopped else SessionStatus.READY
        staged.status = StageStatus.STAGED
        return result

    def resume_initial_note_preflight_after_resolution(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        if state.pending_interaction is not None:
            raise RuntimeError("clear the pending interaction before resuming")
        for node in state.graph.nodes.values():
            if node.status in {NodeStatus.FAILED, NodeStatus.CANCELLED, NodeStatus.RUNNING}:
                node.reset_for_execution()
        return self.execute_until_note_confirmation(on_progress=on_progress)

    def resume_staged_note_preflight_after_resolution(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        staged = self._require_staged()
        if state.pending_interaction is not None:
            raise RuntimeError("clear the pending interaction before resuming")
        for node in staged.graph.nodes.values():
            if node.status in {NodeStatus.FAILED, NodeStatus.CANCELLED, NodeStatus.RUNNING}:
                node.reset_for_execution()
        return self.execute_staged_until_note_confirmation(on_progress=on_progress)

    def resume_initial_after_resolution(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        if state.staged is not None:
            raise RuntimeError(
                "initial-execution resume cannot run with a staged update"
            )
        if state.pending_interaction is not None:
            raise RuntimeError(
                "clear the pending interaction before resuming"
            )
        for node in state.graph.nodes.values():
            if node.status in {
                NodeStatus.FAILED,
                NodeStatus.CANCELLED,
                NodeStatus.RUNNING,
            }:
                node.reset_for_execution()
        return self.execute(on_progress=on_progress)

    def resume_after_resolution(self, on_progress=None) -> RuntimeResult:
        state = self._require_state()
        staged = self._require_staged()
        if state.pending_interaction is not None:
            raise RuntimeError("clear the pending interaction before resuming")
        for node in staged.graph.nodes.values():
            if node.status in {
                NodeStatus.FAILED,
                NodeStatus.CANCELLED,
                NodeStatus.RUNNING,
            }:
                node.reset_for_execution()
        return self.execute_staged(on_progress=on_progress)

    def resume_stopped(self, on_progress=None) -> RuntimeResult:
        """Resume a cooperatively stopped graph without discarding completed work."""
        state = self._require_state()
        if state.status is not SessionStatus.CANCELLED:
            raise RuntimeError("the session is not stopped")
        if state.staged is not None:
            for node in state.staged.graph.nodes.values():
                if node.status in {NodeStatus.CANCELLED, NodeStatus.RUNNING}:
                    node.reset_for_execution()
            return self.execute_staged(on_progress=on_progress)
        for node in state.graph.nodes.values():
            if node.status in {NodeStatus.CANCELLED, NodeStatus.RUNNING}:
                node.reset_for_execution()
        return self.execute(on_progress=on_progress)

    def request_correction_fallback_confirmation(
        self, *, original_message: str, reason: str
    ) -> None:
        state = self._require_state()
        state.pending_interaction = PendingInteraction.correction_fallback_confirmation(
            original_message=original_message, reason=reason
        )
        state.status = SessionStatus.WAITING_FOR_HUMAN

    def request_note_execution_confirmation(
        self,
        *,
        continuation: ContinuationKind,
        original_message: str | None = None,
    ) -> PendingInteraction:
        state = self._require_state()
        pending = PendingInteraction.note_execution_confirmation(
            continuation=continuation,
            original_message=original_message,
            question=(
                "This cohort uses discharge-note evidence. The first run may need to build "
                "a note index and can take substantially longer than structured criteria. "
                "Proceed with note retrieval and LLM adjudication?"
            ),
        )
        state.pending_interaction = pending
        state.status = SessionStatus.WAITING_FOR_HUMAN
        if state.staged is not None:
            state.staged.status = StageStatus.WAITING_FOR_HUMAN
        return pending

    def prepare_without_notes(self, *, on_progress=None) -> ReuseReport:
        """Prune note criteria from the pending initial/staged task and rebuild safely.

        The source TaskSpec has already been parsed.  This method performs only
        deterministic structural pruning and graph reconciliation. No parser or
        LLM is invoked.
        """
        from agents.cohort.note_policy import without_note_requirements

        state = self._require_state()
        pending = state.pending_interaction
        if (
            pending is None
            or pending.kind is not PendingInteractionKind.NOTE_EXECUTION_CONFIRMATION
        ):
            raise RuntimeError("no note-execution confirmation is pending")

        continuation = pending.continuation
        if continuation is ContinuationKind.INITIAL_EXECUTION:
            task = without_note_requirements(state.task)
            graph = self.domain_agent.run_module(
                "notes_without.build_graph",
                lambda: self.domain_agent.build_graph(task),
                on_progress=on_progress,
            )
            artifacts, report = self.domain_agent.run_module(
                "notes_without.reconcile_graphs",
                lambda: reconcile_graphs(state.graph, graph, self.artifacts),
                on_progress=on_progress,
            )
            self.domain_agent.run_module(
                "notes_without.put_inputs",
                lambda: self._put_inputs(task, artifacts),
                on_progress=on_progress,
            )
            state.task = task
            state.graph = graph
            self.artifacts = artifacts
            state.pending_interaction = None
            state.status = SessionStatus.READY
            return report

        if continuation is not ContinuationKind.STAGED_EXECUTION:
            raise RuntimeError(f"unsupported note continuation: {continuation}")
        staged = self._require_staged()
        task = without_note_requirements(staged.task)
        graph = self.domain_agent.run_module(
            "notes_without.build_staged_graph",
            lambda: self.domain_agent.build_graph(task),
            on_progress=on_progress,
        )
        # Reconcile against the committed graph so the staged reuse report remains
        # meaningful relative to the state the correction is replacing.
        artifacts, report = self.domain_agent.run_module(
            "notes_without.reconcile_staged_graph",
            lambda: reconcile_graphs(state.graph, graph, self.artifacts),
            on_progress=on_progress,
        )
        self.domain_agent.run_module(
            "notes_without.put_staged_inputs",
            lambda: self._put_inputs(task, artifacts),
            on_progress=on_progress,
        )
        state.staged = StagedExecution(
            base_version=staged.base_version,
            task=task,
            graph=graph,
            artifacts=artifacts,
            reuse_report=report,
            rationale=staged.rationale,
            correction_plan=staged.correction_plan,
        )
        state.pending_interaction = None
        state.status = SessionStatus.READY
        return report

    def clear_pending_interaction(self) -> None:
        state = self._require_state()
        state.pending_interaction = None
        state.status = SessionStatus.READY

    def supersede_pending_for_correction(self) -> None:
        """Make the current pending semantic question yield to newer task intent.

        A staged graph is deliberately retained and marked as the stopped base so
        correction analysis/diffing can compare against the user's latest intended
        (but unresolved) task and preserve completed artifacts. Initial execution
        similarly keeps its graph/artifacts while clearing only the stale question.
        """
        state = self._require_state()
        pending = state.pending_interaction
        if pending is None:
            return
        state.pending_interaction = None
        self.mark_current_base_stopped()

    def mark_current_base_stopped(self) -> None:
        """Mark the current committed/staged graph as a reconcilable stopped base."""
        state = self._require_state()
        state.status = SessionStatus.CANCELLED
        if state.staged is not None:
            state.staged.status = StageStatus.STAGED

    def discard_staged(self) -> None:
        state = self._require_state()
        state.staged = None
        state.pending_interaction = None
        state.status = SessionStatus.READY


    def enter_manual_fallback(self) -> ManualEditForm:
        state = self._require_state()
        if state.staged is not None:
            raise RuntimeError(
                "cannot enter manual fallback while a task update is staged"
            )
        form = self.domain_agent.build_manual_edit_form(
            state.task,
            version=state.version,
        )
        state.pending_interaction = PendingInteraction.manual_edit(form)
        state.status = SessionStatus.WAITING_FOR_HUMAN
        return form

    def cancel_manual_fallback(self) -> None:
        state = self._require_state()
        if (
            state.pending_interaction is not None
            and state.pending_interaction.kind
            is PendingInteractionKind.MANUAL_EDIT
        ):
            state.pending_interaction = None
        state.status = SessionStatus.READY

    def compile_manual_edit(
        self,
        submitted: dict[str, Any],
    ) -> ManualEditResult:
        state = self._require_state()
        pending = state.pending_interaction
        if (
            pending is None
            or pending.kind is not PendingInteractionKind.MANUAL_EDIT
        ):
            raise RuntimeError("manual fallback mode is not active")
        return self.domain_agent.compile_manual_edit(
            state.task,
            submitted,
        )

    def apply_manual_edit(
        self,
        submitted: dict[str, Any],
    ) -> tuple[ManualEditResult, ReuseReport | None]:
        result = self.compile_manual_edit(submitted)
        if not result.valid:
            return result, None
        self.cancel_manual_fallback()
        report = self.stage(result.task_delta)
        return result, report

    def request_stop(
        self, *, reason: str = "user_stop", checkpoint_current: bool = False
    ) -> None:
        """Request a boundary yield, optionally at a resumable batch checkpoint."""
        self._yield_reason = reason
        self._yield_event.set()
        self.runtime.request_stop(checkpoint_current=checkpoint_current)

    def request_checkpoint_stop(self) -> None:
        """Stop checkpointable infrastructure after its current completed batch."""
        self.runtime.request_checkpoint_stop()

    def begin_parallel_checkpoint(self) -> None:
        self.runtime.begin_parallel_checkpoint()

    def finish_parallel_checkpoint(self, *, cancel_current: bool) -> None:
        self.runtime.finish_parallel_checkpoint(cancel_current=cancel_current)

    def clear_stop_request(self) -> None:
        self._yield_event.clear()
        self._yield_reason = None
        self.runtime.clear_stop()



    def has_pending_note_work(self, *, staged: bool | None = None) -> bool:
        """Return whether the selected graph still has an unevaluated note criterion.

        This is derived from graph state rather than persisted separately.  When
        ``staged`` is omitted, prefer the staged graph when one exists.
        """
        state = self._require_state()
        use_staged = (state.staged is not None) if staged is None else bool(staged)
        graph = state.staged.graph if use_staged and state.staged is not None else state.graph
        return any(
            node.operator_name == "evaluate_requirement"
            and node.config.get("requirement_kind") == "note"
            and node.status is not NodeStatus.COMPLETED
            for node in graph.nodes.values()
        )

    def requirement_statuses(self) -> dict[str, str]:
        """Derive requirement lifecycle from graph state without duplicating it."""
        state = self._require_state()
        graph = state.staged.graph if state.staged is not None else state.graph
        task = state.staged.task if state.staged is not None else state.task
        result: dict[str, str] = {}
        for requirement in task.requirements:
            nodes = [
                node for node in graph.nodes.values()
                if node.config.get("requirement_id") == requirement.requirement_id
            ]
            statuses = {node.status for node in nodes}
            if NodeStatus.RUNNING in statuses:
                result[requirement.requirement_id] = "active"
            elif nodes and all(node.status is NodeStatus.COMPLETED for node in nodes):
                result[requirement.requirement_id] = "resolved"
            else:
                result[requirement.requirement_id] = "unresolved"
        return result

    @property
    def yield_requested(self) -> bool:
        return self._yield_event.is_set()

    @property
    def yield_reason(self) -> str | None:
        return self._yield_reason

    def set_execution_backend(
        self,
        backend: ExecutionBackend | str,
    ) -> None:
        selected = (
            backend
            if isinstance(backend, ExecutionBackend)
            else ExecutionBackend(str(backend))
        )
        if selected is self.execution_backend:
            return
        if self.state is not None and self.state.status is SessionStatus.RUNNING:
            raise RuntimeError("cannot switch execution backend while a run is active")

        was_stopped = self.runtime.stop_requested
        self.runtime = self._build_runtime(
            selected,
            max_workers=self.runtime.max_workers,
            execution_mode=self.runtime.execution_mode,
        )
        if was_stopped:
            self.runtime.request_stop()
        self.execution_backend = selected

    def set_execution_mode(
        self,
        mode: ExecutionMode | str,
        *,
        workers: int = 4,
    ) -> None:
        selected_mode = mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
        selected_workers = 1
        self.max_workers = selected_workers
        self.runtime.max_workers = selected_workers
        self.runtime.execution_mode = selected_mode
    def summary(self) -> dict[str, Any]:
        return self.domain_agent.summarize(self.artifacts)

    def _commit_staged(self, result: RuntimeResult) -> None:
        old = self._require_state()
        staged = self._require_staged()
        self.artifacts = staged.artifacts
        self.state = AgentState(
            task=staged.task,
            graph=staged.graph,
            version=old.version + 1,
            status=SessionStatus.COMPLETED,
            last_reuse_report=staged.reuse_report,
            last_correction_plan=staged.correction_plan,
        )

    def _node_cache_key(self, graph: ExecutionGraph, node_id: str) -> str:
        """Fingerprint a node together with its full semantic dependency lineage.

        A direct-parent-only key is not sufficient: a parent computation can keep
        the same node fingerprint while one of *its* semantic inputs changes.
        Folding parent cache keys recursively makes cached outputs specific to the
        complete upstream semantics without storing another lifecycle/state field.
        """
        memo: dict[str, str] = {}

        def lineage_key(current_id: str) -> str:
            cached_key = memo.get(current_id)
            if cached_key is not None:
                return cached_key
            current = graph.nodes[current_id]
            payload = {
                "node": current.semantic_fingerprint,
                "parents": sorted(
                    lineage_key(parent)
                    for parent in graph.parents(current_id)
                ),
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            key = hashlib.sha256(encoded).hexdigest()
            memo[current_id] = key
            return key

        return lineage_key(node_id)

    def _archive_completed(
        self, graph: ExecutionGraph, store: InMemoryArtifactStore
    ) -> None:
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if node.operator_name is None or node.status is not NodeStatus.COMPLETED:
                continue
            if not all(store.contains(output_id) for output_id in node.outputs):
                continue
            self.artifact_cache[self._node_cache_key(graph, node_id)] = tuple(
                store.get(output_id) for output_id in node.outputs
            )

    def _restore_cached(
        self, graph: ExecutionGraph, store: InMemoryArtifactStore
    ) -> tuple[str, ...]:
        restored: list[str] = []
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if node.operator_name is None or node.status is NodeStatus.COMPLETED:
                continue
            # A cached child is only valid once every graph dependency is itself
            # known-valid for the current graph. Some operators consume input
            # artifacts directly while depending on computation nodes for semantic
            # ordering, so checking artifact presence alone can admit stale output.
            if not all(
                graph.nodes[parent].status is NodeStatus.COMPLETED
                for parent in graph.parents(node_id)
            ):
                continue
            if not all(store.contains(input_id) for input_id in node.inputs):
                continue
            cached = self.artifact_cache.get(self._node_cache_key(graph, node_id))
            if cached is None:
                continue
            if {artifact.artifact_id for artifact in cached} != set(node.outputs):
                continue
            store.put_many(cached)
            node.status = NodeStatus.COMPLETED
            node.error = None
            durations = [
                artifact.metadata.get("producer_duration_ms")
                for artifact in cached
                if artifact.metadata.get("producer_duration_ms") is not None
            ]
            if durations:
                node.duration_ms = float(max(durations))
            for artifact in cached:
                components = artifact.metadata.get("producer_component_timings_ms")
                if isinstance(components, dict):
                    node.component_timings_ms = {
                        str(key): float(value) for key, value in components.items()
                    }
                    break
            restored.append(node_id)
        return tuple(restored)

    def _put_inputs(
        self, task: TaskSpec, store: InMemoryArtifactStore
    ) -> None:
        for artifact_id, value in self.domain_agent.build_input_values(task).items():
            store.put(Artifact(
                artifact_id=artifact_id,
                artifact_type=(
                    "task_spec" if artifact_id == "task_spec" else "requirement"
                ),
                value=value,
                metadata={"domain": self.domain_agent.domain},
            ))

    def _require_state(self) -> AgentState:
        if self.state is None:
            raise RuntimeError("session has not been initialized")
        return self.state

    def _require_staged(self) -> StagedExecution:
        state = self._require_state()
        if state.staged is None:
            raise RuntimeError("no staged task update exists")
        return state.staged
