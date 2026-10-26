from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import threading
from copy import deepcopy
import time
from typing import Any, Callable

from framework.artifact import Artifact, ArtifactProvenance
from framework.failure import FailureEvent
from framework.artifact_store import ArtifactStore
from framework.graph import ExecutionGraph, ExecutionNode, NodeStatus
from framework.operator import (
    SideEffectLevel,
    OperatorContext,
    OperatorRegistry,
    operator_context_scope,
)
from framework.scheduler import Scheduler
from framework.timing import measure


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    stopped: bool
    completed_nodes: tuple[str, ...]
    failed_nodes: tuple[str, ...]
    unresolved_nodes: tuple[str, ...]


@dataclass(slots=True)
class GraphRuntime:
    registry: OperatorRegistry
    scheduler: Scheduler = field(default_factory=Scheduler)
    max_workers: int = 1
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    recovery_handler: Callable | None = field(default=None, repr=False, compare=False)
    recovery_enabled: Callable[[], bool] | None = field(default=None, repr=False, compare=False)
    _stop_event: threading.Event = field(
        init=False,
        default_factory=threading.Event,
        repr=False,
    )
    _checkpoint_stop_event: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )
    _parallel_checkpoint_pending: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )
    _parallel_checkpoint_done: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )

    def request_stop(self, *, checkpoint_current: bool = False) -> None:
        self._stop_event.set()
        if checkpoint_current:
            self._checkpoint_stop_event.set()

    def request_checkpoint_stop(self) -> None:
        self._checkpoint_stop_event.set()

    def begin_parallel_checkpoint(self) -> None:
        self._parallel_checkpoint_done.clear()
        self._parallel_checkpoint_pending.set()

    def finish_parallel_checkpoint(self, *, cancel_current: bool) -> None:
        if cancel_current:
            self._checkpoint_stop_event.set()
        self._parallel_checkpoint_done.set()

    def clear_stop(self) -> None:
        self._stop_event.clear()
        self._checkpoint_stop_event.clear()
        self._parallel_checkpoint_pending.clear()
        self._parallel_checkpoint_done.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def run(
        self,
        graph: ExecutionGraph,
        artifact_store: ArtifactStore,
        *,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None = None,
        defer_node: Callable[[ExecutionNode], bool] | None = None,
    ) -> RuntimeResult:
        self._emit(on_progress, "execution_started", f"Graph execution v{execution_version} started")
        for node in graph.nodes.values():
            if node.operator_name is not None and node.status is NodeStatus.COMPLETED:
                self._emit(
                    on_progress,
                    "node_preserved",
                    self._node_progress_payload(node, phase="preserved"),
                )

        while not self.stop_requested:
            with measure("execution.scheduler_ready", category="execution_internal"):
                ready = self.scheduler.ready_nodes(graph, artifact_store)
            if not ready:
                break

            if defer_node is not None:
                ready = [node for node in ready if not defer_node(node)]
                if not ready:
                    break

            self._execute_ready_nodes(
                ready,
                artifact_store,
                execution_version,
                on_progress,
            )

        with measure("execution.finalize_status", category="execution_internal"):
            failed = tuple(
                node.node_id
                for node in graph.nodes.values()
                if node.status == NodeStatus.FAILED
            )
            completed = tuple(
                node.node_id
                for node in graph.nodes.values()
                if node.status == NodeStatus.COMPLETED
            )
            unresolved = self.scheduler.unresolved_nodes(graph)

        if failed:
            raise RuntimeError(
                f"execution failed at nodes: {list(failed)}"
            )

        deferred_remaining = (
            defer_node is not None
            and any(
                node.kind is not None
                and node.status is not NodeStatus.COMPLETED
                and defer_node(node)
                for node in graph.nodes.values()
            )
        )
        if unresolved and not self.stop_requested and not deferred_remaining:
            raise RuntimeError(
                "execution stalled. Unresolved nodes: "
                + ", ".join(unresolved)
            )

        self._emit(
            on_progress,
            "execution_stopped" if self.stop_requested else "execution_finished",
            (
                f"Execution stopped with {len(completed)} completed node(s) preserved"
                if self.stop_requested
                else f"Execution finished with {len(completed)} completed node(s)"
            ),
        )
        return RuntimeResult(
            stopped=self.stop_requested,
            completed_nodes=completed,
            failed_nodes=failed,
            unresolved_nodes=unresolved,
        )


    def _execute_ready_nodes(
        self,
        nodes: list[ExecutionNode],
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ) -> None:
        # Cohort graph execution itself stays on one lane in both modes.
        # PARALLEL is reserved for overlapping one active LLM/manual action with
        # interpretation of a newly arrived correction.
        self._execute_sequential(nodes, artifact_store, execution_version, on_progress)

    def _execute_sequential(
        self,
        nodes: list[ExecutionNode],
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ) -> None:
        for node in nodes:
            if self.stop_requested:
                break
            self._execute_node(node, artifact_store, execution_version, on_progress)

    def _execute_parallel(
        self,
        nodes: list[ExecutionNode],
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ) -> None:
        with measure("execution.parallel_dispatch", category="execution_internal"):
            pool = ThreadPoolExecutor(max_workers=self.max_workers)
            futures = {
                pool.submit(
                    self._call_operator,
                    node,
                    artifact_store,
                    execution_version,
                    on_progress,
                ): node
                for node in nodes
            }
        try:
            for future in as_completed(futures):
                node = futures[future]
                try:
                    output = future.result()
                    if node.execution_version == execution_version:
                        self._timed_commit(
                            node,
                            output,
                            artifact_store,
                            execution_version,
                        )
                        self._emit(
                            on_progress,
                            "node_committed",
                            self._node_progress_payload(
                                node, phase="completed", duration_ms=node.duration_ms
                            ),
                        )
                    elif node.status == NodeStatus.RUNNING:
                        node.status = NodeStatus.CANCELLED
                except InterruptedError:
                    node.status = NodeStatus.CANCELLED
                except Exception as exc:
                    node.status = NodeStatus.FAILED
                    node.error = f"{type(exc).__name__}: {exc}"
                    node.failure = node.failure or FailureEvent.capture(
                        exc, origin="operator", node_id=node.node_id,
                        operator_name=node.operator_name,
                        metadata={"config": dict(node.config)},
                    )
                    raise
        finally:
            pool.shutdown(wait=True)

    def _execute_node(
        self,
        node: ExecutionNode,
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ) -> None:
        try:
            output = self._call_operator(
                node,
                artifact_store,
                execution_version,
                on_progress,
            )
            if node.execution_version == execution_version:
                self._timed_commit(
                    node,
                    output,
                    artifact_store,
                    execution_version,
                )
                self._emit(
                    on_progress,
                    "node_committed",
                    self._node_progress_payload(
                        node, phase="completed", duration_ms=node.duration_ms
                    ),
                )
            elif node.status == NodeStatus.RUNNING:
                node.status = NodeStatus.CANCELLED
        except InterruptedError:
            node.status = NodeStatus.CANCELLED
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.error = f"{type(exc).__name__}: {exc}"
            node.failure = node.failure or FailureEvent.capture(
                exc, origin="operator", node_id=node.node_id,
                operator_name=node.operator_name,
                metadata={"config": dict(node.config)},
            )
            raise

    def _call_operator(
        self,
        node: ExecutionNode,
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ) -> dict[str, Any]:
        if self.stop_requested:
            node.status = NodeStatus.CANCELLED
            return {}

        registered = self.registry.get(node.operator_name or "")
        node.status = NodeStatus.RUNNING
        node.failure = None
        node.execution_version = execution_version
        node.started_at = time.perf_counter()

        context = OperatorContext(
            execution_version=execution_version,
            # Ordinary operators remain atomic. Long note infrastructure uses
            # the separate checkpoint/boundary events below at explicit safe
            # batch boundaries.
            stop_event=threading.Event(),
            node_id=node.node_id,
            metadata={
                "interruptibility": registered.spec.interruptibility.value,
                "requirement_id": node.config.get("requirement_id"),
                "requirement_kind": node.config.get("requirement_kind"),
            },
            checkpoint_stop_event=self._checkpoint_stop_event,
            boundary_stop_event=self._stop_event,
            parallel_checkpoint_pending=self._parallel_checkpoint_pending,
            parallel_checkpoint_done=self._parallel_checkpoint_done,
            progress_callback=on_progress,
        )
        self._emit(
            on_progress,
            "node_started",
            self._node_progress_payload(node, phase="started"),
        )

        recovery_enabled = self.recovery_handler is not None and (
            self.recovery_enabled is None or self.recovery_enabled())
        retry_eligible = registered.spec.retryable and registered.spec.side_effects is SideEffectLevel.NONE
        safe_to_retry = False
        detached_inputs = None
        detached_config = None
        outcome = "completed"
        error_type: str | None = None
        try:
            with context.measure("input_materialization"):
                inputs = {
                    artifact_id: artifact_store.get(artifact_id).value
                    for artifact_id in node.inputs
                }
            if recovery_enabled and retry_eligible:
                # Capture pristine inputs before the first attempt, not after an
                # operator might have partially mutated its own working copy.
                try:
                    detached_inputs, detached_config = deepcopy((inputs, node.config))
                    inputs = deepcopy(detached_inputs)
                    safe_to_retry = True
                except Exception:
                    safe_to_retry = False
            with context.measure("operator_body"):
                with operator_context_scope(context):
                    config = deepcopy(detached_config) if safe_to_retry else dict(node.config)
                    return registered.function(inputs, config, context)
        except InterruptedError:
            outcome = "cancelled"
            error_type = "InterruptedError"
            raise
        except Exception as exc:
            from framework.resolution import ResolutionRequired
            from framework.steering import SteeringYield
            from framework.repair.models import RecoveryCheck, RepairStatus
            outcome = "raised"
            error_type = type(exc).__name__
            if recovery_enabled and not isinstance(exc, (ResolutionRequired, SteeringYield)):
                node.failure = FailureEvent.capture(exc, origin="operator", node_id=node.node_id,
                    operator_name=node.operator_name, metadata={"config": dict(node.config)})
                def retry():
                    if self.stop_requested or self._checkpoint_stop_event.is_set():
                        raise InterruptedError("Execution stopped before recovery retry")
                    with operator_context_scope(context):
                        value = registered.function(deepcopy(detached_inputs), deepcopy(detached_config), context)
                    valid = isinstance(value, dict) and set(value) == set(node.outputs)
                    return RecoveryCheck(valid, True, "Operator output contract checked.", value=value)
                try:
                    repaired = self.recovery_handler(node.failure, node, retry, safe_to_retry)
                except (InterruptedError, ResolutionRequired, SteeringYield):
                    raise
                except Exception:
                    # A malfunctioning repair service must not hide the original error.
                    repaired = None
                if repaired is not None and repaired.status is RepairStatus.RECOVERED:
                    if self.stop_requested:
                        raise InterruptedError("Execution stopped during recovery")
                    outcome, error_type = "recovered", None
                    return repaired.recovery_value
            raise
        finally:
            completed_at = time.perf_counter()
            duration_ms = (completed_at - node.started_at) * 1000.0
            # Crucially, persist subcomponent timings even if the operator raises
            # ResolutionRequired (or any other exception). Previously those
            # timings disappeared because this copy happened only on success.
            node.completed_at = completed_at
            node.duration_ms = duration_ms
            node.component_timings_ms = dict(context.component_timings_ms)
            node.attempt_timings.append({
                "attempt": len(node.attempt_timings) + 1,
                "execution_version": execution_version,
                "outcome": outcome,
                "error_type": error_type,
                "duration_ms": round(duration_ms, 3),
                "component_timings_ms": {
                    key: round(value, 3)
                    for key, value in sorted(context.component_timings_ms.items())
                },
            })
            self._emit(
                on_progress,
                "node_attempt_finished",
                self._node_progress_payload(
                    node, phase=outcome, duration_ms=duration_ms,
                ),
            )


    @staticmethod
    def _emit(callback, event_type: str, payload: str) -> None:
        if callback is not None:
            callback(event_type, payload)

    @staticmethod
    def _node_progress_payload(
        node: ExecutionNode,
        *,
        phase: str,
        duration_ms: float | None = None,
    ) -> str:
        import json
        payload = {
            "node_id": node.node_id,
            "operator": node.operator_name,
            "phase": phase,
            "status": node.status.value,
            "requirement_id": node.config.get("requirement_id"),
            "duration_ms": None if duration_ms is None else round(duration_ms, 3),
            "components_ms": {
                key: round(value, 3)
                for key, value in sorted(node.component_timings_ms.items())
            },
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _timed_commit(
        self,
        node: ExecutionNode,
        output: dict[str, Any],
        artifact_store: ArtifactStore,
        execution_version: int,
    ) -> None:
        started = time.perf_counter()
        try:
            with measure(
                "execution.commit_output",
                category="execution_internal",
                metadata={"node_id": node.node_id},
            ):
                self._commit_output(
                    node,
                    output,
                    artifact_store,
                    execution_version,
                )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            node.component_timings_ms["commit_output"] = (
                node.component_timings_ms.get("commit_output", 0.0)
                + elapsed_ms
            )
            if node.attempt_timings:
                latest = node.attempt_timings[-1]
                components = dict(latest.get("component_timings_ms", {}))
                components["commit_output"] = round(
                    components.get("commit_output", 0.0) + elapsed_ms,
                    3,
                )
                latest["component_timings_ms"] = components

    @staticmethod
    def _commit_output(
        node: ExecutionNode,
        output: dict[str, Any],
        artifact_store: ArtifactStore,
        execution_version: int,
    ) -> None:
        expected = set(node.outputs)
        actual = set(output)
        if expected != actual:
            raise ValueError(
                f"operator {node.operator_name!r} returned "
                f"{sorted(actual)}, expected {sorted(expected)}"
            )

        provenance = ArtifactProvenance(
            producer_node_id=node.node_id,
            producer_operator=node.operator_name,
            execution_version=execution_version,
            source_artifact_ids=node.inputs,
        )
        for artifact_id, value in output.items():
            artifact_store.put(
                Artifact(
                    artifact_id=artifact_id,
                    artifact_type="generic",
                    value=value,
                    provenance=provenance,
                    metadata={
                        "producer_node_fingerprint": node.semantic_fingerprint,
                        "producer_duration_ms": node.duration_ms,
                        "producer_component_timings_ms": dict(
                            node.component_timings_ms
                        ),
                    },
                )
            )

        node.status = NodeStatus.COMPLETED
        node.error = None
        node.failure = None
