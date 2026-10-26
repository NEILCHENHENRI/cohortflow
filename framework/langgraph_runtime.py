from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
import threading
from typing import Annotated, Any, Callable, TypedDict
from uuid import uuid4

from framework.artifact_store import ArtifactStore
from framework.failure import FailureEvent
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus
from framework.runtime import ExecutionMode, GraphRuntime, RuntimeResult
from framework.timing import measure


class _LangGraphExecutionState(TypedDict):
    completed: Annotated[list[str], add]


class _StopRequested(InterruptedError):
    pass


def _load_langgraph():
    """Load LangGraph lazily so Native remains usable without the dependency."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised when dependency absent
        raise RuntimeError(
            "LangGraph backend requires the 'langgraph' package. "
            "Install project requirements or run `pip install langgraph`."
        ) from exc
    return StateGraph, START, END, InMemorySaver


@dataclass(slots=True)
class LangGraphRuntime(GraphRuntime):
    """LangGraph-backed adapter for the framework's existing execution DAG.

    The adapter deliberately reuses GraphRuntime's operator invocation, timing,
    failure capture, artifact commit, and stop semantics. LangGraph owns graph
    scheduling/checkpoint mechanics only. TaskSpec semantics, invalidation,
    artifact reuse, and operator behavior remain framework-owned.
    """

    _commit_lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )
    last_thread_id: str | None = field(init=False, default=None)

    def run(
        self,
        graph: ExecutionGraph,
        artifact_store: ArtifactStore,
        *,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None = None,
        defer_node: Callable[[ExecutionNode], bool] | None = None,
    ) -> RuntimeResult:
        StateGraph, START, END, InMemorySaver = _load_langgraph()

        self._emit(
            on_progress,
            "execution_started",
            f"LangGraph execution v{execution_version} started",
        )
        for node in graph.nodes.values():
            if node.operator_name is not None and node.status is NodeStatus.COMPLETED:
                self._emit(
                    on_progress,
                    "node_preserved",
                    self._node_progress_payload(node, phase="preserved"),
                )

        failed_before_run = tuple(
            node.node_id
            for node in graph.nodes.values()
            if node.status is NodeStatus.FAILED
        )
        if failed_before_run:
            raise RuntimeError(
                f"execution failed at nodes: {list(failed_before_run)}"
            )

        active = self._active_nodes(graph, defer_node=defer_node)
        if active and not self.stop_requested:
            builder = StateGraph(_LangGraphExecutionState)
            for node in active:
                builder.add_node(
                    node.node_id,
                    self._node_action(
                        node,
                        artifact_store,
                        execution_version,
                        on_progress,
                    ),
                )

            parents = self._effective_parents(graph, active)
            active_ids = {node.node_id for node in active}
            children: dict[str, set[str]] = {node_id: set() for node_id in active_ids}
            for node_id, node_parents in parents.items():
                for parent in node_parents:
                    children[parent].add(node_id)

            for node in active:
                node_parents = sorted(parents[node.node_id])
                if not node_parents:
                    builder.add_edge(START, node.node_id)
                elif len(node_parents) == 1:
                    builder.add_edge(node_parents[0], node.node_id)
                else:
                    # StateGraph treats a list source as a synchronization edge:
                    # the child runs once after every listed parent completes.
                    builder.add_edge(node_parents, node.node_id)

            for node_id, node_children in children.items():
                if not node_children:
                    builder.add_edge(node_id, END)

            checkpointer = InMemorySaver()
            compiled = builder.compile(checkpointer=checkpointer)
            self.last_thread_id = (
                f"steerable-{id(graph):x}-v{execution_version}-{uuid4().hex[:8]}"
            )
            config = {"configurable": {"thread_id": self.last_thread_id}}
            try:
                with measure(
                    "execution.langgraph_invoke",
                    category="execution_internal",
                ):
                    compiled.invoke({"completed": []}, config=config)
            except _StopRequested:
                pass

        return self._finalize_langgraph(
            graph,
            on_progress=on_progress,
            allow_deferred=defer_node is not None,
        )

    def _active_nodes(
        self,
        graph: ExecutionGraph,
        *,
        defer_node: Callable[[ExecutionNode], bool] | None = None,
    ) -> list[ExecutionNode]:
        runnable_statuses = {
            NodeStatus.PENDING,
            NodeStatus.INVALIDATED,
            NodeStatus.CANCELLED,
        }
        blocked: set[str] = set()
        if defer_node is not None:
            deferred = {
                node.node_id for node in graph.nodes.values()
                if node.kind is NodeKind.COMPUTATION and defer_node(node)
            }
            blocked = deferred | graph.descendants(deferred)
        return [
            graph.nodes[node_id]
            for node_id in graph.topological_order()
            if (
                graph.nodes[node_id].kind is NodeKind.COMPUTATION
                and graph.nodes[node_id].status in runnable_statuses
                and node_id not in blocked
            )
        ]

    def _effective_parents(
        self,
        graph: ExecutionGraph,
        active: list[ExecutionNode],
    ) -> dict[str, set[str]]:
        active_ids = {node.node_id for node in active}
        parents: dict[str, set[str]] = {
            node.node_id: {
                parent
                for parent in graph.parents(node.node_id)
                if parent in active_ids
            }
            for node in active
        }

        ordered = [node.node_id for node in active]
        # Keep graph execution single-lane in both steering modes. Parallel
        # steering only overlaps a correction interpretation with one active
        # LLM/manual action. It is not general graph parallelism.
        for previous, current in zip(ordered, ordered[1:]):
            parents[current].add(previous)
        return parents

    def _node_action(
        self,
        node: ExecutionNode,
        artifact_store: ArtifactStore,
        execution_version: int,
        on_progress: Callable[[str, str], None] | None,
    ):
        def action(_state: _LangGraphExecutionState) -> dict[str, list[str]]:
            if self.stop_requested:
                node.status = NodeStatus.CANCELLED
                raise _StopRequested()
            try:
                output = self._call_operator(
                    node,
                    artifact_store,
                    execution_version,
                    on_progress,
                )
                if node.execution_version == execution_version:
                    # LangGraph may execute independent nodes concurrently. Keep
                    # artifact commit/status mutation serialized just as the
                    # native runtime commits worker results on its coordinator.
                    with self._commit_lock:
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
                            node,
                            phase="completed",
                            duration_ms=node.duration_ms,
                        ),
                    )
                elif node.status is NodeStatus.RUNNING:
                    node.status = NodeStatus.CANCELLED
                return {"completed": [node.node_id]}
            except _StopRequested:
                node.status = NodeStatus.CANCELLED
                raise
            except InterruptedError:
                node.status = NodeStatus.CANCELLED
                raise _StopRequested()
            except Exception as exc:
                node.status = NodeStatus.FAILED
                node.error = f"{type(exc).__name__}: {exc}"
                node.failure = node.failure or FailureEvent.capture(
                    exc,
                    origin="operator",
                    node_id=node.node_id,
                    operator_name=node.operator_name,
                    metadata={
                        "config": dict(node.config),
                        "execution_backend": "langgraph",
                    },
                )
                raise

        return action

    def _finalize_langgraph(
        self,
        graph: ExecutionGraph,
        *,
        on_progress: Callable[[str, str], None] | None,
        allow_deferred: bool = False,
    ) -> RuntimeResult:
        with measure("execution.finalize_status", category="execution_internal"):
            failed = tuple(
                node.node_id
                for node in graph.nodes.values()
                if node.status is NodeStatus.FAILED
            )
            completed = tuple(
                node.node_id
                for node in graph.nodes.values()
                if node.status is NodeStatus.COMPLETED
            )
            unresolved = self.scheduler.unresolved_nodes(graph)

        if failed:
            raise RuntimeError(f"execution failed at nodes: {list(failed)}")
        if unresolved and not self.stop_requested and not allow_deferred:
            raise RuntimeError(
                "execution stalled. Unresolved nodes: " + ", ".join(unresolved)
            )

        self._emit(
            on_progress,
            "execution_stopped" if self.stop_requested else "execution_finished",
            (
                f"LangGraph execution stopped with {len(completed)} completed node(s) preserved"
                if self.stop_requested
                else f"LangGraph execution finished with {len(completed)} completed node(s)"
            ),
        )
        return RuntimeResult(
            stopped=self.stop_requested,
            completed_nodes=completed,
            failed_nodes=failed,
            unresolved_nodes=unresolved,
        )
