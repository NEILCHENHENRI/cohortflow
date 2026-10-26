from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from heapq import heapify, heappop, heappush
import json
from typing import Any, Iterable
from framework.failure import FailureEvent


class NodeKind(str, Enum):
    INPUT = "input"
    COMPUTATION = "computation"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ExecutionNode:
    node_id: str
    kind: NodeKind
    operator_name: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    error: str | None = None
    failure: FailureEvent | None = None
    execution_version: int | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    component_timings_ms: dict[str, float] = field(default_factory=dict)
    attempt_timings: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id cannot be empty")
        if self.kind is NodeKind.COMPUTATION and not self.operator_name:
            raise ValueError("computation nodes require operator_name")
        if self.kind is NodeKind.INPUT and self.operator_name is not None:
            raise ValueError("input nodes cannot declare operator_name")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"duplicate outputs for node {self.node_id}")

    @property
    def semantic_fingerprint(self) -> str:
        payload = {
            "kind": self.kind.value,
            "operator_name": self.operator_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "config": self.config,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def reset_for_execution(self) -> None:
        self.status = NodeStatus.PENDING
        self.error = None
        self.failure = None
        self.execution_version = None
        self.started_at = None
        self.completed_at = None
        self.duration_ms = None
        self.component_timings_ms = {}
        self.attempt_timings = []


@dataclass(slots=True)
class ExecutionGraph:
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)
    dependencies: dict[str, set[str]] = field(default_factory=dict)

    def add_node(
        self,
        node: ExecutionNode,
        *,
        depends_on: Iterable[str] = (),
    ) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        parents = set(depends_on)
        missing = parents.difference(self.nodes)
        if missing:
            raise KeyError(
                "dependencies must be added before their child: "
                f"{sorted(missing)}"
            )
        self.nodes[node.node_id] = node
        self.dependencies[node.node_id] = parents
        self.topological_order()

    def parents(self, node_id: str) -> set[str]:
        self._require_node(node_id)
        return set(self.dependencies[node_id])

    def children(self, node_id: str) -> set[str]:
        self._require_node(node_id)
        return {
            child
            for child, parents in self.dependencies.items()
            if node_id in parents
        }

    def descendants(self, node_ids: Iterable[str]) -> set[str]:
        frontier = list(node_ids)
        visited: set[str] = set()
        while frontier:
            current = frontier.pop()
            for child in self.children(current):
                if child not in visited:
                    visited.add(child)
                    frontier.append(child)
        return visited

    def invalidate(
        self,
        node_ids: Iterable[str],
        *,
        include_descendants: bool = True,
    ) -> set[str]:
        initial = set(node_ids)
        for node_id in initial:
            self._require_node(node_id)
        affected = set(initial)
        if include_descendants:
            affected.update(self.descendants(initial))
        for node_id in affected:
            node = self.nodes[node_id]
            if node.kind is NodeKind.COMPUTATION:
                node.status = NodeStatus.INVALIDATED
                node.error = None
        return affected

    def topological_order(self) -> list[str]:
        indegree = {
            node_id: len(parents)
            for node_id, parents in self.dependencies.items()
        }
        children = {node_id: set() for node_id in self.nodes}
        for child, parents in self.dependencies.items():
            for parent in parents:
                children[parent].add(child)

        # Keep the lexicographically smallest ready node first without sorting
        # the entire frontier every time a child becomes ready.
        ready = [
            node_id for node_id, degree in indegree.items()
            if degree == 0
        ]
        heapify(ready)
        order: list[str] = []
        while ready:
            current = heappop(ready)
            order.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heappush(ready, child)

        if len(order) != len(self.nodes):
            raise ValueError("execution graph contains a cycle")
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                node_id: {
                    "kind": node.kind.value,
                    "operator_name": node.operator_name,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "config": node.config,
                    "status": node.status.value,
                    "error": node.error,
                    "failure": None if node.failure is None else {
                        "failure_id": node.failure.failure_id,
                        "category": node.failure.category.value,
                        "recovery": node.failure.recovery.value,
                        "exception_type": node.failure.exception_type,
                        "message": node.failure.message,
                    },
                    "execution_version": node.execution_version,
                    "duration_ms": node.duration_ms,
                    "component_timings_ms": dict(node.component_timings_ms),
                    "attempt_timings": [dict(item) for item in node.attempt_timings],
                    "fingerprint": node.semantic_fingerprint,
                    "depends_on": sorted(self.dependencies[node_id]),
                }
                for node_id, node in self.nodes.items()
            }
        }

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"unknown node: {node_id}")
