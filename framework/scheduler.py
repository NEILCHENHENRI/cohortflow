from __future__ import annotations

from dataclasses import dataclass

from framework.artifact_store import ArtifactStore
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus


@dataclass(slots=True)
class Scheduler:
    def ready_nodes(
        self,
        graph: ExecutionGraph,
        artifact_store: ArtifactStore,
    ) -> list[ExecutionNode]:
        ready: list[ExecutionNode] = []
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if node.kind is not NodeKind.COMPUTATION:
                continue
            if node.status not in {
                NodeStatus.PENDING,
                NodeStatus.INVALIDATED,
                NodeStatus.CANCELLED,
            }:
                continue
            if not all(
                graph.nodes[parent].status == NodeStatus.COMPLETED
                for parent in graph.parents(node_id)
            ):
                continue
            if not all(
                artifact_store.contains(artifact_id)
                for artifact_id in node.inputs
            ):
                continue
            ready.append(node)
        return ready

    def unresolved_nodes(
        self,
        graph: ExecutionGraph,
    ) -> tuple[str, ...]:
        return tuple(
            node.node_id
            for node in graph.nodes.values()
            if (
                node.kind is NodeKind.COMPUTATION
                and node.status in {
                    NodeStatus.PENDING,
                    NodeStatus.INVALIDATED,
                    NodeStatus.RUNNING,
                    NodeStatus.CANCELLED,
                }
            )
        )
