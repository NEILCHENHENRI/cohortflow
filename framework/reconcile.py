from __future__ import annotations

from dataclasses import dataclass

from framework.artifact_store import ArtifactStore, InMemoryArtifactStore
from framework.graph import ExecutionGraph, NodeKind, NodeStatus


@dataclass(frozen=True, slots=True)
class ReuseReport:
    reused_nodes: tuple[str, ...] = ()
    invalidated_nodes: tuple[str, ...] = ()
    added_nodes: tuple[str, ...] = ()
    removed_nodes: tuple[str, ...] = ()


def reconcile_graphs(
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    old_store: ArtifactStore,
) -> tuple[InMemoryArtifactStore, ReuseReport]:
    reused: list[str] = []
    # Membership checks are independent of the ordered report below. Both
    # stable inputs and reused computations can satisfy a child's parents.
    reusable_parents: set[str] = set()
    invalidated: list[str] = []
    added: list[str] = []
    removed = sorted(set(old_graph.nodes) - set(new_graph.nodes))
    new_store = InMemoryArtifactStore()

    for node_id in new_graph.topological_order():
        new_node = new_graph.nodes[node_id]
        old_node = old_graph.nodes.get(node_id)

        if new_node.kind is NodeKind.INPUT:
            if (
                old_node is not None
                and old_node.kind is NodeKind.INPUT
                and old_node.semantic_fingerprint
                == new_node.semantic_fingerprint
            ):
                reusable_parents.add(node_id)
                for output_id in new_node.outputs:
                    if old_store.contains(output_id):
                        new_store.put(old_store.get(output_id))
            elif old_node is None:
                added.append(node_id)
            continue

        if old_node is None:
            added.append(node_id)
            new_node.status = NodeStatus.PENDING
            continue

        unchanged = (
            old_node.semantic_fingerprint
            == new_node.semantic_fingerprint
        )
        outputs_available = all(
            old_store.contains(output_id)
            for output_id in new_node.outputs
        )
        parents_reusable = all(
            parent in reusable_parents
            for parent in new_graph.parents(node_id)
        )

        if (
            unchanged
            and outputs_available
            and parents_reusable
            and old_node.status == NodeStatus.COMPLETED
        ):
            new_node.status = NodeStatus.COMPLETED
            new_node.duration_ms = old_node.duration_ms
            new_node.component_timings_ms = dict(old_node.component_timings_ms)
            new_node.attempt_timings = [dict(item) for item in old_node.attempt_timings]
            for output_id in new_node.outputs:
                new_store.put(old_store.get(output_id))
            reused.append(node_id)
            reusable_parents.add(node_id)
        else:
            new_node.status = NodeStatus.INVALIDATED
            invalidated.append(node_id)

    return new_store, ReuseReport(
        reused_nodes=tuple(reused),
        invalidated_nodes=tuple(invalidated),
        added_nodes=tuple(added),
        removed_nodes=tuple(removed),
    )
