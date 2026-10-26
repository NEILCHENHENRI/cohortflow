from presentation.models import GraphNodeView, GraphView


def test_graph_view_is_domain_agnostic():
    graph = GraphView(
        nodes=(
            GraphNodeView(
                node_id="n1",
                label="Do work",
                status="completed",
                category="transform",
            ),
        ),
        edges=(),
        version=2,
    )
    assert graph.nodes[0].label == "Do work"
    assert graph.version == 2
