from __future__ import annotations
from presentation.models import GraphView

# Bump this whenever _node_content/_node_style/_gate_svg's rendering logic
# changes. The cache signature below is otherwise built entirely from graph
# DATA (labels, categories, statuses) -- it has no way to know the code that
# turns that data into pixels has changed. st.session_state legitimately
# persists across a file-edit-triggered rerun (that's normal, intentional
# Streamlit behavior), so without this, a session that already cached a
# render before a graph.py update keeps replaying the stale pre-update
# output forever, since the underlying cohort data never actually changed.
_RENDER_VERSION = 10

_LAYER_HEIGHT = 185
_NODE_SPACING = 170


def _compute_layered_layout(
    node_ids: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    """A small, synchronous, server-side replacement for the frontend's
    async ElkJS-based auto-layout.

    The underlying React Flow library has a long-standing, still only
    partially resolved history of `fitView` failing to correctly fit the
    viewport right after nodes are added/repositioned asynchronously (e.g.
    xyflow/xyflow #4793, #4801, #3946, #1712) -- exactly the "everything
    collapses toward the origin" symptom this graph kept showing. Computing
    final pixel positions here, before anything reaches the frontend,
    removes the async race entirely: there is no "wait for the client-side
    layout engine to finish, then fit the view" handoff to get wrong,
    because positions are already correct in the same synchronous call that
    builds the node objects.

    Standard layered-DAG technique: each node's layer is the longest path
    from any root (so an edge that skips levels, e.g. input:task straight
    to validate:cohort, still places validate:cohort below everything it
    actually depends on). Within a layer, nodes are ordered by the average
    x of their already-placed parents (a simple barycenter heuristic) to
    keep related branches visually near each other and reduce edge
    crossings, then centered around x=0.
    """
    children: dict[str, list[str]] = {nid: [] for nid in node_ids}
    parents: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for source, target in edges:
        if source in children and target in parents:
            children[source].append(target)
            parents[target].append(source)

    # Longest-path layering via a simple topological pass (the graph is a
    # DAG by construction -- a Boolean expression tree plus a linear
    # pipeline never cycles).
    depth: dict[str, int] = {}
    in_degree = {nid: len(parents[nid]) for nid in node_ids}
    ready = [nid for nid in node_ids if in_degree[nid] == 0]
    for nid in ready:
        depth[nid] = 0
    while ready:
        nid = ready.pop(0)
        for child in children[nid]:
            depth[child] = max(depth.get(child, 0), depth[nid] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    # Any node not reached (shouldn't happen for a real DAG, but stay
    # defensive against unexpected input) falls back to layer 0.
    for nid in node_ids:
        depth.setdefault(nid, 0)

    # Compact uneven branches toward their merge points. Longest-path
    # layering correctly places every merge below all of its inputs, but a
    # shallow root that feeds a much deeper merge would otherwise remain at
    # layer 0 and create a visually misleading edge spanning most of the
    # graph. Push each node down to the latest layer that still leaves at
    # least one layer before every child. This is topology-only: no knowledge
    # of cohort criteria, Boolean operators, or node categories is needed.
    #
    # Example:
    #   A -> B -> C -> MERGE
    #   D ------------> MERGE
    # becomes
    #   A -> B -> C -> MERGE
    #             D --> MERGE
    # while preserving all edge directions and relative dependency order.
    topo = sorted(node_ids, key=lambda nid: depth[nid], reverse=True)
    for nid in topo:
        if not children[nid]:
            continue
        latest_valid = min(depth[child] for child in children[nid]) - 1
        if latest_valid > depth[nid]:
            depth[nid] = latest_valid

    layers: dict[int, list[str]] = {}
    for nid in node_ids:
        layers.setdefault(depth[nid], []).append(nid)

    positions: dict[str, tuple[float, float]] = {}
    for layer_index in sorted(layers):
        layer_nodes = layers[layer_index]
        if layer_index == 0 or not positions:
            ordered = layer_nodes
        else:
            def barycenter(nid: str) -> float:
                parent_xs = [
                    positions[p][0] for p in parents[nid] if p in positions
                ]
                return sum(parent_xs) / len(parent_xs) if parent_xs else 0.0
            ordered = sorted(layer_nodes, key=barycenter)
        offset = -(len(ordered) - 1) / 2.0
        for index, nid in enumerate(ordered):
            positions[nid] = (
                (offset + index) * _NODE_SPACING,
                layer_index * _LAYER_HEIGHT,
            )
    return positions


def render_graph(view: GraphView, *, selected_node_id: str | None = None) -> str | None:
    import streamlit as st
    try:
        from streamlit_flow import streamlit_flow
        from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
        from streamlit_flow.layouts import ManualLayout
        from streamlit_flow.state import StreamlitFlowState
    except ImportError:
        return _render_fallback(view, selected_node_id)

    # StreamlitFlowState stamps a fresh timestamp on every construction,
    # which the frontend treats as "Python changed me, resync" -- so
    # rebuilding it on every Streamlit rerun (which happens far more often
    # than the graph's actual data changes, e.g. on any unrelated widget
    # interaction) stalls the layout mid-computation and can round-trip
    # back into another rerun, producing exactly the stuck-at-origin /
    # flickering symptoms. The fix, matching the component's own intended
    # usage: persist the returned state in st.session_state and only build
    # a new one when the underlying data actually changed.
    #
    # widget_key is passed to streamlit_flow() as its `key=` argument, so
    # Streamlit claims that exact session_state slot for the widget's own
    # lifecycle bookkeeping. cache_key/sig_key are deliberately DIFFERENT
    # strings for our own manual cache entries -- reusing the widget's key
    # for a plain dict-style read/write collides with Streamlit's widget
    # state management and raises "st.session_state has no key ...".
    #
    # widget_key also deliberately changes on every structural rebuild (see
    # `generation` below). The underlying React Flow library has a long,
    # still not fully resolved history of `fit_view` failing to correctly
    # fit the viewport right after nodes are added/changed asynchronously
    # (xyflow/xyflow #4793, #4801, #3946, #1712) -- exactly the "everything
    # collapses toward the origin" symptom this graph kept showing. Rather
    # than keep fighting that timing race from the Python side, positions
    # are now computed synchronously in Python (_compute_layered_layout)
    # before anything reaches the frontend, so there is no async client-side
    # layout pass to race against. Layout=ManualLayout() is used always.
    # The key still changes per structural rebuild so fit_view reliably
    # re-centers on a genuinely new graph shape rather than inheriting
    # whatever viewport a much earlier, smaller graph happened to leave
    # behind. Selection-only updates (below) deliberately do NOT bump this,
    # so a click never remounts anything.
    cache_key = "_cohort_graph_cache"
    sig_key = "_cohort_graph_sig"
    selection_key = "_cohort_graph_selection"
    generation_key = "_cohort_graph_generation"

    # Selecting a node is deliberately NOT part of the structural signature.
    # It used to be, which meant every single click -- which changes
    # selected_node_id -- invalidated the whole cache and forced a full,
    # unnecessary rebuild. Selection is a pure style toggle. It should
    # never require recomputing positions or anything else.
    structural_signature = (
        _RENDER_VERSION,
        view.version,
        tuple(
            (
                node.node_id, node.label, node.status, node.category,
                node.subtitle, node.execution_mode, node.primary_metric,
            )
            for node in view.nodes
        ),
        tuple((e.source, e.target, e.changed, e.animated) for e in view.edges),
    )

    rebuilt = st.session_state.get(sig_key) != structural_signature
    if rebuilt:
        # A node with no incoming edge (a genuine root, e.g. a per-requirement
        # input node or the whole-task input) still gets a target handle
        # drawn by default -- an empty connector dot that reads as "something
        # is missing above" rather than "this is where the flow starts".
        # React Flow's own 'input'/'output' node types exist for exactly
        # this: 'input' renders only a source handle, 'output' only a target
        # handle. Computed purely from edge topology, not from any
        # cohort-specific category, so this stays correct for any graph.
        has_incoming = {edge.target for edge in view.edges}
        has_outgoing = {edge.source for edge in view.edges}

        def node_type_for(node_id: str) -> str:
            if node_id not in has_incoming:
                return "input"
            if node_id not in has_outgoing:
                return "output"
            return "default"

        positions = _compute_layered_layout(
            [item.node_id for item in view.nodes],
            [(edge.source, edge.target) for edge in view.edges],
        )

        nodes = [
            StreamlitFlowNode(
                id=item.node_id,
                pos=positions[item.node_id],
                data={"content": _node_content(item)},
                node_type=node_type_for(item.node_id),
                source_position="bottom",
                target_position="top",
                selected=(item.node_id == selected_node_id),
                selectable=item.selectable,
                draggable=False,
                connectable=False,
                style=_node_style(item),
            )
            for item in view.nodes
        ]
        edges = [
            StreamlitFlowEdge(
                id=f"{edge.source}->{edge.target}",
                source=edge.source,
                target=edge.target,
                label=edge.label,
                edge_type="smoothstep",
                animated=edge.animated or edge.changed,
                marker_end={"type": "arrowclosed"},
                style=(
                    {"stroke": "#B7791F", "strokeWidth": 2.5}
                    if edge.changed else {"stroke": "#9AA0A6"}
                ),
            )
            for edge in view.edges
        ]
        st.session_state[cache_key] = StreamlitFlowState(nodes, edges)
        st.session_state[sig_key] = structural_signature
        st.session_state[selection_key] = selected_node_id
        st.session_state[generation_key] = st.session_state.get(generation_key, 0) + 1
    elif st.session_state.get(selection_key) != selected_node_id:
        # Same graph, only the selection changed: update the existing,
        # already-positioned nodes in place rather than rebuilding anything.
        cached = st.session_state[cache_key]
        for node in cached.nodes:
            node.selected = (node.id == selected_node_id)
        st.session_state[selection_key] = selected_node_id

    st.caption(view.description or f"Version {view.version}")
    widget_key = f"cohort-graph-{st.session_state.get(generation_key, 0)}"
    result = streamlit_flow(
        widget_key,
        st.session_state[cache_key],
        # Positions are already computed correctly in Python
        # (_compute_layered_layout), so the frontend never needs to run its
        # own async auto-arrange pass -- ManualLayout tells it to just use
        # the positions nodes already carry, for every render.
        layout=ManualLayout(),
        fit_view=rebuilt,
        height=520,
        show_minimap=True,
        show_controls=True,
        get_node_on_click=True,
        allow_new_edges=False,
        pan_on_drag=True,
        allow_zoom=True,
        hide_watermark=True,
    )
    if result is not None:
        # Persist whatever the component echoes back (it may carry a fresh
        # timestamp of its own) so the *next* rerun's signature check sees
        # the same underlying nodes/edges and skips rebuilding again.
        st.session_state[cache_key] = result
        if result.selected_id:
            st.session_state[selection_key] = result.selected_id
            return result.selected_id
    return selected_node_id


_GATE_LABELS = {"AND", "OR", "NOT"}


def _node_content(item) -> str:
    if item.category == "logic" and item.label in _GATE_LABELS:
        fill, stroke = _node_colors(item.status, item.execution_mode, item.category)
        return _gate_svg(item.label, fill, stroke)
    prefix = "🗄 " if item.category == "source" else ""
    lines = [f"**{prefix}{item.label}**"]
    if item.primary_metric:
        lines.append(item.primary_metric)
    elif item.subtitle:
        lines.append(item.subtitle)
    if item.badges:
        # Small side-annotations attached to the node (e.g. the dictionary
        # table that resolved a fact) -- reference detail, not another
        # step in the main flow, so it renders as a compact tag on the
        # node itself rather than a separate connected box.
        lines.append(" · ".join(item.badges))
    return "\n\n".join(lines)


def _gate_svg(kind: str, fill: str, stroke: str) -> str:
    """Draw vertical logic gates for the top-to-bottom cohort flow.

    The classic ANSI glyphs face right, so the path geometry is rotated 90°
    while the text remains upright. Inputs therefore enter from the top and the
    gate output leaves from the bottom, matching the graph's layered layout.
    """
    text_color = "#202124"
    if kind == "AND":
        shape = (
            '<path d="M20,15 L80,15 A30,30 0 0 1 80,75 L20,75 Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5" />'
        )
    elif kind == "OR":
        shape = (
            '<path d="M25,15 Q45,45 25,75 Q75,80 115,45 Q75,10 25,15 Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5" />'
        )
    else:
        shape = (
            '<path d="M25,15 L25,75 L105,45 Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5" />'
            f'<circle cx="115" cy="45" r="9" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.5" />'
        )
    return (
        '<svg viewBox="0 0 90 140" width="90" height="140" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<g transform="matrix(0 1 -1 0 90 0)">' + shape + '</g>'
        f'<text x="45" y="76" text-anchor="middle" font-size="15" '
        f'font-weight="700" font-family="sans-serif" fill="{text_color}">{kind}</text>'
        '</svg>'
    )


def _render_fallback(view: GraphView, selected_node_id: str | None) -> str | None:
    import streamlit as st
    st.warning("Interactive graph renderer is unavailable. Using node selector.")
    labels = {n.node_id: f"{n.label} — {n.status}" for n in view.nodes}
    ids = list(labels)
    if not ids:
        return None
    index = ids.index(selected_node_id) if selected_node_id in ids else 0
    return st.selectbox("Graph node", ids, index=index, format_func=lambda x: labels[x])


def _node_style(item) -> dict:
    if item.category == "logic" and item.label in _GATE_LABELS:
        # The SVG gate glyph draws its own shape and border. A surrounding
        # box would just frame it in an unrelated rectangle.
        return {
            "background": "transparent",
            "border": "none",
            "padding": "0px",
            "width": "90px",
            "height": "140px",
        }
    background, border = _node_colors(item.status, item.execution_mode, item.category)
    style = {
        "background": background,
        "border": (
            f"2px dashed {border}" if item.category == "source"
            else f"2px solid {border}"
        ),
        "borderRadius": "10px" if item.category not in {"concept", "source"} else "999px",
        "padding": "6px 4px" if item.category in {"concept", "source"} else "10px 8px",
        "fontSize": "12px" if item.category in {"concept", "source"} else "13px",
        "fontFamily": "monospace" if item.category == "source" else "inherit",
        "color": "#202124",
        "minWidth": "90px" if item.category in {"concept", "source"} else "140px",
    }
    if item.node_id and item.category == "concept":
        style["opacity"] = 0.92
    return style


def _node_colors(status: str, mode: str, category: str) -> tuple[str, str]:
    if status == "failed":
        return "#FCE8E6", "#C5221F"
    if status == "running":
        return "#FFF3CD", "#B7791F"
    if status in {"pending", "invalidated"}:
        return "#F1F3F4", "#80868B"
    if category == "concept":
        return "#F3E8FD", "#8430CE"
    if category == "source":
        return "#E0F2F1", "#00796B"
    if mode == "reused":
        return "#E6F4EA", "#188038"
    if mode in {"new", "recomputed"}:
        return "#E8F0FE", "#1967D2"
    return "#FFFFFF", "#5F6368"
