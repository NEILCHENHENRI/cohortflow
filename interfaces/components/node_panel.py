from __future__ import annotations
from interfaces.renderers.sections import render_section
from presentation.models import NodeDetailView

def render_node_panel(detail: NodeDetailView) -> None:
    import streamlit as st
    st.markdown(f"### {detail.title}")
    st.caption(detail.description)
    if detail.metrics:
        cols=st.columns(min(len(detail.metrics),4))
        for col, metric in zip(cols,detail.metrics): col.metric(metric.label,metric.value,help=metric.help)
    for section in detail.sections: render_section(section)
