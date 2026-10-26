from __future__ import annotations

from presentation.models import SectionKind, SectionView


def render_section(section: SectionView) -> None:
    """Render one presentation section with Streamlit."""
    import streamlit as st

    with st.expander(section.title, expanded=section.expanded):
        if section.description:
            st.caption(section.description)

        if section.kind is SectionKind.TEXT:
            st.markdown(str(section.data))
        elif section.kind is SectionKind.WARNING:
            st.warning(str(section.data))
        elif section.kind is SectionKind.KEY_VALUE:
            st.dataframe([section.data], width="stretch", hide_index=True)
        elif section.kind in {SectionKind.TABLE, SectionKind.ENTITY_LIST}:
            st.dataframe(section.data, width="stretch", hide_index=True)
        elif section.kind is SectionKind.JSON:
            st.json(section.data)
        elif section.kind is SectionKind.METRICS:
            items = list(section.data)
            if not items:
                st.caption("No metrics available.")
                return
            columns = st.columns(min(len(items), 4))
            for column, item in zip(columns, items):
                column.metric(
                    item.get("label", "Metric"),
                    item.get("value", "—"),
                )
        else:
            st.write(section.data)
