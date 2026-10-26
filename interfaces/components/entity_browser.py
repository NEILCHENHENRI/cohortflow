from __future__ import annotations
from typing import Callable
from interfaces.renderers.sections import render_section
from presentation.models import EntityCollectionView, EntityDetailView

def render_entity_browser(collections: tuple[EntityCollectionView,...], *, selected_collection_id: str|None, selected_entity_id: str|None, load_entity: Callable[[str,str],EntityDetailView]) -> tuple[str|None,str|None]:
    import streamlit as st
    if not collections: return None,None
    collection_ids=[c.collection_id for c in collections]
    cidx=collection_ids.index(selected_collection_id) if selected_collection_id in collection_ids else 0
    collection_id=st.selectbox("Patient collection",collection_ids,index=cidx,format_func=lambda cid: next(c.title for c in collections if c.collection_id==cid))
    collection=next(c for c in collections if c.collection_id==collection_id)
    st.caption(f"{collection.count} patients · {collection.description}")
    st.dataframe(list(collection.rows),width="stretch",hide_index=True)
    ids=[str(row[collection.id_field]) for row in collection.rows]
    if not ids: return collection_id,None
    eidx=ids.index(selected_entity_id) if selected_entity_id in ids else 0
    entity_id=st.selectbox("Inspect patient",ids,index=eidx)
    detail=load_entity(collection_id,entity_id)
    st.markdown(f"#### {detail.title}")
    st.caption(detail.subtitle)
    if detail.summary_fields:
        st.dataframe([detail.summary_fields],width="stretch",hide_index=True)
    for section in detail.sections: render_section(section)
    return collection_id,entity_id
