from __future__ import annotations

import json
from typing import Any

from framework.manual_edit import (
    ManualEditForm,
    ManualFieldType,
)


def render_manual_editor(
    form: ManualEditForm,
) -> tuple[str | None, dict[str, Any] | None]:
    import streamlit as st

    st.warning(
        "Manual fallback mode is active. Changes remain staged until "
        "validation and execution succeed."
    )
    st.subheader(form.title)
    st.caption(form.description)

    submitted: dict[str, Any] = {}
    with st.form("manual_fallback_form"):
        for section in form.sections:
            with st.expander(section.title, expanded=True):
                if section.description:
                    st.caption(section.description)
                for field in section.fields:
                    submitted[field.field_id] = _render_field(field)
        left, middle, right = st.columns(3)
        review = left.form_submit_button(
            "Review and apply",
            type="primary",
        )
        reset = middle.form_submit_button("Reset")
        cancel = right.form_submit_button("Cancel")

    if cancel:
        return "cancel", None
    if reset:
        return "reset", None
    if review:
        return "submit", submitted
    return None, None


def _render_field(field):
    import streamlit as st

    kwargs = {
        "help": field.help_text,
        "disabled": not field.editable,
        "key": f"manual_{field.field_id}",
    }
    if field.field_type is ManualFieldType.TEXT:
        return st.text_input(
            field.label,
            value="" if field.value is None else str(field.value),
            **kwargs,
        )
    if field.field_type is ManualFieldType.INTEGER:
        return st.text_input(
            field.label,
            value="" if field.value is None else str(field.value),
            **kwargs,
        )
    if field.field_type is ManualFieldType.BOOLEAN:
        return st.checkbox(
            field.label,
            value=bool(field.value),
            **kwargs,
        )
    if field.field_type is ManualFieldType.SELECT:
        values = [choice.value for choice in field.choices]
        labels = {choice.value: choice.label for choice in field.choices}
        current = (
            field.value
            if field.value in values
            else values[0]
        )
        return st.selectbox(
            field.label,
            values,
            index=values.index(current),
            format_func=lambda value: labels[value],
            **kwargs,
        )
    if field.field_type is ManualFieldType.JSON:
        initial = (
            field.value
            if isinstance(field.value, str)
            else json.dumps(field.value, indent=2, default=str)
        )
        return st.text_area(
            field.label,
            value=initial,
            height=160,
            **kwargs,
        )
    raise ValueError(f"unsupported manual field type: {field.field_type}")
