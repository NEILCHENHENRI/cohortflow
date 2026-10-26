from __future__ import annotations

from presentation.models import ExecutionEvidenceView


def render_execution_evidence(
    evidence: ExecutionEvidenceView,
) -> None:
    import streamlit as st

    st.subheader("Execution evidence")
    if evidence.status_message:
        if evidence.status in {"failed", "cancelled"}:
            st.warning(evidence.status_message)
        else:
            st.info(evidence.status_message)
    logic_col, count_col = st.columns([3, 1])
    with logic_col:
        st.markdown("**Eligibility logic**")
        st.markdown(f"`{evidence.operator}`")
    with count_col:
        st.metric(
            "Final cohort size",
            "Not produced" if evidence.final_count is None else evidence.final_count,
        )

    st.markdown("#### Criterion results")
    if not evidence.criteria:
        st.info("No criterion evidence artifacts are available yet.")
    else:
        rows = []
        for item in evidence.criteria:
            rows.append({
                "criterion": item.label,
                "patients": item.patient_count,
                "concept": item.concept_name or item.concept_id,
                "vocabulary": item.vocabulary_id,
                "resolution": item.resolution_source,
            })
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        f"Final patient IDs ({len(evidence.final_patient_ids)})",
        expanded=False,
    ):
        st.dataframe(
            [{"patient_id": item} for item in evidence.final_patient_ids],
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        "Patient membership evidence",
        expanded=False,
    ):
        if evidence.membership_rows:
            st.dataframe(
                list(evidence.membership_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No patient membership evidence is available.")
