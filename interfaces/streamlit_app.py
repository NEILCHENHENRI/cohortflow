from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from agents.cohort.presentation import CohortPresentationAdapter
from agents.cohort.parser import render_structured_task
from framework.parsing import ParsingMode
from framework.correction import CorrectionMode
from infrastructure.config import AppMode, InfrastructureConfig, ProjectPaths
from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.factory import build_local_cohort_stack
from infrastructure.llm.ollama import OllamaClient, OllamaError
from duckdb import Error as DuckDBError
from infrastructure.mimic_hosp.presentation import MimicHospPatientDataProvider
from infrastructure.omop.presentation import OMOPPatientDataProvider
from interfaces.components.entity_browser import render_entity_browser
from interfaces.components.graph import render_graph
from interfaces.components.node_panel import render_node_panel
from interfaces.components.evidence_panel import render_execution_evidence
from interfaces.components.manual_editor import render_manual_editor
from framework.pending_interaction import PendingInteractionKind
from framework.resolution import ResponseMode
from framework.execution_backend import ExecutionBackend
from framework.runtime import ExecutionMode
from framework.repair import RepairMode, RepairPolicy, RepairStatus
from infrastructure.repair.cloud import CommercialRepairBackend, CommercialRepairConfig
from interfaces.controller import AgentInteractionController
from interfaces.background import BackgroundInteractionJob, llm_activity_from_events
from interfaces.session import UIState
from presentation.models import ConversationMessageView


def _parse_mode() -> AppMode:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=[AppMode.LOCAL.value, AppMode.DEMO.value])
    args, _ = parser.parse_known_args()
    return AppMode(args.mode or "local")


def _build_session(mode: AppMode) -> tuple[
    AgentInteractionController,
    CohortPresentationAdapter,
    UIState,
    OllamaClient,
]:
    root = Path(__file__).resolve().parents[1]
    adapter_dir = Path(st.session_state["custom_adapter_directory"].strip()).expanduser().resolve() if mode is AppMode.CUSTOM else None
    paths = (ProjectPaths(root, adapter_dir / "dataset.duckdb", None, root / "data")
             if adapter_dir else ProjectPaths.from_project_root(root, mode=mode))
    config = InfrastructureConfig(
        paths=paths,
        mode=mode,
        adapter_directory=adapter_dir,
        ollama_model="qwen3:4b",
        # Parsing thinking is controlled per pass. Note adjudication has
        # its own default and control.
        ollama_think=False,
        # Thinking traces add real generation time on top of an already
        # local, modest-hardware model. A short timeout would kill a
        # slow-but-correct generation rather than a genuinely stuck one.
        ollama_timeout_seconds=900.0,
    )
    stack = build_local_cohort_stack(config)
    # Application and direct cohort parser default to localized correction.
    stack.session.domain_agent.parser.configure_correction(CorrectionMode.ROBUST_DELTA)

    clinical = DuckDBConnectionFactory(
        DuckDBConfig(paths.clinical_database, read_only=True)
    )
    if mode is AppMode.DEMO:
        vocabulary = DuckDBConnectionFactory(
            DuckDBConfig(paths.vocabulary_database, read_only=True)
        )
        provider = OMOPPatientDataProvider(clinical, vocabulary)
    elif mode is AppMode.CUSTOM:
        provider = stack.agent.parser.dataset
    else:
        provider = MimicHospPatientDataProvider(clinical)
    return (
        stack.interaction_controller,
        CohortPresentationAdapter(stack.session, provider),
        UIState(),
        stack.llm,
    )


def _ensure_session(mode: AppMode) -> None:
    directory = st.session_state.get("custom_adapter_directory", "").strip()
    mode_key = mode.value + (
        ":" + str(Path(directory).expanduser().resolve()) if mode is AppMode.CUSTOM else ""
    )
    changed = st.session_state.get("app_mode") != mode_key
    if not changed and "interaction_controller" in st.session_state:
        return
    if st.session_state.get("background_interaction") is not None:
        raise ValueError("Finish or stop active work before switching datasets.")
    # Validate/build first. A mistyped path must not erase the current session.
    controller, adapter, ui_state, llm_client = _build_session(mode)
    previous = st.session_state.get("interaction_controller")
    if previous is not None and previous.repair_manager is not None:
        previous.repair_manager.close()
    for key in (
        "background_interaction", "background_result", "last_operation_log",
        "last_llm_activity", "last_timing_report", "workspace_panel_target",
        "workspace_panel_generation", "workspace_job_active",
    ):
        st.session_state.pop(key, None)
    st.session_state.app_mode = mode_key
    st.session_state.interaction_controller = controller
    st.session_state.presentation_adapter = adapter
    st.session_state.ui_state = ui_state
    st.session_state.llm_client = llm_client


def _user_message(
    content: str,
    graph_version: int | None,
) -> ConversationMessageView:
    return ConversationMessageView(
        message_id=uuid4().hex,
        role="user",
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        graph_version=graph_version,
    )


def _apply_result(ui_state: UIState, result) -> None:
    # The user turn is inserted as soon as the background job starts, so only
    # append assistant/system turns returned by the controller.
    ui_state.messages.extend(
        message for message in result.messages if message.role != "user"
    )
    if result.graph_version is not None:
        ui_state.graph_version = result.graph_version
    if result.selected_node_id is not None:
        ui_state.selected_node_id = result.selected_node_id
        ui_state.clear_entity_selection()
    ui_state.notifications.extend(result.notifications)


def _latest_node_states(events) -> list[dict]:
    states: dict[str, dict] = {}
    for event in events:
        if event.event_type not in {
            "node_started", "node_attempt_finished", "node_committed", "node_preserved"
        }:
            continue
        try:
            payload = json.loads(event.payload)
        except Exception:
            continue
        node_id = str(payload.get("node_id") or "")
        if not node_id:
            continue
        states[node_id] = {
            "node": node_id,
            "operator": payload.get("operator"),
            "phase": payload.get("phase"),
            "requirement": payload.get("requirement_id"),
            "elapsed ms": payload.get("duration_ms"),
            "components": ", ".join(
                f"{name}={value:.1f}ms"
                for name, value in (payload.get("components_ms") or {}).items()
            ),
        }
    return list(states.values())


def _requirement_label(controller, requirement_id: str | None) -> str:
    if not requirement_id:
        return "requirement"
    state = controller.session.state
    if state is None:
        return requirement_id
    task = state.staged.task if state.staged is not None else state.task
    requirement = next(
        (item for item in task.requirements if item.requirement_id == requirement_id),
        None,
    )
    if requirement is None:
        return requirement_id
    payload = requirement.payload
    for key in ("term", "condition", "medication", "name", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{requirement.kind} ({requirement_id})"


def _node_operation_text(controller, event) -> str | None:
    try:
        payload = json.loads(event.payload)
    except Exception:
        return None
    operator = str(payload.get("operator") or "")
    phase = str(payload.get("phase") or "")
    duration = payload.get("duration_ms")
    label = _requirement_label(controller, payload.get("requirement_id"))
    if operator == "evaluate_requirement":
        if event.event_type == "node_preserved":
            return f"Reused completed work for {label}."
        if event.event_type == "node_started":
            return f"Evaluating {label}."
        if event.event_type == "node_committed":
            suffix = f" ({duration / 1000:.2f}s)" if isinstance(duration, (int, float)) and duration >= 1000 else ""
            return f"Finished evaluating {label}{suffix}."
        return None
    names = {
        "aggregate_cohort": "Combining criterion patient sets into the cohort.",
        "validate_cohort": "Validating the cohort result.",
        "summarize_cohort": "Preparing the cohort summary.",
        "cohort_universe": "Loading the patient universe for NOT logic.",
    }
    if event.event_type == "node_started" and operator in names:
        return names[operator]
    if event.event_type == "node_preserved" and operator in names:
        return "Reused " + names[operator][0].lower() + names[operator][1:]
    return None


def _operation_log_rows(events, controller, *, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    last_text: str | None = None
    for event in events:
        text: str | None = None
        if event.event_type == "module_started":
            spec = controller.session.domain_agent.module_spec(event.payload)
            if spec.name != "execution.runtime":
                text = spec.description.rstrip(".") + "."
        elif event.event_type == "predicate_compilation_reused":
            try:
                payload = json.loads(event.payload)
                kind = str(payload.get("kind") or "criterion")
                label = str(payload.get("text") or payload.get("predicate_id") or "criterion")
                text = f"Reused compiled {kind} semantics for “{label}”."
            except Exception:
                text = "Reused an unchanged compiled criterion."
        elif event.event_type == "requirement_context":
            try:
                payload = json.loads(event.payload)
                category = str(payload.get("category") or "criterion")
                database = str(payload.get("database") or "configured database")
                term = payload.get("term")
                if isinstance(term, str) and term.strip():
                    text = f"Looking up {category} “{term.strip()}” in {database}."
                else:
                    text = f"Evaluating {category} criterion in {database}."
            except Exception:
                text = None
        elif event.event_type == "concept_resolved":
            try:
                payload = json.loads(event.payload)
                category = str(payload.get("category") or "concept")
                original = payload.get("term")
                resolved = payload.get("resolved_to") or payload.get("concept_id")
                vocabulary = payload.get("vocabulary")
                source = payload.get("resolution_source")
                database = payload.get("database")
                left = f"{category} “{original}”" if original else category
                suffix_bits = [str(x) for x in (vocabulary, source) if x]
                suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
                db = f" using {database}" if database else ""
                text = f"Resolved {left} → {resolved}{suffix}{db}."
            except Exception:
                text = None
        elif event.event_type in {
            "correction_queued", "execution_yielded", "delivery_deferred",
            "control", "execution_started", "execution_finished", "execution_stopped",
            "disambiguation_started", "disambiguation_llm_started",
            "disambiguation_llm_finished", "note_stage", "note_index_progress",
            "note_llm_stage",
        }:
            text = event.payload
        elif event.event_type in {"node_started", "node_committed", "node_preserved"}:
            text = _node_operation_text(controller, event)
        elif event.event_type == "module_interrupted":
            module_name, _, reason = event.payload.partition("|")
            spec = controller.session.domain_agent.module_spec(module_name)
            if reason == "ResolutionRequired":
                text = "Paused for a clarification before continuing graph execution."
            elif reason == "SteeringYield":
                text = f"Yielded after {spec.description[0].lower() + spec.description[1:]}"
            else:
                text = f"Paused while {spec.description[0].lower() + spec.description[1:]}"
        elif event.event_type == "module_failed":
            spec = controller.session.domain_agent.module_spec(event.payload)
            text = f"Failed while {spec.description[0].lower() + spec.description[1:]}"
        if not text or text == last_text:
            continue
        rows.append({
            "time": f"{event.elapsed_ms / 1000:.1f}s",
            "operation": text,
        })
        last_text = text
    return rows if limit is None else rows[-limit:]


def _render_llm_activity(calls: list[dict]) -> None:
    st.markdown("#### LLM activity")
    if not calls:
        st.caption("No LLM calls recorded for this interaction yet.")
        return
    for index, call in enumerate(calls, 1):
        status = call.get("status", "running")
        elapsed = call.get("elapsed_seconds")
        suffix = f" · {float(elapsed):.2f}s" if elapsed is not None else ""
        label = (
            f"{index}. {call.get('stage', 'LLM call')} · "
            f"{call.get('model', 'unknown')} · think={call.get('think', False)} "
            f"· {status}{suffix}"
        )
        with st.expander(label, expanded=(status == "running")):
            thinking = str(call.get("thinking") or "")
            raw = str(call.get("raw_output") or "")
            structured = str(call.get("structured_output") or "")
            if thinking:
                st.caption("Thinking")
                st.code(thinking, language="text")
            st.caption("Raw model output")
            if raw:
                st.code(raw, language="json")
            else:
                st.caption("Waiting for model output...")
            if structured:
                st.caption("Structured result")
                st.code(structured, language="json")
            if call.get("error"):
                st.error(str(call["error"]))


def _render_current_task_spec(controller) -> None:
    state = controller.session.state
    if state is None or state.task is None:
        return
    st.markdown("#### Current structured representation")
    st.code(render_structured_task(state.task), language="json")


def _render_timing_summary(timing: dict | None) -> None:
    if not isinstance(timing, dict):
        return
    summary = timing.get("summary", {})
    if summary:
        cols = st.columns(5)
        cols[0].metric("Total", f"{summary.get('interaction_wall_ms', 0.0) / 1000:.2f}s")
        cols[1].metric("LLM", f"{summary.get('llm_generation_ms', 0.0) / 1000:.2f}s")
        cols[2].metric("Graph", f"{summary.get('graph_execution_wall_ms', 0.0) / 1000:.2f}s")
        cols[3].metric("Reused", int(summary.get("reused_node_count", 0)))
        cols[4].metric("Preserved", f"{summary.get('reused_node_compute_ms', 0.0) / 1000:.2f}s")

    spans = [
        item for item in timing.get("spans", [])
        if item.get("name") != "interaction.total"
        and item.get("category") in {"llm", "semantic", "correction", "execution"}
    ]
    graph_nodes = timing.get("graph_nodes", [])
    if spans or graph_nodes:
        with st.expander("Timing details", expanded=False):
            if spans:
                # Keep this diagnostic concise: longest high-level spans first.
                top = sorted(
                    spans,
                    key=lambda item: float(item.get("duration_ms") or 0.0),
                    reverse=True,
                )[:12]
                st.dataframe(
                    [
                        {
                            "component": item.get("name"),
                            "category": item.get("category"),
                            "ms": round(float(item.get("duration_ms") or 0.0), 3),
                        }
                        for item in top
                    ],
                    hide_index=True,
                    width="stretch",
                )
            if graph_nodes:
                rows = []
                for item in graph_nodes:
                    row = {
                        "node": item.get("node_id"),
                        "status": item.get("status"),
                        "reused": item.get("reused"),
                        "ms": item.get("duration_ms"),
                    }
                    components = item.get("component_timings_ms", {})
                    for key in (
                        "concept_resolution", "disambiguation_llm",
                        "repository_query", "materialize_patient_set",
                    ):
                        if key in components:
                            row[key] = components[key]
                    rows.append(row)
                st.dataframe(rows, hide_index=True, width="stretch")


def _render_log_rows(rows: list[dict], *, height: int = 500) -> None:
    if not rows:
        st.caption("No operations recorded yet.")
        return
    # A fixed-height container keeps the workspace stable while retaining the
    # complete operation history. Streamlit makes height-constrained containers
    # scrollable, so long runs no longer push the rest of the page downward.
    with st.container(height=height, border=True):
        for row in rows:
            st.write(f"{row['time']} · {row['operation']}")


@st.fragment(run_every=0.5)
def _render_operation_panel(job: BackgroundInteractionJob | None, controller) -> None:
    if job is None:
        _render_timing_summary(st.session_state.get("last_timing_report"))
        _render_log_rows(st.session_state.get("last_operation_log") or [], height=420)
        _render_llm_activity(st.session_state.get("last_llm_activity") or [])
        # The canonical TaskSpec is intentionally a post-run view. While a
        # background interaction is running, this panel is reserved for live
        # LLM activity so a staged/intermediate task is never presented as the
        # settled representation.
        _render_current_task_spec(controller)
        return

    snapshot = job.snapshot()
    cols = st.columns(4)
    cols[0].metric("Elapsed", f"{int(snapshot.elapsed_ms / 1000)}s")
    cols[1].metric("Graph", "running" if snapshot.execution_started else "not started")
    cols[2].metric("Queued", snapshot.queue_depth)
    cols[3].metric("State", "stopping" if snapshot.stop_requested else "running")

    _render_log_rows(_operation_log_rows(snapshot.events, controller), height=420)
    _render_llm_activity(llm_activity_from_events(snapshot.events))

    node_rows = _latest_node_states(snapshot.events)
    if node_rows:
        with st.expander("Current graph progress", expanded=False):
            st.dataframe(node_rows, hide_index=True, width="stretch")

    if snapshot.running:
        if st.button(
            "Stop execution",
            type="secondary",
            disabled=snapshot.stop_requested,
            key="midrun_stop_execution",
        ):
            if not job.request_stop():
                st.warning("Execution has already finished.")
            else:
                st.rerun()
        st.caption(
            "Boundary-cooperative: the current atomic module finishes and its "
            "valid output is preserved before execution yields."
        )
        return

    st.session_state.last_operation_log = _operation_log_rows(
        snapshot.events, controller
    )
    st.session_state.last_llm_activity = llm_activity_from_events(snapshot.events)
    result = job.take_result()
    if result is not None:
        timing = result.diagnostics.get("timing") if isinstance(result.diagnostics, dict) else None
        if isinstance(timing, dict):
            st.session_state.last_timing_report = timing
        st.session_state.background_result = result
    st.session_state.background_interaction = None
    st.session_state.workspace_panel_target = "Evidence"
    st.session_state.workspace_panel_generation = int(
        st.session_state.get("workspace_panel_generation", 0)
    ) + 1
    st.session_state.workspace_job_active = False
    st.rerun()



def _render_last_failure(controller) -> None:
    failure = getattr(controller.session, "last_failure", None)
    if failure is None:
        return
    outcome = getattr(controller.session, "last_repair_outcome", None)
    with st.expander("Last failure", expanded=outcome is not None and outcome.status is not RepairStatus.DISABLED):
        st.write(f"**`{failure.exception_type}`**")
        st.caption(failure.message)
        if failure.node_id:
            st.caption(f"Node: {failure.node_id} · Operator: {failure.operator_name or 'unknown'}")
        if outcome is None or outcome.status is RepairStatus.DISABLED:
            return
        st.divider()
        st.write(f"**Self repair · {outcome.status.value.replace('_', ' ').title()}**")
        st.write(outcome.message)
        st.caption(f"Original failure: {outcome.verification.value} · retries: {outcome.retry_attempts}")
        if outcome.assessment is not None:
            a = outcome.assessment
            st.caption(
                f"Analyst: {a.action.value} · {a.scope.value} · {a.confidence:.0%} confidence"
            )
            if a.diagnosis:
                st.write(a.diagnosis)
        proposal = outcome.proposal
        if proposal is not None:
            st.write(f"**Patch · {proposal.backend}**")
            st.write(proposal.summary or proposal.diagnosis)
            if proposal.unified_diff.strip():
                st.code(proposal.unified_diff, language="diff")
        if outcome.validation is not None:
            report = outcome.validation
            if report.passed:
                st.success(
                    f"Validated outside the live tree · {len(report.changed_files)} file(s), "
                    f"{report.changed_lines} changed line(s)"
                )
            else:
                st.warning(report.rejection_reason or "Repair validation failed.")
            for step in report.steps:
                icon = "✓" if step.passed else "✗"
                with st.expander(f"{icon} {step.name}", expanded=not step.passed):
                    if step.command:
                        st.code(" ".join(step.command), language="bash")
                    if step.output.strip():
                        st.code(step.output[-8000:], language="text")
        if outcome.persistent_branch:
            st.success(f"Saved as {outcome.persistent_branch} · {(outcome.persistent_commit or '')[:10]}")
        elif outcome.status is RepairStatus.VALIDATED and controller.repair_manager is not None:
            st.caption(f"Isolated workspace: {outcome.workspace_path}")
            st.info("After reviewing and adopting a patch, restart the process and start a fresh session. Existing results belong to the old implementation.")
            left, right = st.columns(2)
            if left.button("Save repair branch", key=f"persist_{failure.failure_id}", disabled=bool(st.session_state.get("background_interaction"))):
                try:
                    saved = controller.repair_manager.persist(outcome)
                    controller.session.last_repair_outcome = saved
                    st.success(saved.message)
                except Exception as exc:
                    st.error(f"Could not save repair branch: {exc}")
            if right.button("Discard repair", key=f"discard_{failure.failure_id}", disabled=bool(st.session_state.get("background_interaction"))):
                controller.repair_manager.discard(failure.failure_id)
                controller.session.last_repair_outcome = None
                controller.session.last_repair_proposal = None
                st.rerun()

def _render_settings(controller, llm_client, *, job_active: bool) -> None:
    with st.sidebar:
        cohort_parser = controller.session.domain_agent.parser
        if job_active:
            st.caption("Settings are locked while work is running. Stop or finish the current work to change them.")

        st.caption("Settings apply to future calls. Completed results are not recomputed.")
        st.subheader("Parsing")
        mode_labels = {
            ParsingMode.CARBON: "Carbon · 1 call",
            ParsingMode.GRAPHITE: "Graphite · 2 calls",
            ParsingMode.DIAMOND: "Diamond · adaptive",
            ParsingMode.GRAPHENE: "Graphene · adaptive atomic",
        }
        selected_mode = st.selectbox(
            "Strategy",
            options=list(mode_labels),
            format_func=lambda value: mode_labels[value],
            index=list(mode_labels).index(cohort_parser.parsing_mode),
            help="Applies to initial parsing, Full Regeneration and mode-aware localized correction. Call counts shown describe initial parsing, before retries. Graphite is the default.",
            key="parsing_strategy",
            disabled=job_active,
        )
        if not job_active:
            cohort_parser.configure_parsing(mode=selected_mode)

        correction_options = {
            "Robust Delta": CorrectionMode.ROBUST_DELTA,
            "Full Regeneration": CorrectionMode.FULL_REGENERATION,
        }
        current_correction = getattr(cohort_parser, "correction_mode", CorrectionMode.ROBUST_DELTA)
        correction_label = next(
            (label for label, value in correction_options.items() if value is current_correction),
            "Robust Delta",
        )
        selected_correction_label = st.radio(
            "Correction strategy",
            options=list(correction_options),
            index=list(correction_options).index(correction_label),
            help="Robust Delta uses compact v2 correction by default and changes only targeted criteria. Unsafe interpretations ask before Full Regeneration.",
            key="correction_strategy",
            disabled=job_active,
        )
        if not job_active:
            cohort_parser.configure_correction(correction_options[selected_correction_label])
        if selected_correction_label == "Robust Delta":
            strategies = {"Compact TaskDelta (v2)": "compact", "Follow parsing mode": "mode_aware", "Original localized steering (v1)": "legacy"}
            selected_localized = st.selectbox(
                "Localized parsing",
                options=list(strategies),
                index=list(strategies.values()).index(cohort_parser.localized_strategy),
                help="Compact v2 uses one edit-interpretation stage, then compiles changed criteria with the selected parsing mode. Follow parsing mode also adapts interpretation to that mode. Original localized steering (v1) uses one interpretation call plus one call per added or modified criterion.",
                key="localized_parsing_strategy",
                disabled=job_active,
            )
            if not job_active:
                cohort_parser.configure_localized_correction(strategies[selected_localized])

        pass_keys = []
        if selected_mode is ParsingMode.CARBON:
            pass_keys = [("carbon", "Combined")]
        elif selected_mode is ParsingMode.GRAPHITE:
            pass_keys = [("relationships", "Structure"), ("compilation", "Semantics")]
        else:
            pass_keys = [
                ("extraction", "Extraction"),
                ("relationships", "Relations"),
                ("table_selection", "Table selection"),
                ("compilation", "Semantics"),
            ]
        thinking_updates = {}
        with st.expander("Thinking", expanded=False):
            st.caption("Off by default. Enable only where extra reasoning helps.")
            for pass_key, label in pass_keys:
                thinking_updates[pass_key] = st.toggle(
                    label,
                    value=cohort_parser.thinking_by_pass.get(pass_key, False),
                    key=f"parsing_think_{pass_key}",
                    disabled=job_active,
                )
            correction_thinking = st.toggle(
                "Corrections",
                value=cohort_parser.thinking_by_pass.get("correction_delta", False),
                key="parsing_think_corrections",
                disabled=job_active,
                help="Controls Graphite/Original-baseline correction interpretation and Full Regeneration merging. Carbon uses Combined. Diamond/Graphene use Extraction and Relations above.",
            )
            thinking_updates["correction_target"] = correction_thinking
            thinking_updates["correction_delta"] = correction_thinking
            thinking_updates["correction_merge"] = correction_thinking
        if not job_active:
            cohort_parser.configure_parsing(thinking_by_pass=thinking_updates)
            llm_client.think = False

        note_matcher = getattr(controller.session.domain_agent, "note_matcher", None)
        if note_matcher is not None:
            with st.expander("Note matching", expanded=False):
                st.caption("Per-patient BM25 and BGE retrieval, fixed relevance filtering, then patient-level LLM adjudication.")
                note_thinking = st.toggle(
                    "Adjudication thinking",
                    value=getattr(note_matcher, "think", True),
                    help="On by default because the final note-evidence decision must handle negation, attribution, and conflicting excerpts.",
                    key="note_adjudication_thinking",
                    disabled=job_active,
                )
                if not job_active:
                    note_matcher.think = note_thinking
                retriever = getattr(note_matcher, "retriever", None)
                retrieval_config = getattr(retriever, "config", None)
                if retrieval_config is not None:
                    with st.popover("Advanced note retrieval settings"):
                        st.caption(
                            "Set a fixed relevance cutoff. Each patient's evidence is selected independently of other candidate patients."
                        )
                        relevance_threshold = st.number_input(
                            "Minimum passage relevance",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(retrieval_config.relevance_threshold),
                            step=0.01,
                            key="note_relevance_threshold",
                            disabled=job_active,
                        )
                        if not job_active:
                            retriever.config = replace(
                                retrieval_config,
                                relevance_threshold=float(relevance_threshold),
                            )

        st.subheader("Execution")
        backend_options = {
            "Native": ExecutionBackend.NATIVE,
            "LangGraph": ExecutionBackend.LANGGRAPH,
        }
        current_backend = getattr(
            controller.session, "execution_backend", ExecutionBackend.NATIVE
        )
        current_backend_label = next(
            (label for label, value in backend_options.items() if value is current_backend),
            "Native",
        )
        selected_backend_label = st.radio(
            "Backend",
            options=list(backend_options),
            index=list(backend_options).index(current_backend_label),
            horizontal=True,
            help="Native is the default. LangGraph runs the same execution DAG through LangGraph with in-memory checkpointing.",
            key="execution_backend",
            disabled=job_active,
        )
        if not job_active:
            controller.session.set_execution_backend(
                backend_options[selected_backend_label]
            )

        execution_options = {
            "Sequential": ExecutionMode.SEQUENTIAL,
            "Parallel": ExecutionMode.PARALLEL,
        }
        current_execution_mode = getattr(
            controller.session.runtime, "execution_mode", ExecutionMode.SEQUENTIAL
        )
        current_execution_label = next(
            (label for label, value in execution_options.items() if value is current_execution_mode),
            "Sequential",
        )
        selected_execution_label = st.radio(
            "Steering mode",
            options=list(execution_options),
            index=list(execution_options).index(current_execution_label),
            help="Sequential interprets queued corrections at a safe boundary. Parallel can overlap correction interpretation with active work. Graph operators still execute sequentially.",
            key="execution_strategy",
            disabled=job_active,
        )
        if not job_active:
            controller.session.set_execution_mode(
                execution_options[selected_execution_label], workers=4
            )

        st.subheader("Self Repair (Experimental)")
        repair_enabled = st.toggle(
            "Enable",
            value=False,
            key="self_repair_enabled",
            disabled=job_active,
            help="Off by default. Diagnose failures, retry safe operations at most twice, or validate a patch in an isolated worktree. Code patches require review, restart and a fresh session.",
        )
        repair_mode = RepairMode.OFF
        cloud_backend = None
        cloud_provider = None
        cloud_model = None
        if repair_enabled:
            repair_choice = st.radio(
                "Repair path",
                options=["Local", "Local → Cloud"],
                horizontal=False,
                help="Rules handle clear input and transient failures. Other failures use local analysis. Cloud can follow a local backend failure, rejected patch, or explicit escalation.",
                key="repair_path",
                disabled=job_active,
            )
            repair_mode = (
                RepairMode.LOCAL
                if repair_choice == "Local"
                else RepairMode.LOCAL_THEN_CLOUD
            )
            if repair_mode is RepairMode.LOCAL_THEN_CLOUD:
                cloud_provider = "openai"
                st.caption("Cloud provider: OpenAI")
                default_model = "gpt-5.4"
                cloud_model = st.text_input(
                    "Model", value=default_model, key=f"repair_model_{cloud_provider}", disabled=job_active
                )
                api_key = st.text_input(
                    "API key",
                    type="password",
                    key=f"repair_key_{cloud_provider}",
                    disabled=job_active,
                    help="Kept only in this Streamlit session. Never written to the repository.",
                )
                if api_key.strip():
                    try:
                        cloud_backend = CommercialRepairBackend(CommercialRepairConfig(
                            provider=cloud_provider, api_key=api_key.strip(), model=cloud_model.strip()
                        ))
                    except ValueError as exc:
                        st.warning(str(exc))
                else:
                    st.caption("Cloud escalation stays unavailable until a key is entered.")
        if controller.repair_manager is not None and not job_active:
            controller.repair_manager.configure(
                RepairPolicy(
                    mode=repair_mode,
                    cloud_provider=cloud_provider,
                    cloud_model=cloud_model,
                ),
                cloud_backend=cloud_backend,
            )

        if repair_enabled:
            st.caption("Passing patch checks does not verify the original failure without an agent replay hook.")

        st.caption(
            f"{llm_client.config.model} · {selected_mode.value} · {selected_execution_label}"
        )


def main() -> None:
    st.set_page_config(
        page_title="CohortFlow",
        page_icon="🧭",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 0.75rem;}
        h1 {margin-bottom: 0.15rem;}
        [data-testid="stChatInput"] {padding-bottom: 0.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    prepared_mode = _parse_mode()
    selection = st.sidebar.radio("Dataset", ("Prepared", "Custom"), disabled=bool(st.session_state.get("background_interaction")))
    mode = AppMode.CUSTOM if selection == "Custom" else prepared_mode
    if mode is AppMode.CUSTOM:
        st.sidebar.text_input("Adapter directory", key="custom_adapter_directory", disabled=bool(st.session_state.get("background_interaction")))
        if not st.session_state.get("custom_adapter_directory", "").strip():
            st.info("Choose the directory produced by the adapter's --build command. It contains the DuckDB database and adaptation log.")
            st.stop()
    try:
        _ensure_session(mode)
    except (ValueError, OSError, KeyError, OllamaError, DuckDBError) as exc:
        st.error(str(exc))
        st.stop()

    controller = st.session_state.interaction_controller
    adapter = st.session_state.presentation_adapter
    ui_state: UIState = st.session_state.ui_state
    llm_client: OllamaClient = st.session_state.llm_client

    completed_result = st.session_state.pop("background_result", None)
    if completed_result is not None:
        _apply_result(ui_state, completed_result)

    _render_settings(controller, llm_client, job_active=bool(st.session_state.get("background_interaction")))

    metadata = adapter.metadata()
    st.title(f"{metadata.icon} {metadata.display_name}")
    st.caption(metadata.description)
    st.caption(
        "Research prototype: use this interface for cohort requests, corrections, "
        "and responses to active clarifications or confirmations."
    )
    if mode is AppMode.DEMO:
        st.info(
            "Demo mode uses 24 entirely synthetic patients and demo-only "
            "concept IDs. No MIMIC or other real clinical data is included.",
            icon="🧪",
        )

    active_job = st.session_state.get("background_interaction")
    job_active_now = active_job is not None
    job_active_before = bool(st.session_state.get("workspace_job_active", False))
    if "workspace_panel_target" not in st.session_state:
        st.session_state.workspace_panel_target = (
            "Operation log" if job_active_now else "Evidence"
        )
    if "workspace_panel_generation" not in st.session_state:
        st.session_state.workspace_panel_generation = 0
    if job_active_now and not job_active_before:
        st.session_state.workspace_panel_target = "Operation log"
        st.session_state.workspace_panel_generation += 1
    elif not job_active_now and job_active_before:
        st.session_state.workspace_panel_target = "Evidence"
        st.session_state.workspace_panel_generation += 1
    st.session_state.workspace_job_active = job_active_now

    left, right = st.columns([1.25, 1], gap="large")

    with right:
        st.subheader("Conversation")
        chat_container = st.container(height=545, border=True)
        with chat_container:
            if not ui_state.messages:
                if mode is AppMode.DEMO:
                    st.caption("Try: Find patients with type 2 diabetes mellitus. Then: Also require metformin exposure.")
                elif mode is AppMode.CUSTOM:
                    st.caption("Describe a cohort using fields in the selected adapter catalog.")
                else:
                    st.caption("Describe the patients and eligibility criteria for your cohort.")
            for message in ui_state.messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)
                    # Graph versions are visible in the execution workspace.
                    # repeating them under every chat turn adds noise and makes
                    # clarification loops especially hard to read.

        pending_interaction = (
            controller.session.state.pending_interaction
            if controller.session.state is not None
            else None
        )
        if (
            pending_interaction is not None
            and pending_interaction.kind is PendingInteractionKind.NOTE_EXECUTION_CONFIRMATION
            and active_job is None
        ):
            st.warning(pending_interaction.question, icon="📝")
            confirm_col, without_notes_col, cancel_col = st.columns(3)
            if confirm_col.button("Proceed", type="primary", key="confirm_note_processing"):
                current_version = controller.session.state.version
                ui_state.messages.append(_user_message("Proceed", current_version))
                job = BackgroundInteractionJob(controller, "proceed")
                st.session_state.background_interaction = job
                st.session_state.workspace_panel_target = "Operation log"
                st.session_state.workspace_panel_generation = int(st.session_state.get("workspace_panel_generation", 0)) + 1
                st.session_state.workspace_job_active = True
                job.start()
                st.rerun()
            if without_notes_col.button(
                "Proceed without notes",
                key="confirm_without_note_processing",
                help="Remove note criteria from this already-parsed cohort and continue with structured criteria only.",
            ):
                current_version = controller.session.state.version
                ui_state.messages.append(
                    _user_message("Proceed without notes", current_version)
                )
                job = BackgroundInteractionJob(controller, "proceed without notes")
                st.session_state.background_interaction = job
                st.session_state.workspace_panel_target = "Operation log"
                st.session_state.workspace_panel_generation = int(st.session_state.get("workspace_panel_generation", 0)) + 1
                st.session_state.workspace_job_active = True
                job.start()
                st.rerun()
            if cancel_col.button("Cancel", key="cancel_note_processing"):
                result = controller.handle_message("cancel")
                _apply_result(ui_state, result)
                st.rerun()

        if (
            pending_interaction is not None
            and pending_interaction.kind is PendingInteractionKind.CORRECTION_FALLBACK_CONFIRMATION
            and active_job is None
        ):
            st.warning(pending_interaction.question, icon="↩️")
            fallback_col, keep_col = st.columns(2)
            if fallback_col.button(
                "Use Full Regeneration",
                type="primary",
                key="confirm_correction_full_regeneration",
            ):
                current_version = controller.session.state.version
                ui_state.messages.append(_user_message("Use Full Regeneration", current_version))
                job = BackgroundInteractionJob(controller, "use full regeneration")
                st.session_state.background_interaction = job
                st.session_state.workspace_panel_target = "Operation log"
                st.session_state.workspace_panel_generation = int(st.session_state.get("workspace_panel_generation", 0)) + 1
                st.session_state.workspace_job_active = True
                job.start()
                st.rerun()
            if keep_col.button(
                "Keep current cohort",
                key="decline_correction_full_regeneration",
            ):
                result = controller.handle_message("keep current")
                _apply_result(ui_state, result)
                st.rerun()

        pending_resolution = (
            controller.session.state.pending_resolution
            if controller.session.state is not None
            else None
        )
        if pending_resolution is not None:
            term = str(pending_resolution.request.input_payload.get("term") or "criterion")
            domain = str(pending_resolution.request.input_payload.get("domain") or "semantic concept")
            st.warning(
                f"Clarification required for {domain}: {term}",
                icon="🔎",
            )
            st.caption(pending_resolution.question)
            st.caption(
                "You can select candidates below or reply in natural language: "
                "for example, ‘serum potassium’, ‘none of these’, ‘remove this "
                "requirement’, ‘replace it with sodium’, or ‘cancel’."
            )

            widget_key = f"manual_resolution_{pending_resolution.resolution_id}"
            with st.expander("Select candidates manually", expanded=True):
                option_labels = {
                    option.option_id: option.label
                    for option in pending_resolution.options
                }
                option_ids = list(option_labels)
                allow_many = (
                    pending_resolution.request.response_mode
                    is ResponseMode.SELECT_MANY
                )
                if allow_many:
                    chosen = st.multiselect(
                        "Choose one or more options",
                        options=option_ids,
                        format_func=lambda oid: option_labels[oid],
                        key=f"{widget_key}_input",
                    )
                else:
                    single = st.radio(
                        "Choose one option",
                        options=option_ids,
                        format_func=lambda oid: option_labels[oid],
                        index=None,
                        key=f"{widget_key}_input",
                    )
                    chosen = [single] if single is not None else []

                if st.button(
                    "Submit selection",
                    disabled=not chosen,
                    key=f"{widget_key}_submit",
                ):
                    indexes = [option_ids.index(option_id) + 1 for option_id in chosen]
                    reply_text = " and ".join(str(index) for index in indexes)
                    visible = ", ".join(option_labels[option_id] for option_id in chosen)
                    current_version = controller.session.state.version
                    ui_state.messages.append(
                        _user_message(f"Selected: {visible}", current_version)
                    )
                    # Route manual selections through the same background
                    # interaction path as typed replies. Resumed graph work and
                    # any immediately following ambiguity therefore remain
                    # visible in the operation/LLM activity panels.
                    job = BackgroundInteractionJob(controller, reply_text)
                    st.session_state.background_interaction = job
                    st.session_state.workspace_panel_target = "Operation log"
                    st.session_state.workspace_panel_generation = int(
                        st.session_state.get("workspace_panel_generation", 0)
                    ) + 1
                    st.session_state.workspace_job_active = True
                    job.start()
                    st.rerun()

        prompt = st.chat_input(
            (
                "Send another correction (it will queue after the current module)"
                if active_job is not None
                else (
                    "Reply to the pending clarification or refine/remove/replace the requirement"
                    if pending_resolution is not None
                    else "Describe the cohort or send a correction"
                )
            ),
            key="cohort_chat_input",
        )
        if prompt:
            current_version = (
                controller.session.state.version
                if controller.session.state is not None
                else None
            )
            ui_state.messages.append(_user_message(prompt, current_version))
            if active_job is not None:
                if not active_job.submit_correction(prompt):
                    st.warning("The running interaction could not accept another correction.")
            else:
                job = BackgroundInteractionJob(controller, prompt)
                st.session_state.background_interaction = job
                st.session_state.workspace_panel_target = "Operation log"
                st.session_state.workspace_panel_generation = int(
                    st.session_state.get("workspace_panel_generation", 0)
                ) + 1
                st.session_state.workspace_job_active = True
                job.start()
            st.rerun()

    with left:
        _render_last_failure(controller)
        st.subheader("Execution workspace")
        panel = st.segmented_control(
            "Workspace panel",
            options=["Evidence", "Cohort graph", "Operation log"],
            selection_mode="single",
            default=st.session_state.workspace_panel_target,
            required=True,
            key=f"workspace_panel_{st.session_state.workspace_panel_generation}",
            label_visibility="collapsed",
            width="stretch",
        ) or st.session_state.workspace_panel_target
        st.session_state.workspace_panel_target = panel

        state = controller.session.state
        pending = None if state is None else state.pending_interaction
        manual_active = (
            pending is not None
            and pending.kind is PendingInteractionKind.MANUAL_EDIT
        )

        if panel == "Evidence":
            if state is None:
                st.info(
                    "Describe a cohort in the conversation panel to create "
                    "the first execution graph."
                )
            elif manual_active and active_job is not None:
                st.info("Finish or stop active work before applying manual edits.")
            elif manual_active:
                action, submitted = render_manual_editor(pending.manual_form)
                if action == "cancel":
                    result = controller.cancel_manual_edit()
                    ui_state.messages.extend(result.messages)
                    st.rerun()
                if action == "reset":
                    st.rerun()
                if action == "submit" and submitted is not None:
                    with st.status("Validating manual changes...", expanded=True) as status:
                        result = controller.submit_manual_edit(submitted)
                        if result.error:
                            status.update(
                                label="Manual validation failed",
                                state="error",
                                expanded=True,
                            )
                        else:
                            status.update(
                                label="Manual changes processed",
                                state="complete",
                                expanded=False,
                            )
                    ui_state.messages.extend(result.messages)
                    if result.graph_version is not None:
                        ui_state.graph_version = result.graph_version
                    if result.selected_node_id is not None:
                        ui_state.selected_node_id = result.selected_node_id
                    st.rerun()
            else:
                try:
                    render_execution_evidence(adapter.execution_evidence())
                except Exception as exc:
                    st.error(
                        "Evidence visualization is temporarily unavailable, "
                        f"but the conversation remains usable: {exc}"
                    )

        if panel == "Cohort graph":
            if state is None:
                st.info("The cohort graph will appear once the initial request is parsed.")
            elif manual_active:
                st.caption("Apply or cancel the manual edit before inspecting the updated graph.")
            else:
                try:
                    graph_view = adapter.graph_view()
                    selected = render_graph(
                        graph_view,
                        selected_node_id=ui_state.selected_node_id,
                    )
                    if selected != ui_state.selected_node_id:
                        ui_state.selected_node_id = selected
                        ui_state.clear_entity_selection()

                    if ui_state.selected_node_id:
                        detail = adapter.node_view(ui_state.selected_node_id)
                        st.divider()
                        render_node_panel(detail)
                        collection_id, entity_id = render_entity_browser(
                            detail.entity_collections,
                            selected_collection_id=ui_state.selected_collection_id,
                            selected_entity_id=ui_state.selected_entity_id,
                            load_entity=adapter.entity_view,
                        )
                        ui_state.selected_collection_id = collection_id
                        ui_state.selected_entity_id = entity_id
                except Exception as exc:
                    st.error(
                        "Graph visualization is temporarily unavailable, "
                        f"but the conversation remains usable: {exc}"
                    )

        if panel == "Operation log":
            _render_operation_panel(active_job, controller)


if __name__ == "__main__":
    main()
