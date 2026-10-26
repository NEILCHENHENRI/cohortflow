from __future__ import annotations

import json
from pathlib import Path

from agents.cohort.parser import render_structured_task
from framework.logic import LogicExpression, assign_logic_node_ids
from framework.task import Objective, Requirement, TaskSpec


def _task() -> TaskSpec:
    logic = assign_logic_node_ids(
        LogicExpression.or_(
            LogicExpression.predicate("req_diabetes", "type 2 diabetes"),
            LogicExpression.predicate("req_hypertension", "hypertension"),
        )
    )
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "test"),
        requirements=(
            Requirement("req_diabetes", "condition", {"term": "type 2 diabetes"}),
            Requirement("req_hypertension", "condition", {"term": "hypertension"}),
        ),
        logic=logic,
        context={"relations": []},
    )


def test_structured_task_renderer_uses_stable_refs_and_current_logic():
    payload = json.loads(render_structured_task(_task()))
    assert [item["ref"] for item in payload["predicates"]] == ["p1", "p2"]
    assert payload["logic"]["operator"] == "OR"
    assert payload["logic"]["children"] == ["p1", "p2"]
    assert payload["relations"] == []


def test_operation_panel_shows_taskspec_only_after_background_job_is_gone():
    source_path = Path(__file__).resolve().parents[2] / "interfaces" / "streamlit_app.py"
    source = source_path.read_text()
    start = source.index("def _render_operation_panel")
    end = source.index("\n\ndef _render_last_failure", start)
    function_source = source[start:end]
    before_running_snapshot, after_running_snapshot = function_source.split("snapshot = job.snapshot()", 1)
    assert "_render_current_task_spec(controller)" in before_running_snapshot
    assert "_render_current_task_spec(controller)" not in after_running_snapshot


def test_streamlit_exposes_only_sequential_and_parallel_execution_modes():
    source_path = Path(__file__).resolve().parents[2] / "interfaces" / "streamlit_app.py"
    source = source_path.read_text()
    assert '"Sequential": ExecutionMode.SEQUENTIAL' in source
    assert '"Parallel": ExecutionMode.PARALLEL' in source
    assert "Smart/Safe" not in source
    assert "Full Parallel" not in source


def test_pending_resolution_has_one_natural_language_input_path():
    source_path = Path(__file__).resolve().parents[2] / "interfaces" / "streamlit_app.py"
    source = source_path.read_text()
    assert '"Reply or refine"' not in source
    assert '"Reply to the pending clarification or refine/remove/replace the requirement"' in source
    assert 'with st.expander("Select candidates manually", expanded=True):' in source


def test_title_area_states_intended_research_prototype_usage():
    source_path = Path(__file__).resolve().parents[2] / "interfaces" / "streamlit_app.py"
    source = source_path.read_text()
    assert "Research prototype: use this interface for cohort requests, corrections" in source
    assert "responses to active clarifications or confirmations" in source


# Exercise real Streamlit widgets without databases, model calls or a web server.
def _settings_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string('''
from types import SimpleNamespace
import streamlit as st
from agents.cohort.parser import CohortTaskParser
from framework.correction import CorrectionMode
from framework.operator import OperatorRegistry
from framework.session import SteerableSession
from framework.repair import RepairManager
from infrastructure.llm.ollama import OllamaClient
from infrastructure.mimic_hosp.note_retrieval import NoteRetrievalConfig
from interfaces.streamlit_app import _render_settings
if "controller" not in st.session_state:
    llm = OllamaClient()
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA)
    agent = SimpleNamespace(parser=parser, operator_registry=OperatorRegistry,
        note_matcher=SimpleNamespace(think=True, retriever=SimpleNamespace(config=NoteRetrievalConfig())))
    session = SteerableSession(agent)
    st.session_state.controller = SimpleNamespace(session=session, repair_manager=RepairManager(repository_root="."))
    st.session_state.llm = llm
_render_settings(st.session_state.controller, st.session_state.llm,
    job_active=st.session_state.get("busy", False))
''').run()


def test_streamlit_settings_map_to_parser_runtime_and_repair():
    from framework.parsing import ParsingMode
    from framework.correction import CorrectionMode
    from framework.runtime import ExecutionMode
    from framework.repair import RepairMode
    app = _settings_app()
    assert not app.exception
    app.number_input(key="note_relevance_threshold").set_value(0.6).run()
    assert not app.exception
    assert app.session_state["controller"].session.domain_agent.note_matcher.retriever.config.relevance_threshold == 0.6
    controller = app.session_state["controller"]
    parser = controller.session.domain_agent.parser
    assert parser.parsing_mode is ParsingMode.GRAPHITE
    assert parser.correction_mode is CorrectionMode.ROBUST_DELTA
    assert controller.repair_manager.policy.mode is RepairMode.OFF
    for mode in ParsingMode:
        app.selectbox(key="parsing_strategy").set_value(mode).run()
        assert not app.exception
        assert parser.parsing_mode is mode
    app.selectbox(key="localized_parsing_strategy").select("Original localized steering (v1)").run()
    assert parser.localized_strategy == "legacy"
    app.radio(key="correction_strategy").set_value("Full Regeneration").run()
    assert parser.correction_mode is CorrectionMode.FULL_REGENERATION
    app.radio(key="execution_strategy").set_value("Parallel").run()
    assert controller.session.runtime.execution_mode is ExecutionMode.PARALLEL
    assert controller.session.runtime.max_workers == 1
    app.toggle(key="self_repair_enabled").set_value(True).run()
    assert controller.repair_manager.policy.mode is RepairMode.LOCAL
    app.radio(key="repair_path").set_value("Local → Cloud").run()
    assert app.text_input(key="repair_model_openai").value == "gpt-5.4"
    app.text_input(key="repair_key_openai").set_value("mock-key").run()
    assert controller.repair_manager.cloud_backend.provider == "openai"
    app.text_input(key="repair_model_openai").set_value(" ").run()
    assert controller.repair_manager.cloud_backend is None
    assert any("model cannot be empty" in item.value for item in app.warning)


def test_busy_streamlit_settings_are_disabled_and_do_not_mutate_workers():
    from unittest.mock import patch
    app = _settings_app()
    app.toggle(key="self_repair_enabled").set_value(True).run()
    app.radio(key="repair_path").set_value("Local → Cloud").run()
    controller = app.session_state["controller"]
    parser = controller.session.domain_agent.parser
    app.session_state["busy"] = True
    app.session_state["llm"].think = True
    note_config = controller.session.domain_agent.note_matcher.retriever.config
    with patch.object(parser, "configure_parsing", side_effect=AssertionError("active parser mutated")), \
         patch.object(controller.repair_manager, "configure", side_effect=AssertionError("active repair mutated")):
        app.run()
    assert not app.exception
    assert app.session_state["llm"].think is True
    assert controller.session.domain_agent.note_matcher.retriever.config is note_config
    for kind in ("selectbox", "radio", "toggle", "number_input", "slider", "text_input"):
        assert all(widget.disabled for widget in app.get(kind))


def test_dataset_switch_failure_preserves_session_and_active_job_blocks_switch(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pytest
    import interfaces.streamlit_app as ui
    from infrastructure.config import AppMode
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    old = SimpleNamespace(repair_manager=Mock())
    state = State(app_mode="local", interaction_controller=old, custom_adapter_directory="/missing",
        ui_state="old-ui")
    monkeypatch.setattr(ui.st, "session_state", state)
    build = Mock(side_effect=ValueError("invalid bundle"))
    monkeypatch.setattr(ui, "_build_session", build)
    with pytest.raises(ValueError, match="invalid bundle"):
        ui._ensure_session(AppMode.CUSTOM)
    assert state.interaction_controller is old
    assert state.ui_state == "old-ui" and state.app_mode == "local"
    old.repair_manager.close.assert_not_called()
    state.background_interaction = object()
    build.reset_mock()
    with pytest.raises(ValueError, match="active work"):
        ui._ensure_session(AppMode.CUSTOM)
    build.assert_not_called()
    state.pop("background_interaction")
    build.side_effect = None
    build.return_value = ("new-controller", "new-adapter", "new-ui", "new-llm")
    ui._ensure_session(AppMode.CUSTOM)
    assert state.interaction_controller == "new-controller"
    old.repair_manager.close.assert_called_once()
