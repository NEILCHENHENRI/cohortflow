from pathlib import Path

import pytest

from infrastructure.config import AppMode, ProjectPaths
from infrastructure.demo import build_demo_databases


def test_project_paths_select_demo_database_names(tmp_path):
    paths = ProjectPaths.from_project_root(tmp_path, mode=AppMode.DEMO)

    assert paths.clinical_database == (
        tmp_path.resolve() / "data" / "demo" / "clinical.duckdb"
    )
    assert paths.vocabulary_database == (
        tmp_path.resolve() / "data" / "demo" / "vocabulary.duckdb"
    )


def test_project_paths_select_local_mimic_hosp_database(tmp_path):
    paths = ProjectPaths.from_project_root(tmp_path, mode=AppMode.LOCAL)

    assert paths.clinical_database == (
        tmp_path.resolve() / "data" / "mimic_hosp.duckdb"
    )
    assert paths.vocabulary_database is None


def test_demo_builder_creates_expected_databases(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    clinical, vocabulary = build_demo_databases(tmp_path / "demo")

    assert clinical.exists()
    assert vocabulary.exists()
    # DuckDB's storage header must remain 1.0-compatible even on a 2.x builder.
    assert int.from_bytes(clinical.read_bytes()[12:20], "little") == 64
    assert int.from_bytes(vocabulary.read_bytes()[12:20], "little") == 64
    with duckdb.connect(str(clinical), read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM person"
        ).fetchone()[0] == 24
        assert connection.execute(
            """SELECT COUNT(DISTINCT person_id)
               FROM condition_occurrence
               WHERE condition_concept_id = 9000001"""
        ).fetchone()[0] == 12


def test_demo_factory_executes_documented_request_and_compact_correction(tmp_path, monkeypatch):
    import infrastructure.factory as factory
    from framework.correction import CorrectionMode
    calls = []

    class ScriptedLLM:
        def __init__(self, config):
            self.config = config
            self.think = False
            self.responses = [
                {"predicates": [{"predicate_id": "p1", "text": "type 2 diabetes mellitus"}],
                 "logic": "p1", "ambiguities": [], "requires_clarification": False,
                 "clarification_question": None, "rationale": "Initial cohort."},
                {"requirements": [{"requirement_id": "p1", "kind": "condition",
                 "payload": {"term": "Type 2 diabetes mellitus"}, "polarity": "include", "required": True}],
                 "assumptions": []},
                {"actions": [{"action": "ADD", "local_ref": "new_1",
                  "requirement_text": "metformin exposure", "excluded": False,
                  "anchor": "ROOT", "operator": "AND", "relation_memberships": []}],
                 "requires_clarification": False, "clarification_question": None},
                {"requirements": [{"requirement_id": "p2", "kind": "medication",
                 "payload": {"term": "Metformin"}, "polarity": "include", "required": True}],
                 "assumptions": []},
            ]

        def ensure_ready(self, **kwargs):
            calls.append(self.config.model)

        def generate_structured(self, **kwargs):
            assert self.responses, "Unexpected model call"
            return self.responses.pop(0)

    monkeypatch.setattr(factory, "OllamaClient", ScriptedLLM)
    def forbidden(*args, **kwargs):
        raise AssertionError("Demo must not initialize MIMIC services")
    monkeypatch.setattr(factory, "MimicHospConceptResolver", forbidden)
    monkeypatch.setattr(factory, "BGEHybridNoteRetriever", forbidden)
    from infrastructure.config import InfrastructureConfig
    config = InfrastructureConfig(
        paths=ProjectPaths.from_project_root(tmp_path, mode=AppMode.DEMO),
        mode=AppMode.DEMO,
    )
    stack = factory.build_local_cohort_stack(config)
    assert calls == ["qwen3:4b"]
    assert stack.agent.parser.localized_strategy == "compact"
    assert stack.agent.parser.correction_mode is CorrectionMode.ROBUST_DELTA
    initial = stack.interaction_controller.handle_message("Find patients with type 2 diabetes mellitus")
    assert stack.session.artifacts.contains("cohort_summary"), [m.content for m in initial.messages]
    assert stack.session.summary()["cohort_size"] == 12
    result = stack.interaction_controller.handle_message("Also require metformin exposure")
    assert stack.session.summary()["cohort_size"] == 8
    assert any("reused" in message.content for message in result.messages)
    assert not stack.llm.responses


def test_run_launcher_forwards_demo_mode_and_port(monkeypatch):
    import run
    import sys
    captured = {}
    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return 0
    monkeypatch.setattr(sys, "argv", ["run.py", "--demo", "--port", "8519",
                                     "--server.headless", "true"])
    monkeypatch.setattr(run.subprocess, "call", launch)
    with pytest.raises(SystemExit) as exit_info:
        run.main()
    assert exit_info.value.code == 0
    assert captured["command"][-3:] == ["--", "--mode", "demo"]
    assert captured["command"][:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.headless" in captured["command"]
    assert captured["command"][captured["command"].index("--server.port") + 1] == "8519"
    assert captured["cwd"] == Path(run.__file__).resolve().parent


def test_demo_requirement_payloads_filter_dates_and_counts_and_reject_unsupported(tmp_path):
    from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
    from infrastructure.omop.repository import DuckDBCohortRepository
    clinical, _ = build_demo_databases(tmp_path / "demo")
    repository = DuckDBCohortRepository(DuckDBConnectionFactory(DuckDBConfig(clinical)))
    assert repository.patients_with_condition(9000001, {"count_min": 2}) == set()
    assert repository.patients_with_condition(
        9000001, {"start_time": "2022-03-01"}
    ) == set(range(6, 13))
    assert repository.patients_with_medication(
        9000101, {"end_time": "2022-03-20"}
    ) == {1, 2, 3}
    with pytest.raises(ValueError, match="does not support"):
        repository.patients_with_medication(9000101, {"dose_value": 500})
    with pytest.raises(ValueError, match="any_recorded"):
        repository.patients_with_medication(9000101, {"exposure_mode": "active_at_index"})
