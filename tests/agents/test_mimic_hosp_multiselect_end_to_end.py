"""End-to-end test: a real MIMIC-IV hosp-schema DuckDB, driven entirely
through AgentInteractionController, where an ambiguous condition term
resolves to several ICD codes (spanning ICD-9's "type II" phrasing and
ICD-10's "type 2" phrasing) and the human selects more than one -- which
must be combined with OR rather than requiring a patient to have every
selected code.

This is the scenario from the reported bug: a plain query for "type 2
diabetes" only ever found the ICD-10 candidate, silently missing every
patient whose diabetes was coded under the pre-2015 ICD-9 admissions.
"""

import pytest

duckdb = pytest.importorskip("duckdb")

from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from agents.cohort.parser import CohortTaskParser
from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.llm.recovery import LLMSemanticExceptionResolver
from infrastructure.mimic_hosp.concepts import MimicHospConceptResolver
from infrastructure.mimic_hosp.repository import MimicHospCohortRepository
from interfaces.controller import AgentInteractionController


def make_clinical(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE patients (
                subject_id BIGINT, gender VARCHAR, anchor_age INTEGER,
                anchor_year INTEGER, dod DATE
            )
        """)
        con.execute("""
            CREATE TABLE admissions (
                subject_id BIGINT, hadm_id BIGINT,
                admittime TIMESTAMP, dischtime TIMESTAMP, admission_type VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE diagnoses_icd (
                subject_id BIGINT, hadm_id BIGINT, seq_num INTEGER,
                icd_code VARCHAR, icd_version INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE d_icd_diagnoses (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, drug VARCHAR,
                starttime TIMESTAMP, stoptime TIMESTAMP
            )
        """)
        con.executemany(
            "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
            [
                (1, "M", 40, 2020, None),  # coded ICD-9
                (2, "F", 50, 2020, None),  # coded ICD-10
                (3, "F", 60, 2020, None),  # unrelated condition only
            ],
        )
        con.executemany(
            "INSERT INTO admissions VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, "2020-01-01", "2020-01-03", "URGENT"),
                (2, 502, "2020-01-01", "2020-01-03", "URGENT"),
                (3, 503, "2020-01-01", "2020-01-03", "URGENT"),
            ],
        )
        con.executemany(
            "INSERT INTO diagnoses_icd VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, 1, "25000", 9),
                (2, 502, 1, "E119", 10),
                (3, 503, 1, "I10", 10),
            ],
        )
        con.executemany(
            "INSERT INTO d_icd_diagnoses VALUES (?, ?, ?)",
            [
                (
                    "25000", 9,
                    "Diabetes mellitus without mention of complication, "
                    "type II or unspecified type",
                ),
                ("E119", 10, "Type 2 diabetes mellitus without complications"),
                ("I10", 10, "Essential (primary) hypertension"),
            ],
        )
    return path


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_structured(self, **kwargs):
        return self.responses.pop(0)


class NeverLLM:
    """Asserts the deterministic multi-select reply parser handled the
    human's answer without needing an LLM round trip."""

    def generate_structured(self, **kwargs):
        raise AssertionError("deterministic multi-select should handle this")


def test_multiple_selected_candidates_combine_with_or_not_and(tmp_path):
    clinical = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    resolver = MimicHospConceptResolver(clinical)
    repository = MimicHospCohortRepository(clinical)

    llm = QueueLLM([
        {
            "predicates": [
                {"predicate_id": "diabetes", "text": "patient has type 2 diabetes"},
            ],
            "logic": "diabetes",
            "ambiguities": [],
            "requires_clarification": False,
            "clarification_question": None,
            "rationale": "Initial cohort.",
        },
        {
            "requirements": [{
                "requirement_id": "diabetes",
                "kind": "condition",
                "payload": {"term": "type 2 diabetes"},
                "polarity": "include",
                "required": True,
            }],
            "rationale": "compiled",
            "assumptions": [],
        },
    ])

    agent = CohortAgent(CohortTaskParser(llm), resolver, repository)
    session = agent.create_session()
    semantic = LLMSemanticExceptionResolver(NeverLLM())
    controller = AgentInteractionController(session, CohortInteractionAdapter(), semantic)

    first = controller.handle_message("find patients with type 2 diabetes")
    assert first.error is None
    pending = session.state.pending_resolution
    assert pending is not None
    offered = {option.option_id for option in pending.options}
    assert {"25000", "E119"} <= offered

    second = controller.handle_message("1 and 2")
    assert second.error is None
    assert session.state.pending_resolution is None

    # Both the ICD-9-coded patient (1) and the ICD-10-coded patient (2) must
    # be included -- an AND of the two codes would incorrectly require a
    # single patient to carry both codes and return nobody.
    assert session.summary()["cohort_size"] == 2


def test_manual_selection_bypasses_reply_parsing_but_reaches_the_same_result(tmp_path):
    # The parallel "select manually" path (e.g. a checkbox widget) must
    # apply exactly like a typed reply would -- same OR-combination, same
    # resumed execution -- without going through interpret_human_reply at
    # all, so it must work even with no semantic resolver configured.
    clinical = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    resolver = MimicHospConceptResolver(clinical)
    repository = MimicHospCohortRepository(clinical)

    llm = QueueLLM([
        {
            "predicates": [
                {"predicate_id": "diabetes", "text": "patient has type 2 diabetes"},
            ],
            "logic": "diabetes",
            "ambiguities": [],
            "requires_clarification": False,
            "clarification_question": None,
            "rationale": "Initial cohort.",
        },
        {
            "requirements": [{
                "requirement_id": "diabetes",
                "kind": "condition",
                "payload": {"term": "type 2 diabetes"},
                "polarity": "include",
                "required": True,
            }],
            "rationale": "compiled",
            "assumptions": [],
        },
    ])

    agent = CohortAgent(CohortTaskParser(llm), resolver, repository)
    session = agent.create_session()
    # No semantic_resolver at all -- the manual path must not need one.
    controller = AgentInteractionController(session, CohortInteractionAdapter())

    first = controller.handle_message("find patients with type 2 diabetes")
    assert first.error is None
    pending = session.state.pending_resolution
    assert pending is not None
    option_ids = tuple(
        option.option_id for option in pending.options
        if option.option_id in {"25000", "E119"}
    )
    assert len(option_ids) == 2

    result = controller.submit_pending_resolution(option_ids)
    assert session.state.pending_resolution is None
    assert "Manually selected" in result.messages[0].content
    assert session.summary()["cohort_size"] == 2


def test_submit_pending_resolution_rejects_unknown_option_id(tmp_path):
    clinical = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    resolver = MimicHospConceptResolver(clinical)
    repository = MimicHospCohortRepository(clinical)

    llm = QueueLLM([
        {
            "predicates": [
                {"predicate_id": "diabetes", "text": "patient has type 2 diabetes"},
            ],
            "logic": "diabetes",
            "ambiguities": [],
            "requires_clarification": False,
            "clarification_question": None,
            "rationale": "Initial cohort.",
        },
        {
            "requirements": [{
                "requirement_id": "diabetes",
                "kind": "condition",
                "payload": {"term": "type 2 diabetes"},
                "polarity": "include",
                "required": True,
            }],
            "rationale": "compiled",
            "assumptions": [],
        },
    ])

    agent = CohortAgent(CohortTaskParser(llm), resolver, repository)
    session = agent.create_session()
    controller = AgentInteractionController(session, CohortInteractionAdapter())

    controller.handle_message("find patients with type 2 diabetes")
    assert session.state.pending_resolution is not None

    with pytest.raises(ValueError, match="unknown resolution option"):
        controller.submit_pending_resolution(("not-a-real-option-id",))
