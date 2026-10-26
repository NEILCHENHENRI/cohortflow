import pytest

duckdb = pytest.importorskip("duckdb")

from framework.resolution import (
    ResolutionAction,
    ResolutionDecision,
    ResolutionOption,
    ResolutionRequired,
)
from infrastructure.database.duckdb import (
    DuckDBConfig,
    DuckDBConnectionFactory,
)
from infrastructure.omop.concepts import OMOPConceptResolver


def make_vocabulary(tmp_path):
    path = tmp_path / "vocabulary.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute('''
            CREATE TABLE concept (
                concept_id BIGINT,
                concept_name VARCHAR,
                domain_id VARCHAR,
                vocabulary_id VARCHAR,
                concept_class_id VARCHAR,
                standard_concept VARCHAR,
                concept_code VARCHAR,
                invalid_reason VARCHAR
            )
        ''')
        con.executemany(
            "INSERT INTO concept VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    201826, "Type 2 diabetes mellitus", "Condition", "SNOMED",
                    "Clinical Finding", "S", "44054006", None,
                ),
                (
                    202020, "Type 2 diabetes mellitus with ulcer", "Condition",
                    "SNOMED", "Clinical Finding", "S", "x", None,
                ),
                (
                    19019073, "Metformin", "Drug", "RxNorm",
                    "Ingredient", "S", "6809", None,
                ),
            ],
        )
    return path


def resolver(tmp_path, exception_resolver=None):
    factory = DuckDBConnectionFactory(
        DuckDBConfig(make_vocabulary(tmp_path), read_only=True)
    )
    return OMOPConceptResolver(factory, exception_resolver)


def test_exact_condition_and_medication_match(tmp_path):
    r = resolver(tmp_path)

    condition = r.resolve_condition("Type 2 diabetes mellitus")
    drug = r.resolve_medication("Metformin")

    assert condition.concept_id == 201826
    assert condition.domain_id == "Condition"
    assert drug.domain_id == "Drug"


def test_missing_exact_concept_raises(tmp_path):
    r = resolver(tmp_path)

    with pytest.raises(LookupError):
        r.resolve_condition("Unknown diagnosis")


def test_ambiguous_condition_without_resolver_asks_for_clarification(tmp_path):
    r = resolver(tmp_path)

    with pytest.raises(ResolutionRequired):
        r.resolve_condition("type 2 diabetes")


class SelectTop:
    def resolve(self, request):
        return ResolutionDecision(
            action=ResolutionAction.SELECT_OPTION,
            confidence=0.99,
            rationale="The general phrase matches the general concept.",
            option_id=request.options[0].option_id,
        )


def test_llm_assisted_selection_resolves_ambiguity(tmp_path):
    r = resolver(tmp_path, SelectTop())

    concept = r.resolve_condition("type 2 diabetes")
    assert concept.concept_id == 201826


class AskHuman:
    def resolve(self, request):
        return ResolutionDecision(
            action=ResolutionAction.ASK_HUMAN,
            confidence=0.5,
            rationale="Ambiguous.",
            question="Which diabetes concept did you mean?",
        )


def test_human_escalation_round_trip_reuses_recorded_selection(tmp_path):
    r = resolver(tmp_path, AskHuman())

    with pytest.raises(ResolutionRequired):
        r.resolve_condition("type 2 diabetes")

    r.record_human_selection(
        pending_context={
            "input_payload": {
                "resolution_key": {
                    "resolver_id": "omop_concept",
                    "input_fingerprint": "Condition:type 2 diabetes",
                    "scope_id": "session",
                }
            }
        },
        options=(
            ResolutionOption(
                "201826",
                "Type 2 diabetes mellitus",
                {"concept_id": 201826, "domain_id": "Condition"},
            ),
        ),
        rationale="Human selected the general concept.",
    )

    concept = r.resolve_condition("type 2 diabetes")
    assert concept.concept_id == 201826
    assert concept.resolution_source == "human"
