import pytest

duckdb = pytest.importorskip("duckdb")

from framework.resolution import (
    ResolutionAction,
    ResolutionDecision,
    ResolutionRequired,
)
from infrastructure.database.duckdb import (
    DuckDBConfig,
    DuckDBConnectionFactory,
)
from infrastructure.mimic_hosp.concepts import MimicHospConceptResolver


def make_clinical(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE d_icd_diagnoses (
                icd_code VARCHAR,
                icd_version INTEGER,
                long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT,
                hadm_id BIGINT,
                drug VARCHAR,
                starttime TIMESTAMP,
                stoptime TIMESTAMP
            )
        """)
        con.executemany(
            "INSERT INTO d_icd_diagnoses VALUES (?, ?, ?)",
            [
                ("25000", 9, "Diabetes mellitus without mention of complication, type II or unspecified type"),
                ("25040", 9, "Diabetes with renal manifestations, type II or unspecified type"),
                ("E119", 10, "Type 2 diabetes mellitus without complications"),
                ("E1121", 10, "Type 2 diabetes mellitus with diabetic nephropathy"),
                ("I10", 10, "Essential (primary) hypertension"),
            ],
        )
        con.executemany(
            "INSERT INTO prescriptions VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, "Metformin", "2020-01-05", None),
                (1, 501, "Metformin", "2020-02-05", None),
            ],
        )
    return path


def resolver(tmp_path, exception_resolver=None):
    factory = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    return MimicHospConceptResolver(factory, exception_resolver)


def test_exact_condition_and_medication_match(tmp_path):
    r = resolver(tmp_path)

    condition = r.resolve_condition(
        "Type 2 diabetes mellitus without complications"
    )
    assert condition.concept_id == "E119"
    assert condition.condition_codes == (("E119", 10),)
    assert condition.domain_id == "Condition"

    drug = r.resolve_medication("Metformin")
    assert drug.domain_id == "Drug"
    assert drug.medication_names == ("Metformin",)


def test_unmatched_term_raises_lookup_error(tmp_path):
    r = resolver(tmp_path)

    with pytest.raises(LookupError):
        r.resolve_condition("completely unrelated phrase")


def test_ambiguous_condition_without_resolver_asks_for_clarification(tmp_path):
    r = resolver(tmp_path)

    with pytest.raises(ResolutionRequired):
        r.resolve_condition("diabetes")


def test_search_finds_icd9_phrasing_despite_roman_numeral_and_word_order(tmp_path):
    # ICD-9-CM writes "type II" (Roman numeral) with "diabetes mellitus"
    # first. ICD-10-CM writes "type 2" (Arabic) with "type 2" first. A
    # plain substring search for "type 2 diabetes" would only ever find the
    # ICD-10 rows. The resolver's search must find both.
    r = resolver(tmp_path)

    with pytest.raises(ResolutionRequired) as excinfo:
        r.resolve_condition("type 2 diabetes")

    candidate_ids = {option.option_id for option in excinfo.value.request.options}
    assert "25000" in candidate_ids  # ICD-9, "type II" phrasing
    assert "E119" in candidate_ids   # ICD-10, "type 2" phrasing


class SelectFirst:
    def resolve(self, request):
        return ResolutionDecision(
            action=ResolutionAction.SELECT_OPTION,
            confidence=0.9,
            rationale="Prefer the uncomplicated form.",
            option_id=request.options[0].option_id,
        )


def test_llm_assisted_selection_resolves_ambiguity(tmp_path):
    r = resolver(tmp_path, SelectFirst())

    concept = r.resolve_condition("diabetes")
    assert concept.resolution_source == "llm"
    assert concept.vocabulary_id in {"ICD9CM", "ICD10CM"}


class SelectMultiple:
    def resolve(self, request):
        ids = tuple(option.option_id for option in request.options[:2])
        return ResolutionDecision(
            action=ResolutionAction.SELECT_OPTION,
            confidence=0.9,
            rationale="Both variants apply.",
            option_id=ids[0],
            option_ids=ids,
        )


def test_llm_can_select_multiple_candidates_combined_with_or(tmp_path):
    r = resolver(tmp_path, SelectMultiple())

    concept = r.resolve_condition("diabetes")
    assert len(concept.entries) == 2
    assert len(concept.condition_codes) == 2


class AskHuman:
    def resolve(self, request):
        return ResolutionDecision(
            action=ResolutionAction.ASK_HUMAN,
            confidence=0.4,
            rationale="Too ambiguous.",
            question="Which diabetes code did you mean?",
        )


def test_human_escalation_round_trip_reuses_recorded_selection(tmp_path):
    r = resolver(tmp_path, AskHuman())

    with pytest.raises(ResolutionRequired) as excinfo:
        r.resolve_condition("diabetes")

    request = excinfo.value.request
    chosen = request.options[0]
    r.record_human_selection(
        pending_context={"input_payload": request.input_payload},
        options=(chosen,),
        rationale="Human picked the first option.",
    )

    concept = r.resolve_condition("diabetes")
    assert concept.resolution_source == "human"
    assert concept.concept_id == chosen.option_id


def test_human_can_select_multiple_options_combined_with_or(tmp_path):
    r = resolver(tmp_path, AskHuman())

    with pytest.raises(ResolutionRequired) as excinfo:
        r.resolve_condition("diabetes")

    request = excinfo.value.request
    chosen = request.options[:2]
    assert len(chosen) == 2
    r.record_human_selection(
        pending_context={"input_payload": request.input_payload},
        options=chosen,
        rationale="Human picked two related variants.",
    )

    concept = r.resolve_condition("diabetes")
    assert concept.resolution_source == "human"
    assert len(concept.entries) == 2
    assert {entry.code for entry in concept.entries} == {
        option.option_id for option in chosen
    }


def test_resolver_works_when_dictionary_tables_live_in_a_non_default_schema(tmp_path):
    # Reproduces the exact failure DuckDB raises for an unqualified query
    # against a table that only exists under a non-default schema (e.g.
    # `vocab`): "Table with name d_icd_diagnoses does not exist! Did you
    # mean vocab.d_icd_diagnoses?"
    path = tmp_path / "mimic_hosp_namespaced.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA vocab")
        con.execute("""
            CREATE TABLE vocab.d_icd_diagnoses (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, drug VARCHAR,
                starttime TIMESTAMP, stoptime TIMESTAMP
            )
        """)
        con.execute(
            "INSERT INTO vocab.d_icd_diagnoses VALUES "
            "('25000', 9, 'Diabetes mellitus without mention of complication')"
        )

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    r = MimicHospConceptResolver(factory)

    condition = r.resolve_condition(
        "Diabetes mellitus without mention of complication"
    )
    assert condition.concept_id == "25000"


def test_normalization_folds_roman_numerals_and_ignores_stopwords():
    from infrastructure.mimic_hosp.concepts import _tokenize

    # Word order differs (that's the point -- ICD-9 vs ICD-10 phrasing), but
    # the resulting token SET must match so the index scores them the same.
    assert set(_tokenize("Diabetes mellitus, type II")) == set(
        _tokenize("Type 2 diabetes mellitus")
    )
    assert "with" not in _tokenize("diabetes with complications")
    assert "unspecified" not in _tokenize("diabetes, unspecified")


def test_token_index_ranks_denser_matches_higher():
    from infrastructure.mimic_hosp.concepts import _TokenIndex

    rows = [
        ("A", "Type 2 diabetes mellitus without complications"),
        ("B", "Diabetes mellitus without mention of complication, type II"),
        ("C", "Personal history of malignant neoplasm"),
    ]
    index = _TokenIndex(rows, text_column=1)
    results = index.search("type 2 diabetes", limit=5)
    ids = [row[0] for row, _score in results]
    assert "C" not in ids
    assert set(ids) == {"A", "B"}

class ExpandCOPD:
    def __init__(self):
        self.calls = []

    def expand(self, term, *, domain, on_progress=None):
        self.calls.append((term, domain))
        if on_progress is not None:
            on_progress("stage", "fake terminology expansion")
        return ("chronic obstructive pulmonary disease",)


def test_zero_candidate_terminology_expansion_recovers_dictionary_candidates(tmp_path):
    path = tmp_path / "mimic_hosp_copd.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE d_icd_diagnoses (icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR)")
        con.execute("CREATE TABLE prescriptions (subject_id BIGINT, hadm_id BIGINT, drug VARCHAR, starttime TIMESTAMP, stoptime TIMESTAMP)")
        con.execute("INSERT INTO d_icd_diagnoses VALUES ('J449', 10, 'Chronic obstructive pulmonary disease, unspecified')")
    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    expander = ExpandCOPD()
    r = MimicHospConceptResolver(factory, terminology_expander=expander)

    progress = []
    with pytest.raises(ResolutionRequired) as excinfo:
        r.resolve_condition("COPD", on_progress=lambda event, message: progress.append((event, message)))
    assert expander.calls == [("COPD", "condition")]
    assert any("Trying medical terminology expansion" in message for _event, message in progress)
    assert any("Recovered" in message for _event, message in progress)
    option = excinfo.value.request.options[0]
    assert option.option_id == "J449"
    assert "J449" in option.label


def test_terminology_expansion_is_not_called_when_lexical_candidates_exist(tmp_path):
    expander = ExpandCOPD()
    factory = DuckDBConnectionFactory(DuckDBConfig(make_clinical(tmp_path), read_only=True))
    r = MimicHospConceptResolver(factory, terminology_expander=expander)
    with pytest.raises(ResolutionRequired):
        r.resolve_condition("diabetes")
    assert expander.calls == []


def _medication_identifier_fixture(tmp_path):
    path = tmp_path / "meds.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE d_icd_diagnoses (icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR)")
        con.execute("""CREATE TABLE prescriptions (
            subject_id BIGINT, hadm_id BIGINT, drug VARCHAR, starttime TIMESTAMP, stoptime TIMESTAMP,
            ndc VARCHAR, gsn VARCHAR, formulary_drug_cd VARCHAR
        )""")
        con.executemany(
            "INSERT INTO prescriptions VALUES (?, ?, ?, NULL, NULL, ?, ?, ?)",
            [
                (1, 1, "Furosemide", "0001", "G1", "F1"),
                (2, 2, "furosemide", "0002", "G2", "F2"),
                (3, 3, "Furosemide Solution", "0003", "G3", "F3"),
            ],
        )
    return DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))


def test_medication_exact_match_is_case_deduplicated_and_auto_resolves(tmp_path):
    r = MimicHospConceptResolver(_medication_identifier_fixture(tmp_path))

    concept = r.resolve_medication("FUROSEMIDE")

    assert concept.medication_names == ("Furosemide",)
    assert concept.concept_name.casefold() == "furosemide"


def test_medication_fuzzy_candidates_show_best_identifier_and_unique_option_ids(tmp_path):
    r = MimicHospConceptResolver(_medication_identifier_fixture(tmp_path))

    # No exact match: both the base drug and solution share the token
    # "furosemide", so the normal manual-disambiguation path is exercised.
    with pytest.raises(ResolutionRequired) as excinfo:
        r.resolve_medication("furosemide medication")

    options = excinfo.value.request.options
    labels = [option.label for option in options]
    assert sum(label.casefold().startswith("furosemide (") for label in labels) == 1
    assert any("furosemide solution" in label.casefold() for label in labels)
    assert any("NDC:" in label for label in labels)
    assert len({option.option_id for option in options}) == len(options)
