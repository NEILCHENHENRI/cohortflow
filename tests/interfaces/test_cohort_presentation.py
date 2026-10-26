from agents.cohort.agent import CohortAgent
from agents.cohort.parser import CohortTaskParser
from agents.cohort.presentation import CohortPresentationAdapter
from framework.logic import LogicExpression
from framework.task import Objective, Requirement, TaskSpec
from infrastructure.mimic_hosp.concepts import MimicHospConcept, MimicHospConceptEntry


class UnusedLLM:
    def generate_structured(self, **kwargs):
        raise AssertionError


class Resolver:
    def resolve_condition(self, term):
        return term

    def resolve_medication(self, term):
        return term


class Repository:
    def patients_with_patient_attribute(self, payload):
        return {1, 2}

    def patients_with_condition(self, concept, payload=None):
        return {1, 2}

    def patients_with_medication(self, concept, payload=None):
        return set()


class PatientProvider:
    def patient_detail(self, patient_id):
        return {
            "demographics": {"Patient ID": patient_id},
            "conditions": [{"concept_id": 100}],
            "medications": [],
            "visits": [],
        }


def test_adapter_exposes_graph_and_patient_entities():
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement(
                "diabetes",
                "condition",
                {"term": "Type 2 diabetes mellitus"},
            ),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes mellitus"),
    )
    agent = CohortAgent(
        CohortTaskParser(UnusedLLM()),
        Resolver(),
        Repository(),
    )
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    adapter = CohortPresentationAdapter(session, PatientProvider())
    graph = adapter.graph_view()
    node = adapter.node_view("aggregate:cohort")

    assert graph.version == 1
    # A single bare requirement (no AND/OR wrapper) connects straight to
    # the final cohort -- there's no separate aggregation node to show when
    # there's nothing to aggregate.
    assert any(item.node_id == "logic:criterion:diabetes" for item in graph.nodes)
    assert not any(item.node_id == "logic:op:root" for item in graph.nodes)
    # aggregate:cohort is still a real node in the underlying execution
    # graph (it always runs), even though it isn't shown as its own visual
    # node when there's only one requirement to combine.
    assert node.entity_collections[0].count == 2

    detail = adapter.entity_view(
        node.entity_collections[0].collection_id,
        "1",
    )
    assert detail.summary_fields["Patient ID"] == 1
    assert detail.sections[0].title == "Conditions"


def test_evidence_is_queryable_before_summary_exists():
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement(
                "diabetes",
                "condition",
                {"term": "Type 2 diabetes mellitus"},
            ),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes mellitus"),
    )
    agent = CohortAgent(
        CohortTaskParser(UnusedLLM()),
        Resolver(),
        Repository(),
    )
    session = agent.create_session()
    session.initialize(task)

    adapter = CohortPresentationAdapter(session, PatientProvider())
    state = adapter.execution_state()
    evidence = adapter.execution_evidence()
    logic = adapter.graph_view()

    assert state.summary is None
    assert evidence.summary_available is False
    assert evidence.final_count is None
    assert evidence.criteria[0].requirement_id == "diabetes"
    assert any(node.node_id == "logic:result" for node in logic.nodes)


def test_logic_view_preserves_nested_boolean_structure():
    from framework.logic import LogicExpression

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "Diabetes"}),
            Requirement("hypertension", "condition", {"term": "Hypertension"}),
            Requirement("adult", "patient", {"attribute": "age", "operator": ">=", "value": 40}),
        ),
        logic=LogicExpression.and_(
            LogicExpression.or_(
                LogicExpression.predicate("diabetes", "patient has diabetes"),
                LogicExpression.predicate("hypertension", "patient has hypertension"),
            ),
            LogicExpression.predicate("adult", "patient is at least 40"),
        ),
    )
    agent = CohortAgent(CohortTaskParser(UnusedLLM()), Resolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    view = CohortPresentationAdapter(session, PatientProvider()).graph_view()
    labels = {node.node_id: node.label for node in view.nodes}
    assert labels["logic:op:root"] == "AND"
    assert labels["logic:op:root.0"] == "OR"
    edge_pairs = {(edge.source, edge.target) for edge in view.edges}
    assert ("logic:criterion:diabetes", "logic:op:root.0") in edge_pairs
    assert ("logic:criterion:hypertension", "logic:op:root.0") in edge_pairs
    assert ("logic:op:root.0", "logic:op:root") in edge_pairs
    assert ("logic:criterion:adult", "logic:op:root") in edge_pairs


def test_evidence_survives_non_numeric_concept_ids():
    # Backends without an integer concept vocabulary (e.g. ICD codes like
    # "E119", or verbatim drug names) must not crash the evidence view.
    class AlphanumericConcept:
        concept_id = "E119"
        concept_name = "Type 2 diabetes mellitus without complications"
        domain_id = "Condition"
        vocabulary_id = "ICD10CM"
        resolution_source = "exact"

    class AlphanumericResolver:
        def resolve_condition(self, term):
            return AlphanumericConcept()

        def resolve_medication(self, term):
            return AlphanumericConcept()

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "Type 2 diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes"),
    )
    agent = CohortAgent(
        CohortTaskParser(UnusedLLM()),
        AlphanumericResolver(),
        Repository(),
    )
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    evidence = CohortPresentationAdapter(session, PatientProvider()).execution_evidence()

    assert evidence.criteria[0].concept_id == "E119"


def test_logic_view_includes_resolved_concept_entries_as_nodes():
    # A criterion resolved to multiple dictionary entries (e.g. an ICD-9 and
    # an ICD-10 code for the same clinical concept, OR-combined by a
    # multi-select resolution) shows one node per entry, each feeding into
    # the natural-language criterion node -- not just squashed into a
    # subtitle string.
    class Entry:
        def __init__(self, code, display_name, vocabulary_id):
            self.code = code
            self.display_name = display_name
            self.vocabulary_id = vocabulary_id

    class MultiEntryConcept:
        concept_id = "25000+E119"
        concept_name = "2 codes"
        domain_id = "Condition"
        vocabulary_id = "ICD9CM+ICD10CM"
        resolution_source = "human"
        entries = (
            Entry("25000", "Diabetes mellitus, type II", "ICD9CM"),
            Entry("E119", "Type 2 diabetes mellitus", "ICD10CM"),
        )

    class MultiEntryResolver:
        def resolve_condition(self, term):
            return MultiEntryConcept()

        def resolve_medication(self, term):
            return MultiEntryConcept()

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "type 2 diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes"),
    )
    agent = CohortAgent(
        CohortTaskParser(UnusedLLM()),
        MultiEntryResolver(),
        Repository(),
    )
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    view = CohortPresentationAdapter(session, PatientProvider()).graph_view()

    concept_nodes = [node for node in view.nodes if node.category == "concept"]
    assert {node.label for node in concept_nodes} == {"25000", "E119"}
    assert {node.subtitle for node in concept_nodes} == {"ICD9CM", "ICD10CM"}

    edge_pairs = {(edge.source, edge.target) for edge in view.edges}
    assert ("logic:concept:diabetes:0", "logic:criterion:diabetes") in edge_pairs
    assert ("logic:concept:diabetes:1", "logic:criterion:diabetes") in edge_pairs


def test_criterion_label_stays_the_keyword_not_the_resolved_fact():
    # The criterion node is the extracted keyword from the request. The
    # resolved concept is a separate "fact" shown via its own concept
    # node(s). The fact's name must never overwrite the criterion's own
    # label once resolution succeeds -- that would collapse two distinct
    # pieces of information (what was asked for vs. what it resolved to)
    # into one, and hide the original keyword.
    class Entry:
        code = "E119"
        display_name = "Type 2 diabetes mellitus without complications"
        vocabulary_id = "ICD10CM"

    class ResolvedConcept:
        concept_id = "E119"
        concept_name = "Type 2 diabetes mellitus without complications"
        domain_id = "Condition"
        vocabulary_id = "ICD10CM"
        resolution_source = "exact"
        entries = (Entry(),)

    class FactResolver:
        def resolve_condition(self, term):
            return ResolvedConcept()

        def resolve_medication(self, term):
            return ResolvedConcept()

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "type 2 diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes"),
    )
    agent = CohortAgent(CohortTaskParser(UnusedLLM()), FactResolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    adapter = CohortPresentationAdapter(session, PatientProvider())
    evidence = adapter.execution_evidence()
    view = adapter.graph_view()

    # The extracted keyword ("type 2 diabetes"), not the resolved fact's
    # official name, is the criterion's own label.
    assert evidence.criteria[0].label == "type 2 diabetes"
    criterion_node = next(
        n for n in view.nodes if n.node_id == "logic:criterion:diabetes"
    )
    assert criterion_node.label == "type 2 diabetes"

    # The resolved fact is still fully present -- just as a separate node,
    # not smuggled into the criterion's own label.
    concept_node = next(n for n in view.nodes if n.category == "concept")
    assert concept_node.label == "E119"
    assert concept_node.tooltip == "Type 2 diabetes mellitus without complications"


def test_final_cohort_node_click_shows_the_full_patient_list():
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has diabetes"),
    )
    agent = CohortAgent(CohortTaskParser(UnusedLLM()), Resolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    adapter = CohortPresentationAdapter(session, PatientProvider())
    detail = adapter.node_view("logic:result")

    section = next(
        s for s in detail.sections if s.section_id == "final_patient_ids"
    )
    assert section.data == [{"patient_id": 1}, {"patient_id": 2}]
    assert "2" in section.title


def test_source_table_nodes_are_deduplicated_across_criteria():
    # Two different conditions both compute patient membership against the
    # same EHR fact table (diagnoses_icd). That should render as one shared
    # source node feeding both resolved facts, not two duplicate boxes. The
    # dictionary table (d_icd_diagnoses) that resolved each *term* is
    # reference detail attached to the fact as a badge, not its own main-flow
    # node. The presentation flow is DB -> fact -> keyword -> Boolean logic.
    class TwoConditionResolver:
        def resolve_condition(self, term):
            if term == "diabetes":
                return MimicHospConcept(
                    entries=(MimicHospConceptEntry("E119", "Type 2 diabetes mellitus", "ICD10CM"),),
                    domain_id="Condition",
                )
            return MimicHospConcept(
                entries=(MimicHospConceptEntry("I10", "Essential hypertension", "ICD10CM"),),
                domain_id="Condition",
            )

        def resolve_medication(self, term):
            raise NotImplementedError

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "diabetes"}),
            Requirement("hypertension", "condition", {"term": "hypertension"}),
        ),
        logic=LogicExpression.and_(
            LogicExpression.predicate("diabetes", "patient has diabetes"),
            LogicExpression.predicate("hypertension", "patient has hypertension"),
        ),
    )
    agent = CohortAgent(CohortTaskParser(UnusedLLM()), TwoConditionResolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    view = CohortPresentationAdapter(session, PatientProvider()).graph_view()

    source_nodes = [n for n in view.nodes if n.category == "source"]
    assert {n.label for n in source_nodes} == {"diagnoses_icd"}
    assert len(source_nodes) == 1  # deduplicated, not one per criterion

    concept_nodes = {n.node_id: n for n in view.nodes if n.category == "concept"}
    for node in concept_nodes.values():
        assert node.badges == ("📖 d_icd_diagnoses",)

    source_edges = {
        (e.source, e.target) for e in view.edges
        if e.source == "logic:source:diagnoses_icd"
    }
    # The one shared source node fans out to both resolved fact nodes.
    assert source_edges == {
        ("logic:source:diagnoses_icd", "logic:concept:diabetes:0"),
        ("logic:source:diagnoses_icd", "logic:concept:hypertension:0"),
    }


def test_clicking_a_concept_node_shows_its_own_code_and_free_text_name():
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "type 2 diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has type 2 diabetes"),
    )

    class NamedResolver:
        def resolve_condition(self, term):
            return MimicHospConcept(
                entries=(
                    MimicHospConceptEntry(
                        "E119", "Type 2 diabetes mellitus without complications", "ICD10CM",
                    ),
                ),
                domain_id="Condition",
            )

        def resolve_medication(self, term):
            raise NotImplementedError

    agent = CohortAgent(CohortTaskParser(UnusedLLM()), NamedResolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    adapter = CohortPresentationAdapter(session, PatientProvider())
    detail = adapter.node_view("logic:concept:diabetes:0")

    assert detail.title == "E119"
    assert detail.description == "Type 2 diabetes mellitus without complications"
    concept_section = next(
        s for s in detail.sections if s.section_id == "concept_detail"
    )
    assert concept_section.data["Code"] == "E119"
    assert concept_section.data["Name"] == "Type 2 diabetes mellitus without complications"
    assert concept_section.data["Vocabulary"] == "ICD10CM"
    assert concept_section.data["Source tables"] == "d_icd_diagnoses, diagnoses_icd"


def test_clicking_a_source_node_shows_which_criteria_use_it():
    class NamedResolver:
        def resolve_condition(self, term):
            return MimicHospConcept(
                entries=(MimicHospConceptEntry("E119", "Type 2 diabetes mellitus", "ICD10CM"),),
                domain_id="Condition",
            )

        def resolve_medication(self, term):
            raise NotImplementedError

    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find patients"),
        requirements=(
            Requirement("diabetes", "condition", {"term": "diabetes"}),
        ),
        logic=LogicExpression.predicate("diabetes", "patient has diabetes"),
    )
    agent = CohortAgent(CohortTaskParser(UnusedLLM()), NamedResolver(), Repository())
    session = agent.create_session()
    session.initialize(task)
    session.execute()

    adapter = CohortPresentationAdapter(session, PatientProvider())
    detail = adapter.node_view("logic:source:diagnoses_icd")

    assert detail.title == "diagnoses_icd"
    referenced = next(s for s in detail.sections if s.section_id == "referenced_by")
    assert referenced.data == "diabetes"


def test_non_term_requirements_use_logic_leaf_text_instead_of_bare_refs():
    from tests.cohort_fixtures import encounter_note_task

    task = encounter_note_task()
    by_id = {r.requirement_id: r for r in task.requirements}
    labels = {
        rid: CohortPresentationAdapter._requirement_display_label(req, task.logic)
        for rid, req in by_id.items()
    }
    assert labels["p1"] == "age at first admission is at least 50 years"
    assert labels["p2"] == "age at first admission is at most 80 years"
    assert labels["p3"] == "at least two emergency admissions"
    assert labels["p12"] == "survived the relevant hospitalization"
    assert all(label not in {"p1", "p2", "p3", "p12"} for label in labels.values())
