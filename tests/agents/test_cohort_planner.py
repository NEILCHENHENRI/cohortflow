from agents.cohort.planner import CohortPlanner
from framework.logic import LogicExpression
from framework.task import Objective, Requirement, TaskSpec


def make_task(requirements):
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective(
            "cohort_retrieval",
            "Find matching patients",
        ),
        requirements=tuple(requirements),
        logic=_logic(requirements),
    )



def _logic(requirements):
    leaves = []
    for r in requirements:
        leaf = LogicExpression.predicate(r.requirement_id, r.payload.get("term", r.requirement_id))
        if r.polarity == "exclude":
            leaf = LogicExpression.not_(leaf)
        leaves.append(leaf)
    return leaves[0] if len(leaves) == 1 else LogicExpression.and_(*leaves)

def test_planner_builds_stable_requirement_branches():
    planner = CohortPlanner()
    diabetes = Requirement(
        "condition_diabetes",
        "condition",
        {"term": "Type 2 diabetes mellitus"},
    )
    ckd = Requirement(
        "condition_ckd",
        "condition",
        {"term": "Chronic kidney disease"},
    )

    old_graph = planner.build_graph(make_task([diabetes]))
    new_graph = planner.build_graph(make_task([diabetes, ckd]))

    shared = set(old_graph.nodes).intersection(new_graph.nodes)
    shared_evaluators = {
        node_id for node_id in shared
        if node_id.startswith("evaluate:requirement:")
    }

    assert len(shared_evaluators) == 1
    assert (
        old_graph.nodes["aggregate:cohort"].semantic_fingerprint
        != new_graph.nodes["aggregate:cohort"].semantic_fingerprint
    )


def test_exclusion_requirement_is_recorded_in_aggregation_config():
    planner = CohortPlanner()
    task = make_task([
        Requirement("include_a", "condition", {"term": "A"}),
        Requirement(
            "exclude_b",
            "condition",
            {"term": "B"},
            polarity="exclude",
        ),
    ])

    graph = planner.build_graph(task)
    config = graph.nodes["aggregate:cohort"].config

    assert config["logic"]["kind"] == "and"
    assert set(config["artifact_by_requirement"]) == {"include_a", "exclude_b"}
    assert config["universe_artifact"] == "all_patient_ids"


def test_note_requirement_waits_for_structured_corequirement_candidates():
    planner = CohortPlanner()
    structured = Requirement("p1", "condition", {"term": "diabetes"})
    note = Requirement("p2", "note", {"claim": "medication non-adherence"})
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find matching patients"),
        requirements=(structured, note),
        logic=LogicExpression.and_(
            LogicExpression.predicate("p1", "diabetes"),
            LogicExpression.predicate("p2", "medication non-adherence"),
        ),
    )

    graph = planner.build_graph(task)
    note_node = next(
        node for node in graph.nodes.values()
        if node.operator_name == "evaluate_requirement"
        and node.config.get("requirement_kind") == "note"
    )
    candidate_node_id = next(
        node_id for node_id in graph.parents(note_node.node_id)
        if node_id.startswith("note:candidates:")
    )
    candidate_node = graph.nodes[candidate_node_id]

    assert candidate_node.config["logic"]["predicate_id"] == "p1"
    assert candidate_node.outputs[0] in note_node.inputs
    structured_node = next(
        node.node_id for node in graph.nodes.values()
        if node.operator_name == "evaluate_requirement"
        and node.config.get("requirement_id") == "p1"
    )
    assert structured_node in graph.parents(candidate_node_id)


def test_or_sibling_does_not_become_note_prerequisite():
    planner = CohortPlanner()
    structured = Requirement("p1", "condition", {"term": "diabetes"})
    note = Requirement("p2", "note", {"claim": "non-adherence"})
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "Find matching patients"),
        requirements=(structured, note),
        logic=LogicExpression.or_(
            LogicExpression.predicate("p1", "diabetes"),
            LogicExpression.predicate("p2", "non-adherence"),
        ),
    )
    graph = planner.build_graph(task)
    assert not any(node_id.startswith("note:candidates:") for node_id in graph.nodes)
