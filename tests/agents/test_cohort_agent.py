from agents.cohort.agent import CohortAgent
from agents.cohort.parser import CohortTaskParser
from framework.logic import LogicExpression
from framework.task import (
    Objective,
    Requirement,
    TaskDelta,
    TaskSpec,
)


class UnusedLLM:
    def generate_structured(self, **kwargs):
        raise AssertionError("LLM should not be called in this test")


class Resolver:
    def resolve_condition(self, term):
        return term

    def resolve_medication(self, term):
        return term


class Repository:
    DATA = {
        "Type 2 diabetes mellitus": {1, 2, 3, 4},
        "Chronic kidney disease": {2, 4},
    }

    def patients_with_condition(self, concept, payload=None):
        return set(self.DATA[concept])

    def patients_with_medication(self, concept, payload=None):
        return set()


def make_task(*requirements):
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective(
            "cohort_retrieval",
            "Find matching patients",
        ),
        requirements=tuple(requirements),
        logic=(
            LogicExpression.predicate(requirements[0].requirement_id, requirements[0].payload.get("term", requirements[0].requirement_id))
            if len(requirements) == 1 else
            LogicExpression.and_(*(LogicExpression.predicate(r.requirement_id, r.payload.get("term", r.requirement_id)) for r in requirements))
        ),
    )


def test_agent_selectively_reuses_unchanged_requirement_branch():
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

    agent = CohortAgent(
        parser=CohortTaskParser(UnusedLLM()),
        resolver=Resolver(),
        repository=Repository(),
    )
    session = agent.create_session()
    session.initialize(make_task(diabetes))
    session.execute()

    assert session.summary()["cohort_size"] == 4

    report = session.stage(
        TaskDelta(
            add_requirements=(ckd,),
            logic=LogicExpression.and_(
                LogicExpression.predicate("condition_diabetes", "Type 2 diabetes mellitus"),
                LogicExpression.predicate("condition_ckd", "Chronic kidney disease"),
            ),
            rationale="Add CKD",
        )
    )
    session.execute_staged()

    assert session.summary()["cohort_size"] == 2
    assert len(report.reused_nodes) == 1
    assert any(
        node_id.startswith("evaluate:requirement:")
        for node_id in report.reused_nodes
    )
    assert "aggregate:cohort" in report.invalidated_nodes
    assert "deliver:summary" in report.invalidated_nodes
