from framework.correction import (
    ArtifactMutation,
    CorrectionAnalysis,
    CorrectionIntent,
    CorrectionPlanner,
    CorrectionScope,
    CorrectionTarget,
    MutationKind,
)
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind
from framework.task import Objective, TaskDelta, TaskSpec


def graph(config_value):
    result = ExecutionGraph()
    result.add_node(ExecutionNode(
        node_id="input",
        kind=NodeKind.INPUT,
        outputs=("input_artifact",),
        config={"value": config_value},
    ))
    result.add_node(
        ExecutionNode(
            node_id="compute",
            kind=NodeKind.COMPUTATION,
            operator_name="compute",
            inputs=("input_artifact",),
            outputs=("result",),
            config={"value": config_value},
        ),
        depends_on=("input",),
    )
    return result


def test_planner_localizes_changed_node_and_descendants():
    task = TaskSpec("x", Objective("x", "x"))
    intent = CorrectionIntent(
        user_text="change it",
        summary="Change value",
        desired_effect="Update output",
        scope=CorrectionScope.GLOBAL,
        mutations=(
            ArtifactMutation(
                MutationKind.UPDATE_VALUE,
                CorrectionTarget("task"),
                "Change task",
            ),
        ),
        confidence=0.99,
    )
    analysis = CorrectionAnalysis(intent, TaskDelta(
        metadata_updates={"value": 2},
    ))

    plan = CorrectionPlanner().plan(
        current_task=task,
        current_graph=graph(1),
        analysis=analysis,
        build_graph=lambda proposed: graph(
            proposed.metadata.get("value", 1)
        ),
    )
    assert plan.changed_nodes == ("compute", "input")
    assert set(plan.affected_nodes) == {"input", "compute"}
    assert plan.recompute_boundary == ("input",)
