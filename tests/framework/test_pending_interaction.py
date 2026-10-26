from framework.pending_interaction import (
    ContinuationKind,
    PendingInteraction,
    PendingInteractionKind,
)
from framework.resolution import (
    PendingResolution,
    ResolutionRequest,
)


def test_semantic_pending_interaction_keeps_continuation():
    resolution = PendingResolution(
        resolution_id="r1",
        request=ResolutionRequest(
            resolver_id="x",
            error_code="ambiguous",
            message="ambiguous",
            recoverable=True,
        ),
        question="Which one?",
        graph_version=1,
    )
    pending = PendingInteraction.semantic_resolution(
        resolution,
        continuation=ContinuationKind.INITIAL_EXECUTION,
    )
    assert pending.kind is PendingInteractionKind.SEMANTIC_RESOLUTION
    assert pending.continuation is ContinuationKind.INITIAL_EXECUTION
    assert pending.resolution is resolution
