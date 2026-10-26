import pytest

from framework.resolution import (
    ResolutionAction,
    ResolutionDecision,
    ResolutionOption,
    ResolutionRequest,
    ResolutionRequired,
)


def test_resolution_decision_requires_option_id():
    with pytest.raises(ValueError):
        ResolutionDecision(
            action=ResolutionAction.SELECT_OPTION,
            confidence=0.9,
            rationale="x",
        )


def test_resolution_required_preserves_request():
    request = ResolutionRequest(
        resolver_id="test",
        error_code="ambiguous",
        message="ambiguous",
        recoverable=True,
        options=(ResolutionOption("1", "A"),),
    )
    error = ResolutionRequired(
        request,
        question="Which one?",
        rationale="Low confidence",
    )
    assert error.question == "Which one?"
    assert error.request is request
