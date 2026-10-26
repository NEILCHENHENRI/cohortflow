from framework.failure import FailureEvent


class StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = "req-123"


def test_failure_event_is_raw_and_preserves_runtime_context():
    try:
        raise StatusError("opaque backend failure", 503)
    except StatusError as exc:
        event = FailureEvent.capture(
            exc,
            origin="operator",
            node_id="query:p1",
            operator_name="evaluate_requirement",
            metadata={"requirement_id": "p1"},
        )
    assert event.exception_type == "StatusError"
    assert event.message == "opaque backend failure"
    assert event.status_code == 503
    assert event.request_id == "req-123"
    assert event.node_id == "query:p1"
    assert event.operator_name == "evaluate_requirement"
    assert event.metadata["requirement_id"] == "p1"
    assert "StatusError" in event.traceback
