from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import traceback as traceback_module
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """Raw, implementation-agnostic record of an unhandled failure.

    Captures transport hints through wrapped exceptions. Recovery uses a small
    deterministic set of transient/error-contract rules, then agent hooks or the
    analyst for other failures. Handled exceptions never reach this record.
    """

    failure_id: str
    timestamp: str
    exception_type: str
    message: str
    traceback: str
    origin: str
    status_code: int | None = None
    request_id: str | None = None
    node_id: str | None = None
    operator_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        exc: BaseException,
        *,
        origin: str,
        node_id: str | None = None,
        operator_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "FailureEvent":
        chain = []
        current = exc
        while current is not None and all(current is not item for item in chain):
            chain.append(current)
            current = current.__cause__
        details = dict(metadata or {})
        if any(isinstance(item, (TimeoutError, ConnectionError)) for item in chain):
            details["transient_transport_failure"] = True
        return cls(
            failure_id=f"fail_{uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            ),
            origin=origin,
            status_code=next((code for item in chain if (code := _status_code(item)) is not None), None),
            request_id=_request_id(exc),
            node_id=node_id,
            operator_name=operator_name,
            metadata=details,
        )


def _status_code(exc: BaseException) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _request_id(exc: BaseException) -> str | None:
    for name in ("request_id", "_request_id"):
        value = getattr(exc, name, None)
        if isinstance(value, str) and value:
            return value
    return None
