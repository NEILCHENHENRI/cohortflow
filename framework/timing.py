from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class TimingSpan:
    name: str
    category: str
    duration_ms: float
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "duration_ms": round(self.duration_ms, 3),
            "parent": self.parent,
            "metadata": dict(self.metadata),
        }


class TimingRecorder:
    """Thread-safe collector for diagnostic wall-clock spans.

    Spans may be nested. Callers should not add nested durations together when
    estimating wall time. The report intentionally preserves parent/child
    relationships so the UI can show both coarse stages and detailed work.
    """

    def __init__(self) -> None:
        self._spans: list[TimingSpan] = []
        self._lock = threading.Lock()
        self._started = time.perf_counter()

    def add(
        self,
        name: str,
        duration_ms: float,
        *,
        category: str = "other",
        parent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        span = TimingSpan(
            name=name,
            category=category,
            duration_ms=max(0.0, float(duration_ms)),
            parent=parent,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._spans.append(span)

    def report(self) -> dict[str, Any]:
        with self._lock:
            spans = list(self._spans)
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        by_name: dict[str, float] = {}
        for span in spans:
            by_name[span.name] = by_name.get(span.name, 0.0) + span.duration_ms
        return {
            "total_wall_ms": round(elapsed_ms, 3),
            "totals_by_name_ms": {
                key: round(value, 3)
                for key, value in sorted(by_name.items())
            },
            "spans": [span.to_dict() for span in spans],
        }


_active_recorder: ContextVar[TimingRecorder | None] = ContextVar(
    "active_timing_recorder", default=None
)
_active_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "active_timing_stack", default=()
)


@contextmanager
def timing_session() -> Iterator[TimingRecorder]:
    recorder = TimingRecorder()
    recorder_token = _active_recorder.set(recorder)
    stack_token = _active_stack.set(())
    try:
        yield recorder
    finally:
        _active_stack.reset(stack_token)
        _active_recorder.reset(recorder_token)


@contextmanager
def measure(
    name: str,
    *,
    category: str = "other",
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    recorder = _active_recorder.get()
    if recorder is None:
        yield
        return

    stack = _active_stack.get()
    parent = stack[-1] if stack else None
    token = _active_stack.set(stack + (name,))
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder.add(
            name,
            (time.perf_counter() - started) * 1000.0,
            category=category,
            parent=parent,
            metadata=metadata,
        )
        _active_stack.reset(token)
