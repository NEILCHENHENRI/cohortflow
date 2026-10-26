from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any


class OperatorRole(str, Enum):
    RETRIEVE = "retrieve"
    TRANSFORM = "transform"
    INFER = "infer"
    ACT = "act"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"


class Interruptibility(str, Enum):
    ATOMIC = "atomic"
    CANCELLABLE = "cancellable"
    CHECKPOINTABLE = "checkpointable"
    STREAMING = "streaming"


class SideEffectLevel(str, Enum):
    NONE = "none"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


@dataclass(slots=True)
class OperatorContext:
    execution_version: int
    stop_event: threading.Event
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_stop_event: threading.Event | None = None
    boundary_stop_event: threading.Event | None = None
    parallel_checkpoint_pending: threading.Event | None = None
    parallel_checkpoint_done: threading.Event | None = None
    component_timings_ms: dict[str, float] = field(default_factory=dict)
    progress_callback: Callable[[str, str], None] | None = None

    @contextmanager
    def measure(self, name: str):
        """Measure a subcomponent inside this operator invocation."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.component_timings_ms[name] = (
                self.component_timings_ms.get(name, 0.0) + elapsed
            )

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def emit_progress(self, event_type: str, payload: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event_type, payload)

    def raise_if_cancelled(self) -> None:
        if self.stop_requested:
            raise InterruptedError(
                f"execution cancelled while running node {self.node_id}"
            )

    def raise_if_checkpoint_cancelled(self) -> None:
        if self.checkpoint_stop_event is not None and self.checkpoint_stop_event.is_set():
            raise InterruptedError(
                f"checkpointable execution cancelled while running node {self.node_id}"
            )

    def wait_for_parallel_checkpoint(self) -> None:
        if (
            self.parallel_checkpoint_pending is not None
            and self.parallel_checkpoint_pending.is_set()
            and self.parallel_checkpoint_done is not None
        ):
            self.parallel_checkpoint_done.wait()

    def raise_if_boundary_stop_requested(self) -> None:
        self.wait_for_parallel_checkpoint()
        if self.boundary_stop_event is not None and self.boundary_stop_event.is_set():
            raise InterruptedError(
                f"execution yielded at a safe boundary in node {self.node_id}"
            )


_active_operator_context: ContextVar["OperatorContext | None"] = ContextVar(
    "active_operator_context", default=None
)


@contextmanager
def operator_context_scope(context: "OperatorContext"):
    """Expose the currently running operator to lower-level services.

    Infrastructure code such as semantic disambiguators can use
    ``measure_current_operator_component`` to contribute fine-grained timings
    to the node that invoked them without threading an OperatorContext through
    every service API.
    """
    token = _active_operator_context.set(context)
    try:
        yield
    finally:
        _active_operator_context.reset(token)


def emit_current_operator_progress(event_type: str, payload: str) -> None:
    """Emit a live event from infrastructure invoked inside an operator."""
    context = _active_operator_context.get()
    if context is not None:
        context.emit_progress(event_type, payload)


def raise_if_current_operator_checkpoint_cancelled() -> None:
    """Yield a checkpointable infrastructure loop between completed batches."""
    context = _active_operator_context.get()
    if context is not None:
        context.raise_if_checkpoint_cancelled()


def raise_if_current_operator_boundary_stop_requested() -> None:
    """Yield after a resumable deterministic stage before starting new work."""
    context = _active_operator_context.get()
    if context is not None:
        context.raise_if_boundary_stop_requested()


@contextmanager
def measure_current_operator_component(name: str):
    """Measure work against the active operator, when one exists.

    Outside graph execution this is intentionally a no-op so the same
    infrastructure services remain usable in tests and standalone tools.
    """
    context = _active_operator_context.get()
    if context is None:
        yield
        return
    with context.measure(name):
        yield


OperatorFunction = Callable[
    [dict[str, Any], dict[str, Any], OperatorContext],
    dict[str, Any],
]


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    role: OperatorRole
    description: str
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    interruptibility: Interruptibility = Interruptibility.ATOMIC
    side_effects: SideEffectLevel = SideEffectLevel.NONE
    cacheable: bool = True
    retryable: bool = True
    estimated_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("operator name cannot be empty")
        if not self.description.strip():
            raise ValueError("operator description cannot be empty")


@dataclass(slots=True)
class RegisteredOperator:
    spec: OperatorSpec
    function: OperatorFunction


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, RegisteredOperator] = {}

    def register(
        self,
        spec: OperatorSpec,
        function: OperatorFunction,
    ) -> None:
        if spec.name in self._operators:
            raise ValueError(f"operator already registered: {spec.name}")
        self._operators[spec.name] = RegisteredOperator(spec, function)

    def get(self, name: str) -> RegisteredOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

    def specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(
            item.spec
            for _, item in sorted(self._operators.items())
        )

    def describe_for_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "role": spec.role.value,
                "description": spec.description,
                "input_types": list(spec.input_types),
                "output_types": list(spec.output_types),
                "interruptibility": spec.interruptibility.value,
                "side_effects": spec.side_effects.value,
                "cacheable": spec.cacheable,
                "retryable": spec.retryable,
            }
            for spec in self.specs()
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._operators

    def __iter__(self) -> Iterable[RegisteredOperator]:
        return iter(self._operators.values())
