from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from framework.operator import Interruptibility
from framework.resolution import ResolutionRequired
from framework.steering import SteeringYield, steering_checkpoint
from framework.timing import measure

T = TypeVar("T")


class AgentModuleRole(str, Enum):
    """Lifecycle role for work that sits above individual graph operators."""

    INTERPRET = "interpret"
    COMPILE = "compile"
    VALIDATE = "validate"
    DIFF = "diff"
    PLAN = "plan"
    BUILD = "build"
    RECONCILE = "reconcile"
    PREPARE = "prepare"
    EXECUTE = "execute"
    DELIVER = "deliver"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class AgentModuleSpec:
    """Generic steerable unit in an agent lifecycle.

    Graph operators already model execution-node modularity. AgentModuleSpec
    applies the same idea to interpretation/planning/reconciliation work that
    happens before and around the graph. A module is atomic by default: once it
    starts it is allowed to finish, then steering is checked before the next
    module begins.
    """

    name: str
    role: AgentModuleRole
    description: str
    category: str
    interruptibility: Interruptibility = Interruptibility.ATOMIC
    checkpoint_after: bool = True
    cacheable: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("agent module name cannot be empty")
        if not self.description.strip():
            raise ValueError("agent module description cannot be empty")
        if not self.category.strip():
            raise ValueError("agent module category cannot be empty")


ProgressCallback = Callable[[str, str], None]


class AgentModuleRunner:
    """Runs lifecycle modules with uniform timing, progress and steering.

    This runner intentionally does not know anything about cohorts. Any domain
    agent can register module specs and inherit the same mid-run semantics.
    """

    def run(
        self,
        spec: AgentModuleSpec,
        function: Callable[[], T],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> T:
        if on_progress is not None:
            on_progress("module_started", spec.name)
        try:
            with measure(spec.name, category=spec.category):
                result = function()
        except (ResolutionRequired, SteeringYield) as exc:
            # These are expected control-flow interruptions, not failures.
            # ResolutionRequired pauses for human clarification. SteeringYield
            # hands control to a queued correction at a safe module boundary.
            if on_progress is not None:
                on_progress(
                    "module_interrupted",
                    f"{spec.name}|{type(exc).__name__}",
                )
            raise
        except Exception:
            if on_progress is not None:
                on_progress("module_failed", spec.name)
            raise
        if on_progress is not None:
            on_progress("module_finished", spec.name)
        if spec.checkpoint_after:
            steering_checkpoint(on_progress, spec.name)
        return result
