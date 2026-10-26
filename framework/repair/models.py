from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Protocol, Callable

from framework.failure import FailureEvent


class RepairMode(str, Enum):
    OFF = "off"
    LOCAL = "local"
    LOCAL_THEN_CLOUD = "local_then_cloud"


class FailureAction(str, Enum):
    RETRY = "retry"
    ADAPT_RUNTIME = "adapt_runtime"
    ASK_USER = "ask_user"
    LOCAL_REPAIR = "local_repair"
    CLOUD_ESCALATE = "cloud_escalate"
    STOP = "stop"


class RepairScope(str, Enum):
    NONE = "none"
    RUNTIME = "runtime"
    CONFIG = "config"
    SOURCE = "source"
    UNKNOWN = "unknown"


class RepairStatus(str, Enum):
    DISABLED = "disabled"
    ANALYZED = "analyzed"
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    ERROR = "error"
    RECOVERED = "recovered"
    NEEDS_INPUT = "needs_input"
    UNSUPPORTED = "unsupported"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    mode: RepairMode = RepairMode.OFF
    cloud_provider: str | None = None
    cloud_model: str | None = None
    max_changed_files: int = 3
    max_changed_lines: int = 180
    run_full_suite: bool = True
    test_timeout_seconds: float = 180.0
    max_runtime_retries: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RepairMode(self.mode))
        if type(self.max_runtime_retries) is not int or not 0 <= self.max_runtime_retries <= 5:
            raise ValueError("max_runtime_retries must be an integer between 0 and 5")
        if self.max_changed_files < 1 or self.max_changed_lines < 1:
            raise ValueError("repair patch budgets must be positive")
        if not math.isfinite(self.test_timeout_seconds) or self.test_timeout_seconds <= 0:
            raise ValueError("repair test timeout must be positive and finite")


@dataclass(frozen=True, slots=True)
class FailureAssessment:
    action: FailureAction
    scope: RepairScope
    confidence: float
    diagnosis: str
    rationale: str
    likely_files: tuple[str, ...] = ()
    likely_symbols: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RepairContext:
    failure: FailureEvent
    repository_root: str
    source_context: tuple[tuple[str, str], ...] = ()
    test_context: tuple[tuple[str, str], ...] = ()
    prior_attempt: str | None = None
    task_context: dict[str, Any] = field(default_factory=dict)
    recovery_hooks: RecoveryHooks | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RepairProposal:
    backend: str
    diagnosis: str
    confidence: float
    summary: str
    unified_diff: str = ""
    tests_to_run: tuple[str, ...] = ()
    requires_human_review: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationStep:
    name: str
    passed: bool
    command: tuple[str, ...] = ()
    return_code: int | None = None
    output: str = ""
    test_results: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: bool
    changed_files: tuple[str, ...] = ()
    changed_lines: int = 0
    steps: tuple[ValidationStep, ...] = ()
    rejection_reason: str | None = None
    baseline_steps: tuple[ValidationStep, ...] = ()
    reproduction: RecoveryCheck | None = None
    replay: RecoveryCheck | None = None


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    status: RepairStatus
    failure: FailureEvent
    assessment: FailureAssessment | None = None
    proposal: RepairProposal | None = None
    validation: ValidationReport | None = None
    workspace_path: str | None = None
    backend: str | None = None
    escalated: bool = False
    persistent_branch: str | None = None
    persistent_commit: str | None = None
    message: str = ""
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    retry_attempts: int = 0
    state_token: str | None = None
    recovery_value: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RecoveryCheck:
    """Evidence supplied by trusted host code, never a model's confidence."""
    succeeded: bool
    task_preserved: bool
    message: str = ""
    reproduced_failure: bool = False
    value: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if any(type(x) is not bool for x in (self.succeeded, self.task_preserved, self.reproduced_failure)):
            raise ValueError("recovery check flags must be booleans")
        if self.succeeded and self.reproduced_failure:
            raise ValueError("a successful operation cannot reproduce the failure")


@dataclass(frozen=True, slots=True)
class RecoveryHooks:
    """Host-owned operation contract. Models cannot authorize retry or replay.

    Retry must check the output contract and avoid external side effects. Replay
    must execute the supplied workspace code in isolation, with bounded resources
    and detached inputs. Phases are 'baseline' and 'candidate'. No live-code replay
    is provided by default. snapshot() identifies the unchanged task/operation.
    """
    operation_key: str
    snapshot: Callable[[], str]
    retry: Callable[[], RecoveryCheck] | None = None
    safe_to_retry: bool = False
    replay: Callable[[str, str], RecoveryCheck] | None = None
    assessment: FailureAssessment | None = None
    task_context: dict[str, Any] = field(default_factory=dict)


class RepairBackend(Protocol):
    name: str
    def analyze(self, context: RepairContext) -> FailureAssessment: ...
    def propose(
        self,
        context: RepairContext,
        assessment: FailureAssessment,
    ) -> RepairProposal: ...
