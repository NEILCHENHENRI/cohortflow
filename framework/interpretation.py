from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InteractionAct(str, Enum):
    ANSWER_PENDING = "answer_pending"
    MODIFY_TASK = "modify_task"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    CHANGE_LOGIC = "change_logic"
    OVERRIDE_ENTITY = "override_entity"
    REQUEST_INFORMATION = "request_information"
    INSPECT_RESULT = "inspect_result"
    CONTROL_EXECUTION = "control_execution"


class GroundingIssueCode(str, Enum):
    UNKNOWN_REFERENCE = "unknown_reference"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    AMBIGUOUS_OPTION_REFERENCE = "ambiguous_option_reference"
    UNSUPPORTED_CHANGE = "unsupported_change"
    INVALID_LOGIC = "invalid_logic"


@dataclass(frozen=True, slots=True)
class SemanticReference:
    text: str
    reference_type: str
    reference_id: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class GroundingIssue:
    code: GroundingIssueCode
    message: str
    candidates: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ("ask_human",)


@dataclass(frozen=True, slots=True)
class GroundingResult:
    valid: bool
    references: tuple[SemanticReference, ...] = ()
    issues: tuple[GroundingIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingReplyInterpretation:
    action: str
    confidence: float
    rationale: str
    option_id: str | None = None
    option_ids: tuple[str, ...] = ()
    residual_message: str | None = None
    question: str | None = None
    acts: tuple[InteractionAct, ...] = ()
    references: tuple[SemanticReference, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.option_ids and self.option_id is not None:
            object.__setattr__(self, "option_ids", (self.option_id,))
        elif self.option_ids and self.option_id is None:
            object.__setattr__(self, "option_id", self.option_ids[0])

    @property
    def has_residual_instruction(self) -> bool:
        return bool(self.residual_message and self.residual_message.strip())
