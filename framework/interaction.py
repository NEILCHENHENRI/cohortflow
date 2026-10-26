from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from framework.agent import ProgressCallback
from framework.manual_edit import is_fallback_command
from framework.reconcile import ReuseReport
from framework.resolution import (
    PendingResolution,
    ResolutionAction,
    ResolutionOption,
    ResolutionRequired,
    SemanticResolver,
)
from framework.session import SteerableSession
from framework.task import TaskDelta


class MessageRoute(str, Enum):
    INITIAL_TASK = "initial_task"
    ANSWER_PENDING = "answer_pending"
    MODIFY_TASK = "modify_task"
    INSPECT_RESULT = "inspect_result"
    CONTROL_EXECUTION = "control_execution"
    ENTER_MANUAL_FALLBACK = "enter_manual_fallback"


class MessageRouter:
    def route(self, message: str, session: SteerableSession) -> MessageRoute:
        cleaned = message.strip().casefold()
        if session.state is None:
            if session.pending_initial_interaction is not None:
                return MessageRoute.ANSWER_PENDING
            return MessageRoute.INITIAL_TASK
        if is_fallback_command(message):
            return MessageRoute.ENTER_MANUAL_FALLBACK
        if session.state.pending_interaction is not None:
            return MessageRoute.ANSWER_PENDING
        if cleaned in {"pause", "stop", "cancel", "resume"}:
            return MessageRoute.CONTROL_EXECUTION
        if cleaned.startswith(("why ", "show ", "inspect ", "explain ")):
            return MessageRoute.INSPECT_RESULT
        return MessageRoute.MODIFY_TASK


@runtime_checkable
class InteractionAdapter(Protocol):
    def format_initial(self, summary: dict) -> str: ...
    def format_update(
        self,
        summary: dict,
        old_summary: dict | None,
        report: ReuseReport,
        rationale: str,
    ) -> str: ...
    def format_pending(self, pending: PendingResolution) -> str: ...
    def format_resolution(
        self,
        options: tuple[ResolutionOption, ...],
        summary: dict,
    ) -> str: ...
