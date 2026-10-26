from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from framework.interpretation import PendingReplyInterpretation
from uuid import uuid4


class ResolutionAction(str, Enum):
    SELECT_OPTION = "select_option"
    REPLACE_INPUT = "replace_input"
    RETRY = "retry"
    ASK_HUMAN = "ask_human"
    ABORT = "abort"


class ResponseMode(str, Enum):
    SELECT_ONE = "select_one"
    SELECT_MANY = "select_many"
    FREE_TEXT = "free_text"
    CONFIRM = "confirm"


@dataclass(frozen=True, slots=True)
class ResolutionOption:
    option_id: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolutionRequest:
    resolver_id: str
    error_code: str
    message: str
    recoverable: bool
    input_payload: dict[str, Any] = field(default_factory=dict)
    options: tuple[ResolutionOption, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    response_mode: ResponseMode = ResponseMode.SELECT_ONE


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    action: ResolutionAction
    confidence: float
    rationale: str
    option_id: str | None = None
    option_ids: tuple[str, ...] = ()
    replacement: dict[str, Any] | None = None
    question: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.action is ResolutionAction.SELECT_OPTION:
            if self.option_id is None and not self.option_ids:
                raise ValueError("select_option requires option_id or option_ids")
            if not self.option_ids:
                object.__setattr__(self, "option_ids", (self.option_id,))
            elif self.option_id is None:
                object.__setattr__(self, "option_id", self.option_ids[0])
        if self.action is ResolutionAction.ASK_HUMAN and not self.question:
            raise ValueError("ask_human requires question")

    @property
    def selected_option_ids(self) -> tuple[str, ...]:
        """The full set of selected option ids, whether one or many.

        Prefer this over the singular ``option_id`` when consuming a
        decision, since a SELECT_MANY response mode may select more than
        one option even though ``option_id`` (kept for older single-select
        callers) only ever reflects the first.
        """
        return self.option_ids


@dataclass(slots=True)
class PendingResolution:
    resolution_id: str
    request: ResolutionRequest
    question: str
    graph_version: int
    node_id: str | None = None
    selected_option_ids: tuple[str, ...] = ()
    human_response: str | None = None

    @property
    def options(self) -> tuple[ResolutionOption, ...]:
        return self.request.options

    @property
    def context(self) -> dict[str, Any]:
        return {
            **self.request.context,
            "input_payload": self.request.input_payload,
            "resolver_id": self.request.resolver_id,
            "error_code": self.request.error_code,
        }

    def resolve(self, option_ids: str | tuple[str, ...], human_response: str) -> None:
        ids = (option_ids,) if isinstance(option_ids, str) else tuple(option_ids)
        if not ids:
            raise ValueError("at least one option_id is required")
        valid = {option.option_id for option in self.options}
        unknown = [option_id for option_id in ids if option_id not in valid]
        if unknown:
            raise ValueError(f"unknown resolution option(s): {unknown}")
        self.selected_option_ids = ids
        self.human_response = human_response


class ResolutionRequired(RuntimeError):
    """Normal control-flow signal: execution needs semantic clarification."""

    def __init__(
        self,
        request: ResolutionRequest,
        *,
        question: str,
        rationale: str = "",
        node_id: str | None = None,
    ) -> None:
        super().__init__(question)
        self.request = request
        self.question = question
        self.rationale = rationale
        self.node_id = node_id

    def pending(self, graph_version: int) -> PendingResolution:
        return PendingResolution(
            resolution_id=uuid4().hex,
            request=self.request,
            question=self.question,
            graph_version=graph_version,
            node_id=self.node_id,
        )


@runtime_checkable
class SemanticResolver(Protocol):
    def resolve(self, request: ResolutionRequest) -> ResolutionDecision: ...

    def resolve_human_answer(
        self,
        pending: PendingResolution,
        human_response: str,
    ) -> ResolutionDecision: ...

    def interpret_human_reply(
        self,
        pending: PendingResolution,
        human_response: str,
    ) -> PendingReplyInterpretation: ...


@dataclass(frozen=True, slots=True)
class ResolutionKey:
    resolver_id: str
    input_fingerprint: str
    scope_id: str = "session"


@dataclass(frozen=True, slots=True)
class ResolutionRecord:
    key: ResolutionKey
    option_id: str
    option_label: str
    option_payload: dict[str, Any]
    source: str
    rationale: str = ""


class ResolutionStore(ABC):
    @abstractmethod
    def get(self, key: ResolutionKey) -> ResolutionRecord | None: ...

    @abstractmethod
    def put(self, record: ResolutionRecord) -> None: ...


class InMemoryResolutionStore(ResolutionStore):
    def __init__(self) -> None:
        self._records: dict[ResolutionKey, ResolutionRecord] = {}

    def get(self, key: ResolutionKey) -> ResolutionRecord | None:
        return self._records.get(key)

    def put(self, record: ResolutionRecord) -> None:
        self._records[record.key] = record

    def snapshot(self) -> tuple[ResolutionRecord, ...]:
        return tuple(self._records.values())
