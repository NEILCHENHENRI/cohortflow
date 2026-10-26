from __future__ import annotations

from prompts.resolution import SEMANTIC_RESOLUTION_SYSTEM_PROMPT, HUMAN_RESOLUTION_REPLY_SYSTEM_PROMPT

import json
import re
from typing import Any

from framework.interpretation import (
    InteractionAct,
    PendingReplyInterpretation,
    SemanticReference,
)
from framework.resolution import (
    PendingResolution,
    ResolutionAction,
    ResolutionDecision,
    ResolutionRequest,
    ResponseMode,
)
from framework.operator import (
    emit_current_operator_progress,
    measure_current_operator_component,
)
from framework.timing import measure
from infrastructure.llm.ollama import OllamaClient


RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "confidence", "rationale",
        "candidate_id", "replacement", "question",
    ],
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "select_candidate", "replace_input", "retry",
                "ask_human", "abort",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "candidate_id": {"type": ["string", "null"]},
        "replacement": {"type": ["object", "null"]},
        "question": {"type": ["string", "null"]},
    },
}

HUMAN_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "confidence", "candidate_id", "candidate_ids", "rationale",
        "question", "residual_message", "acts", "references",
    ],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["select_candidate", "ask_human", "abort"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "candidate_id": {"type": ["string", "null"]},
        "candidate_ids": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
        "question": {"type": ["string", "null"]},
        "residual_message": {"type": ["string", "null"]},
        "acts": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "answer_pending", "modify_task", "add_constraint",
                    "remove_constraint", "change_logic", "override_entity",
                    "request_information", "inspect_result",
                    "control_execution",
                ],
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "text", "reference_type",
                    "reference_id", "confidence",
                ],
                "properties": {
                    "text": {"type": "string"},
                    "reference_type": {"type": "string"},
                    "reference_id": {"type": ["string", "null"]},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
    },
}




class LLMSemanticExceptionResolver:
    def __init__(
        self,
        llm: OllamaClient,
        *,
        auto_accept_confidence: float = 0.90,
        human_answer_confidence: float = 0.80,
    ) -> None:
        if not 0 <= auto_accept_confidence <= 1:
            raise ValueError("auto_accept_confidence must be between 0 and 1")
        if not 0 <= human_answer_confidence <= 1:
            raise ValueError("human_answer_confidence must be between 0 and 1")
        self._llm = llm
        self.auto_accept_confidence = auto_accept_confidence
        self.human_answer_confidence = human_answer_confidence

    @staticmethod
    def _operator_progress(event_type: str, payload: str) -> None:
        emit_current_operator_progress(event_type, payload)

    def resolve(self, request: ResolutionRequest) -> ResolutionDecision:
        metadata = {
            "resolver_id": request.resolver_id,
            "error_code": request.error_code,
            "candidate_count": len(request.options),
        }
        emit_current_operator_progress(
            "disambiguation_started",
            f"Resolving {len(request.options)} semantic candidate(s)",
        )
        with measure_current_operator_component("disambiguation_total"):
            with measure(
                "disambiguation.total",
                category="disambiguation",
                metadata=metadata,
            ):
                if not request.recoverable:
                    return ResolutionDecision(
                        ResolutionAction.ABORT,
                        1.0,
                        "The request is marked as non-recoverable.",
                    )

                with measure_current_operator_component("disambiguation_payload"):
                    with measure(
                        "disambiguation.payload_build",
                        category="disambiguation",
                        metadata=metadata,
                    ):
                        payload = {
                            "resolver_id": request.resolver_id,
                            "error_code": request.error_code,
                            "message": request.message,
                            "input": request.input_payload,
                            "context": request.context,
                            "candidates": [
                                {
                                    "candidate_id": option.option_id,
                                    "label": option.label,
                                    "payload": option.payload,
                                }
                                for option in request.options
                            ],
                        }
                        user_prompt = json.dumps(
                            payload,
                            ensure_ascii=False,
                            default=str,
                        )

                llm_metadata = {
                    **metadata,
                    "user_prompt_chars": len(user_prompt),
                }
                emit_current_operator_progress(
                    "disambiguation_llm_started",
                    f"Comparing {len(request.options)} candidate(s) with the semantic resolver",
                )
                with measure_current_operator_component("disambiguation_llm"):
                    with measure(
                        "llm.semantic_disambiguation",
                        category="llm",
                        metadata=llm_metadata,
                    ):
                        raw = self._llm.generate_structured(
                            system_prompt=SEMANTIC_RESOLUTION_SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                            schema=RECOVERY_SCHEMA,
                            on_progress=self._operator_progress,
                            stage_label="Semantic candidate disambiguation",
                        )
                emit_current_operator_progress(
                    "disambiguation_llm_finished",
                    "Semantic comparison finished. Validating the decision",
                )

                with measure_current_operator_component("disambiguation_decision"):
                    with measure(
                        "disambiguation.decision_build",
                        category="disambiguation",
                        metadata=metadata,
                    ):
                        decision = ResolutionDecision(
                            action=_action(raw["action"]),
                            confidence=float(raw["confidence"]),
                            rationale=str(raw["rationale"]),
                            option_id=raw.get("candidate_id"),
                            replacement=raw.get("replacement"),
                            question=raw.get("question"),
                        )

                with measure_current_operator_component("disambiguation_validation"):
                    with measure(
                        "disambiguation.validation",
                        category="disambiguation",
                        metadata=metadata,
                    ):
                        return self._validate_request(request, decision)

    def interpret_human_reply(
        self,
        pending: PendingResolution,
        human_response: str,
        *,
        on_progress=None,
    ) -> PendingReplyInterpretation:
        normalized = _normalize(human_response)
        if not normalized:
            return PendingReplyInterpretation(
                action="ask_human",
                confidence=0.0,
                rationale="The response was empty.",
                question=pending.question,
                acts=(InteractionAct.ANSWER_PENDING,),
            )

        none_phrases = {
            "none", "none of these", "none of them", "neither",
            "no option", "no options", "not listed",
        }
        if normalized in none_phrases:
            term = str(pending.request.input_payload.get("term") or "this criterion")
            return PendingReplyInterpretation(
                action="ask_human",
                confidence=1.0,
                rationale="The user explicitly rejected every proposed candidate.",
                question=(
                    f"None of the current candidates for {term!r} will be selected. "
                    "Please refine what you mean, replace/remove this requirement, "
                    "or type cancel to abandon the pending interaction."
                ),
                acts=(InteractionAct.ANSWER_PENDING,),
            )

        allow_many = pending.request.response_mode is ResponseMode.SELECT_MANY

        superseding = _deterministic_pending_supersession(pending, human_response)
        if superseding is not None:
            return superseding

        deterministic = _deterministic_reply_match(
            pending,
            human_response,
            allow_many=allow_many,
        )
        if deterministic is not None:
            return deterministic

        payload = {
            "question": pending.question,
            "context": pending.context,
            "response_mode": pending.request.response_mode.value,
            "options": [
                {
                    "candidate_id": option.option_id,
                    "label": option.label,
                    "payload": option.payload,
                }
                for option in pending.options
            ],
            "human_response": human_response,
        }
        raw = self._llm.generate_structured(
            system_prompt=HUMAN_RESOLUTION_REPLY_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
            schema=HUMAN_REPLY_SCHEMA,
            on_progress=on_progress,
            stage_label="Pending clarification interpretation",
        )
        candidate_ids = raw.get("candidate_ids") or (
            [raw["candidate_id"]] if raw.get("candidate_id") else []
        )
        reply = PendingReplyInterpretation(
            action=str(raw["action"]),
            confidence=float(raw["confidence"]),
            rationale=str(raw["rationale"]),
            option_id=raw.get("candidate_id"),
            option_ids=tuple(candidate_ids),
            residual_message=raw.get("residual_message"),
            question=raw.get("question"),
            acts=tuple(
                InteractionAct(value) for value in raw.get("acts", [])
            ),
            references=tuple(
                SemanticReference(
                    text=item["text"],
                    reference_type=item["reference_type"],
                    reference_id=item.get("reference_id"),
                    confidence=float(item["confidence"]),
                )
                for item in raw.get("references", [])
            ),
        )
        return self._validate_reply(pending, reply, allow_many=allow_many)

    def resolve_human_answer(
        self,
        pending: PendingResolution,
        human_response: str,
        *,
        on_progress=None,
    ) -> ResolutionDecision:
        reply = self.interpret_human_reply(
            pending, human_response, on_progress=on_progress
        )
        return ResolutionDecision(
            action=_action(reply.action),
            confidence=reply.confidence,
            rationale=reply.rationale,
            option_id=reply.option_id,
            option_ids=reply.option_ids,
            question=reply.question,
        )

    def _validate_request(
        self,
        request: ResolutionRequest,
        decision: ResolutionDecision,
    ) -> ResolutionDecision:
        if decision.action is not ResolutionAction.SELECT_OPTION:
            return decision
        valid = {option.option_id for option in request.options}
        if decision.option_id not in valid:
            return ResolutionDecision(
                ResolutionAction.ASK_HUMAN,
                0.0,
                "The model selected an unavailable option.",
                question=_default_question(request),
            )
        if decision.confidence < self.auto_accept_confidence:
            return ResolutionDecision(
                ResolutionAction.ASK_HUMAN,
                decision.confidence,
                decision.rationale,
                question=decision.question or _default_question(request),
            )
        return decision

    def _validate_reply(
        self,
        pending: PendingResolution,
        reply: PendingReplyInterpretation,
        *,
        allow_many: bool = False,
    ) -> PendingReplyInterpretation:
        if reply.action != "select_candidate":
            return reply
        valid = {option.option_id for option in pending.options}
        selected = reply.option_ids or (
            (reply.option_id,) if reply.option_id else ()
        )
        if (
            not selected
            or any(option_id not in valid for option_id in selected)
            or (not allow_many and len(selected) > 1)
            or reply.confidence < self.human_answer_confidence
        ):
            return PendingReplyInterpretation(
                action="ask_human",
                confidence=reply.confidence,
                rationale="The response could not be mapped safely.",
                residual_message=reply.residual_message,
                question=reply.question or pending.question,
                acts=reply.acts,
                references=reply.references,
            )
        return reply


def _deterministic_pending_supersession(
    pending: PendingResolution,
    human_response: str,
) -> PendingReplyInterpretation | None:
    """Catch high-confidence task corrections that make a pending choice moot."""
    text = _normalize(human_response)
    if not text:
        return None
    term = _normalize(str(pending.request.input_payload.get("term") or ""))
    if not term:
        return None

    escaped = re.escape(term)
    retract_patterns = (
        rf"^(?:actually\s+)?no\s+{escaped}\b",
        rf"^(?:actually\s+)?(?:remove|drop)\s+(?:the\s+)?{escaped}(?:\s+requirement)?\b",
        rf"^(?:actually\s+)?(?:do\s+not|don't)\s+include\s+{escaped}\b",
        rf"^(?:never\s+mind|nevermind)\s+(?:the\s+)?{escaped}\b",
        rf"^(?:actually\s+)?replace\s+{escaped}\s+with\b",
    )
    if not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in retract_patterns):
        return None

    return PendingReplyInterpretation(
        action="abort",
        confidence=1.0,
        rationale=(
            "The reply changes or retracts the task criterion that caused the "
            "pending clarification, so selecting a candidate would apply stale intent."
        ),
        residual_message=human_response.strip(),
        acts=(InteractionAct.MODIFY_TASK, InteractionAct.REMOVE_CONSTRAINT),
    )


def _deterministic_reply_match(
    pending: PendingResolution,
    human_response: str,
    *,
    allow_many: bool = False,
) -> PendingReplyInterpretation | None:
    normalized_full = _normalize(human_response)

    if allow_many:
        many = _deterministic_multi_select(pending, human_response)
        if many is not None:
            return many

    exact = []
    for index, option in enumerate(pending.options, 1):
        if normalized_full in {
            _normalize(option.option_id),
            _normalize(option.label),
            str(index),
            f"option {index}",
            f"the {index}",
        }:
            exact.append(option)
    if len(exact) == 1:
        return PendingReplyInterpretation(
            action="select_candidate",
            confidence=1.0,
            rationale="The response exactly matched one pending option.",
            option_id=exact[0].option_id,
            acts=(InteractionAct.ANSWER_PENDING,),
        )

    # Numeric/"option N" answer followed by an additional correction. This
    # common pattern should not need another LLM call just to preserve the
    # residual instruction. SELECT_MANY replies are handled above before this
    # branch, so "1 and 3" is not misread as a correction there.
    prefixed = re.match(
        r"^\s*(?:option\s+)?(\d+)\s*(?:[,;:.]|\band\b|\bbut\b|\bthen\b)\s*(.+)$",
        human_response,
        flags=re.IGNORECASE,
    )
    if prefixed is not None:
        index = int(prefixed.group(1))
        if 1 <= index <= len(pending.options):
            option = pending.options[index - 1]
            residual = prefixed.group(2).strip()
            normalized_residual = _normalize(residual)
            residual_is_another_option = any(
                normalized_residual in {
                    str(i),
                    f"option {i}",
                    _normalize(candidate.option_id),
                    _normalize(candidate.label),
                }
                for i, candidate in enumerate(pending.options, 1)
            )
            if residual_is_another_option:
                return None
            return PendingReplyInterpretation(
                action="select_candidate",
                confidence=1.0,
                rationale=(
                    "The reply selected one numbered option and included "
                    "an additional instruction."
                ),
                option_id=option.option_id,
                residual_message=residual or None,
                acts=(
                    InteractionAct.ANSWER_PENDING,
                    InteractionAct.MODIFY_TASK,
                ),
            )

    # Full option label followed by another clause.
    compound = []
    for option in pending.options:
        label = _normalize(option.label)
        if normalized_full.startswith(label + " "):
            residual = normalized_full[len(label):].strip(" ,.;:-")
            compound.append((option, residual))
    if len(compound) == 1:
        option, residual = compound[0]
        return PendingReplyInterpretation(
            action="select_candidate",
            confidence=1.0,
            rationale=(
                "The reply selected one option and included "
                "an additional instruction."
            ),
            option_id=option.option_id,
            residual_message=residual or None,
            acts=(
                InteractionAct.ANSWER_PENDING,
                InteractionAct.MODIFY_TASK,
            ),
        )

    response_tokens = set(normalized_full.split())
    partial = []
    if response_tokens:
        for option in pending.options:
            label_tokens = set(_normalize(option.label).split())
            if response_tokens <= label_tokens:
                partial.append(option)
    if len(partial) == 1:
        return PendingReplyInterpretation(
            action="select_candidate",
            confidence=0.95,
            rationale="The response uniquely matched one option by tokens.",
            option_id=partial[0].option_id,
            acts=(InteractionAct.ANSWER_PENDING,),
        )
    if len(partial) > 1:
        labels = ", ".join(option.label for option in partial)
        return PendingReplyInterpretation(
            action="ask_human",
            confidence=0.0,
            rationale="The partial response matched multiple options.",
            question=(
                f"Your reply matches several options: {labels}. "
                "Which exact one did you mean?"
            ),
            acts=(InteractionAct.ANSWER_PENDING,),
        )
    return None


_ALL_PHRASES = {"all", "all of them", "all of the above", "everything", "every option"}


def _deterministic_multi_select(
    pending: PendingResolution,
    human_response: str,
) -> PendingReplyInterpretation | None:
    normalized_full = _normalize(human_response)
    if normalized_full in _ALL_PHRASES:
        return PendingReplyInterpretation(
            action="select_candidate",
            confidence=1.0,
            rationale="The response selected every pending option.",
            option_ids=tuple(option.option_id for option in pending.options),
            acts=(InteractionAct.ANSWER_PENDING,),
        )

    # Only plain, unambiguous "list the ones I mean" phrasing is handled
    # deterministically here. Exclusion phrasing ("everything except 1 and
    # 3") is deliberately NOT pattern-matched: getting negation inverted
    # would silently produce the wrong cohort rather than fail loudly, and
    # unlike a positive list, there's no cheap way to verify a negation
    # match is unambiguous. It falls through to the LLM, which is given
    # explicit instructions to invert the selection (see HUMAN_REPLY_PROMPT).
    matched = _match_option_list(pending, human_response)
    if matched is None:
        return None
    return PendingReplyInterpretation(
        action="select_candidate",
        confidence=1.0,
        rationale="The response listed several pending options explicitly.",
        option_ids=matched,
        acts=(InteractionAct.ANSWER_PENDING,),
    )


def _match_option_list(
    pending: PendingResolution,
    text: str,
) -> tuple[str, ...] | None:
    """Match a comma/'and'/'&'-separated list of option references (by
    number, id, or label) against the pending options. Segmentation happens
    on the raw text first (before _normalize collapses commas to spaces), so
    "1, 2, and 3" and "option 1 and option 3" both split correctly. Returns
    None if there are fewer than two segments, or if any segment fails to
    match exactly one option -- an unmapped segment falls through to the
    LLM rather than guessing at a partial selection.
    """
    segments = [
        segment.strip()
        for segment in re.split(r",|\band\b|&", text.strip())
        if segment.strip()
    ]
    if len(segments) < 2:
        return None

    by_index = {str(i): option for i, option in enumerate(pending.options, 1)}
    matched: list[str] = []
    for raw_segment in segments:
        segment = _normalize(raw_segment)
        segment = re.sub(r"^(options?|the)\s+", "", segment).strip()
        candidate = by_index.get(segment)
        if candidate is None:
            candidate = next(
                (
                    option for option in pending.options
                    if segment in {
                        _normalize(option.option_id),
                        _normalize(option.label),
                    }
                ),
                None,
            )
        if candidate is None:
            return None
        if candidate.option_id not in matched:
            matched.append(candidate.option_id)

    if len(matched) < 2:
        return None
    return tuple(matched)


def _action(value: str) -> ResolutionAction:
    return {
        "select_candidate": ResolutionAction.SELECT_OPTION,
        "replace_input": ResolutionAction.REPLACE_INPUT,
        "retry": ResolutionAction.RETRY,
        "ask_human": ResolutionAction.ASK_HUMAN,
        "abort": ResolutionAction.ABORT,
    }[value]


def _default_question(request: ResolutionRequest) -> str:
    labels = ", ".join(option.label for option in request.options)
    return f"Which option did you mean: {labels}?"


def _normalize(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    )
