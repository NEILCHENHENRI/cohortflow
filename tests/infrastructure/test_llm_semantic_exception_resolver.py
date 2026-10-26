"""Tests for infrastructure.llm.recovery.LLMSemanticExceptionResolver.

Consolidates what used to be three files, each covering one of the
resolver's three public methods in isolation: resolve() (machine-triggered
resolution of a ResolutionRequest), resolve_human_answer() (a human's free
text reply to a pending question), and interpret_human_reply() (the richer
interpretation that also detects residual instructions and ambiguity).
"""

from framework.resolution import (
    PendingResolution,
    ResolutionAction,
    ResolutionOption,
    ResolutionRequest,
    ResponseMode,
)
from infrastructure.llm.recovery import LLMSemanticExceptionResolver
import pytest


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, **kwargs):
        return self.payload


class NeverLLM:
    def generate_structured(self, **kwargs):
        raise AssertionError("deterministic matching should handle this")


def request():
    return ResolutionRequest(
        resolver_id="omop_concept",
        error_code="no_exact_concept_match",
        message="No exact match",
        recoverable=True,
        input_payload={"term": "type 2 diabetes"},
        options=(
            ResolutionOption("201826", "Type 2 diabetes mellitus"),
            ResolutionOption("999", "Type 2 diabetes with ulcer"),
        ),
    )


def pending():
    return PendingResolution(
        resolution_id="r1",
        request=ResolutionRequest(
            resolver_id="test",
            error_code="ambiguous",
            message="ambiguous",
            recoverable=True,
            options=(
                ResolutionOption("101", "Hypertension stage 1"),
                ResolutionOption("102", "Hypertension stage 2"),
            ),
        ),
        question="Which concept?",
        graph_version=3,
    )


# -- resolve(): machine-triggered resolution of a ResolutionRequest --------

def test_high_confidence_valid_candidate_is_selected():
    resolver = LLMSemanticExceptionResolver(FakeLLM({
        "action": "select_candidate",
        "confidence": 0.98,
        "rationale": "General phrase matches general concept.",
        "candidate_id": "201826",
        "replacement": None,
        "question": None,
    }))
    decision = resolver.resolve(request())
    assert decision.action is ResolutionAction.SELECT_OPTION
    assert decision.option_id == "201826"


def test_low_confidence_selection_is_escalated():
    resolver = LLMSemanticExceptionResolver(FakeLLM({
        "action": "select_candidate",
        "confidence": 0.60,
        "rationale": "Some ambiguity remains.",
        "candidate_id": "201826",
        "replacement": None,
        "question": None,
    }), auto_accept_confidence=0.9)
    assert resolver.resolve(request()).action is ResolutionAction.ASK_HUMAN


def test_unknown_candidate_is_rejected():
    resolver = LLMSemanticExceptionResolver(FakeLLM({
        "action": "select_candidate",
        "confidence": 0.99,
        "rationale": "Invented candidate.",
        "candidate_id": "not-provided",
        "replacement": None,
        "question": None,
    }))
    assert resolver.resolve(request()).action is ResolutionAction.ASK_HUMAN


# -- resolve_human_answer(): a human's free-text reply ----------------------

def test_exact_label_is_resolved_without_llm():
    resolver = LLMSemanticExceptionResolver(FakeLLM({"unused": True}))
    decision = resolver.resolve_human_answer(pending(), "Hypertension stage 1")
    assert decision.action is ResolutionAction.SELECT_OPTION
    assert decision.option_id == "101"


def test_option_number_is_resolved_without_llm():
    resolver = LLMSemanticExceptionResolver(FakeLLM({"unused": True}))
    decision = resolver.resolve_human_answer(pending(), "2")
    assert decision.option_id == "102"


def test_ambiguous_llm_answer_stays_pending():
    resolver = LLMSemanticExceptionResolver(FakeLLM({
        "action": "ask_human",
        "confidence": 0.4,
        "candidate_id": None,
        "rationale": "Ambiguous response.",
        "question": "Did you mean stage 1 or stage 2?",
    }))
    decision = resolver.resolve_human_answer(pending(), "the hypertension one")
    assert decision.action is ResolutionAction.ASK_HUMAN


# -- interpret_human_reply(): richer reply interpretation -------------------

def test_compound_pending_reply_preserves_residual_instruction():
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        pending(),
        "Hypertension stage 1 and the relationship should be OR",
    )
    assert reply.option_id == "101"
    assert reply.residual_message == "and the relationship should be or"


def test_ambiguous_partial_reply_is_not_first_match():
    resolution = PendingResolution(
        resolution_id="r",
        request=ResolutionRequest(
            resolver_id="omop",
            error_code="ambiguous",
            message="ambiguous",
            recoverable=True,
            options=(
                ResolutionOption("1", "Diabetes mellitus type 1 stage 1"),
                ResolutionOption("2", "Diabetes mellitus type 1 stage 2"),
                ResolutionOption("3", "Diabetes mellitus type 1 stage 3"),
            ),
        ),
        question="Which diabetes?",
        graph_version=1,
    )
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        resolution, "type 1"
    )
    assert reply.action == "ask_human"
    assert "several options" in reply.question


# -- interpret_human_reply(): SELECT_MANY response mode ---------------------

def select_many_pending():
    return PendingResolution(
        resolution_id="r2",
        request=ResolutionRequest(
            resolver_id="mimic_hosp_concept",
            error_code="no_exact_concept_match",
            message="ambiguous",
            recoverable=True,
            response_mode=ResponseMode.SELECT_MANY,
            options=(
                ResolutionOption("25000", "Type 2 diabetes mellitus"),
                ResolutionOption("25040", "Type 2 diabetes with renal manifestations"),
                ResolutionOption("25050", "Type 2 diabetes with ophthalmic manifestations"),
            ),
        ),
        question="Which diabetes codes apply?",
        graph_version=1,
    )


def test_multi_select_by_number_list_without_llm():
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        select_many_pending(), "1 and 3"
    )
    assert reply.action == "select_candidate"
    assert reply.option_ids == ("25000", "25050")
    assert reply.option_id == "25000"


def test_multi_select_by_comma_separated_numbers_without_llm():
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        select_many_pending(), "options 1, 2, and 3"
    )
    assert reply.action == "select_candidate"
    assert set(reply.option_ids) == {"25000", "25040", "25050"}


def test_multi_select_all_keyword_without_llm():
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        select_many_pending(), "all of them"
    )
    assert set(reply.option_ids) == {"25000", "25040", "25050"}


def test_single_select_still_works_in_select_many_mode():
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        select_many_pending(), "1"
    )
    assert reply.option_ids == ("25000",)


def test_multi_select_rejected_when_mode_is_select_one():
    # The base fixture's request defaults to SELECT_ONE.
    reply = LLMSemanticExceptionResolver(FakeLLM({
        "action": "ask_human",
        "confidence": 0.3,
        "candidate_id": None,
        "candidate_ids": None,
        "rationale": "Multiple options is not allowed here.",
        "question": "Please pick exactly one.",
        "residual_message": None,
        "acts": [],
        "references": [],
    })).interpret_human_reply(pending(), "1 and 2")
    # SELECT_ONE mode: the deterministic multi-select path must not fire,
    # so this falls through to the (stubbed) LLM.
    assert reply.action == "ask_human"


# -- interpret_human_reply(): exclusion phrasing ("everything except X") ----

# -- interpret_human_reply(): exclusion phrasing falls through to the LLM ---
#
# Exclusion ("everything except X") is deliberately NOT pattern-matched
# deterministically -- getting a negation inverted would silently produce
# the wrong cohort, and unlike a positive list there's no cheap way to
# verify a negation match is unambiguous. It always reaches the LLM, which
# is given explicit instructions (HUMAN_REPLY_PROMPT) to invert the
# selection itself.

def eight_option_pending():
    # Mirrors a real ICD-10 diabetes-family disambiguation list.
    return PendingResolution(
        resolution_id="r3",
        request=ResolutionRequest(
            resolver_id="mimic_hosp_concept",
            error_code="no_exact_concept_match",
            message="ambiguous",
            recoverable=True,
            response_mode=ResponseMode.SELECT_MANY,
            options=(
                ResolutionOption("E11", "Type 2 diabetes mellitus"),
                ResolutionOption("E110", "Type 2 diabetes mellitus with hyperosmolarity"),
                ResolutionOption("E111", "Type 2 diabetes mellitus with ketoacidosis"),
                ResolutionOption("E116", "Type 2 diabetes mellitus with other specified complications"),
                ResolutionOption("E1164", "Type 2 diabetes mellitus with hypoglycemia"),
                ResolutionOption("E1165", "Type 2 diabetes mellitus with hyperglycemia"),
                ResolutionOption("E1169", "Type 2 diabetes mellitus with other specified complication"),
                ResolutionOption("E118", "Type 2 diabetes mellitus with unspecified complications"),
            ),
        ),
        question="Which condition do you mean?",
        graph_version=1,
    )


def test_exclusion_phrasing_is_not_misread_as_a_positive_list():
    # The safety property that actually matters: even with no LLM
    # available, "everything except 1 and 3" must never be silently
    # misread as "select 1 and 3". It should raise (no deterministic
    # match) rather than resolve to the wrong cohort.
    with pytest.raises(AssertionError, match="deterministic matching should handle this"):
        LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
            eight_option_pending(), "everything except 1 and 3"
        )


def test_exclusion_reaches_llm_which_correctly_inverts_the_selection():
    # The LLM is instructed (see HUMAN_REPLY_PROMPT) to return every option
    # EXCEPT the ones named -- this exercises that the plumbing carries a
    # correctly-inverted LLM response through unchanged.
    complement = {"E110", "E116", "E1164", "E1165", "E1169", "E118"}
    reply = LLMSemanticExceptionResolver(FakeLLM({
        "action": "select_candidate",
        "confidence": 0.97,
        "candidate_id": next(iter(complement)),
        "candidate_ids": sorted(complement),
        "rationale": "Selected every option except E11 and E111.",
        "question": None,
        "residual_message": None,
        "acts": [],
        "references": [],
    })).interpret_human_reply(eight_option_pending(), "everything except 1 and 3")

    assert reply.action == "select_candidate"
    assert set(reply.option_ids) == complement
    assert "E11" not in reply.option_ids
    assert "E111" not in reply.option_ids


def test_exclusion_naming_every_option_is_rejected_even_from_the_llm():
    # If the LLM itself returns an empty/invalid selection (e.g. because
    # excluding everything leaves nothing), validation must still catch it
    # rather than accept an empty candidate_ids list.
    reply = LLMSemanticExceptionResolver(FakeLLM({
        "action": "select_candidate",
        "confidence": 0.9,
        "candidate_id": None,
        "candidate_ids": [],
        "rationale": "Excluding everything leaves nothing selected.",
        "question": None,
        "residual_message": None,
        "acts": [],
        "references": [],
    })).interpret_human_reply(
        eight_option_pending(),
        "everything except 1, 2, 3, 4, 5, 6, 7, and 8",
    )
    assert reply.action == "ask_human"


def test_direct_selection_is_not_misread_as_exclusion():
    # Sanity check: ordinary "1 and 3" (no exclusion wording) still selects
    # exactly those two, not their complement.
    reply = LLMSemanticExceptionResolver(NeverLLM()).interpret_human_reply(
        eight_option_pending(), "1 and 3"
    )
    assert set(reply.option_ids) == {"E11", "E111"}
