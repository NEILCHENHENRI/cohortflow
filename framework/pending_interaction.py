from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from framework.correction import CorrectionAnalysis
from framework.resolution import PendingResolution
from framework.manual_edit import ManualEditForm


class PendingInteractionKind(str, Enum):
    SEMANTIC_RESOLUTION = "semantic_resolution"
    INITIAL_PARSE_CLARIFICATION = "initial_parse_clarification"
    CORRECTION_CLARIFICATION = "correction_clarification"
    CORRECTION_FALLBACK_CONFIRMATION = "correction_fallback_confirmation"
    MANUAL_EDIT = "manual_edit"
    NOTE_EXECUTION_CONFIRMATION = "note_execution_confirmation"


class ContinuationKind(str, Enum):
    INITIAL_INTERPRETATION = "initial_interpretation"
    INITIAL_NOTE_PREFLIGHT = "initial_note_preflight"
    INITIAL_EXECUTION = "initial_execution"
    STAGED_NOTE_PREFLIGHT = "staged_note_preflight"
    STAGED_EXECUTION = "staged_execution"
    CORRECTION_INTERPRETATION = "correction_interpretation"
    MANUAL_EDIT_REVIEW = "manual_edit_review"


@dataclass(slots=True)
class PendingInteraction:
    kind: PendingInteractionKind
    continuation: ContinuationKind
    question: str
    resolution: PendingResolution | None = None
    original_message: str | None = None
    correction_analysis: CorrectionAnalysis | None = None
    manual_form: ManualEditForm | None = None
    deferred_corrections: tuple[str, ...] = ()
    deferred_correction_analyses: tuple[CorrectionAnalysis, ...] = ()
    rationale: str | None = None

    @classmethod
    def semantic_resolution(
        cls,
        resolution: PendingResolution,
        *,
        continuation: ContinuationKind,
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.SEMANTIC_RESOLUTION,
            continuation=continuation,
            question=resolution.question,
            resolution=resolution,
        )


    @classmethod
    def initial_parse_clarification(
        cls, *, original_message: str, question: str, rationale: str
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.INITIAL_PARSE_CLARIFICATION,
            continuation=ContinuationKind.INITIAL_INTERPRETATION,
            question=question,
            original_message=original_message,
            rationale=rationale,
        )

    @classmethod
    def correction_clarification(
        cls,
        *,
        original_message: str,
        question: str,
        analysis: CorrectionAnalysis | None = None,
        rationale: str | None = None,
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.CORRECTION_CLARIFICATION,
            continuation=ContinuationKind.CORRECTION_INTERPRETATION,
            question=question,
            original_message=original_message,
            correction_analysis=analysis,
            rationale=rationale,
        )



    @classmethod
    def correction_fallback_confirmation(
        cls, *, original_message: str, reason: str
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.CORRECTION_FALLBACK_CONFIRMATION,
            continuation=ContinuationKind.CORRECTION_INTERPRETATION,
            question=(
                "Robust correction could not be applied safely. "
                "Use Full Regeneration for this correction?\n\n"
                f"Reason: {reason}"
            ),
            original_message=original_message,
        )

    @classmethod
    def note_execution_confirmation(
        cls,
        *,
        continuation: ContinuationKind,
        question: str,
        original_message: str | None = None,
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.NOTE_EXECUTION_CONFIRMATION,
            continuation=continuation,
            question=question,
            original_message=original_message,
        )

    @classmethod
    def manual_edit(
        cls,
        form: ManualEditForm,
    ) -> "PendingInteraction":
        return cls(
            kind=PendingInteractionKind.MANUAL_EDIT,
            continuation=ContinuationKind.MANUAL_EDIT_REVIEW,
            question=(
                "Manual fallback mode is active. Edit the structured fields "
                "and submit them for review."
            ),
            manual_form=form,
        )
