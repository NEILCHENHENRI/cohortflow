from __future__ import annotations

from framework.interaction import InteractionAdapter
from framework.reconcile import ReuseReport
from framework.resolution import PendingResolution, ResolutionOption


class CohortInteractionAdapter(InteractionAdapter):
    """Only cohort-specific user-facing wording remains here."""

    def format_initial(self, summary: dict) -> str:
        return (
            "The initial cohort has been created. "
            f"Cohort size: **{summary.get('cohort_size', 0)}**."
        )

    def format_update(
        self,
        summary: dict,
        old_summary: dict | None,
        report: ReuseReport,
        rationale: str,
    ) -> str:
        old_size = None if old_summary is None else old_summary.get("cohort_size")
        new_size = summary.get("cohort_size", 0)
        transition = (
            f"{old_size} → {new_size}"
            if old_size is not None
            else str(new_size)
        )
        return (
            "The correction was committed.\n\n"
            f"**Cohort size:** {transition}\n\n"
            f"**Graph update:** {len(report.reused_nodes)} reused, "
            f"{len(report.added_nodes)} added, "
            f"{len(report.invalidated_nodes)} recomputed."
        )

    def format_pending(self, pending: PendingResolution) -> str:
        options = "\n".join(
            f"{i}. {option.label}"
            for i, option in enumerate(pending.options, 1)
        )
        return (
            f"I need your confirmation before continuing. {pending.question}"
            + (f"\n\n{options}" if options else "")
        )

    def format_resolution(
        self,
        options: tuple[ResolutionOption, ...],
        summary: dict,
    ) -> str:
        if len(options) == 1:
            described = f"**{options[0].label}** (concept {options[0].option_id})"
        else:
            labels = ", ".join(
                f"**{option.label}** (concept {option.option_id})"
                for option in options
            )
            described = f"{len(options)} options, combined with OR: {labels}"
        return (
            f"Using {described}. "
            "The interrupted branch resumed and the staged task was committed.\n\n"
            f"**Cohort size:** {summary.get('cohort_size', 0)}"
        )
