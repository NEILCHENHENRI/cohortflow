from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from framework.agent import ComposedDomainAgent, DomainAgent, ProgressCallback
from agents.cohort.operators import (
    CohortOperatorServices,
    CohortRepository,
    ConceptResolver,
    NoteMatcher,
    register_cohort_operators,
)
from agents.cohort.parser import CohortTaskParser
from agents.cohort.planner import CohortPlanner
from agents.cohort.manual_edit import CohortManualEditProvider
from framework.artifact_store import ArtifactStore
from framework.operator import OperatorRegistry
from framework.module import AgentModuleRole, AgentModuleSpec
from framework.resolution import PendingResolution, ResolutionOption
from framework.session import SteerableSession
from framework.task import TaskDelta, TaskSpec


@dataclass(slots=True)
class CohortAgent(ComposedDomainAgent):
    """Cohort-specific parser, planner, operators, and result semantics."""

    parser: CohortTaskParser
    resolver: ConceptResolver
    repository: CohortRepository
    note_matcher: NoteMatcher | None = None
    max_workers: int = 1
    planner: CohortPlanner = field(default_factory=CohortPlanner)
    domain: str = field(init=False, default="cohort_curation")
    _registry: OperatorRegistry = field(init=False)
    _manual_editor: CohortManualEditProvider = field(
        init=False, default_factory=CohortManualEditProvider
    )

    def assess_failure(self, failure):
        if failure.exception_type == "UnitConversionError":
            from framework.repair.models import FailureAction, FailureAssessment, RepairScope
            return FailureAssessment(FailureAction.ASK_USER, RepairScope.NONE, 1.0,
                "Unit conversion needs clarification",
                "Clarify the value/unit or ask a clinician to resolve the conversion. Preserve the criterion.")
        return DomainAgent.assess_failure(self, failure)

    def __post_init__(self) -> None:
        self._bind_parser_modules()
        if getattr(self.parser, "dataset", None) is not None:
            self._manual_editor = CohortManualEditProvider(self.parser.dataset)
        self._registry = register_cohort_operators(
            CohortOperatorServices(self.resolver, self.repository, self.note_matcher)
        )


    def module_specs(self) -> tuple[AgentModuleSpec, ...]:
        return (
            *DomainAgent.module_specs(self),
            AgentModuleSpec(
                "semantic.predicate_extraction", AgentModuleRole.INTERPRET,
                "Extract atomic cohort requirements without interpreting relationships.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.initial_logic_parse", AgentModuleRole.INTERPRET,
                "Parse the initial cohort request into Boolean logic.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.relationship_parse", AgentModuleRole.INTERPRET,
                "Attach logical relationships to already extracted requirements.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.correction_interpret", AgentModuleRole.INTERPRET,
                "Interpret a correction into localized semantic actions over existing refs.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.correction_extraction", AgentModuleRole.INTERPRET,
                "Extract fixed change intents and predicate text without structural attachment.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.correction_attachment", AgentModuleRole.INTERPRET,
                "Attach fixed change intents to current predicate, group and relation references.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.correction_merge", AgentModuleRole.INTERPRET,
                "Merge a correction into a complete cohort request.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.logic_reparse", AgentModuleRole.INTERPRET,
                "Reparse the corrected complete cohort request.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.table_selection", AgentModuleRole.INTERPRET,
                "Assign predicates to semantic tables when the field catalog is large.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.predicate_compilation", AgentModuleRole.COMPILE,
                "Compile predicate leaves into executable requirements.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.canonicalization", AgentModuleRole.VALIDATE,
                "Canonicalize compiled requirements and Boolean logic.", "semantic"
            ),
            AgentModuleSpec(
                "semantic.structured_diff", AgentModuleRole.DIFF,
                "Diff the verified corrected task against the current task.", "semantic"
            ),
        )


    def summarize(self, artifacts: ArtifactStore) -> dict[str, Any]:
        if not artifacts.contains("cohort_summary"):
            raise RuntimeError("cohort summary has not been produced")
        return dict(artifacts.get("cohort_summary").value)


    def record_resolution(
        self,
        pending: PendingResolution,
        options: tuple[ResolutionOption, ...],
        *,
        rationale: str,
    ) -> None:
        recorder = getattr(self.resolver, "record_human_selection", None)
        if recorder is None:
            raise RuntimeError("concept resolver cannot persist selections")
        recorder(
            pending_context=pending.context,
            options=options,
            rationale=rationale,
        )
