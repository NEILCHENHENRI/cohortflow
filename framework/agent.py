from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from framework.artifact_store import ArtifactStore
from framework.correction import (
    CorrectionAnalysis,
)
from framework.graph import ExecutionGraph
from framework.manual_edit import ManualEditForm, ManualEditResult, StructuredManualEditProvider
from framework.module import AgentModuleRole, AgentModuleRunner, AgentModuleSpec
from framework.operator import OperatorRegistry
from framework.resolution import PendingResolution, ResolutionOption
from framework.task import TaskDelta, TaskSpec

if TYPE_CHECKING:
    from framework.session import SteerableSession


ProgressCallback = Callable[[str, str], None]


@runtime_checkable
class StructuredLLM(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        on_progress: ProgressCallback | None = None,
        stage_label: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]: ...


class DomainAgent(ABC):
    """Domain plugin. Generic execution/steering lives in SteerableSession."""

    domain: str

    def assess_failure(self, failure):
        """Optional deterministic domain guidance. Must not rewrite the task."""
        return None

    def repair_replay(self, failure, task_context):
        """Optional bounded, isolated replay(workspace, phase) callback.

        Return None when unavailable. Implementations must use the supplied
        workspace code and detached inputs, and verify the original task.
        """
        return None



    def module_specs(self) -> tuple[AgentModuleSpec, ...]:
        """Lifecycle modules supplied by the generic agent abstraction.

        Domains can extend this catalog with semantic modules. The framework
        owns how modules are timed, surfaced and yielded for steering.
        """
        return (
            AgentModuleSpec(
                "session.build_graph", AgentModuleRole.BUILD,
                "Build an execution graph for the current task.", "planning"
            ),
            AgentModuleSpec(
                "session.put_inputs", AgentModuleRole.PREPARE,
                "Materialize graph input artifacts.", "planning"
            ),
            AgentModuleSpec(
                "session.restore_artifact_cache", AgentModuleRole.RECONCILE,
                "Restore reusable artifacts from unfinished or prior revisions.", "reuse"
            ),
            AgentModuleSpec(
                "correction.interpret", AgentModuleRole.INTERPRET,
                "Interpret an incoming correction against the active task.", "correction"
            ),
            AgentModuleSpec(
                "correction.plan", AgentModuleRole.PLAN,
                "Plan the correction and its invalidation scope.", "correction"
            ),
            AgentModuleSpec(
                "correction.apply_delta", AgentModuleRole.PREPARE,
                "Apply a structured task delta to produce a candidate task.", "correction"
            ),
            AgentModuleSpec(
                "correction.build_graph", AgentModuleRole.BUILD,
                "Build the candidate execution graph.", "planning"
            ),
            AgentModuleSpec(
                "correction.reconcile_graphs", AgentModuleRole.RECONCILE,
                "Reconcile old and new graphs and preserve equivalent work.", "reuse"
            ),
            AgentModuleSpec(
                "correction.put_inputs", AgentModuleRole.PREPARE,
                "Materialize candidate graph inputs.", "planning"
            ),
            AgentModuleSpec(
                "correction.restore_artifact_cache", AgentModuleRole.RECONCILE,
                "Restore reusable session artifacts into the corrected graph.", "reuse"
            ),
            AgentModuleSpec(
                "execution.runtime", AgentModuleRole.EXECUTE,
                "Execute the current graph.", "execution", checkpoint_after=False
            ),
        )

    def module_spec(self, name: str) -> AgentModuleSpec:
        for spec in self.module_specs():
            if spec.name == name:
                return spec
        # A domain may expose dynamically named validation/retry checkpoints.
        # Unknown modules remain steerable without forcing every domain to
        # duplicate framework mechanics.
        return AgentModuleSpec(
            name=name,
            role=AgentModuleRole.OTHER,
            description=f"Agent lifecycle module {name}.",
            category=name.split(".", 1)[0] if "." in name else "agent",
        )

    def run_module(
        self,
        name: str,
        function,
        *,
        on_progress: ProgressCallback | None = None,
    ):
        return AgentModuleRunner().run(
            self.module_spec(name), function, on_progress=on_progress
        )

    @abstractmethod
    def parse_task(
        self,
        request: str,
        on_progress: ProgressCallback | None = None,
    ) -> TaskSpec: ...

    @abstractmethod
    def interpret_correction(
        self,
        current_task: TaskSpec,
        message: str,
        on_progress: ProgressCallback | None = None,
    ) -> CorrectionAnalysis: ...

    @abstractmethod
    def build_graph(self, task: TaskSpec) -> ExecutionGraph: ...

    @abstractmethod
    def build_input_values(self, task: TaskSpec) -> dict[str, Any]: ...

    @abstractmethod
    def operator_registry(self) -> OperatorRegistry: ...

    @abstractmethod
    def summarize(self, artifacts: ArtifactStore) -> dict[str, Any]: ...


    def build_manual_edit_form(
        self,
        task: TaskSpec,
        *,
        version: int,
    ) -> ManualEditForm:
        raise NotImplementedError(
            f"{type(self).__name__} does not support manual fallback editing"
        )

    def compile_manual_edit(
        self,
        original_task: TaskSpec,
        submitted: dict[str, Any],
    ) -> ManualEditResult:
        raise NotImplementedError(
            f"{type(self).__name__} does not support manual fallback editing"
        )

    def record_resolution(
        self,
        pending: PendingResolution,
        options: tuple[ResolutionOption, ...],
        *,
        rationale: str,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support persisted resolutions"
        )


class TaskParser(Protocol):
    """Semantic interpretation boundary for component-based agents."""

    def bind_module_runner(self, runner) -> None: ...

    def parse_task(
        self, request: str, on_progress: ProgressCallback | None = None,
    ) -> TaskSpec: ...

    def interpret_correction(
        self, current_task: TaskSpec, message: str,
        on_progress: ProgressCallback | None = None,
    ) -> CorrectionAnalysis: ...


class TaskPlanner(Protocol):
    """Deterministic planning boundary, independent of the agent's domain."""

    def build_graph(self, task: TaskSpec) -> ExecutionGraph: ...

    def build_input_values(self, task: TaskSpec) -> dict[str, Any]: ...


class ComposedDomainAgent(DomainAgent):
    """Reusable delegation for agents assembled from a parser and a planner.

    Subclasses own construction, operator registration, summaries, and optional
    manual editors. DomainAgent remains available for agents with other designs.
    """

    parser: TaskParser
    planner: TaskPlanner
    max_workers: int = 1
    _registry: OperatorRegistry
    _manual_editor: StructuredManualEditProvider

    def _bind_parser_modules(self) -> None:
        self.parser.bind_module_runner(self.run_module)

    def create_session(self) -> SteerableSession:
        from framework.session import SteerableSession

        return SteerableSession(self, max_workers=self.max_workers)

    def parse_task(
        self, request: str, on_progress: ProgressCallback | None = None,
    ) -> TaskSpec:
        return self.parser.parse_task(request, on_progress=on_progress)

    def interpret_correction(
        self, current_task: TaskSpec, message: str,
        on_progress: ProgressCallback | None = None,
    ) -> CorrectionAnalysis:
        return self.parser.interpret_correction(
            current_task, message, on_progress=on_progress
        )

    def build_graph(self, task: TaskSpec) -> ExecutionGraph:
        return self.planner.build_graph(task)

    def build_input_values(self, task: TaskSpec) -> dict[str, Any]:
        return self.planner.build_input_values(task)

    def operator_registry(self) -> OperatorRegistry:
        return self._registry

    def build_manual_edit_form(self, task: TaskSpec, *, version: int) -> ManualEditForm:
        if not hasattr(self, "_manual_editor"):
            return super().build_manual_edit_form(task, version=version)
        return self._manual_editor.build_form(task, version=version)

    def compile_manual_edit(
        self, original_task: TaskSpec, submitted: dict[str, Any],
    ) -> ManualEditResult:
        if not hasattr(self, "_manual_editor"):
            return super().compile_manual_edit(original_task, submitted)
        return self._manual_editor.compile(original_task, submitted)
