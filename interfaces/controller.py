from __future__ import annotations

from dataclasses import replace
import inspect
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from framework.agent import ProgressCallback
from framework.correction import CorrectionFallbackRequired, CorrectionMode
from framework.failure import FailureEvent
from framework.repair.manager import RepairManager
from framework.interpretation import InteractionAct, PendingReplyInterpretation
from framework.interaction import (
    InteractionAdapter,
    MessageRoute,
    MessageRouter,
)
from framework.resolution import ResolutionAction, ResolutionRequired, SemanticResolver
from framework.pending_interaction import (
    ContinuationKind,
    PendingInteractionKind,
)
from framework.session import SteerableSession
from framework.timing import measure, timing_session
from framework.steering import SteeringYield
from framework.runtime import ExecutionMode
from presentation.models import ConversationMessageView, InteractionResult


def _message(
    role: str,
    content: str,
    version: int | None = None,
    metadata: dict | None = None,
):
    return ConversationMessageView(
        uuid4().hex,
        role,
        content,
        datetime.now(timezone.utc).isoformat(),
        version,
        metadata=metadata or {},
    )




def _task_has_note(task) -> bool:
    return any(requirement.kind == "note" for requirement in task.requirements)


@runtime_checkable
class InteractionController(Protocol):
    session: SteerableSession
    def handle_message(
        self,
        text: str,
        on_progress: ProgressCallback | None = None,
    ) -> InteractionResult: ...


class AgentInteractionController:
    """Generic chat lifecycle. Domain wording is supplied by an adapter."""

    def __init__(
        self,
        session: SteerableSession,
        adapter: InteractionAdapter,
        semantic_resolver: SemanticResolver | None = None,
        repair_manager: RepairManager | None = None,
    ) -> None:
        self.session = session
        self.adapter = adapter
        self.semantic_resolver = semantic_resolver
        self.repair_manager = repair_manager
        if isinstance(repair_manager, RepairManager):
            self.session.repair_manager = repair_manager
        self.router = MessageRouter()
        # Most controller calls produce a timing report as part of their result.
        # SteeringYield is intentionally re-raised, so keep the just-finished
        # interaction report here as well. The background supervisor can fold
        # yielded revisions into one end-to-end timing report.
        self._latest_timing_report: dict | None = None

    @property
    def agent(self):
        """Compatibility alias used by the existing Streamlit page."""
        return self.session.domain_agent

    def handle_message(self, text: str, on_progress=None) -> InteractionResult:
        self.session.last_failure = None
        self.session.last_repair_proposal = None
        self.session.last_repair_outcome = None
        with timing_session() as recorder:
            try:
                with measure("interaction.total", category="interaction"):
                    result = self._handle_message(text, on_progress=on_progress)
            except SteeringYield:
                # The yielded revision still consumed real compute. Preserve its
                # timing even though there is no InteractionResult to carry it.
                diagnostic = self._attach_timing_diagnostics(
                    InteractionResult(), recorder.report()
                )
                self._latest_timing_report = diagnostic.diagnostics.get("timing")
                raise
            result = self._attach_timing_diagnostics(result, recorder.report())
            self._latest_timing_report = result.diagnostics.get("timing")
        return result

    @property
    def latest_timing_report(self) -> dict | None:
        return self._latest_timing_report

    def _handle_message(self, text: str, on_progress=None) -> InteractionResult:
        cleaned = text.strip()
        if not cleaned:
            return InteractionResult(error="Message cannot be empty.")
        route = self.router.route(cleaned, self.session)
        if route is MessageRoute.ANSWER_PENDING:
            if self.session.state is None:
                return self._answer_initial_parse_clarification(cleaned, on_progress)
            return self._answer_pending_interaction(cleaned, on_progress)
        if route is MessageRoute.CONTROL_EXECUTION:
            return self._handle_control(cleaned, on_progress)

        version = self.session.state.version if self.session.state else None
        if route is MessageRoute.ENTER_MANUAL_FALLBACK:
            user = _message("user", cleaned, version)
            form = self.session.enter_manual_fallback()
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        (
                            "Manual fallback mode is active. Edit the "
                            "structured cohort fields in the execution "
                            "workspace, then review and apply the changes."
                        ),
                        version,
                        metadata={"manual_form_id": form.form_id},
                    ),
                ),
                graph_changed=False,
                graph_version=version,
                notifications=("manual_fallback_entered",),
            )

        user = _message("user", cleaned, version)
        try:
            if route is MessageRoute.INITIAL_TASK:
                self._emit(on_progress, "Interpreting the initial request...")
                task = self.session.domain_agent.parse_task(cleaned, on_progress)
                if task.metadata.get("logic_requires_clarification"):
                    rationale = str(
                        task.metadata.get("logic_clarification_rationale")
                        or "The request has materially different plausible interpretations."
                    )
                    question = str(
                        task.metadata.get("logic_clarification_question")
                        or "Please restate the full cohort request more clearly."
                    )
                    pending = self.session.request_initial_parse_clarification(
                        original_message=cleaned, question=question, rationale=rationale
                    )
                    return InteractionResult(
                        messages=(
                            user,
                            _message(
                                "assistant",
                                f"{rationale}\n\n{question}\n\nPlease restate the full cohort request so I can reinterpret it from scratch.",
                                None,
                            ),
                        ),
                        graph_changed=False,
                        graph_version=None,
                        notifications=("initial_parse_clarification_required",),
                    )
                self.session.initialize(
                    task,
                    on_progress=on_progress,
                )
                self._emit(on_progress, "Initial graph built.")
                if _task_has_note(task):
                    self._emit(
                        on_progress,
                        "Resolving structured requirements before note confirmation...",
                    )
                    try:
                        preflight = self.session.execute_until_note_confirmation(
                            on_progress=on_progress
                        )
                    except ResolutionRequired:
                        pending_resolution = self.session.state.pending_resolution
                        return InteractionResult(
                            messages=(
                                user,
                                _message(
                                    "assistant",
                                    self.adapter.format_pending(pending_resolution),
                                    self.session.state.version,
                                ),
                            ),
                            graph_changed=True,
                            graph_version=self.session.state.version,
                        )
                    if preflight.stopped:
                        return self._stopped_result(
                            user,
                            preflight.completed_nodes,
                            self.session.state.version,
                            staged=False,
                        )
                    pending = self.session.request_note_execution_confirmation(
                        continuation=ContinuationKind.INITIAL_EXECUTION,
                        original_message=cleaned,
                    )
                    return InteractionResult(
                        messages=(
                            user,
                            _message("assistant", pending.question, self.session.state.version),
                        ),
                        graph_changed=True,
                        graph_version=self.session.state.version,
                        notifications=("note_confirmation_required",),
                    )
                self._emit(on_progress, "Executing the graph...")
                execution = self.session.execute(on_progress=on_progress)
                version = self.session.state.version
                if execution.stopped:
                    return self._stopped_result(
                        user, execution.completed_nodes, version, staged=False
                    )
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            self.adapter.format_initial(self.session.summary()),
                            version,
                        ),
                    ),
                    graph_changed=True,
                    graph_version=version,
                    selected_node_id="deliver:summary",
                )

            if route is not MessageRoute.MODIFY_TASK:
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            "That interaction type is recognized but is not yet implemented.",
                            version,
                        ),
                    )
                )

            return self._handle_correction(
                cleaned,
                cleaned,
                on_progress,
                user_message=user,
            )
        except ResolutionRequired:
            pending = self.session.state.pending_resolution
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        self.adapter.format_pending(pending),
                        self.session.state.version,
                    ),
                ),
                graph_changed=True,
                graph_version=self.session.state.version,
            )
        except SteeringYield:
            raise
        except Exception as exc:
            return self._failure_result(exc, user=user, version=version, origin="interaction")



    def _answer_initial_parse_clarification(
        self, response: str, on_progress=None
    ) -> InteractionResult:
        pending = self.session.pending_initial_interaction
        if pending is None or pending.kind is not PendingInteractionKind.INITIAL_PARSE_CLARIFICATION:
            raise RuntimeError("no initial parsing clarification is pending")
        deferred = tuple(pending.deferred_corrections)
        # The user was explicitly asked to restate the whole request. Drop the
        # ambiguous wording and run the ordinary initial parser again. If newer
        # corrections were already waiting, stop before doing obsolete graph
        # work once the clarified TaskSpec materializes.
        self.session.clear_initial_parse_clarification()
        if deferred:
            self.session.request_stop(reason="correction")
        result = self._handle_message(response, on_progress=on_progress)

        # The restatement itself can still require clarification. Keep queued
        # corrections parked until a valid initial TaskSpec exists.
        if self.session.state is None:
            self.session.clear_stop_request()
            next_pending = self.session.pending_initial_interaction
            if next_pending is not None and deferred:
                next_pending.deferred_corrections = (
                    *next_pending.deferred_corrections, *deferred
                )
            return result

        if not deferred:
            return result
        self.session.clear_stop_request()
        correction_result = self._apply_deferred_corrections_fifo(
            deferred, on_progress=on_progress
        )
        return InteractionResult(
            messages=(*result.messages, *correction_result.messages),
            graph_changed=(result.graph_changed or correction_result.graph_changed),
            graph_version=(correction_result.graph_version or result.graph_version),
            selected_node_id=(correction_result.selected_node_id or result.selected_node_id),
            notifications=(*result.notifications, *correction_result.notifications),
            error=correction_result.error or result.error,
        )


    def _failure_result(
        self,
        exc: Exception,
        *,
        user: ConversationMessageView | None,
        version: int | None,
        origin: str,
    ) -> InteractionResult:
        failure = self.session.last_failure or FailureEvent.capture(exc, origin=origin)
        self.session.last_failure = failure
        proposal = None
        outcome = None
        repair_error = None
        if self.repair_manager is not None:
            try:
                previous = self.session.last_repair_outcome
                if previous is not None and previous.failure.failure_id == failure.failure_id:
                    outcome = previous
                elif isinstance(self.repair_manager, RepairManager):
                    outcome = self.session.repair_failure(failure)
                else:
                    outcome = self.repair_manager.repair(failure)
                proposal = outcome.proposal
            except Exception as repair_exc:
                # Repair must never mask the original failure.
                repair_error = f"{type(repair_exc).__name__}: {repair_exc}"
        self.session.last_repair_proposal = proposal
        self.session.last_repair_outcome = outcome
        diagnostics = {
            "failure": {
                "failure_id": failure.failure_id,
                "origin": failure.origin,
                "exception_type": failure.exception_type,
                "message": failure.message,
                "status_code": failure.status_code,
                "request_id": failure.request_id,
                "node_id": failure.node_id,
                "operator_name": failure.operator_name,
            }
        }
        if outcome is not None:
            diagnostics["repair"] = {
                "status": outcome.status.value,
                "verification": outcome.verification.value,
                "retry_attempts": outcome.retry_attempts,
                "backend": outcome.backend,
                "escalated": outcome.escalated,
                "message": outcome.message,
                "workspace_path": outcome.workspace_path,
                "assessment": None if outcome.assessment is None else {
                    "action": outcome.assessment.action.value,
                    "scope": outcome.assessment.scope.value,
                    "confidence": outcome.assessment.confidence,
                    "diagnosis": outcome.assessment.diagnosis,
                    "rationale": outcome.assessment.rationale,
                },
                "validation": None if outcome.validation is None else {
                    "passed": outcome.validation.passed,
                    "changed_files": list(outcome.validation.changed_files),
                    "changed_lines": outcome.validation.changed_lines,
                    "rejection_reason": outcome.validation.rejection_reason,
                },
            }
            if proposal is not None:
                diagnostics["repair"]["proposal"] = {
                    "backend": proposal.backend,
                    "diagnosis": proposal.diagnosis,
                    "confidence": proposal.confidence,
                    "summary": proposal.summary,
                    "unified_diff": proposal.unified_diff,
                }
        if repair_error:
            diagnostics["repair_error"] = repair_error
        messages = []
        if user is not None:
            messages.append(user)
        description = f"Execution failed: {failure.exception_type}: {failure.message}"
        if outcome is not None and outcome.status.value != "disabled":
            description += f"\n\nSelf repair: {outcome.message}"
        messages.append(_message("assistant", description, version))
        return InteractionResult(
            messages=tuple(messages),
            error=f"{failure.exception_type}: {failure.message}",
            diagnostics=diagnostics,
            graph_version=version,
        )

    def submit_manual_edit(
        self,
        submitted: dict,
        on_progress=None,
    ) -> InteractionResult:
        self.session.last_failure = None
        self.session.last_repair_proposal = None
        self.session.last_repair_outcome = None
        with timing_session() as recorder:
            with measure("interaction.total", category="interaction"):
                result = self._submit_manual_edit(submitted, on_progress=on_progress)
            result = self._attach_timing_diagnostics(result, recorder.report())
        return result

    def _submit_manual_edit(
        self,
        submitted: dict,
        on_progress=None,
    ) -> InteractionResult:
        version = self.session.state.version
        result, report = self.session.apply_manual_edit(submitted)
        if not result.valid:
            message = "\n".join(
                f"- **{issue.field_id}:** {issue.message}"
                for issue in result.issues
            )
            return InteractionResult(
                messages=(
                    _message(
                        "assistant",
                        "Manual edit validation failed:\n" + message,
                        version,
                    ),
                ),
                graph_changed=False,
                graph_version=version,
                error="Manual edit validation failed",
            )

        self._emit(on_progress, "Executing manual fallback changes...")
        try:
            execution = self.session.execute_staged(on_progress=on_progress)
            if execution.stopped:
                return self._stopped_result(
                    None,
                    execution.completed_nodes,
                    self.session.state.version,
                    staged=True,
                )
        except ResolutionRequired:
            pending = self.session.state.pending_resolution
            return InteractionResult(
                messages=(
                    _message(
                        "assistant",
                        (
                            "The manual fields are structurally valid, but "
                            "a clinical concept still needs confirmation.\n\n"
                            + self.adapter.format_pending(pending)
                        ),
                        version,
                    ),
                ),
                graph_changed=True,
                graph_version=version,
            )

        new_version = self.session.state.version
        return InteractionResult(
            messages=(
                _message(
                    "assistant",
                    (
                        "Manual fallback changes were validated, executed, "
                        "and committed.\n\n"
                        + self.adapter.format_update(
                            self.session.summary(),
                            None,
                            report,
                            result.rationale,
                        )
                    ),
                    new_version,
                ),
            ),
            graph_changed=True,
            graph_version=new_version,
            selected_node_id="deliver:summary",
            notifications=("manual_fallback_committed",),
        )

    def cancel_manual_edit(self) -> InteractionResult:
        result = self._cancel_manual_edit()
        return result

    def _cancel_manual_edit(self) -> InteractionResult:
        version = self.session.state.version
        self.session.cancel_manual_fallback()
        return InteractionResult(
            messages=(
                _message(
                    "assistant",
                    "Manual fallback mode was cancelled. No changes were applied.",
                    version,
                ),
            ),
            graph_changed=False,
            graph_version=version,
        )

    def handle_preanalyzed_correction(
        self, text: str, analysis, on_progress=None
    ) -> InteractionResult:
        """Finish a correction whose LLM interpretation overlapped one active action."""
        version = self.session.state.version if self.session.state else None
        user = _message("user", text, version)
        return self._handle_correction(
            text, text, on_progress, user_message=user, preanalyzed=analysis
        )

    def _handle_correction(
        self,
        interpretation_text: str,
        displayed_text: str,
        on_progress,
        *,
        user_message=None,
        preanalyzed=None,
    ) -> InteractionResult:
        version = self.session.state.version
        user = user_message or _message("user", displayed_text, version)
        try:
            old_summary = self.session.summary()
        except Exception:
            old_summary = None

        self._emit(on_progress, "Interpreting correction intent...")
        try:
            analysis = preanalyzed or self.session.analyze_correction(
                interpretation_text,
                on_progress=on_progress,
            )
        except CorrectionFallbackRequired as exc:
            self.session.request_correction_fallback_confirmation(
                original_message=interpretation_text, reason=exc.reason
            )
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        self.session.state.pending_interaction.question,
                        version,
                    ),
                ),
                graph_changed=False,
                graph_version=version,
                notifications=("correction_fallback_confirmation_required",),
            )
        intent = analysis.intent

        # Model-reported confidence is not calibrated enough to be a hard
        # interaction gate. The semantic verifier must explicitly declare an
        # ambiguity before we interrupt the user.
        if intent.requires_clarification:
            question = (
                intent.clarification_question
                or "Could you clarify what should change?"
            )
            rationale = intent.rationale or "The correction has materially different plausible interpretations."
            self.session.request_correction_clarification(
                original_message=interpretation_text,
                question=question,
                analysis=analysis,
                rationale=rationale,
            )
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        f"{rationale}\n\n{question}\n\nPlease restate the full correction so I can reinterpret it from scratch.",
                        version,
                    ),
                ),
                graph_changed=False,
                graph_version=version,
            )

        plan = self.session.plan_correction(analysis, on_progress=on_progress)
        self._emit(
            on_progress,
            (
                f"Correction scope: {intent.scope.value}. "
                f"Affected nodes: {len(plan.affected_nodes)}, "
                f"recompute boundary: "
                f"{', '.join(plan.recompute_boundary) or 'none'}."
            ),
        )
        report = self.session.stage_correction(
            analysis,
            plan,
            on_progress=on_progress,
        )
        self._emit(on_progress, "Corrected graph staged.")
        staged_task = self.session.state.staged.task
        if _task_has_note(staged_task):
            self._emit(
                on_progress,
                "Resolving structured requirements before note confirmation...",
            )
            try:
                preflight = self.session.execute_staged_until_note_confirmation(
                    on_progress=on_progress
                )
            except ResolutionRequired:
                pending_resolution = self.session.state.pending_resolution
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            self.adapter.format_pending(pending_resolution),
                            self.session.state.version,
                        ),
                    ),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                )
            if preflight.stopped:
                return self._stopped_result(
                    user,
                    preflight.completed_nodes,
                    self.session.state.version,
                    staged=True,
                )
            if self.session.has_pending_note_work(staged=True):
                pending = self.session.request_note_execution_confirmation(
                    continuation=ContinuationKind.STAGED_EXECUTION,
                    original_message=interpretation_text,
                )
                return InteractionResult(
                    messages=(
                        user,
                        _message("assistant", pending.question, version),
                    ),
                    graph_changed=True,
                    graph_version=version,
                    notifications=("note_confirmation_required",),
                )
        self._emit(on_progress, "Executing the staged correction...")
        try:
            execution = self.session.execute_staged(on_progress=on_progress)
        except ResolutionRequired:
            # A correction can introduce a new ambiguous concept while this
            # helper is called directly (for example, immediately after a
            # deferred correction is released by a prior manual resolution).
            # In that path there is no outer handle_message() exception guard,
            # so convert the expected pause back into an InteractionResult here
            # instead of letting Streamlit crash.
            pending = self.session.state.pending_resolution
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        self.adapter.format_pending(pending),
                        self.session.state.version,
                    ),
                ),
                graph_changed=True,
                graph_version=self.session.state.version,
            )
        version = self.session.state.version
        if execution.stopped:
            return self._stopped_result(
                user, execution.completed_nodes, version, staged=True
            )
        return InteractionResult(
            messages=(
                user,
                _message(
                    "assistant",
                    self.adapter.format_update(
                        self.session.summary(),
                        old_summary,
                        report,
                        intent.rationale or analysis.task_delta.rationale,
                    ),
                    version,
                ),
            ),
            graph_changed=True,
            graph_version=version,
            selected_node_id="deliver:summary",
        )

    def _answer_pending_interaction(
        self,
        response: str,
        on_progress,
    ) -> InteractionResult:
        state = self.session.state
        pending_interaction = state.pending_interaction
        if pending_interaction is None:
            raise RuntimeError("no pending interaction exists")

        if pending_interaction.kind is PendingInteractionKind.NOTE_EXECUTION_CONFIRMATION:
            version = state.version
            user = _message("user", response, version)
            normalized = response.strip().casefold()
            proceed = normalized in {"yes", "y", "proceed", "continue", "run", "confirm"}
            without_notes = normalized in {
                "proceed without notes", "without notes", "skip notes", "continue without notes"
            }
            cancel = normalized in {"no", "n", "cancel", "stop"}
            if not proceed and not without_notes and not cancel:
                correction = response.strip()
                already_waiting = bool(pending_interaction.deferred_corrections)
                pending_interaction.deferred_corrections = (
                    *pending_interaction.deferred_corrections, correction
                )
                if (
                    self.session.runtime.execution_mode is ExecutionMode.PARALLEL
                    and not already_waiting
                ):
                    analysis = self.session.analyze_correction(
                        correction, on_progress=on_progress
                    )
                    pending_interaction.deferred_correction_analyses = (analysis,)
                    note = "The correction has been interpreted in parallel and is waiting for the note confirmation action to finish."
                else:
                    note = "The correction is queued until the note confirmation action finishes and the prior semantic state is materialized."
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            f"{note}\n\n{pending_interaction.question}",
                            version,
                        ),
                    ),
                    graph_changed=False,
                    graph_version=version,
                    notifications=("correction_deferred_until_note_confirmation",),
                )
            continuation = pending_interaction.continuation
            deferred_corrections = tuple(pending_interaction.deferred_corrections)
            deferred_analyses = tuple(pending_interaction.deferred_correction_analyses)
            if cancel:
                self.session.clear_pending_interaction()
                if continuation is ContinuationKind.STAGED_EXECUTION:
                    self.session.discard_staged()
                base = InteractionResult(
                    messages=(
                        user,
                        _message("assistant", "Note processing cancelled. No note-based execution was run.", version),
                    ),
                    graph_changed=False,
                    graph_version=version,
                    notifications=("note_execution_cancelled",),
                )
                if not deferred_corrections:
                    return base
                queued = self._apply_deferred_corrections_fifo(
                    deferred_corrections, deferred_analyses, on_progress=on_progress
                )
                return InteractionResult(
                    messages=(*base.messages, *queued.messages),
                    graph_changed=queued.graph_changed,
                    graph_version=queued.graph_version or version,
                    selected_node_id=queued.selected_node_id,
                    notifications=(*base.notifications, *queued.notifications),
                    error=queued.error,
                )
            if without_notes:
                self._emit(
                    on_progress,
                    "Proceeding without notes. Removing note criteria and rebuilding the structured graph...",
                )
                try:
                    staged_report = self.session.prepare_without_notes(
                        on_progress=on_progress
                    )
                except ValueError as exc:
                    return InteractionResult(
                        messages=(
                            user,
                            _message("assistant", str(exc), version),
                        ),
                        graph_changed=False,
                        graph_version=version,
                        notifications=("note_skip_unavailable",),
                    )
                if deferred_corrections:
                    self.session.mark_current_base_stopped()
                    queued = self._apply_deferred_corrections_fifo(
                        deferred_corrections, deferred_analyses, on_progress=on_progress
                    )
                    return InteractionResult(
                        messages=(
                            user,
                            _message("assistant", "Note criteria were removed. Applying the queued correction before continuing execution.", version),
                            *queued.messages,
                        ),
                        graph_changed=True,
                        graph_version=queued.graph_version or version,
                        selected_node_id=queued.selected_node_id,
                        notifications=("note_execution_skipped", *queued.notifications),
                        error=queued.error,
                    )
                try:
                    execution = (
                        self.session.execute(on_progress=on_progress)
                        if continuation is ContinuationKind.INITIAL_EXECUTION
                        else self.session.execute_staged(on_progress=on_progress)
                    )
                except ResolutionRequired:
                    pending = self.session.state.pending_resolution
                    return InteractionResult(
                        messages=(
                            user,
                            _message(
                                "assistant",
                                self.adapter.format_pending(pending),
                                self.session.state.version,
                            ),
                        ),
                        graph_changed=True,
                        graph_version=self.session.state.version,
                    )
                if execution.stopped:
                    return self._stopped_result(
                        user,
                        execution.completed_nodes,
                        self.session.state.version,
                        staged=(continuation is ContinuationKind.STAGED_EXECUTION),
                    )
                if continuation is ContinuationKind.INITIAL_EXECUTION:
                    content = self.adapter.format_initial(self.session.summary())
                else:
                    content = self.adapter.format_update(
                        self.session.summary(),
                        None,
                        staged_report,
                        "Note criteria were skipped at user confirmation.",
                    )
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            content,
                            self.session.state.version,
                        ),
                    ),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                    selected_node_id="deliver:summary",
                    notifications=("note_execution_skipped",),
                )

            if deferred_corrections:
                # Confirmation is the singular active action. Once it finishes,
                # apply queued semantic changes before starting note retrieval.
                self.session.supersede_pending_for_correction()
                queued = self._apply_deferred_corrections_fifo(
                    deferred_corrections, deferred_analyses, on_progress=on_progress
                )
                return InteractionResult(
                    messages=(
                        user,
                        _message("assistant", "Note processing was confirmed. Applying the queued correction before starting retrieval.", version),
                        *queued.messages,
                    ),
                    graph_changed=queued.graph_changed,
                    graph_version=queued.graph_version or version,
                    selected_node_id=queued.selected_node_id,
                    notifications=("note_execution_confirmed_before_correction", *queued.notifications),
                    error=queued.error,
                )

            self.session.clear_pending_interaction()
            self._emit(on_progress, "Note processing confirmed. Starting retrieval pipeline...")
            staged_report = None
            staged_rationale = "Note-based correction."
            if continuation is ContinuationKind.STAGED_EXECUTION and state.staged is not None:
                staged_report = state.staged.reuse_report
                staged_rationale = state.staged.rationale
            try:
                execution = (
                    self.session.execute(on_progress=on_progress)
                    if continuation is ContinuationKind.INITIAL_EXECUTION
                    else self.session.execute_staged(on_progress=on_progress)
                )
            except ResolutionRequired:
                pending = self.session.state.pending_resolution
                return InteractionResult(
                    messages=(user, _message("assistant", self.adapter.format_pending(pending), self.session.state.version)),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                )
            if execution.stopped:
                return self._stopped_result(user, execution.completed_nodes, self.session.state.version, staged=(continuation is ContinuationKind.STAGED_EXECUTION))
            if continuation is ContinuationKind.INITIAL_EXECUTION:
                return InteractionResult(
                    messages=(user, _message("assistant", self.adapter.format_initial(self.session.summary()), self.session.state.version)),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                    selected_node_id="deliver:summary",
                    notifications=("note_execution_completed",),
                )
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        self.adapter.format_update(
                            self.session.summary(),
                            None,
                            staged_report,
                            staged_rationale,
                        ) if staged_report is not None else self.adapter.format_initial(self.session.summary()),
                        self.session.state.version,
                    ),
                ),
                graph_changed=True,
                graph_version=self.session.state.version,
                selected_node_id="deliver:summary",
                notifications=("note_execution_completed",),
            )

        if (
            pending_interaction.kind
            is PendingInteractionKind.CORRECTION_FALLBACK_CONFIRMATION
        ):
            version = state.version
            user = _message("user", response, version)
            normalized = response.strip().casefold()
            yes = normalized in {"yes", "y", "proceed", "use full regeneration", "full regeneration", "confirm"}
            no = normalized in {"no", "n", "cancel", "keep current", "do not fallback"}
            if not yes and not no:
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            "Please choose Full Regeneration or Keep current cohort.",
                            version,
                        ),
                    ),
                    graph_changed=False,
                    graph_version=version,
                )
            original = pending_interaction.original_message or ""
            self.session.clear_pending_interaction()
            if no:
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            "The correction was not applied. The current cohort is unchanged.",
                            version,
                        ),
                    ),
                    graph_changed=False,
                    graph_version=version,
                    notifications=("correction_fallback_declined",),
                )
            parser = getattr(self.session.domain_agent, "parser", None)
            if parser is None or not hasattr(parser, "configure_correction"):
                raise RuntimeError("Full Regeneration fallback is unavailable for this agent")
            previous_mode = getattr(parser, "correction_mode", CorrectionMode.ROBUST_DELTA)
            parser.configure_correction(CorrectionMode.FULL_REGENERATION)
            try:
                result = self._handle_correction(
                    original,
                    original,
                    on_progress,
                    user_message=user,
                )
            finally:
                parser.configure_correction(previous_mode)
            return replace(
                result,
                notifications=(*result.notifications, "correction_full_regeneration_confirmed"),
            )

        if (
            pending_interaction.kind
            is PendingInteractionKind.CORRECTION_CLARIFICATION
        ):
            # The clarification response is a complete replacement correction,
            # not an amendment to the ambiguous text. Reinterpret it from scratch
            # against the same latest semantic base. Newer corrections remain
            # FIFO-queued until this replacement materializes its TaskSpec.
            version = state.version
            deferred = tuple(pending_interaction.deferred_corrections)
            self.session.clear_pending_interaction()
            result = self._handle_correction(
                response,
                response,
                on_progress,
                user_message=_message("user", response, version),
            )
            if not deferred:
                return result
            next_pending = self.session.state.pending_interaction
            if next_pending is not None:
                next_pending.deferred_corrections = (
                    *next_pending.deferred_corrections, *deferred
                )
                return result
            queued_result = self._apply_deferred_corrections_fifo(
                deferred, on_progress=on_progress
            )
            return InteractionResult(
                messages=(*result.messages, *queued_result.messages),
                graph_changed=(result.graph_changed or queued_result.graph_changed),
                graph_version=(queued_result.graph_version or result.graph_version),
                selected_node_id=(queued_result.selected_node_id or result.selected_node_id),
                notifications=(*result.notifications, *queued_result.notifications),
                error=queued_result.error or result.error,
            )

        pending = pending_interaction.resolution
        deferred_corrections = pending_interaction.deferred_corrections
        deferred_analyses = pending_interaction.deferred_correction_analyses
        if pending is None:
            raise RuntimeError(
                "semantic pending interaction has no resolution request"
            )
        user = _message("user", response, state.version)
        if self.semantic_resolver is None:
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        "No clarification resolver is configured.",
                        state.version,
                    ),
                ),
                error="No clarification resolver configured",
            )

        interpret_reply = getattr(
            self.semantic_resolver,
            "interpret_human_reply",
            None,
        )
        if interpret_reply is not None:
            try:
                parameters = inspect.signature(interpret_reply).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "on_progress" in parameters:
                reply = interpret_reply(
                    pending, response, on_progress=on_progress
                )
            else:
                reply = interpret_reply(pending, response)
            decision = _reply_to_resolution_decision(reply)
        else:
            reply = None
            resolve_answer = self.semantic_resolver.resolve_human_answer
            try:
                parameters = inspect.signature(resolve_answer).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "on_progress" in parameters:
                decision = resolve_answer(
                    pending, response, on_progress=on_progress
                )
            else:
                decision = resolve_answer(pending, response)

        if decision.action is ResolutionAction.ASK_HUMAN:
            if reply is not None and reply.has_residual_instruction:
                pending_interaction.deferred_corrections = (
                    *pending_interaction.deferred_corrections,
                    reply.residual_message or "",
                )
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        decision.question or pending.question,
                        state.version,
                    ),
                ),
                notifications=(
                    ("correction_deferred_until_resolution",)
                    if reply is not None and reply.has_residual_instruction
                    else ()
                ),
            )
        if decision.action is not ResolutionAction.SELECT_OPTION:
            # A user can correct away the very criterion being clarified. In
            # that case the pending question is stale: preserve its staged/
            # completed work as a reusable base, then run the full correction
            # pipeline instead of forcing an irrelevant option choice.
            if (
                reply is not None
                and reply.has_residual_instruction
                and InteractionAct.MODIFY_TASK in reply.acts
            ):
                correction = (reply.residual_message or response).strip()
                already_waiting = bool(pending_interaction.deferred_corrections)
                pending_interaction.deferred_corrections = (
                    *pending_interaction.deferred_corrections, correction
                )
                if (
                    self.session.runtime.execution_mode is ExecutionMode.PARALLEL
                    and not already_waiting
                ):
                    # Only the first correction may overlap the still-active
                    # manual action. Any later correction must wait until this
                    # one materializes an updated TaskSpec, preserving FIFO
                    # semantic ordering rather than analyzing against stale state.
                    analysis = self.session.analyze_correction(
                        correction, on_progress=on_progress
                    )
                    pending_interaction.deferred_correction_analyses = (analysis,)
                    note = "The correction has been interpreted in parallel and is waiting for this selection to finish."
                else:
                    note = "The correction is queued and will be interpreted after the earlier action/correction is materialized."
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            f"{note}\n\n{decision.question or pending.question}",
                            state.version,
                        ),
                    ),
                    graph_changed=False,
                    graph_version=state.version,
                    notifications=("correction_deferred_until_resolution",),
                )
            if response.strip().casefold() in {"cancel", "stop", "abort"}:
                if state.staged is not None:
                    self.session.discard_staged()
                else:
                    self.session.clear_pending_interaction()
                return InteractionResult(
                    messages=(
                        user,
                        _message(
                            "assistant",
                            "The pending interaction was cancelled.",
                            state.version,
                        ),
                    )
                )
            # A failed/unrecognized clarification must never destroy the staged
            # task. Keep the same pending resolution alive so the user can
            # refine the term, remove/replace the criterion, or cancel explicitly.
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        decision.question or (
                            "I couldn't safely map that reply to the current "
                            "candidates. Refine the term, remove/replace this "
                            "requirement, or type cancel."
                        ),
                        state.version,
                    ),
                ),
                graph_changed=False,
                graph_version=state.version,
            )

        options = tuple(
            item for item in pending.options
            if item.option_id in decision.selected_option_ids
        )
        if not options:
            raise RuntimeError(
                "resolution decision selected no known pending option"
            )
        user = _message("user", response, state.version)
        corrections = [item.strip() for item in deferred_corrections if item.strip()]
        if reply is not None and reply.has_residual_instruction:
            residual = (reply.residual_message or "").strip()
            if residual:
                corrections.append(residual)
        resolution_result = self._apply_pending_resolution(
            pending, options, decision.rationale, response,
            on_progress=on_progress,
            resume_execution=not bool(corrections),
        )
        if corrections:
            residual_result = self._apply_deferred_corrections_fifo(
                corrections,
                deferred_analyses,
                on_progress=on_progress,
            )
            return InteractionResult(
                messages=(
                    user,
                    *resolution_result.messages,
                    *residual_result.messages,
                ),
                graph_changed=(
                    resolution_result.graph_changed or residual_result.graph_changed
                ),
                graph_version=(
                    residual_result.graph_version or resolution_result.graph_version
                ),
                selected_node_id=(
                    residual_result.selected_node_id or resolution_result.selected_node_id
                ),
                notifications=(
                    *resolution_result.notifications, *residual_result.notifications
                ),
                error=residual_result.error,
            )

        return InteractionResult(
            messages=(user, *resolution_result.messages),
            graph_changed=resolution_result.graph_changed,
            graph_version=resolution_result.graph_version,
            selected_node_id=resolution_result.selected_node_id,
        )

    def _apply_deferred_corrections_fifo(
        self,
        corrections,
        preanalyzed=(),
        *,
        on_progress=None,
    ) -> InteractionResult:
        """Apply deferred corrections one at a time against the latest TaskSpec.

        At most the first correction may have been preinterpreted while a manual
        action was active. Every later correction is interpreted only after its
        predecessor has materialized semantic state. If a correction opens a new
        human gate, remaining corrections stay individually queued on that gate.
        """
        queue = [str(item).strip() for item in corrections if str(item).strip()]
        analyses = list(preanalyzed or ())
        messages = []
        notifications = []
        graph_changed = False
        graph_version = self.session.state.version if self.session.state else None
        selected_node_id = None
        error = None

        for index, correction in enumerate(queue):
            analysis = analyses[index] if index < len(analyses) else None
            if analysis is not None:
                result = self.handle_preanalyzed_correction(
                    correction, analysis, on_progress=on_progress
                )
            else:
                version = self.session.state.version if self.session.state else None
                result = self._handle_correction(
                    correction,
                    correction,
                    on_progress,
                    user_message=_message("user", correction, version),
                )
            # The correction text was already surfaced when it was queued. Only
            # append assistant-side continuation here.
            messages.extend(
                message for message in result.messages if message.role != "user"
            )
            notifications.extend(result.notifications)
            graph_changed = graph_changed or result.graph_changed
            graph_version = result.graph_version or graph_version
            selected_node_id = result.selected_node_id or selected_node_id
            error = result.error or error

            pending = (
                self.session.state.pending_interaction
                if self.session.state is not None else None
            )
            if pending is not None:
                remaining = tuple(queue[index + 1 :])
                if remaining:
                    pending.deferred_corrections = (
                        *pending.deferred_corrections, *remaining
                    )
                break

        return InteractionResult(
            messages=tuple(messages),
            graph_changed=graph_changed,
            graph_version=graph_version,
            selected_node_id=selected_node_id,
            notifications=tuple(notifications),
            error=error,
        )

    def submit_pending_resolution(
        self,
        option_ids: tuple[str, ...],
        *,
        on_progress=None,
    ) -> InteractionResult:
        """Apply a manually chosen selection (e.g. from a checkbox/dropdown
        widget) for the current pending resolution, bypassing free-text
        interpretation entirely. A parallel path to answering in the chat
        box with typed text -- both end up calling the same underlying
        session/resolver machinery.
        """
        state = self.session.state
        if state is None:
            raise RuntimeError("no active session")
        pending_interaction = state.pending_interaction
        if (
            pending_interaction is None
            or pending_interaction.kind is not PendingInteractionKind.SEMANTIC_RESOLUTION
        ):
            raise RuntimeError("no pending resolution to answer")
        pending = pending_interaction.resolution
        deferred_corrections = pending_interaction.deferred_corrections
        deferred_analyses = pending_interaction.deferred_correction_analyses
        if pending is None:
            raise RuntimeError(
                "semantic pending interaction has no resolution request"
            )
        ids = tuple(dict.fromkeys(option_ids))
        if not ids:
            raise ValueError("at least one option_id is required")
        options = tuple(
            item for item in pending.options if item.option_id in ids
        )
        unknown = set(ids) - {item.option_id for item in pending.options}
        if unknown:
            raise ValueError(f"unknown resolution option(s): {sorted(unknown)}")

        summary_text = ", ".join(option.label for option in options)
        user = _message(
            "user", f"(Manually selected: {summary_text})", state.version,
        )
        resolution_result = self._apply_pending_resolution(
            pending, options, "Manually selected by the user.", summary_text,
            on_progress=on_progress,
            resume_execution=not bool(deferred_corrections),
        )
        if deferred_corrections:
            correction_result = self._apply_deferred_corrections_fifo(
                [item.strip() for item in deferred_corrections if item.strip()],
                deferred_analyses,
                on_progress=on_progress,
            )
            return InteractionResult(
                messages=(
                    user,
                    *resolution_result.messages,
                    *correction_result.messages,
                ),
                graph_changed=(
                    resolution_result.graph_changed or correction_result.graph_changed
                ),
                graph_version=(
                    correction_result.graph_version or resolution_result.graph_version
                ),
                selected_node_id=(
                    correction_result.selected_node_id or resolution_result.selected_node_id
                ),
                notifications=(
                    *resolution_result.notifications, *correction_result.notifications
                ),
                error=correction_result.error,
            )
        return InteractionResult(
            messages=(user, *resolution_result.messages),
            graph_changed=resolution_result.graph_changed,
            graph_version=resolution_result.graph_version,
            selected_node_id=resolution_result.selected_node_id,
        )

    def _apply_pending_resolution(
        self,
        pending,
        options,
        rationale: str,
        response_text: str,
        *,
        on_progress=None,
        resume_execution: bool = True,
    ) -> InteractionResult:
        """Shared tail for resolving a pending semantic question, whatever
        chose the options (typed free text or a manual selection widget):
        record the selection, resume execution, and describe the result.
        """
        state = self.session.state
        pending.resolve(tuple(option.option_id for option in options), response_text)
        self.session.domain_agent.record_resolution(
            pending, options, rationale=rationale,
        )
        continuation = state.pending_interaction.continuation
        if not resume_execution:
            # The singular manual action is complete. Do not resume obsolete
            # graph work before a queued correction is interpreted. Retain the
            # recorded human selection in the resolver and mark this graph as a
            # stopped/reconcilable base for the correction.
            self.session.supersede_pending_for_correction()
            return InteractionResult(
                messages=(
                    _message(
                        "assistant",
                        "Selection recorded. Applying the queued correction before continuing execution.",
                        state.version,
                    ),
                ),
                graph_changed=False,
                graph_version=state.version,
                notifications=("resolution_recorded_before_correction",),
            )

        self.session.clear_pending_interaction()

        try:
            if continuation is ContinuationKind.INITIAL_EXECUTION:
                self.session.resume_initial_after_resolution(on_progress=on_progress)
            elif continuation is ContinuationKind.STAGED_EXECUTION:
                self.session.resume_after_resolution(on_progress=on_progress)
            elif continuation is ContinuationKind.INITIAL_NOTE_PREFLIGHT:
                self.session.resume_initial_note_preflight_after_resolution(
                    on_progress=on_progress
                )
                note_pending = self.session.request_note_execution_confirmation(
                    continuation=ContinuationKind.INITIAL_EXECUTION,
                )
                return InteractionResult(
                    messages=(
                        _message(
                            "assistant",
                            note_pending.question,
                            self.session.state.version,
                        ),
                    ),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                    notifications=("note_confirmation_required",),
                )
            elif continuation is ContinuationKind.STAGED_NOTE_PREFLIGHT:
                self.session.resume_staged_note_preflight_after_resolution(
                    on_progress=on_progress
                )
                note_pending = self.session.request_note_execution_confirmation(
                    continuation=ContinuationKind.STAGED_EXECUTION,
                )
                return InteractionResult(
                    messages=(
                        _message(
                            "assistant",
                            note_pending.question,
                            self.session.state.version,
                        ),
                    ),
                    graph_changed=True,
                    graph_version=self.session.state.version,
                    notifications=("note_confirmation_required",),
                )
            else:
                raise RuntimeError(
                    f"unsupported semantic continuation: {continuation}"
                )
        except ResolutionRequired:
            next_pending = self.session.state.pending_resolution
            return InteractionResult(
                messages=(
                    _message(
                        "assistant",
                        self.adapter.format_pending(next_pending),
                        self.session.state.version,
                    ),
                ),
                graph_changed=True,
                graph_version=self.session.state.version,
            )

        version = self.session.state.version
        resolution_message = _message(
            "assistant",
            self.adapter.format_resolution(
                options,
                self.session.summary(),
            ),
            version,
        )
        return InteractionResult(
            messages=(resolution_message,),
            graph_changed=True,
            graph_version=version,
            selected_node_id="deliver:summary",
        )

    def _handle_control(self, cleaned: str, on_progress=None) -> InteractionResult:
        state = self.session.state
        if state is None:
            return InteractionResult(error="No active session.")
        version = state.version
        user = _message("user", cleaned, version)
        command = cleaned.casefold()
        if command in {"stop", "cancel", "pause"}:
            self.session.request_stop()
            return InteractionResult(
                messages=(
                    user,
                    _message(
                        "assistant",
                        "Stop requested. Completed graph work will be preserved.",
                        version,
                    ),
                ),
                graph_changed=True,
                graph_version=version,
            )
        if command == "resume":
            try:
                execution = self.session.resume_stopped(on_progress=on_progress)
            except Exception as exc:
                return InteractionResult(
                    messages=(
                        user,
                        _message("assistant", f"I could not resume execution: {exc}", version),
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            if execution.stopped:
                return self._stopped_result(
                    user,
                    execution.completed_nodes,
                    self.session.state.version,
                    staged=self.session.state.staged is not None,
                )
            new_version = self.session.state.version
            if self.session.state.staged is None:
                content = self.adapter.format_initial(self.session.summary())
            else:
                content = "Execution resumed."
            return InteractionResult(
                messages=(user, _message("assistant", content, new_version)),
                graph_changed=True,
                graph_version=new_version,
                selected_node_id="deliver:summary",
            )
        return InteractionResult(error=f"Unsupported control command: {cleaned}")

    def _stopped_result(
        self,
        user_message,
        completed_nodes,
        version: int,
        *,
        staged: bool,
    ) -> InteractionResult:
        count = len(completed_nodes)
        scope = "staged correction" if staged else "current graph"
        messages = []
        if user_message is not None:
            messages.append(user_message)
        messages.append(
            _message(
                "assistant",
                (
                    f"Execution stopped. **{count} completed node(s)** in the {scope} "
                    "were preserved. The running node was not committed if the stop "
                    "arrived before it finished. You can resume or send another correction."
                ),
                version,
                metadata={"kind": "execution_stopped"},
            )
        )
        return InteractionResult(
            messages=tuple(messages),
            graph_changed=True,
            graph_version=version,
            notifications=("execution_stopped",),
        )

    def _attach_timing_diagnostics(
        self,
        result: InteractionResult,
        report: dict,
    ) -> InteractionResult:
        """Attach a diagnostic timing report without changing user-facing text."""
        state = self.session.state
        graph_nodes: list[dict] = []
        reused_nodes: set[str] = set()
        graph_source = "committed"
        if state is not None:
            # During a correction that pauses for semantic resolution, execution
            # happens on state.staged.graph. Reading state.graph here reports the
            # previous committed graph and can make runtime/node timings appear
            # wildly inconsistent. Prefer the staged graph while it exists. After
            # a successful commit state.staged is cleared and state.graph is the
            # correct source again.
            if state.staged is not None:
                diagnostic_graph = state.staged.graph
                reuse_report = state.staged.reuse_report
                graph_source = "staged"
            else:
                diagnostic_graph = state.graph
                reuse_report = state.last_reuse_report

            if reuse_report is not None:
                reused_nodes = set(reuse_report.reused_nodes)
            for node_id in diagnostic_graph.topological_order():
                node = diagnostic_graph.nodes[node_id]
                if node.operator_name is None:
                    continue
                graph_nodes.append({
                    "node_id": node.node_id,
                    "operator": node.operator_name,
                    "status": node.status.value,
                    "reused": node.node_id in reused_nodes,
                    "duration_ms": (
                        None if node.duration_ms is None
                        else round(node.duration_ms, 3)
                    ),
                    "component_timings_ms": {
                        key: round(value, 3)
                        for key, value in sorted(
                            node.component_timings_ms.items()
                        )
                    },
                    "attempt_timings": [
                        dict(item) for item in node.attempt_timings
                    ],
                })

        totals = report.get("totals_by_name_ms", {})
        semantic_ms = sum(
            value for name, value in totals.items()
            if name.startswith("semantic.")
        )
        llm_ms = sum(
            value for name, value in totals.items()
            if name.startswith("llm.")
        )
        executed_nodes = [
            item for item in graph_nodes
            if item["duration_ms"] is not None and not item["reused"]
        ]
        reused_timed_nodes = [
            item for item in graph_nodes
            if item["duration_ms"] is not None and item["reused"]
        ]
        report["summary"] = {
            "interaction_wall_ms": totals.get(
                "interaction.total", report.get("total_wall_ms", 0.0)
            ),
            "semantic_stage_ms": round(semantic_ms, 3),
            "llm_generation_ms": round(llm_ms, 3),
            "graph_execution_wall_ms": round(
                totals.get("execution.runtime", 0.0), 3
            ),
            "graph_node_compute_sum_ms": round(
                sum(item["duration_ms"] or 0.0 for item in executed_nodes),
                3,
            ),
            "reused_node_count": len(reused_nodes),
            "reused_node_compute_ms": round(
                sum(item["duration_ms"] or 0.0 for item in reused_timed_nodes),
                3,
            ),
            "executed_node_count": len(executed_nodes),
        }
        report["graph_nodes"] = graph_nodes
        report["graph_source"] = graph_source

        internal_ms = sum(
            value for name, value in totals.items()
            if name.startswith("execution.")
            and name != "execution.runtime"
        )
        node_attempt_ms = sum(
            sum(
                float(attempt.get("duration_ms", 0.0) or 0.0)
                for attempt in item.get("attempt_timings", [])
            )
            for item in graph_nodes
            if not item["reused"]
        )
        graph_wall_ms = float(
            report["summary"].get("graph_execution_wall_ms", 0.0) or 0.0
        )
        # For the default single-worker runtime, this is a useful sanity check:
        # graph wall time should be almost entirely explained by node attempts
        # plus scheduler/commit/finalization overhead. In parallel mode these
        # quantities overlap, so the remainder is not interpreted as missing work.
        if self.session.max_workers <= 1:
            accounted_ms = node_attempt_ms + internal_ms
            unattributed_ms = max(0.0, graph_wall_ms - accounted_ms)
        else:
            accounted_ms = None
            unattributed_ms = None
        report["summary"].update({
            "graph_node_attempt_sum_ms": round(node_attempt_ms, 3),
            "graph_internal_overhead_sum_ms": round(internal_ms, 3),
            "graph_accounted_ms": (
                None if accounted_ms is None else round(accounted_ms, 3)
            ),
            "graph_unattributed_ms": (
                None if unattributed_ms is None else round(unattributed_ms, 3)
            ),
        })
        report["note"] = (
            "Nested spans overlap. Do not sum all pipeline spans. "
            f"Graph/operator rows are read from the {graph_source} graph. "
            "Failed or resolution-required operator attempts are retained. "
            "With one worker, graph_unattributed_ms should be close to zero. "
            "parallel node attempts can overlap in wall time."
        )

        messages = list(result.messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "assistant":
                metadata = dict(message.metadata)
                metadata["timing_report"] = report
                messages[index] = replace(message, metadata=metadata)
                break
        return replace(
            result,
            messages=tuple(messages),
            diagnostics={**result.diagnostics, "timing": report},
        )

    @staticmethod
    def _emit(callback, text: str) -> None:
        if callback is not None:
            callback("stage", text)


def _reply_to_resolution_decision(
    reply: PendingReplyInterpretation,
):
    from framework.resolution import ResolutionDecision

    action_map = {
        "select_candidate": ResolutionAction.SELECT_OPTION,
        "ask_human": ResolutionAction.ASK_HUMAN,
        "abort": ResolutionAction.ABORT,
    }
    return ResolutionDecision(
        action=action_map[reply.action],
        confidence=reply.confidence,
        rationale=reply.rationale,
        option_id=reply.option_id,
        option_ids=reply.option_ids,
        question=reply.question,
    )
