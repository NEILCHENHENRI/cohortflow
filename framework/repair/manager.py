from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from threading import RLock

from framework.failure import FailureEvent
from framework.repair.locator import RepairContextBuilder
from framework.repair.models import (
    RecoveryHooks, RecoveryCheck, VerificationStatus, RepairScope,
    FailureAction,
    FailureAssessment,
    RepairBackend,
    RepairContext,
    RepairMode,
    RepairOutcome,
    RepairPolicy,
    RepairProposal,
    RepairStatus,
    ValidationReport,
)
from framework.repair.validator import RepairValidator
from framework.repair.workspace import RepairWorkspace, RepairWorkspaceError


class RepairManager:
    """Analyze, patch and validate unhandled failures outside the live tree."""

    def __init__(
        self,
        *,
        repository_root: str,
        local_backend: RepairBackend | None = None,
        cloud_backend: RepairBackend | None = None,
        policy: RepairPolicy | None = None,
    ) -> None:
        self.repository_root = str(Path(repository_root).resolve())
        self.context_builder = RepairContextBuilder(self.repository_root)
        self.local_backend = local_backend
        self.cloud_backend = cloud_backend
        self.policy = policy or RepairPolicy()
        self._recovery_lock = RLock()
        self._workspaces: dict[str, RepairWorkspace] = {}
        self._retry_counts: dict[tuple[str, str], int] = {}
        self._validated: dict[str, tuple[RepairOutcome, str]] = {}

    def configure(self, policy: RepairPolicy, *, cloud_backend: RepairBackend | None = None) -> None:
        self.policy = policy
        self.cloud_backend = cloud_backend

    def repair(self, failure: FailureEvent, *, hooks: RecoveryHooks | None = None) -> RepairOutcome:
        # Parallel graph branches share one budget/workspace registry.
        with self._recovery_lock:
            return self._repair_guarded(failure, hooks=hooks)

    def _repair_guarded(self, failure, *, hooks):
        if self.policy.mode is RepairMode.OFF:
            return RepairOutcome(RepairStatus.DISABLED, failure, message="Self repair is disabled.")
        token = hooks.snapshot() if hooks else None
        outcome = self._repair(failure, hooks=hooks)
        if hooks and hooks.snapshot() != token:
            self.discard(failure.failure_id)
            return RepairOutcome(RepairStatus.STOPPED, failure, state_token=token,
                message="Task or operation changed during recovery. The candidate cannot be used.")
        outcome = replace(outcome, state_token=token)
        if outcome.status is RepairStatus.VALIDATED:
            workspace = self._workspaces[failure.failure_id]
            self._validated[failure.failure_id] = (outcome, workspace.fingerprint())
        return outcome

    @staticmethod
    def _rule_assessment(failure: FailureEvent) -> FailureAssessment | None:
        if (failure.exception_type in {"TimeoutError", "ConnectionError", "ConnectionResetError", "ConnectionRefusedError"}
                or failure.metadata.get("transient_transport_failure") is True
                or failure.status_code in {408, 429, 502, 503, 504}):
            return FailureAssessment(FailureAction.RETRY, RepairScope.RUNTIME, 1.0,
                "Temporary transport/service failure", "Retry only if the host declares the operation safe.")
        if failure.exception_type in {"DomainValidationError", "FileNotFoundError"} or failure.status_code in {401, 403}:
            return FailureAssessment(FailureAction.ASK_USER, RepairScope.NONE, 1.0,
                "Input, configuration or access needs attention", "Please correct the reported input, missing resource or access configuration before continuing.")
        if failure.exception_type == "UnsupportedCriterionError":
            return FailureAssessment(FailureAction.STOP, RepairScope.NONE, 1.0,
                "Operation is unsupported", "The requested criterion must remain unchanged. Report the limitation.")
        return None

    def _repair(self, failure: FailureEvent, *, hooks: RecoveryHooks | None = None) -> RepairOutcome:
        if self.policy.mode is RepairMode.OFF:
            return RepairOutcome(
                RepairStatus.DISABLED,
                failure,
                message="Self repair is disabled.",
            )
        assessment = (hooks.assessment if hooks else None) or self._rule_assessment(failure)
        if assessment is not None:
            context = RepairContext(failure, self.repository_root, recovery_hooks=hooks,
                                    task_context=deepcopy(hooks.task_context) if hooks else {})
            if assessment.action not in {FailureAction.LOCAL_REPAIR, FailureAction.CLOUD_ESCALATE}:
                return self._non_patch(context, assessment, backend="rules")
        if self.local_backend is None:
            return RepairOutcome(
                RepairStatus.ERROR,
                failure,
                message="No local repair backend is configured.",
            )

        try:
            context = replace(self.context_builder.build(failure), recovery_hooks=hooks,
                              task_context=deepcopy(hooks.task_context) if hooks else {})
        except Exception as exc:
            return self._error(failure, self.local_backend.name, exc)
        try:
            assessment = assessment or self.local_backend.analyze(context)
        except Exception as exc:
            return self._local_error(context, exc)

        if assessment.action not in {
            FailureAction.LOCAL_REPAIR,
            FailureAction.CLOUD_ESCALATE,
        }:
            return self._non_patch(context, assessment, backend=self.local_backend.name)

        if assessment.scope not in {RepairScope.SOURCE, RepairScope.CONFIG}:
            return RepairOutcome(RepairStatus.STOPPED, failure, assessment=assessment,
                message="A source patch requires a diagnosed implementation/configuration defect.")

        if assessment.action is FailureAction.CLOUD_ESCALATE:
            if self.policy.mode is RepairMode.LOCAL_THEN_CLOUD and self.cloud_backend is not None:
                return self._cloud_attempt(context, assessment, prior=None)
            return RepairOutcome(
                RepairStatus.ANALYZED,
                failure,
                assessment=assessment,
                backend=self.local_backend.name,
                message="Failure analyst recommends cloud escalation, but cloud repair is not enabled.",
            )

        try:
            local = self.local_backend.propose(context, assessment)
        except Exception as exc:
            return self._local_error(context, exc)
        local_outcome = self._validate_attempt(context, assessment, local, escalated=False)
        if local_outcome.status is RepairStatus.VALIDATED:
            return local_outcome

        if self.policy.mode is RepairMode.LOCAL_THEN_CLOUD and self.cloud_backend is not None:
            prior = self._attempt_summary(local_outcome)
            cloud_context = replace(context, prior_attempt=prior)
            return self._cloud_attempt(cloud_context, assessment, prior=local_outcome)
        return local_outcome

    def _non_patch(self, context, assessment, *, backend, escalated=False):
        if assessment.action is FailureAction.RETRY:
            return self._retry(context, assessment, backend=backend, escalated=escalated)
        status = RepairStatus.ANALYZED
        if assessment.action is FailureAction.ASK_USER:
            status = RepairStatus.NEEDS_INPUT
        elif assessment.action is FailureAction.STOP:
            status = RepairStatus.UNSUPPORTED if context.failure.exception_type == "UnsupportedCriterionError" else RepairStatus.STOPPED
        message = assessment.rationale or assessment.diagnosis
        if assessment.action is FailureAction.ADAPT_RUNTIME:
            message += " Runtime/configuration changes require explicit host support. No automatic change was made."
        return RepairOutcome(status, context.failure, assessment=assessment, backend=backend,
                             escalated=escalated, message=message)

    def _retry(self, context, assessment, *, backend, escalated):
        from framework.steering import SteeringYield
        from framework.resolution import ResolutionRequired
        hooks = context.recovery_hooks
        def stopped(message, attempts=0, verification=VerificationStatus.UNVERIFIED):
            return RepairOutcome(RepairStatus.STOPPED, context.failure, assessment=assessment,
                backend=backend, escalated=escalated, message=message, retry_attempts=attempts, verification=verification)
        if hooks is None or hooks.safe_to_retry is not True or hooks.retry is None:
            return stopped("Retry requires a host-provided safe operation. No operation was repeated.")
        token = hooks.snapshot()
        key = (hooks.operation_key, token)
        attempts = self._retry_counts.get(key, 0)
        while attempts < self.policy.max_runtime_retries:
            if hooks.snapshot() != token:
                return stopped("Task changed before retry. Recovery stopped.", attempts)
            attempts += 1
            self._retry_counts[key] = attempts
            try:
                check = hooks.retry()
                if not isinstance(check, RecoveryCheck):
                    raise TypeError("retry hook must return RecoveryCheck")
                if hooks.snapshot() != token or not check.task_preserved:
                    return stopped("Task changed during retry. Result rejected.", attempts, VerificationStatus.FAILED)
                if check.succeeded:
                    return RepairOutcome(RepairStatus.RECOVERED, context.failure, assessment=assessment,
                        backend=backend, escalated=escalated, retry_attempts=attempts,
                        verification=VerificationStatus.RESOLVED, recovery_value=check.value,
                        message="Failed operation completed on safe retry and passed its output contract. Execution can continue.")
                return stopped(check.message or "Retry did not pass its output contract.", attempts, VerificationStatus.FAILED)
            except (InterruptedError, SteeringYield, ResolutionRequired):
                raise
            except Exception as exc:
                latest = FailureEvent.capture(exc, origin="recovery.retry")
                transient = self._rule_assessment(latest)
                if transient is None or transient.action is not FailureAction.RETRY:
                    return stopped(f"Retry encountered a different failure: {type(exc).__name__}: {exc}", attempts, VerificationStatus.FAILED)
        return stopped("Safe retry limit reached. Execution remains stopped.", attempts, VerificationStatus.FAILED)

    # Compatibility for callers that only display a repair proposal.
    def propose(self, failure: FailureEvent) -> RepairProposal | None:
        outcome = self.repair(failure)
        return outcome.proposal

    def persist(self, outcome: RepairOutcome, *, branch_name: str | None = None) -> RepairOutcome:
        if outcome.status is not RepairStatus.VALIDATED or not outcome.workspace_path:
            raise ValueError("Only validated repairs can be persisted")
        workspace = self._workspaces.get(outcome.failure.failure_id)
        if workspace is None:
            raise ValueError("Validated repair workspace is no longer available")
        validated = self._validated.get(outcome.failure.failure_id)
        if validated is None or validated[0] != outcome or str(workspace.path) != outcome.workspace_path:
            raise ValueError("Repair outcome is stale. Only the latest validated candidate can be saved")
        if workspace.fingerprint() != validated[1]:
            raise ValueError("Repair workspace changed after validation. Revalidation is required")
        if outcome.persistent_commit:
            return outcome
        branch = branch_name or (
            f"codex/repair/{outcome.failure.failure_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        saved_branch, commit = workspace.save_branch(
            branch,
            message=f"Validated self-repair for {outcome.failure.failure_id}",
            changed_files=outcome.validation.changed_files if outcome.validation else (),
        )
        saved = replace(
            outcome,
            persistent_branch=saved_branch,
            persistent_commit=commit,
            message=f"Saved repair as {saved_branch} ({commit[:10]}). Review and adopt the patch, restart the process, and use a fresh session before continuing.",
        )
        self._validated[outcome.failure.failure_id] = (saved, workspace.fingerprint())
        return saved

    def discard(self, failure_id: str) -> None:
        self._validated.pop(failure_id, None)
        workspace = self._workspaces.pop(failure_id, None)
        if workspace is not None:
            workspace.cleanup()

    def close(self) -> None:
        """Release session-owned worktrees. Persisted branches remain in Git."""
        with self._recovery_lock:
            for failure_id in tuple(self._workspaces):
                self.discard(failure_id)
            self._retry_counts.clear()

    @staticmethod
    def _error(failure, backend, exc, *, escalated=False):
        return RepairOutcome(
            RepairStatus.ERROR, failure, backend=backend, escalated=escalated,
            message=f"Repair backend failed: {type(exc).__name__}: {exc}",
        )

    def _local_error(self, context, exc):
        if self.policy.mode is RepairMode.LOCAL_THEN_CLOUD and self.cloud_backend is not None:
            context = replace(context, prior_attempt=f"Local backend failed: {type(exc).__name__}: {exc}")
            return self._cloud_attempt(context, None, prior=None)
        return self._error(context.failure, self.local_backend.name, exc)

    def _cloud_attempt(self, context, local_assessment, *, prior):
        try:
            return self._run_cloud_attempt(context, local_assessment, prior=prior)
        except Exception as exc:
            return self._error(context.failure, self.cloud_backend.name, exc, escalated=True)

    def _run_cloud_attempt(
        self,
        context: RepairContext,
        local_assessment: FailureAssessment,
        *,
        prior: RepairOutcome | None,
    ) -> RepairOutcome:
        assert self.cloud_backend is not None
        cloud_assessment = self.cloud_backend.analyze(context)
        # Cloud may conclude that source repair is inappropriate. Respect that.
        if cloud_assessment.action not in {
            FailureAction.LOCAL_REPAIR,
            FailureAction.CLOUD_ESCALATE,
        }:
            return self._non_patch(context, cloud_assessment, backend=self.cloud_backend.name, escalated=True)
        if cloud_assessment.scope not in {RepairScope.SOURCE, RepairScope.CONFIG}:
            return RepairOutcome(RepairStatus.STOPPED, context.failure, assessment=cloud_assessment,
                escalated=True, message="Cloud diagnosis did not establish a source/configuration defect.")
        proposal = self.cloud_backend.propose(context, cloud_assessment)
        return self._validate_attempt(
            context,
            cloud_assessment,
            proposal,
            escalated=True,
        )

    def _validate_attempt(
        self,
        context: RepairContext,
        assessment: FailureAssessment,
        proposal: RepairProposal,
        *,
        escalated: bool,
    ) -> RepairOutcome:
        validator = RepairValidator(self.policy)
        try:
            workspace = RepairWorkspace.create(self.repository_root)
        except Exception as exc:
            return RepairOutcome(
                RepairStatus.ERROR,
                context.failure,
                assessment=assessment,
                proposal=proposal,
                backend=proposal.backend,
                escalated=escalated,
                message=f"Could not create isolated repair workspace: {exc}",
            )

        try:
            # Scope is inspected by the validator before any patch is applied.
            scope = validator._scope(proposal.unified_diff)
            if scope.rejection_reason:
                workspace.cleanup()
                return RepairOutcome(
                    RepairStatus.REJECTED,
                    context.failure,
                    assessment=assessment,
                    proposal=proposal,
                    validation=scope,
                    backend=proposal.backend,
                    escalated=escalated,
                    message=scope.rejection_reason,
                )
            workspace.validate_paths(scope.changed_files)
            snapshot_fingerprint = workspace.fingerprint()
            baseline = validator.capture_baseline(workspace, context)
            if workspace.fingerprint() != snapshot_fingerprint:
                raise RepairWorkspaceError("baseline tests modified the isolated source snapshot")
            hooks = context.recovery_hooks
            reproduction = None
            if hooks and hooks.replay:
                reproduction = hooks.replay(str(workspace.path), "baseline")
                if (not isinstance(reproduction, RecoveryCheck) or reproduction.succeeded
                        or not reproduction.reproduced_failure or not reproduction.task_preserved):
                    raise RepairWorkspaceError("Replay did not reproduce the original failure while preserving the task")
                if workspace.fingerprint() != snapshot_fingerprint:
                    raise RepairWorkspaceError("baseline replay modified source files")
            workspace.apply_diff(proposal.unified_diff)
            candidate_fingerprint = workspace.fingerprint()
            report = validator.validate(workspace, context, proposal.unified_diff, baseline=baseline)
            if workspace.fingerprint() != candidate_fingerprint:
                raise RepairWorkspaceError("validation modified the candidate workspace")
            if not report.passed:
                workspace.cleanup()
                return RepairOutcome(
                    RepairStatus.REJECTED,
                    context.failure,
                    assessment=assessment,
                    proposal=proposal,
                    validation=report,
                    backend=proposal.backend,
                    escalated=escalated,
                    message=report.rejection_reason or "Patch validation failed.",
                )
            replay = None
            if hooks and hooks.replay:
                replay = hooks.replay(str(workspace.path), "candidate")
                if not isinstance(replay, RecoveryCheck) or not replay.succeeded or not replay.task_preserved:
                    raise RepairWorkspaceError("Candidate replay did not resolve the failure with the task preserved")
                if workspace.fingerprint() != candidate_fingerprint:
                    raise RepairWorkspaceError("candidate replay modified source files")
            report = replace(report, reproduction=reproduction, replay=replay)
            self.discard(context.failure.failure_id)
            self._workspaces[context.failure.failure_id] = workspace
            outcome = RepairOutcome(
                RepairStatus.VALIDATED,
                context.failure,
                assessment=assessment,
                proposal=proposal,
                validation=report,
                workspace_path=str(workspace.path),
                backend=proposal.backend,
                escalated=escalated,
                verification=VerificationStatus.RESOLVED if replay else VerificationStatus.UNVERIFIED,
                message=("Checks passed. Isolated replay verified the failed operation. Review, adopt and reload before continuing."
                         if replay else "Checks passed. Original failure remains unverified (no replay hook). Review, adopt and reload before continuing."),
            )
            self._validated[context.failure.failure_id] = (outcome, candidate_fingerprint)
            return outcome
        except RepairWorkspaceError as exc:
            workspace.cleanup()
            return RepairOutcome(
                RepairStatus.REJECTED,
                context.failure,
                assessment=assessment,
                proposal=proposal,
                validation=ValidationReport(False, rejection_reason=str(exc)),
                backend=proposal.backend,
                escalated=escalated,
                message=str(exc),
            )
        except Exception as exc:
            workspace.cleanup()
            return RepairOutcome(
                RepairStatus.ERROR,
                context.failure,
                assessment=assessment,
                proposal=proposal,
                backend=proposal.backend,
                escalated=escalated,
                message=f"Repair validation crashed: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _attempt_summary(outcome: RepairOutcome) -> str:
        parts = [f"Local repair status: {outcome.status.value}", outcome.message]
        if outcome.proposal is not None:
            parts.append("Local diagnosis: " + outcome.proposal.diagnosis)
            parts.append("Local patch:\n" + outcome.proposal.unified_diff[-8000:])
        if outcome.validation is not None:
            parts.append("Validation: " + ("passed" if outcome.validation.passed else "failed"))
            for step in outcome.validation.steps:
                parts.append(f"{step.name}: {'passed' if step.passed else 'failed'}\n{step.output[-4000:]}")
        return "\n\n".join(parts)
