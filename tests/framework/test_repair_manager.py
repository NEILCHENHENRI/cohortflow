from pathlib import Path
import subprocess
import pytest
from unittest.mock import Mock
from framework.steering import SteeringYield
from framework.repair.models import RecoveryCheck, RecoveryHooks, VerificationStatus, ValidationReport

from framework.failure import FailureEvent
from framework.repair.manager import RepairManager
from framework.repair.models import (
    FailureAction,
    FailureAssessment,
    RepairMode,
    RepairPolicy,
    RepairProposal,
    RepairScope,
    RepairStatus,
)


class Backend:
    def __init__(self, name, *, action=FailureAction.LOCAL_REPAIR, diff=""):
        self.name = name
        self.action = action
        self.diff = diff
        self.analyze_calls = 0
        self.propose_calls = 0

    def analyze(self, context):
        self.analyze_calls += 1
        return FailureAssessment(
            action=self.action,
            scope=RepairScope.SOURCE,
            confidence=0.8,
            diagnosis="small bug",
            rationale="traceback points to x.py",
            likely_files=("x.py",),
        )

    def propose(self, context, assessment):
        self.propose_calls += 1
        return RepairProposal(
            backend=self.name,
            diagnosis="bug",
            confidence=0.8,
            summary="fix it",
            unified_diff=self.diff,
        )


def _failure():
    try:
        raise AttributeError("broken")
    except AttributeError as exc:
        return FailureEvent.capture(exc, origin="test")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "x.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "x.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "base"],
        cwd=root,
        check=True,
    )
    return root


GOOD_DIFF = """--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
BAD_DIFF = """--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = (
"""


def test_repair_off_does_not_call_models(tmp_path):
    local = Backend("local", diff=GOOD_DIFF)
    manager = RepairManager(repository_root=_repo(tmp_path), local_backend=local)
    outcome = manager.repair(_failure())
    assert outcome.status is RepairStatus.DISABLED
    assert local.analyze_calls == 0


def test_local_repair_is_applied_only_in_isolated_workspace_and_validated(tmp_path):
    root = _repo(tmp_path)
    local = Backend("local", diff=GOOD_DIFF)
    manager = RepairManager(
        repository_root=root,
        local_backend=local,
        policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False),
    )
    outcome = manager.repair(_failure())
    assert outcome.status is RepairStatus.VALIDATED
    assert outcome.validation and outcome.validation.passed
    assert Path(outcome.workspace_path, "x.py").read_text() == "VALUE = 2\n"
    assert (root / "x.py").read_text() == "VALUE = 1\n"  # live tree untouched
    manager.discard(outcome.failure.failure_id)


def test_failed_local_validation_escalates_to_cloud(tmp_path):
    root = _repo(tmp_path)
    local = Backend("local", diff=BAD_DIFF)
    cloud = Backend("cloud", diff=GOOD_DIFF)
    manager = RepairManager(
        repository_root=root,
        local_backend=local,
        cloud_backend=cloud,
        policy=RepairPolicy(mode=RepairMode.LOCAL_THEN_CLOUD, run_full_suite=False),
    )
    outcome = manager.repair(_failure())
    assert outcome.status is RepairStatus.VALIDATED
    assert outcome.backend == "cloud"
    assert outcome.escalated
    assert local.propose_calls == 1 and cloud.propose_calls == 1
    manager.discard(outcome.failure.failure_id)


def test_analyst_can_decline_source_repair(tmp_path):
    local = Backend("local", action=FailureAction.ASK_USER, diff=GOOD_DIFF)
    manager = RepairManager(
        repository_root=_repo(tmp_path),
        local_backend=local,
        policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False),
    )
    outcome = manager.repair(_failure())
    assert outcome.status is RepairStatus.NEEDS_INPUT
    assert local.propose_calls == 0


@pytest.mark.parametrize("branch_name", ["repair/test-fix", None])
def test_validated_repair_can_be_saved_as_branch_without_touching_live_tree(tmp_path, branch_name):
    root = _repo(tmp_path)
    manager = RepairManager(
        repository_root=root,
        local_backend=Backend("local", diff=GOOD_DIFF),
        policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False),
    )
    outcome = manager.repair(_failure())
    saved = manager.persist(outcome, branch_name=branch_name)
    if branch_name:
        assert saved.persistent_branch == branch_name
    else:
        assert saved.persistent_branch.startswith("codex/repair/")
    assert saved.persistent_commit
    assert (root / "x.py").read_text() == "VALUE = 1\n"
    branches = subprocess.run(
        ["git", "branch", "--list", saved.persistent_branch], cwd=root, text=True, capture_output=True, check=True
    ).stdout
    assert saved.persistent_branch in branches
    workspace = Path(saved.workspace_path)
    manager.close()
    assert not workspace.exists()
    assert not manager._workspaces


@pytest.mark.parametrize("diff", [
    GOOD_DIFF + "--- a/tests/test_x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-assert False\n",
    "--- a/tests/test_x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
    GOOD_DIFF.replace("a/x.py", "a/../x.py"),
    GOOD_DIFF.replace("x.py", ".git/config"),
    GOOD_DIFF.replace("x.py", "conftest.py"),
    "diff --git a/x.py b/x.py\nold mode 100644\nnew mode 120000\n" + GOOD_DIFF,
    GOOD_DIFF.replace("@@ -1 +1 @@", "@@ -3 +3 @@").replace("-VALUE = 1\n", ""),
    GOOD_DIFF + "diff --git a/hidden.py b/hidden.py\nnew file mode 100644\nindex 0000000..e69de29\n",
])
def test_patch_scope_rejects_protected_deletions_renames_modes_and_bad_hunks(diff):
    from framework.repair.validator import RepairValidator
    report = RepairValidator(RepairPolicy())._scope(diff)
    assert not report.passed
    assert report.rejection_reason


def test_patch_scope_does_not_confuse_source_text_with_file_headers():
    from framework.repair.validator import RepairValidator
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
    report = RepairValidator(RepairPolicy())._scope(diff)
    assert report.passed
    assert report.changed_files == ("x.py",)
    assert report.changed_lines == 2


def test_timeout_with_bytes_output_is_a_failed_step(tmp_path, monkeypatch):
    from framework.repair.validator import RepairValidator
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial stdout", stderr=b"partial stderr")
    monkeypatch.setattr(subprocess, "run", timeout)
    step = RepairValidator(RepairPolicy())._run(tmp_path, "test", ["fake-command"])
    assert not step.passed
    assert "Timed out" in step.output and "partial stdout" in step.output


def test_local_backend_failure_is_reported_and_can_escalate(tmp_path):
    class BrokenBackend(Backend):
        def analyze(self, context):
            raise RuntimeError("model unavailable")
    root = _repo(tmp_path)
    manager = RepairManager(repository_root=root, local_backend=BrokenBackend("local"),
                            policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False))
    failed = manager.repair(_failure())
    assert failed.status is RepairStatus.ERROR
    assert "model unavailable" in failed.message
    manager.configure(RepairPolicy(mode=RepairMode.LOCAL_THEN_CLOUD, run_full_suite=False),
                      cloud_backend=Backend("cloud", diff=GOOD_DIFF))
    repaired = manager.repair(_failure())
    assert repaired.status is RepairStatus.VALIDATED and repaired.escalated
    manager.discard(repaired.failure.failure_id)


def test_full_suite_accepts_existing_failure_but_rejects_new_regression(tmp_path):
    root = _repo(tmp_path)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "from x import VALUE\n"
        "def test_bug(): assert VALUE == 2\n"
        "def test_existing_failure(): assert False\n"
    )
    manager = RepairManager(repository_root=root, local_backend=Backend("local", diff=GOOD_DIFF),
                            policy=RepairPolicy(mode=RepairMode.LOCAL))
    outcome = manager.repair(_failure())
    try:
        assert outcome.status is RepairStatus.VALIDATED, outcome.message
        baseline = outcome.validation.baseline_steps[0]
        candidate = outcome.validation.steps[-1]
        assert baseline.return_code == candidate.return_code == 1
        assert candidate.passed
        assert sum(status == "failure" for _, status in baseline.test_results) == 2
        assert sum(status == "failure" for _, status in candidate.test_results) == 1
    finally:
        manager.discard(outcome.failure.failure_id)
    (tests / "test_x.py").write_text("from x import VALUE\ndef test_regression(): assert VALUE == 1\n")
    failed = manager.repair(_failure())
    assert failed.status is RepairStatus.REJECTED
    assert "test_regression" in failed.validation.steps[-1].output


def test_baseline_comparison_rejects_missing_or_newly_skipped_tests():
    from framework.repair.models import ValidationStep
    from framework.repair.validator import RepairValidator
    baseline = ValidationStep("tests", True, return_code=0, test_results=(("a", "passed"), ("b", "passed")))
    for results in [(("a", "passed"),), (("a", "passed"), ("b", "skipped"))]:
        after = ValidationStep("tests", True, return_code=0, test_results=results)
        assert not RepairValidator._compare_tests(baseline, after).passed


def test_duplicate_test_identities_cannot_hide_failures(tmp_path, monkeypatch):
    from framework.repair.models import ValidationStep
    from framework.repair.validator import RepairValidator

    validator = RepairValidator(RepairPolicy())

    def run(cwd, name, command):
        report = next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml="))
        Path(report).write_text(
            '<testsuite><testcase classname="a" name="same"><failure/></testcase>'
            '<testcase classname="a" name="same"/></testsuite>'
        )
        return ValidationStep(name, False, return_code=1)

    monkeypatch.setattr(validator, "_run", run)
    result = validator._run_tests(tmp_path, "tests", ("tests",))
    assert not result.passed
    assert not result.test_results
    assert "duplicate" in result.output


def test_workspace_fingerprint_detects_permission_changes(tmp_path):
    from framework.repair.workspace import RepairWorkspace

    source = tmp_path / "x.py"
    source.write_text("VALUE = 1\n")
    workspace = RepairWorkspace(tmp_path, tmp_path, "archive", is_git_worktree=False)
    before = workspace.fingerprint()
    source.chmod(source.stat().st_mode ^ 0o100)
    assert workspace.fingerprint() != before


def test_persist_rejects_modified_and_stale_validated_workspaces(tmp_path):
    root = _repo(tmp_path)
    manager = RepairManager(repository_root=root, local_backend=Backend("local", diff=GOOD_DIFF),
                            policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False))
    failure = _failure()
    first = manager.repair(failure)
    Path(first.workspace_path, "x.py").write_text("VALUE = 999\n")
    with pytest.raises(ValueError, match="changed after validation"):
        manager.persist(first)
    second = manager.repair(failure)
    try:
        with pytest.raises(ValueError, match="stale"):
            manager.persist(first)
        assert Path(second.workspace_path, "x.py").read_text() == "VALUE = 2\n"
    finally:
        manager.discard(failure.failure_id)


def test_persist_keeps_dirty_snapshot_separate_from_repair_commit(tmp_path):
    root = _repo(tmp_path)
    (root / "x.py").write_text("VALUE = 10\n")
    (root / "helper.py").write_text("HELPER = True\n")
    diff = GOOD_DIFF.replace("-VALUE = 1\n", "-VALUE = 10\n")
    manager = RepairManager(repository_root=root, local_backend=Backend("local", diff=diff),
                            policy=RepairPolicy(mode=RepairMode.LOCAL, run_full_suite=False))
    outcome = manager.repair(_failure())
    try:
        assert outcome.status is RepairStatus.VALIDATED, outcome.message
        saved = manager.persist(outcome, branch_name="repair/snapshot")
        changed = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", saved.persistent_commit], cwd=root, text=True)
        assert changed.strip() == "x.py"
        assert subprocess.check_output(["git", "show", saved.persistent_commit + ":helper.py"], cwd=root, text=True) == "HELPER = True\n"
        assert (root / "x.py").read_text() == "VALUE = 10\n"
        assert (root / "helper.py").read_text() == "HELPER = True\n"
        assert manager.persist(saved) == saved
    finally:
        manager.discard(outcome.failure.failure_id)


def test_workspace_rejects_symlink_patch_targets(tmp_path):
    from framework.repair.workspace import RepairWorkspace, RepairWorkspaceError
    root = _repo(tmp_path)
    external = tmp_path / "external.py"
    external.write_text("VALUE = 1\n")
    (root / "linked.py").symlink_to(external)
    workspace = RepairWorkspace.create(root)
    try:
        with pytest.raises(RepairWorkspaceError, match="escapes|symlinks"):
            workspace.validate_paths(("linked.py",))
    finally:
        workspace.cleanup()
    assert external.read_text() == "VALUE = 1\n"


def test_validation_cannot_silently_modify_candidate_source(tmp_path):
    root = _repo(tmp_path)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_mutating.py").write_text(
        "from pathlib import Path\n"
        "def test_mutation():\n"
        "    Path('x.py').write_text('VALUE = 999\\n')\n"
    )
    manager = RepairManager(repository_root=root, local_backend=Backend("local", diff=GOOD_DIFF),
                            policy=RepairPolicy(mode=RepairMode.LOCAL))
    outcome = manager.repair(_failure())
    assert outcome.status is RepairStatus.REJECTED
    assert "modified the isolated source" in outcome.message
    assert (root / "x.py").read_text() == "VALUE = 1\n"

# Recovery orchestration is tested with host/model mocks, not live repair trials.


def test_safe_retry_is_bounded_across_failure_events(tmp_path):
    manager = RepairManager(repository_root=str(tmp_path), policy=RepairPolicy(mode=RepairMode.LOCAL))
    retry = Mock(side_effect=TimeoutError("temporary"))
    hooks = RecoveryHooks("operation", lambda: "same-task", retry, True)
    for _ in range(2):
        outcome = manager.repair(FailureEvent.capture(TimeoutError(), origin="mock"), hooks=hooks)
        assert outcome.status is RepairStatus.STOPPED
        assert outcome.retry_attempts == 2
    assert retry.call_count == 2


def test_retry_rules_need_no_model_and_require_host_permission(tmp_path):
    backend = Mock()
    manager = RepairManager(repository_root=str(tmp_path), local_backend=backend,
                            policy=RepairPolicy(mode=RepairMode.LOCAL))
    failure = FailureEvent.capture(TimeoutError(), origin="mock")
    retry = Mock(return_value=RecoveryCheck(True, True, value={"result": 1}))
    unsafe = manager.repair(failure, hooks=RecoveryHooks("op", lambda: "v1", retry))
    assert unsafe.status is RepairStatus.STOPPED
    retry.assert_not_called()
    recovered = manager.repair(failure, hooks=RecoveryHooks("op", lambda: "v1", retry, True))
    assert recovered.status is RepairStatus.RECOVERED
    assert recovered.verification is VerificationStatus.RESOLVED
    assert recovered.recovery_value == {"result": 1}
    backend.analyze.assert_not_called()


def test_retry_rejects_task_change(tmp_path):
    manager = RepairManager(repository_root=str(tmp_path), policy=RepairPolicy(mode=RepairMode.LOCAL))
    version = [1]
    def retry():
        version[0] += 1
        return RecoveryCheck(True, True, value="stale")
    outcome = manager.repair(FailureEvent.capture(TimeoutError(), origin="mock"),
        hooks=RecoveryHooks("op", lambda: str(version[0]), retry, True))
    assert outcome.status is RepairStatus.STOPPED
    assert outcome.recovery_value is None


@pytest.mark.parametrize("exception", [InterruptedError(), SteeringYield("mock")])
def test_retry_propagates_control_signals(tmp_path, exception):
    manager = RepairManager(repository_root=str(tmp_path), policy=RepairPolicy(mode=RepairMode.LOCAL))
    with pytest.raises(type(exception)):
        manager.repair(FailureEvent.capture(TimeoutError(), origin="mock"),
            hooks=RecoveryHooks("op", lambda: "v1", Mock(side_effect=exception), True))


@pytest.mark.parametrize("limit", [-1, 6, True, 1.5])
def test_retry_budget_validation(limit):
    with pytest.raises(ValueError):
        RepairPolicy(max_runtime_retries=limit)


@pytest.mark.parametrize("checks,expected", [
    (None, VerificationStatus.UNVERIFIED),
    ([RecoveryCheck(False, True, reproduced_failure=True), RecoveryCheck(True, True)], VerificationStatus.RESOLVED),
    ([RecoveryCheck(False, True, reproduced_failure=False)], None),
    ([RecoveryCheck(False, True, reproduced_failure=True), RecoveryCheck(True, False)], None),
])
def test_patch_verification_requires_baseline_and_candidate_evidence(tmp_path, monkeypatch, checks, expected):
    import framework.repair.manager as module
    from framework.repair.models import RepairContext
    workspace = Mock(path=tmp_path)
    workspace.fingerprint.return_value = "unchanged-source"
    monkeypatch.setattr(module.RepairWorkspace, "create", Mock(return_value=workspace))
    validator = Mock()
    validator._scope.return_value = ValidationReport(True, changed_files=("x.py",))
    validator.validate.return_value = ValidationReport(True)
    monkeypatch.setattr(module, "RepairValidator", Mock(return_value=validator))
    backend = Backend("local", diff=GOOD_DIFF)
    manager = RepairManager(repository_root=str(tmp_path), local_backend=backend,
                            policy=RepairPolicy(mode=RepairMode.LOCAL))
    failure = _failure()
    manager.context_builder = Mock()
    manager.context_builder.build.return_value = RepairContext(failure, str(tmp_path))
    replay = Mock(side_effect=checks) if checks is not None else None
    outcome = manager.repair(failure, hooks=RecoveryHooks("op", lambda: "v1", replay=replay))
    if expected is None:
        assert outcome.status is RepairStatus.REJECTED
        workspace.cleanup.assert_called_once()
    else:
        assert outcome.status is RepairStatus.VALIDATED
        assert outcome.verification is expected
        if replay:
            assert [call.args[1] for call in replay.call_args_list] == ["baseline", "candidate"]
        manager.discard(failure.failure_id)
