from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import os
import shlex
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace

from framework.repair.models import RepairContext, RepairPolicy, ValidationReport, ValidationStep
from framework.repair.workspace import RepairWorkspace

_PROTECTED = {
    ".gitignore", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "setup.py", "setup.cfg", "tox.ini",
}


class RepairValidator:
    def __init__(self, policy: RepairPolicy) -> None:
        self.policy = policy

    def validate(
        self,
        workspace: RepairWorkspace,
        context: RepairContext,
        unified_diff: str,
        *,
        baseline: tuple[ValidationStep, ...] = (),
    ) -> ValidationReport:
        scope = self._scope(unified_diff)
        if scope.rejection_reason:
            return scope
        steps: list[ValidationStep] = []

        changed_py = [path for path in scope.changed_files if path.endswith(".py") and (workspace.path / path).exists()]
        if changed_py:
            steps.append(self._run(
                workspace.path,
                "compile",
                [sys.executable, "-m", "compileall", "-q", *changed_py],
            ))
            if not steps[-1].passed:
                return replace(self._finish(scope, steps), baseline_steps=baseline)

        before = {step.name: step for step in baseline}
        for name, paths in self._test_suites(workspace, context):
            step = self._run_tests(workspace.path, name, paths)
            steps.append(self._compare_tests(before.get(name), step))
            if not steps[-1].passed:
                break
        return replace(self._finish(scope, steps), baseline_steps=baseline)

    def capture_baseline(self, workspace: RepairWorkspace, context: RepairContext) -> tuple[ValidationStep, ...]:
        """Measure the exact isolated snapshot before applying a candidate patch."""
        return tuple(self._run_tests(workspace.path, name, paths)
                     for name, paths in self._test_suites(workspace, context))

    def _test_suites(self, workspace: RepairWorkspace, context: RepairContext):
        targeted = []
        for path, _ in context.test_context:
            candidate = (workspace.path / path).resolve()
            if not candidate.is_relative_to(workspace.path.resolve()) or not candidate.is_file():
                raise ValueError(f"test path is outside the repair workspace or missing: {path}")
            if path.endswith(".py"):
                targeted.append(path)
        if targeted:
            yield "targeted tests", tuple(targeted)
        if self.policy.run_full_suite and (workspace.path / "tests").is_dir():
            yield "full test suite", ("tests",)

    def _run_tests(self, cwd: Path, name: str, paths: tuple[str, ...]) -> ValidationStep:
        with tempfile.TemporaryDirectory(prefix="repair-test-results-") as directory:
            report = Path(directory) / "results.xml"
            step = self._run(cwd, name, [
                sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                f"--junitxml={report}", *paths,
            ])
            try:
                cases = ET.parse(report).getroot().iter("testcase")
                results = []
                for case in cases:
                    status = "passed"
                    for kind in ("failure", "error", "skipped"):
                        if case.find(kind) is not None:
                            status = kind
                            break
                    results.append((case.get("classname", "") + "::" + case.get("name", ""), status))
            except (OSError, ET.ParseError):
                return replace(step, passed=False, output=step.output + "\nMissing or invalid pytest report.")
            if len({name for name, _ in results}) != len(results):
                return replace(step, passed=False, output=step.output + "\nAmbiguous duplicate pytest test identities.")
            return replace(step, test_results=tuple(results))

    @staticmethod
    def _compare_tests(before: ValidationStep | None, after: ValidationStep) -> ValidationStep:
        if before is None or before.return_code not in {0, 1} or not before.test_results:
            return after
        if after.return_code not in {0, 1} or not after.test_results:
            return replace(after, passed=False)
        old, new = dict(before.test_results), dict(after.test_results)
        regressions = sorted(
            name for name, status in new.items()
            if status in {"failure", "error"} and old.get(name) != status
        )
        missing = sorted(set(old) - set(new))
        skipped = sorted(name for name, status in new.items()
                         if status == "skipped" and name in old and old[name] != "skipped")
        passed = not (regressions or missing or skipped)
        detail = f"\nBaseline comparison: new failures={regressions}. Missing tests={missing}. Newly skipped={skipped}."
        return replace(after, passed=passed, output=after.output + detail)

    def _scope(self, diff: str) -> ValidationReport:
        """Validate both sides of every patch, including deleted files.

        Parse hunk lengths so header-like source lines cannot hide a second file.
        Renames, binary patches, and mode changes are outside source repair scope.
        """
        files: list[str] = []
        changed_lines = 0
        lines = diff.splitlines()
        index = 0
        hunks = 0
        metadata_paths = None

        def path(header: str, prefix: str) -> str | None:
            value = header.split("\t", 1)[0]
            if value == "/dev/null":
                return None
            if value.startswith('"'):
                raise ValueError("quoted patch paths are unsupported")
            if not value.startswith(prefix):
                raise ValueError(f"unsafe patch path: {value}")
            relative = value[len(prefix):]
            parts = Path(relative).parts
            if not relative or Path(relative).is_absolute() or ".." in parts or "\\" in relative:
                raise ValueError(f"unsafe patch path: {relative}")
            if any(part.startswith(".") for part in parts):
                raise ValueError(f"protected file: {relative}")
            if "tests" in parts or Path(relative).name.startswith("test_") or Path(relative).name == "conftest.py":
                raise ValueError("repair may not modify tests")
            if Path(relative).name in _PROTECTED or Path(relative).suffix in {".toml", ".ini", ".cfg", ".yaml", ".yml"}:
                raise ValueError(f"protected file: {relative}")
            return relative

        try:
            while index < len(lines):
                line = lines[index]
                if not line.strip():
                    index += 1
                    continue
                if line.startswith("diff --git "):
                    if metadata_paths is not None:
                        raise ValueError("patch metadata has no text hunks")
                    names = shlex.split(line[len("diff --git "):])
                    if len(names) != 2:
                        raise ValueError("invalid diff header")
                    metadata_paths = {path(names[0], "a/"), path(names[1], "b/")}
                    index += 1
                    continue
                if line.startswith("index "):
                    if metadata_paths is None or not re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100(?:644|755))?", line):
                        raise ValueError("unsupported patch file mode")
                    index += 1
                    continue
                if line in {"new file mode 100644", "deleted file mode 100644"}:
                    if metadata_paths is None:
                        raise ValueError("file mode requires a diff header")
                    index += 1
                    continue
                if not line.startswith("--- ") or index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                    raise ValueError("unsupported or malformed patch metadata")
                old = path(line[4:], "a/")
                new = path(lines[index + 1][4:], "b/")
                if old is None and new is None:
                    raise ValueError("patch must name a repository file")
                if old and new and old != new:
                    raise ValueError("repair may not rename files")
                if metadata_paths is not None and metadata_paths != {p for p in (old, new) if p}:
                    raise ValueError("patch metadata and file headers disagree")
                metadata_paths = None
                files.extend(p for p in (old, new) if p and p not in files)
                index += 2
                file_hunks = 0
                while index < len(lines) and lines[index].startswith("@@ "):
                    match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*", lines[index])
                    if match is None:
                        raise ValueError("invalid patch hunk")
                    old_count = int(match.group(2) or 1)
                    new_count = int(match.group(4) or 1)
                    index += 1
                    while old_count or new_count:
                        if index >= len(lines) or not lines[index]:
                            raise ValueError("truncated patch hunk")
                        marker = lines[index][0]
                        if marker == " " or marker == "-":
                            old_count -= 1
                        if marker == " " or marker == "+":
                            new_count -= 1
                        if marker in {"+", "-"}:
                            changed_lines += 1
                        elif marker != " ":
                            raise ValueError("invalid patch hunk content")
                        if old_count < 0 or new_count < 0:
                            raise ValueError("patch hunk length mismatch")
                        index += 1
                        if index < len(lines) and lines[index] == "\\ No newline at end of file":
                            index += 1
                    file_hunks += 1
                    hunks += 1
                if not file_hunks:
                    raise ValueError("patch file has no text hunks")
            if metadata_paths is not None:
                raise ValueError("patch metadata has no text hunks")
            if not hunks or not changed_lines:
                raise ValueError("repair returned an empty patch")
            if len(files) > self.policy.max_changed_files:
                raise ValueError(f"patch touches {len(files)} files. Limit is {self.policy.max_changed_files}")
            if changed_lines > self.policy.max_changed_lines:
                raise ValueError(f"patch changes {changed_lines} lines. Limit is {self.policy.max_changed_lines}")
        except ValueError as exc:
            return ValidationReport(False, tuple(files), changed_lines, rejection_reason=str(exc))
        return ValidationReport(True, tuple(files), changed_lines)

    def _run(self, cwd: Path, name: str, command: list[str]) -> ValidationStep:
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=self.policy.test_timeout_seconds,
                env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = (result.stdout + "\n" + result.stderr)[-16000:]
            return ValidationStep(name, result.returncode == 0, tuple(command), result.returncode, output)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            output = (decoded(exc.stdout) + "\n" + decoded(exc.stderr))[-16000:]
            return ValidationStep(name, False, tuple(command), None, "Timed out\n" + output)
        except OSError as exc:
            return ValidationStep(name, False, tuple(command), None, str(exc))

    @staticmethod
    def _finish(base: ValidationReport, steps: list[ValidationStep]) -> ValidationReport:
        passed = bool(steps) and all(step.passed for step in steps)
        return ValidationReport(
            passed=passed,
            changed_files=base.changed_files,
            changed_lines=base.changed_lines,
            steps=tuple(steps),
            rejection_reason=None if passed else next((s.name + " failed" for s in steps if not s.passed), base.rejection_reason or "no validation checks ran"),
        )
