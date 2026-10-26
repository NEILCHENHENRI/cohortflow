from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import hashlib


class RepairWorkspaceError(RuntimeError):
    pass


@dataclass(slots=True)
class RepairWorkspace:
    repository_root: Path
    path: Path
    base_commit: str
    is_git_worktree: bool = True
    snapshot_commit: str | None = None

    @classmethod
    def create(cls, repository_root: str | Path) -> "RepairWorkspace":
        root = Path(repository_root).resolve()
        if not (root / ".git").exists():
            # Some tests/distributions are source archives rather than Git
            # checkouts.  Keep the same isolation guarantee with a disposable
            # copy, but persistent branches require a real Git repository.
            temp = Path(tempfile.mkdtemp(prefix="steerable-repair-copy-"))
            target = temp / "repo"
            try:
                shutil.copytree(
                    root,
                    target,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
                    symlinks=True,
                )
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                raise
            return cls(root, target, "archive", is_git_worktree=False)

        base = _run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
        temp = Path(tempfile.mkdtemp(prefix="steerable-repair-"))
        target = temp / "worktree"
        result = _run(
            ["git", "worktree", "add", "--detach", str(target), base],
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            shutil.rmtree(temp, ignore_errors=True)
            raise RepairWorkspaceError(result.stderr.strip() or result.stdout.strip())

        workspace = cls(root, target, base, is_git_worktree=True)
        try:
            # Snapshot the user's current working-tree state into the isolated
            # worktree.  This preserves staged/unstaged source changes without ever
            # modifying or switching the live checkout.
            dirty = _run(["git", "diff", "--binary", "HEAD"], cwd=root).stdout
            if dirty.strip():
                snapshot_patch = temp / "working-tree.patch"
                snapshot_patch.write_text(dirty, encoding="utf-8")
                applied = _run(
                    ["git", "apply", "--whitespace=nowarn", str(snapshot_patch)],
                    cwd=target,
                    check=False,
                )
                if applied.returncode != 0:
                    _run(["git", "worktree", "remove", "--force", str(target)], cwd=root, check=False)
                    shutil.rmtree(temp, ignore_errors=True)
                    raise RepairWorkspaceError(
                        "Could not snapshot current working-tree changes: "
                        + (applied.stderr.strip() or applied.stdout.strip())
                    )
            untracked = _run(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=root,
            ).stdout.split("\0")
            for relative in untracked:
                if not relative:
                    continue
                src = root / relative
                dst = target / relative
                if src.is_file() or src.is_symlink():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_symlink():
                        dst.symlink_to(src.readlink())
                    else:
                        shutil.copy2(src, dst)
            _run(["git", "add", "-A"], cwd=target)
            dirty_snapshot = _run(["git", "diff", "--cached", "--quiet"], cwd=target, check=False)
            if dirty_snapshot.returncode == 1:
                _run(["git", "-c", "user.name=CohortFlow Repair", "-c", "user.email=repair@localhost",
                      "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                      "commit", "-qm", "Snapshot working tree for isolated repair"], cwd=target)
            elif dirty_snapshot.returncode != 0:
                raise RepairWorkspaceError(dirty_snapshot.stderr)
            workspace.snapshot_commit = _run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
            return workspace
        except Exception:
            workspace.cleanup()
            raise

    def apply_diff(self, unified_diff: str) -> None:
        patch_file = self.path.parent / "repair.patch"
        patch_file.write_text(unified_diff, encoding="utf-8")
        if self.is_git_worktree:
            check = _run(
                ["git", "apply", "--check", "--whitespace=nowarn", str(patch_file)],
                cwd=self.path,
                check=False,
            )
            if check.returncode != 0:
                raise RepairWorkspaceError(
                    "Patch does not apply cleanly: "
                    + (check.stderr.strip() or check.stdout.strip())
                )
            applied = _run(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=self.path,
                check=False,
            )
            if applied.returncode != 0:
                raise RepairWorkspaceError(applied.stderr.strip() or applied.stdout.strip())
            return
        # Archive fallback still uses `patch`, never edits the source tree.
        applied = _run(
            ["patch", "--batch", "-p1", "-i", str(patch_file)],
            cwd=self.path,
            check=False,
        )
        if applied.returncode != 0:
            raise RepairWorkspaceError(applied.stderr.strip() or applied.stdout.strip())

    def fingerprint(self) -> str:
        """Detect edits to the validated snapshot before it is persisted."""
        digest = hashlib.sha256()
        for path in sorted(self.path.rglob("*")):
            relative = path.relative_to(self.path)
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if path.is_symlink():
                digest.update(b"symlink\0" + str(relative).encode() + b"\0" + str(path.readlink()).encode() + b"\0")
            elif path.is_file():
                metadata = f"{relative}\0{path.stat().st_mode}\0{path.stat().st_size}\0"
                digest.update(b"file\0" + metadata.encode())
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                digest.update(b"\0")
        return digest.hexdigest()

    def validate_paths(self, paths: tuple[str, ...]) -> None:
        for relative in paths:
            candidate = self.path / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or not candidate.resolve().is_relative_to(self.path.resolve()):
                raise RepairWorkspaceError(f"patch path escapes workspace: {relative}")
            current = self.path
            for part in Path(relative).parts:
                current = current / part
                if current.is_symlink():
                    raise RepairWorkspaceError(f"patch may not traverse symlinks: {relative}")

    def save_branch(self, branch: str, *, message: str, changed_files: tuple[str, ...] = ()) -> tuple[str, str]:
        if not self.is_git_worktree:
            raise RepairWorkspaceError("Persistent repair branches require a Git checkout")
        safe = branch.strip().replace(" ", "-")
        if not safe.startswith(("codex/repair/", "repair/")):
            safe = "codex/repair/" + safe
        switched = _run(["git", "switch", "-c", safe], cwd=self.path, check=False)
        if switched.returncode != 0:
            raise RepairWorkspaceError(switched.stderr.strip() or switched.stdout.strip())
        _run(["git", "add", "-A", "--", *changed_files] if changed_files else ["git", "add", "-A"], cwd=self.path)
        committed = _run(
            [
                "git", "-c", "user.name=CohortFlow Repair",
                "-c", "user.email=repair@localhost", "-c", "core.hooksPath=/dev/null",
                "-c", "commit.gpgsign=false", "commit", "-m", message,
            ],
            cwd=self.path,
            check=False,
        )
        if committed.returncode != 0:
            raise RepairWorkspaceError(committed.stderr.strip() or committed.stdout.strip())
        commit = _run(["git", "rev-parse", "HEAD"], cwd=self.path).stdout.strip()
        return safe, commit

    def cleanup(self) -> None:
        parent = self.path.parent
        if self.is_git_worktree and self.path.exists():
            _run(
                ["git", "worktree", "remove", "--force", str(self.path)],
                cwd=self.repository_root,
                check=False,
            )
        shutil.rmtree(parent, ignore_errors=True)


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RepairWorkspaceError(result.stderr.strip() or result.stdout.strip())
    return result
