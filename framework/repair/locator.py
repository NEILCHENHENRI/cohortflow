from __future__ import annotations

from pathlib import Path
import re

from framework.failure import FailureEvent
from framework.repair.models import RepairContext

_TRACE_FILE = re.compile(r'File "([^"]+)", line (\d+)')


class RepairContextBuilder:
    def __init__(self, repository_root: str | Path, *, radius: int = 80) -> None:
        self.root = Path(repository_root).resolve()
        self.radius = max(10, int(radius))

    def build(self, failure: FailureEvent) -> RepairContext:
        source: list[tuple[str, str]] = []
        seen: set[Path] = set()
        for filename, line_text in _TRACE_FILE.findall(failure.traceback):
            path = Path(filename)
            path = (path if path.is_absolute() else self.root / path).resolve()
            if path in seen or not self._inside_root(path) or not path.is_file():
                continue
            seen.add(path)
            line = max(1, int(line_text))
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, line - self.radius - 1)
            end = min(len(text), line + self.radius)
            numbered = "\n".join(f"{i + 1:04d}: {text[i]}" for i in range(start, end))
            source.append((str(path.relative_to(self.root)), numbered))

        tests = self._related_tests(tuple(path for path in seen))
        return RepairContext(
            failure=failure,
            repository_root=str(self.root),
            source_context=tuple(source),
            test_context=tests,
        )

    def _related_tests(self, source_paths: tuple[Path, ...]) -> tuple[tuple[str, str], ...]:
        tests_root = self.root / "tests"
        if not tests_root.exists():
            return ()
        needles = {path.stem.casefold() for path in source_paths}
        ranked: list[Path] = []
        for path in sorted(tests_root.rglob("test_*.py")):
            if not self._inside_root(path.resolve()) or not path.is_file():
                continue
            name = path.name.casefold()
            if any(needle in name or needle in path.read_text(encoding="utf-8", errors="ignore").casefold() for needle in needles):
                ranked.append(path)
            if len(ranked) >= 3:
                break
        return tuple(
            (str(path.relative_to(self.root)), path.read_text(encoding="utf-8", errors="replace")[:14000])
            for path in ranked
        )

    def _inside_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False
