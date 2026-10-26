from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[str, str], None]


def scoped_progress(
    callback: ProgressCallback | None,
    *,
    scope: str,
    label: str,
) -> ProgressCallback | None:
    """Namespace model-stream events while keeping stage events readable."""
    if callback is None:
        return None

    def emit(event_type: str, payload: str) -> None:
        if event_type == "stage":
            callback("stage", f"{label}: {payload}")
            return
        callback(f"{event_type}:{scope}", payload)

    return emit
