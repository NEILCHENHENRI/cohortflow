from __future__ import annotations

from enum import Enum


class ExecutionBackend(str, Enum):
    """Execution engine used to run the framework's ExecutionGraph."""

    NATIVE = "native"
    LANGGRAPH = "langgraph"
