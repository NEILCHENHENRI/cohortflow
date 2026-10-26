from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SteeringYield(RuntimeError):
    """Raised only at an atomic module boundary when newer user input is waiting.

    The current module has already completed. Callers should preserve any
    committed graph artifacts, stop scheduling obsolete work, and process the
    queued input before continuing.
    """

    checkpoint: str

    def __str__(self) -> str:
        return f"execution yielded for steering at {self.checkpoint}"


def should_yield(callback) -> bool:
    probe = getattr(callback, "should_yield", None)
    return bool(probe()) if callable(probe) else False


def steering_checkpoint(callback, checkpoint: str) -> None:
    if callback is not None:
        callback("module_checkpoint", checkpoint)
    if should_yield(callback):
        raise SteeringYield(checkpoint)
