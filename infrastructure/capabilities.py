from __future__ import annotations

from dataclasses import dataclass

from framework.errors import UnsupportedCriterionError


@dataclass(frozen=True, slots=True)
class DataCapabilities:
    """Backend feature declaration used by the deterministic planner."""

    supports_conditions: bool
    supports_medications: bool
    supports_active_medication: bool = False
    supports_procedures: bool = False
    supports_labs: bool = False
    supports_patient_attributes: bool = False
    supports_admissions: bool = False
    supports_notes: bool = False

    def validate_requirement(self, kind: str, payload: dict) -> None:
        supported = {
            "condition": self.supports_conditions,
            "procedure": self.supports_procedures,
            "lab": self.supports_labs,
            "medication": self.supports_medications,
            "patient": self.supports_patient_attributes,
            "admission": self.supports_admissions,
            "note": self.supports_notes,
        }
        if not supported.get(kind, False):
            raise UnsupportedCriterionError(
                f"backend does not support {kind!r} requirements"
            )
        if (
            kind == "medication"
            and payload.get("exposure_mode") == "active_at_index"
            and not self.supports_active_medication
        ):
            raise UnsupportedCriterionError(
                "backend cannot determine active medication at index date"
            )
