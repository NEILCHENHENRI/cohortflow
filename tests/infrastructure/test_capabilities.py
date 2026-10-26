import pytest

from framework.errors import UnsupportedCriterionError
from infrastructure.capabilities import DataCapabilities


def test_capability_rejects_unsupported_semantics():
    capabilities = DataCapabilities(
        supports_conditions=True,
        supports_medications=True,
        supports_patient_attributes=True,
        supports_admissions=True,
        supports_active_medication=False,
    )
    capabilities.validate_requirement(
        "patient", {"attribute": "age", "operator": ">=", "value": 18}
    )
    capabilities.validate_requirement("admission", {"start_date": "2020-01-01"})
    with pytest.raises(UnsupportedCriterionError):
        capabilities.validate_requirement("lab", {"term": "creatinine"})
    with pytest.raises(UnsupportedCriterionError):
        capabilities.validate_requirement(
            "medication", {"term": "metformin", "exposure_mode": "active_at_index"}
        )
