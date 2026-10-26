import pytest

from framework.operator import (
    OperatorRegistry,
    OperatorRole,
    OperatorSpec,
)


def noop(inputs, config, context):
    return {"out": 1}


def test_operator_registry_registers_and_describes():
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            name="noop",
            role=OperatorRole.TRANSFORM,
            description="Return a constant.",
            output_types=("number",),
        ),
        noop,
    )

    assert "noop" in registry
    assert registry.get("noop").function is noop
    assert registry.describe_for_prompt()[0]["role"] == "transform"


def test_duplicate_operator_rejected():
    registry = OperatorRegistry()
    spec = OperatorSpec(
        name="noop",
        role=OperatorRole.TRANSFORM,
        description="Return a constant.",
    )
    registry.register(spec, noop)
    with pytest.raises(ValueError):
        registry.register(spec, noop)
