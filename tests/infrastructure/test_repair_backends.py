from framework.failure import FailureEvent
from framework.repair.models import FailureAction, RepairContext, RepairScope
from infrastructure.repair.local import LocalRepairBackend
from infrastructure.repair.cloud import CommercialRepairBackend, CommercialRepairConfig
import pytest


class FakeLLM:
    def __init__(self):
        self.calls = []
    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stage_label"] == "Local failure analysis":
            return {
                "action": "local_repair",
                "scope": "source",
                "confidence": 0.86,
                "diagnosis": "qualified column missing",
                "rationale": "trace points to SQL construction",
                "likely_files": ["x.py"],
                "likely_symbols": ["build_sql"],
            }
        return {
            "diagnosis": "qualified column missing",
            "confidence": 0.88,
            "summary": "Qualify the joined column.",
            "unified_diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
            "tests_to_run": ["pytest -q tests/test_x.py"],
        }


def _context():
    try:
        raise AttributeError("broken")
    except AttributeError as exc:
        failure = FailureEvent.capture(exc, origin="test")
    return RepairContext(failure=failure, repository_root=".")


def test_local_repair_backend_separates_analysis_and_patch_calls():
    llm = FakeLLM()
    backend = LocalRepairBackend(llm)
    assessment = backend.analyze(_context())
    proposal = backend.propose(_context(), assessment)
    assert assessment.action is FailureAction.LOCAL_REPAIR
    assert assessment.scope is RepairScope.SOURCE
    assert proposal.backend == "local"
    assert "--- a/x.py" in proposal.unified_diff
    assert [c["stage_label"] for c in llm.calls] == [
        "Local failure analysis", "Local repair patch"
    ]


def test_commercial_repair_config_keeps_key_in_memory_only():
    backend = CommercialRepairBackend(CommercialRepairConfig(
        provider="openai", api_key="sk-secret-test", model="gpt-test"
    ))
    assert backend.config.api_key == "sk-secret-test"
    assert "sk-secret-test" not in repr(backend.config)


@pytest.mark.parametrize("confidence", [True, "0.9", float("nan"), float("inf"), -0.1, 1.1])
def test_repair_backend_rejects_invalid_confidence_instead_of_clamping(confidence):
    from infrastructure.repair.local import _assessment
    payload = FakeLLM().generate_structured(stage_label="Local failure analysis")
    payload["confidence"] = confidence
    with pytest.raises(ValueError, match="confidence"):
        _assessment(payload)


def test_repair_backend_rejects_command_string_in_place_of_test_array():
    from infrastructure.repair.local import _proposal
    payload = FakeLLM().generate_structured(stage_label="Local repair patch")
    payload["tests_to_run"] = "pytest -q"
    with pytest.raises(ValueError, match="array"):
        _proposal("local", payload)


@pytest.mark.parametrize("provider", ["anthropic", "claude", "other"])
def test_cloud_repair_rejects_removed_or_unknown_providers(provider):
    with pytest.raises(ValueError, match="provider must be 'openai'"):
        CommercialRepairBackend(CommercialRepairConfig(
            provider=provider, api_key="test-key", model="test-model",
        ))


def test_cloud_repair_uses_openai_responses_without_network(monkeypatch):
    backend = CommercialRepairBackend(CommercialRepairConfig(
        provider="OpenAI", api_key="test-key", model="gpt-5.4",
    ))
    calls = []
    def post(url, body, headers):
        calls.append((url, body, headers))
        return {"output": [{"content": [{"type": "output_text", "text": '{"result": "ok"}'}]}]}
    monkeypatch.setattr(backend, "_post", post)
    assert backend._structured(system="diagnose", user="failure", schema={"type": "object"}) == {"result": "ok"}
    url, body, headers = calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert body["model"] == "gpt-5.4"
    assert body["store"] is False
    assert headers == {"Authorization": "Bearer test-key"}
