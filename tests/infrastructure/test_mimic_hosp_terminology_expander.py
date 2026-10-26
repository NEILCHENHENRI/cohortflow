from __future__ import annotations

from infrastructure.mimic_hosp.terminology import LLMMedicalTerminologyExpander


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_medical_terminology_expander_deduplicates_case_whitespace_and_caps_eight():
    llm = FakeLLM({
        "canonical_form": " Chronic obstructive pulmonary disease ",
        "equivalent_terms": [
            "COPD",
            "copd",
            " chronic   obstructive pulmonary disease ",
            "Chronic obstructive lung disease",
            "Term 4", "Term 5", "Term 6", "Term 7", "Term 8", "Term 9",
        ],
    })
    progress = []
    result = LLMMedicalTerminologyExpander(llm).expand(
        "COPD", domain="condition", on_progress=lambda event, text: progress.append((event, text))
    )
    assert result[0] == "Chronic obstructive pulmonary disease"
    assert result[1] == "COPD"
    assert len(result) == 8
    assert len({" ".join(v.casefold().split()) for v in result}) == len(result)
    assert llm.calls[0]["stage_label"] == "Medical terminology expansion"
    assert llm.calls[0]["on_progress"] is not None
    assert "DOMAIN: condition" in llm.calls[0]["user_prompt"]


def test_medical_terminology_expander_handles_empty_or_malformed_payload_values():
    llm = FakeLLM({"canonical_form": " ", "equivalent_terms": [None, "", 4, "CHF"]})
    assert LLMMedicalTerminologyExpander(llm).expand("heart failure", domain="condition") == ("CHF",)
