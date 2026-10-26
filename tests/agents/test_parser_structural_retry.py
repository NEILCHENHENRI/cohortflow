from agents.cohort.parser import CohortTaskParser


class RetryLLM:
    def __init__(self):
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "predicates": [
                    {"predicate_id": "adult", "text": "patient is at least 18 years old"}
                ],
                "logic": "adult AND",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "",
            }
        if len(self.calls) == 2:
            return {
                "predicates": [
                    {"predicate_id": "adult", "text": "patient is at least 18 years old"}
                ],
                "logic": "adult",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Corrected expression.",
            }
        return {
            "requirements": [{
                "requirement_id": "adult",
                "kind": "patient",
                "payload": {"attribute": "age", "operator": ">=", "value": 18},
                "polarity": "include",
                "required": True,
            }],
            "assumptions": [],
        }


def test_invalid_boolean_string_is_reprompted_once_before_compilation():
    llm = RetryLLM()
    parser = CohortTaskParser(llm, structural_retry_limit=1)
    task = parser.parse_task("Find adults")

    assert len(llm.calls) == 3
    assert "previous output failed deterministic validation" in llm.calls[1]["user_prompt"].lower()
    assert task.logic is not None
    assert task.logic.kind == "predicate"
    assert task.requirements[0].payload == {"attribute": "age", "operator": ">=", "value": 18}
