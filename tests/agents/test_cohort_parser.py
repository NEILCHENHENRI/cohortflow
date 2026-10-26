from agents.cohort.parser import CohortTaskParser
from framework.correction import CorrectionMode
from framework.parsing import ParsingMode
from framework.errors import DomainValidationError
import pytest


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def flat(predicates, logic, rationale="Parsed cohort logic."):
    return {
        "predicates": predicates,
        "logic": logic,
        "ambiguities": [],
        "requires_clarification": False,
        "clarification_question": None,
        "rationale": rationale,
    }


def requirement(rid, kind, payload):
    return {
        "requirement_id": rid,
        "kind": kind,
        "payload": payload,
        "polarity": "include",
        "required": True,
    }


def test_parse_task_and_full_state_correction():
    initial = [
        {"predicate_id": "condition_diabetes", "text": "patient has type 2 diabetes mellitus"},
        {"predicate_id": "age_adult", "text": "patient is at least 18 years old"},
    ]
    revised = [*initial, {"predicate_id": "condition_ckd", "text": "patient has chronic kidney disease"}]
    llm = FakeLLM([
        flat(initial, "condition_diabetes AND age_adult"),
        {
            "requirements": [
                requirement("condition_diabetes", "condition", {"term": "Type 2 diabetes mellitus"}),
                requirement("age_adult", "patient", {"attribute": "age", "operator": ">=", "value": 18}),
            ],
            "assumptions": [],
        },
        {"merged_request": "Find adults with diabetes and also require CKD"},
        flat(revised, "condition_diabetes AND age_adult AND condition_ckd", "Add CKD as mandatory."),
        {
            "requirements": [
                requirement("condition_diabetes", "condition", {"term": "Type 2 diabetes mellitus"}),
                requirement("age_adult", "patient", {"attribute": "age", "operator": ">=", "value": 18}),
                requirement("condition_ckd", "condition", {"term": "Chronic kidney disease"}),
            ],
            "assumptions": [],
        },
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.FULL_REGENERATION)
    task = parser.parse_task("Find adults with diabetes")
    delta = parser.interpret_correction(task, "Also require CKD").task_delta

    assert {r.kind for r in task.requirements} == {"condition", "patient"}
    assert [r.payload.get("term") for r in delta.add_requirements] == ["Chronic kidney disease"]
    assert delta.logic is not None
    assert len(set(delta.logic.predicate_ids())) == 3
    assert len(llm.calls) == 5


def extraction(predicates):
    return {
        "predicates": predicates,
        "ambiguities": [],
        "requires_clarification": False,
        "clarification_question": None,
        "rationale": "Extracted atomic predicates.",
    }


def relationships(logic):
    return {
        "logic": logic,
        "ambiguities": [],
        "requires_clarification": False,
        "clarification_question": None,
        "rationale": "Attached relationships.",
    }


def test_parsing_modes_have_expected_initial_call_counts():
    from framework.parsing import ParsingMode

    predicates = [
        {"predicate_id": "p1", "text": "patient has diabetes"},
        {"predicate_id": "p2", "text": "patient is at least 65 years old"},
    ]
    compiled = {
        "requirements": [
            requirement("p1", "condition", {"term": "diabetes"}),
            requirement("p2", "patient", {"attribute": "age", "operator": ">=", "value": 65}),
        ],
        "assumptions": [],
    }

    carbon = FakeLLM([{**flat(predicates, "p1 AND p2"), **compiled}])
    CohortTaskParser(carbon, parsing_mode=ParsingMode.CARBON).parse_task("older adults with diabetes")
    assert len(carbon.calls) == 1

    graphite = FakeLLM([flat(predicates, "p1 AND p2"), compiled])
    CohortTaskParser(graphite, parsing_mode=ParsingMode.GRAPHITE).parse_task("older adults with diabetes")
    assert len(graphite.calls) == 2

    diamond = FakeLLM([
        extraction(predicates),
        relationships("p1 AND p2"),
        {"assignments": [
            {"predicate_id": "p1", "domain": "condition"},
            {"predicate_id": "p2", "domain": "patient"},
        ]},
        compiled,
    ])
    CohortTaskParser(diamond, parsing_mode=ParsingMode.DIAMOND).parse_task("older adults with diabetes")
    assert len(diamond.calls) == 4

    graphene = FakeLLM([
        extraction(predicates),
        relationships("p1 AND p2"),
        {"assignments": [{"predicate_id": "p1", "domain": "condition"}]},
        {"requirements": [compiled["requirements"][0]], "assumptions": []},
        {"assignments": [{"predicate_id": "p2", "domain": "patient"}]},
        {"requirements": [compiled["requirements"][1]], "assumptions": []},
    ])
    CohortTaskParser(graphene, parsing_mode=ParsingMode.GRAPHENE).parse_task("older adults with diabetes")
    assert len(graphene.calls) == 6  # 2 + 2N for N=2 when the field catalog exceeds 30


def test_per_pass_thinking_defaults_off_and_can_be_overridden():
    from framework.parsing import ParsingMode

    class ThinkingLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.think = True
            self.think_seen = []

        def generate_structured(self, **kwargs):
            self.think_seen.append(self.think)
            return super().generate_structured(**kwargs)

    predicates = [{"predicate_id": "p1", "text": "patient has diabetes"}]
    llm = ThinkingLLM([
        extraction(predicates),
        relationships("p1"),
        {"assignments": [{"predicate_id": "p1", "domain": "condition"}]},
        {"requirements": [requirement("p1", "condition", {"term": "diabetes"})], "assumptions": []},
    ])
    parser = CohortTaskParser(
        llm,
        parsing_mode=ParsingMode.DIAMOND,
        thinking_by_pass={"relationships": True, "table_selection": True},
    )
    parser.parse_task("patients with diabetes")

    assert llm.think_seen == [False, True, True, False]
    assert llm.think is True  # base client state is restored after each pass


def test_graphite_compilation_receives_all_semantic_fields_without_example_values():
    from framework.parsing import ParsingMode
    predicates = [{"predicate_id": "p1", "text": "oral furosemide"}]
    llm = FakeLLM([
        flat(predicates, "p1"),
        {"requirements": [requirement("p1", "medication", {"term": "furosemide", "route": "oral"})], "assumptions": []},
    ])
    CohortTaskParser(llm, parsing_mode=ParsingMode.GRAPHITE).parse_task("oral furosemide")
    prompt = llm.calls[1]["user_prompt"]
    assert "medication.route" not in prompt  # rendered as table-scoped field name
    assert "route: route of medication administration" in prompt
    assert "admission_type: type or class of hospital admission" in prompt
    assert "EW EMER." not in prompt
    assert "route: PO" not in prompt


def test_diamond_table_assignment_cannot_rewrite_resolved_logic():
    from framework.parsing import ParsingMode
    predicates = [
        {"predicate_id": "p1", "text": "diabetes"},
        {"predicate_id": "p2", "text": "hypertension"},
    ]
    llm = FakeLLM([
        extraction(predicates),
        relationships("p1 OR p2"),
        {"assignments": [
            {"predicate_id": "p1", "domain": "condition"},
            {"predicate_id": "p2", "domain": "condition"},
        ]},
        {"requirements": [
            requirement("p1", "condition", {"term": "diabetes"}),
            requirement("p2", "condition", {"term": "hypertension"}),
        ], "assumptions": []},
    ])
    task = CohortTaskParser(llm, parsing_mode=ParsingMode.DIAMOND).parse_task("diabetes or hypertension")
    assert task.logic is not None
    assert task.logic.kind == "or"
    assert "logic" not in llm.calls[-1]["schema"]["properties"]


def test_robust_correction_routes_add_then_compiles_only_new_predicate():
    initial = [
        {"predicate_id": "p1", "text": "patient has diabetes"},
        {"predicate_id": "p2", "text": "patient is at least 18 years old"},
    ]
    llm = FakeLLM([
        flat(initial, "p1 AND p2"),
        {"requirements": [
            requirement("p1", "condition", {"term": "diabetes"}),
            requirement("p2", "patient", {"attribute": "age", "operator": ">=", "value": 18}),
        ], "assumptions": []},
        {
            "actions": [{
                "action": "ADD", "local_ref": "new_1",
                "requirement_text": "patient has CKD", "excluded": False,
                "anchor": "ROOT", "operator": "AND", "placement": "WITH",
                "relation_memberships": [],
            }],
            "requires_clarification": False, "clarification_question": None,
        },
        {"requirements": [requirement("p3", "condition", {"term": "CKD"})], "assumptions": []},
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA)
    task = parser.parse_task("adults with diabetes")
    analysis = parser.interpret_correction(task, "also require CKD")
    updated = __import__("framework.task", fromlist=["apply_task_delta"]).apply_task_delta(task, analysis.task_delta)
    assert any(r.payload.get("term") == "CKD" for r in updated.requirements)
    assert updated.logic.kind == "and"
    assert len(llm.calls) == 4
    assert llm.calls[2].get("stage_label") == "Correction interpretation"
    assert llm.calls[3].get("stage_label") == "Predicate compilation"
    assert not any(call.get("stage_label") == "Correction merge" for call in llm.calls)


def test_robust_correction_clarification_does_not_fallback_or_compile():
    initial = [{"predicate_id": "p1", "text": "patient has diabetes"}]
    llm = FakeLLM([
        flat(initial, "p1"),
        {"requirements": [requirement("p1", "condition", {"term": "diabetes"})], "assumptions": []},
        {
            "actions": [],
            "requires_clarification": True,
            "clarification_question": "Which kidney criterion do you mean?",
        },
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA)
    task = parser.parse_task("diabetes")
    analysis = parser.interpret_correction(task, "also add that other kidney criterion")
    assert analysis.intent.requires_clarification is True
    assert analysis.intent.clarification_question == "Which kidney criterion do you mean?"
    assert len(llm.calls) == 3
    assert not any(call.get("stage_label") == "Correction merge" for call in llm.calls)


def test_robust_correction_invalid_ref_retries_once_then_requests_fallback():
    from framework.correction import CorrectionFallbackRequired
    initial = [{"predicate_id": "p1", "text": "patient has diabetes"}]
    llm = FakeLLM([
        flat(initial, "p1"),
        {"requirements": [requirement("p1", "condition", {"term": "diabetes"})], "assumptions": []},
        {
            "actions": [{"action": "REMOVE", "target": "p99"}],
            "requires_clarification": False, "clarification_question": None,
        },
        {
            "actions": [{"action": "REMOVE", "target": "p99"}],
            "requires_clarification": False, "clarification_question": None,
        },
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA)
    task = parser.parse_task("diabetes")
    try:
        parser.interpret_correction(task, "remove diabetes")
        assert False, "expected CorrectionFallbackRequired"
    except CorrectionFallbackRequired as exc:
        assert "unknown predicate ref" in exc.reason
    assert len(llm.calls) == 4


@pytest.mark.parametrize("mode", [ParsingMode.DIAMOND, ParsingMode.GRAPHENE])
def test_relationship_prompt_leads_with_fixed_bullets_and_retains_grouping_context(mode):
    predicates = [{"predicate_id": "p1", "text": "diabetes"}, {"predicate_id": "p2", "text": "hypertension"}]
    llm = FakeLLM([relationships("p1 OR p2")])
    parser = CohortTaskParser(llm, parsing_mode=mode)
    from agents.cohort.parser import _predicate_extraction_from_payload
    parsed = parser._attach_relationships(
        "diabetes OR hypertension", _predicate_extraction_from_payload(extraction(predicates)), on_progress=None,
    )
    prompt = llm.calls[0]["user_prompt"]
    assert prompt.startswith("FIXED PREDICATES")
    assert '- p1: "diabetes"\n- p2: "hypertension"' in prompt
    assert prompt.index("- p2:") < prompt.index("ORIGINAL REQUEST")
    assert "diabetes OR hypertension" in prompt
    assert parsed.logic.kind == "or"
    assert [p.text for p in parsed.predicates] == ["diabetes", "hypertension"]


@pytest.mark.parametrize("bad", [
    {**relationships("p1"), "predicates": [{"predicate_id": "p1", "text": "rewritten"}]},
    relationships("p1"),
    relationships("p1 OR p99"),
])
def test_relationship_pass_cannot_rewrite_omit_or_invent_fixed_predicates(bad):
    from agents.cohort.parser import _predicate_extraction_from_payload
    predicates = [{"predicate_id": "p1", "text": "diabetes"}, {"predicate_id": "p2", "text": "hypertension"}]
    llm = FakeLLM([bad])
    parser = CohortTaskParser(llm, structural_retry_limit=0)
    with pytest.raises(DomainValidationError):
        parser._attach_relationships("diabetes or hypertension", _predicate_extraction_from_payload(extraction(predicates)), on_progress=None)


@pytest.mark.parametrize("mode,expected_calls", [(ParsingMode.CARBON, 2), (ParsingMode.GRAPHITE, 3),
                                                  (ParsingMode.DIAMOND, 5), (ParsingMode.GRAPHENE, 7)])
def test_full_regeneration_still_merges_then_uses_selected_mode(mode, expected_calls):
    from framework.logic import LogicExpression
    from framework.task import Objective, Requirement, TaskSpec, apply_task_delta
    base = TaskSpec("cohort_curation", Objective("cohort_retrieval", "diabetes"),
                    requirements=(Requirement("a", "condition", {"term": "diabetes"}),),
                    logic=LogicExpression.predicate("a", "diabetes"))
    predicates = [{"predicate_id": "p1", "text": "diabetes"}, {"predicate_id": "p2", "text": "hypertension"}]
    compiled = {"requirements": [requirement("p1", "condition", {"term": "diabetes"}),
                                  requirement("p2", "condition", {"term": "hypertension"})], "assumptions": []}
    responses = [{"merged_request": "diabetes OR hypertension"}]
    if mode is ParsingMode.CARBON:
        responses += [{**flat(predicates, "p1 OR p2"), **compiled}]
    elif mode is ParsingMode.GRAPHITE:
        responses += [flat(predicates, "p1 OR p2"), compiled]
    else:
        responses += [extraction(predicates), relationships("p1 OR p2")]
        if mode is ParsingMode.DIAMOND:
            responses += [{"assignments": [{"predicate_id": p["predicate_id"], "domain": "condition"} for p in predicates]}, compiled]
        else:
            for p, r in zip(predicates, compiled["requirements"]):
                responses += [{"assignments": [{"predicate_id": p["predicate_id"], "domain": "condition"}]},
                              {"requirements": [r], "assumptions": []}]
    llm = FakeLLM(responses)
    parser = CohortTaskParser(llm, parsing_mode=mode, correction_mode=CorrectionMode.FULL_REGENERATION)
    updated = apply_task_delta(base, parser.interpret_correction(base, "hypertension can also qualify").task_delta)
    assert updated.logic.kind == "or"
    assert {r.payload["term"] for r in updated.requirements} == {"diabetes", "hypertension"}
    assert len(llm.calls) == expected_calls
    assert llm.calls[0]["stage_label"] == "Correction merge"
