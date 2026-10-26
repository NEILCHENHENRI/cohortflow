from agents.cohort.parser import CohortTaskParser
from framework.correction import CorrectionFallbackRequired, CorrectionMode
from framework.logic import LogicExpression
from framework.task import Objective, Requirement, TaskSpec, apply_task_delta
from framework.task_delta import predicate_ref_maps
from framework.parsing import ParsingMode
from copy import deepcopy
import json
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


def req(rid, kind, payload, polarity="include"):
    return Requirement(rid, kind, payload, polarity=polarity)


def task(requirements, logic, relations=()):
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "test"),
        deliverables=(),
        requirements=tuple(requirements),
        logic=logic,
        context={"relations": list(relations)},
        original_request="test",
    )


def compiled(rid, kind, payload):
    return {
        "requirements": [{
            "requirement_id": rid,
            "kind": kind,
            "payload": payload,
            "polarity": "include",
            "required": True,
        }],
        "assumptions": [],
    }


def interpreted(actions):
    return {
        "actions": actions,
        "requires_clarification": False,
        "clarification_question": None,
    }


def extraction_for(actions):
    return {"intents": [
        {"intent_id": f"c{i}", "action": action["action"],
         "instruction": "Apply only this requested change: " + json.dumps(action),
         "requirement_text": action.get("requirement_text")}
        for i, action in enumerate(actions, 1)
    ], "requires_clarification": False, "clarification_question": None, "clarification_rationale": None}


def attachments_for(actions):
    return interpreted([{**action, "intent_id": f"c{i}"} for i, action in enumerate(actions, 1)])


def add_action(text="metformin", local="new_1", anchor="ROOT", operator="AND"):
    return {"action": "ADD", "local_ref": local, "requirement_text": text,
            "excluded": False, "anchor": anchor, "operator": operator, "relation_memberships": []}


def pair_task():
    return task(
        [req("a", "condition", {"term": "diabetes"}), req("b", "medication", {"term": "metformin"})],
        LogicExpression.and_(LogicExpression.predicate("a", "diabetes"),
                             LogicExpression.predicate("b", "metformin"), node_id="g1"),
    )


@pytest.mark.parametrize("mode,expected_calls", [(ParsingMode.CARBON, 1), (ParsingMode.GRAPHITE, 2),
                                                  (ParsingMode.DIAMOND, 4), (ParsingMode.GRAPHENE, 6)])
def test_localized_modes_preserve_untouched_state_and_compile_only_changes(mode, expected_calls):
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    actions = [{"action": "MODIFY", "target": "p2", "requirement_text": "insulin"},
               {**add_action("CKD"), "excluded": True}]
    changed = [*compiled("p2", "medication", {"term": "insulin"})["requirements"],
               *compiled("p3", "condition", {"term": "CKD"})["requirements"]]
    batch = {"requirements": changed, "assumptions": []}
    assignments = [{"predicate_id": "p2", "domain": "medication"}, {"predicate_id": "p3", "domain": "condition"}]
    if mode is ParsingMode.CARBON:
        responses = [{**interpreted(actions), "requirements": [changed[0], {**changed[1], "requirement_id": "new_1"}], "assumptions": []}]
    elif mode is ParsingMode.GRAPHITE:
        responses = [interpreted(actions), batch]
    elif mode is ParsingMode.DIAMOND:
        responses = [extraction_for(actions), attachments_for(actions), {"assignments": assignments}, batch]
    else:
        responses = [extraction_for(actions), attachments_for(actions),
                     {"assignments": assignments[:1]}, {"requirements": changed[:1], "assumptions": []},
                     {"assignments": assignments[1:]}, {"requirements": changed[1:], "assumptions": []}]
    llm = FakeLLM(responses)
    parser = CohortTaskParser(llm, parsing_mode=mode, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    delta = parser.interpret_correction(base, "replace metformin with insulin and exclude CKD").task_delta
    updated = apply_task_delta(base, delta)
    assert base.to_dict() == snapshot
    assert next(r for r in updated.requirements if r.requirement_id == "a") == base.requirements[0]
    assert next(r for r in updated.requirements if r.requirement_id == "b").payload == {"term": "insulin"}
    ckd = next(r for r in updated.requirements if r.payload.get("term") == "CKD")
    assert ckd.polarity == "exclude"
    assert updated.logic.contains_not()
    assert len(llm.calls) == expected_calls
    assert not llm.responses
    for call in llm.calls:
        if call["stage_label"] == "Predicate compilation":
            assert '"predicate_id": "p1"' not in call["user_prompt"]
    if mode in {ParsingMode.DIAMOND, ParsingMode.GRAPHENE}:
        assert llm.calls[1]["user_prompt"].startswith("FIXED CHANGE INTENTS")
        assert '- {"intent_id": "c1"' in llm.calls[1]["user_prompt"]
        for call in llm.calls:
            if call["stage_label"] == "Predicate compilation":
                assert "FIXED DOMAIN ASSIGNMENTS" in call["user_prompt"]
                assert "admission_type:" not in call["user_prompt"]


@pytest.mark.parametrize("mode", list(ParsingMode))
def test_relation_only_correction_never_recompiles_endpoints(mode):
    base = pair_task()
    actions = [{"action": "ADD_RELATION", "local_ref": "new_r1", "relation": "BEFORE",
                "left": "p1", "right": "p2", "max_gap_value": 2, "max_gap_unit": "day", "same_admission": True}]
    if mode is ParsingMode.CARBON:
        responses = [{**interpreted(actions), "requirements": [], "assumptions": []}]
    elif mode is ParsingMode.GRAPHITE:
        responses = [interpreted(actions)]
    else:
        responses = [extraction_for(actions), attachments_for(actions)]
    llm = FakeLLM(responses)
    updated = apply_task_delta(base, CohortTaskParser(llm, parsing_mode=mode, correction_mode="robust_delta", localized_strategy="mode_aware")
                               .interpret_correction(base, "diabetes before metformin within two days during one admission").task_delta)
    assert updated.requirements == base.requirements
    assert updated.context["relations"][0]["max_gap_hours"] == 48
    assert len(llm.calls) == (1 if mode in {ParsingMode.CARBON, ParsingMode.GRAPHITE} else 2)
    assert not any(c["stage_label"] in {"Predicate compilation", "Table assignment"} for c in llm.calls)


@pytest.mark.parametrize("mode", [ParsingMode.DIAMOND, ParsingMode.GRAPHENE])
@pytest.mark.parametrize("field_count,expected_calls", [(30, 3), (31, 4)])
def test_localized_domain_selection_boundary(mode, field_count, expected_calls, monkeypatch):
    monkeypatch.setattr("agents.cohort.parser.semantic_field_count", lambda: field_count)
    actions = [add_action("insulin")]
    responses = [extraction_for(actions), attachments_for(actions)]
    if field_count > 30:
        responses.append({"assignments": [{"predicate_id": "p3", "domain": "medication"}]})
    responses.append(compiled("p3", "medication", {"term": "insulin"}))
    llm = FakeLLM(responses)
    CohortTaskParser(llm, parsing_mode=mode, correction_mode="robust_delta", localized_strategy="mode_aware").interpret_correction(pair_task(), "add insulin")
    assert len(llm.calls) == expected_calls


@pytest.mark.parametrize("mode", list(ParsingMode))
def test_original_localized_baseline_remains_one_plus_m_calls(mode):
    base = pair_task()
    actions = [add_action("insulin"), add_action("CKD", "new_2")]
    llm = FakeLLM([interpreted(actions), compiled("p3", "medication", {"term": "insulin"}),
                   compiled("p4", "condition", {"term": "CKD"})])
    updated = apply_task_delta(base, CohortTaskParser(llm, parsing_mode=mode, correction_mode="robust_delta", localized_strategy="legacy")
                               .interpret_correction(base, "add insulin and CKD").task_delta)
    assert len(updated.requirements) == 4
    assert [c["stage_label"] for c in llm.calls] == ["Correction interpretation", "Predicate compilation", "Predicate compilation"]


@pytest.mark.parametrize("mode", list(ParsingMode))
def test_localized_early_clarification_stops_without_mutation(mode):
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    clarification = {"actions": [], "requires_clarification": True,
                     "clarification_question": "Which criterion?", "requirements": [], "assumptions": []}
    llm = FakeLLM([clarification])
    result = CohortTaskParser(llm, parsing_mode=mode, correction_mode="robust_delta", localized_strategy="mode_aware").interpret_correction(base, "change that one")
    assert result.intent.requires_clarification
    assert base.to_dict() == snapshot
    assert len(llm.calls) == 1
    assert not result.task_delta.acts


@pytest.mark.parametrize("corruption", ["text", "missing", "order", "action", "id"])
def test_attachment_cannot_rewrite_or_drop_extracted_intents(corruption):
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    actions = [add_action("insulin"), add_action("CKD", "new_2")]
    output = attachments_for(actions)
    if corruption == "text":
        output["actions"][0]["requirement_text"] = "aspirin"
    elif corruption == "missing":
        output["actions"].pop()
    elif corruption == "order":
        output["actions"].reverse()
    elif corruption == "action":
        output["actions"][0]["action"] = "REMOVE"
    else:
        output["actions"][0]["intent_id"] = "c99"
    llm = FakeLLM([extraction_for(actions), output])
    parser = CohortTaskParser(llm, parsing_mode="diamond", correction_mode="robust_delta", structural_retry_limit=0, localized_strategy="mode_aware")
    with pytest.raises(CorrectionFallbackRequired):
        parser.interpret_correction(base, "add insulin and CKD")
    assert base.to_dict() == snapshot
    assert len(llm.calls) == 2


def test_carbon_rejects_compilation_of_untouched_predicates():
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    llm = FakeLLM([{**interpreted([add_action("insulin")]),
                   **compiled("p1", "condition", {"term": "changed diabetes"})}])
    parser = CohortTaskParser(llm, parsing_mode="carbon", correction_mode="robust_delta", structural_retry_limit=0, localized_strategy="mode_aware")
    with pytest.raises(CorrectionFallbackRequired, match="interpretation failed"):
        parser.interpret_correction(base, "add insulin")
    assert base.to_dict() == snapshot


def test_batch_failure_never_applies_a_partial_correction():
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    actions = [add_action("insulin"), {"action": "REMOVE", "target": "p1"}]
    llm = FakeLLM([interpreted(actions), compiled("wrong_id", "medication", {"term": "insulin"})])
    parser = CohortTaskParser(llm, correction_mode="robust_delta", structural_retry_limit=0, localized_strategy="mode_aware")
    with pytest.raises(CorrectionFallbackRequired):
        parser.interpret_correction(base, "add insulin and remove diabetes")
    assert base.to_dict() == snapshot


def test_domain_assignment_cannot_drift_during_compilation():
    actions = [add_action("insulin")]
    llm = FakeLLM([extraction_for(actions), attachments_for(actions),
                   {"assignments": [{"predicate_id": "p3", "domain": "medication"}]},
                   compiled("p3", "condition", {"term": "insulin"})])
    parser = CohortTaskParser(llm, parsing_mode="diamond", correction_mode="robust_delta", structural_retry_limit=0, localized_strategy="mode_aware")
    with pytest.raises(CorrectionFallbackRequired) as exc:
        parser.interpret_correction(pair_task(), "add insulin")
    assert "changed assigned domain" in exc.value.reason


@pytest.mark.parametrize("mode,settings,expected", [
    ("carbon", {"carbon": True}, [True]),
    ("graphite", {"correction_delta": True, "compilation": False}, [True, False]),
    ("diamond", {"extraction": True, "relationships": False, "table_selection": True, "compilation": False}, [True, False, True, False]),
    ("graphene", {"extraction": False, "relationships": True, "table_selection": False, "compilation": True}, [False, True, False, True]),
])
def test_localized_passes_honor_and_restore_thinking(mode, settings, expected):
    actions = [add_action("insulin")]
    if mode == "carbon":
        responses = [{**interpreted(actions), **compiled("new_1", "medication", {"term": "insulin"})}]
    elif mode == "graphite":
        responses = [interpreted(actions), compiled("p3", "medication", {"term": "insulin"})]
    else:
        responses = [extraction_for(actions), attachments_for(actions),
                     {"assignments": [{"predicate_id": "p3", "domain": "medication"}]},
                     compiled("p3", "medication", {"term": "insulin"})]

    class ThinkingLLM(FakeLLM):
        think = "original"

        def generate_structured(self, **kwargs):
            observed.append(self.think)
            return super().generate_structured(**kwargs)

    observed = []
    llm = ThinkingLLM(responses)
    CohortTaskParser(llm, parsing_mode=mode, correction_mode="robust_delta", thinking_by_pass=settings, localized_strategy="mode_aware").interpret_correction(pair_task(), "add insulin")
    assert observed == expected
    assert llm.think == "original"


def test_steering_interrupt_between_localized_passes_is_not_reported_as_bad_correction():
    from framework.steering import SteeringYield

    class Progress:
        def __call__(self, *args):
            pass

        def should_yield(self):
            return True

    actions = [add_action("insulin")]
    llm = FakeLLM([extraction_for(actions)])
    base = pair_task()
    before = deepcopy(base.to_dict())
    parser = CohortTaskParser(llm, parsing_mode="diamond", correction_mode="robust_delta", localized_strategy="mode_aware")
    with pytest.raises(SteeringYield):
        parser.interpret_correction(base, "add insulin", on_progress=Progress())
    assert len(llm.calls) == 1
    assert base.to_dict() == before


@pytest.mark.parametrize("mode", list(ParsingMode))
def test_localized_modes_execute_and_reuse_existing_patient_results(mode):
    from agents.cohort.agent import CohortAgent

    class Resolver:
        def resolve_condition(self, term):
            return term

    class Repository:
        calls = []

        def patients_with_condition(self, concept, payload=None):
            self.calls.append(concept)
            return {"diabetes": {1, 2}, "hypertension": {2, 3}}[concept]

    base = task([req("a", "condition", {"term": "diabetes"})], LogicExpression.predicate("a", "diabetes"))
    actions = [add_action("hypertension", operator="OR")]
    if mode is ParsingMode.CARBON:
        responses = [{**interpreted(actions), **compiled("new_1", "condition", {"term": "hypertension"})}]
    elif mode is ParsingMode.GRAPHITE:
        responses = [interpreted(actions), compiled("p2", "condition", {"term": "hypertension"})]
    else:
        responses = [extraction_for(actions), attachments_for(actions),
                     {"assignments": [{"predicate_id": "p2", "domain": "condition"}]},
                     compiled("p2", "condition", {"term": "hypertension"})]
    repository = Repository()
    parser = CohortTaskParser(FakeLLM(responses), parsing_mode=mode, correction_mode="robust_delta", localized_strategy="mode_aware")
    session = CohortAgent(parser, Resolver(), repository).create_session()
    session.initialize(base)
    session.execute()
    assert session.summary()["cohort_size"] == 2
    analysis = session.analyze_correction("hypertension can also qualify")
    session.stage_correction(analysis, session.plan_correction(analysis))
    assert session.state.task == base  # staging cannot alter the committed task
    session.execute_staged()
    assert session.summary()["cohort_size"] == 3
    assert repository.calls == ["diabetes", "hypertension"]  # existing evidence reused


def test_modify_after_set_excluded_preserves_the_requested_exclusion():
    actions = [{"action": "SET_EXCLUDED", "target": "p2", "excluded": True},
               {"action": "MODIFY", "target": "p2", "requirement_text": "insulin"}]
    llm = FakeLLM([interpreted(actions), compiled("p2", "medication", {"term": "insulin"})])
    base = pair_task()
    updated = apply_task_delta(base, CohortTaskParser(llm, correction_mode="robust_delta", localized_strategy="mode_aware")
                               .interpret_correction(base, "exclude insulin instead of requiring metformin").task_delta)
    assert next(r for r in updated.requirements if r.requirement_id == "b").polarity == "exclude"
    assert updated.logic.contains_not()


def test_individual_compilation_assumptions_cannot_overwrite_each_other_or_existing_assumptions():
    from dataclasses import replace
    from framework.task import Assumption

    base = replace(pair_task(), assumptions=(Assumption("a1", "Original assumption"),))
    actions = [add_action("insulin"), add_action("CKD", "new_2")]
    one = {**compiled("p3", "medication", {"term": "insulin"}),
           "assumptions": [{"assumption_id": "a1", "statement": "Insulin assumption"}]}
    two = {**compiled("p4", "condition", {"term": "CKD"}),
           "assumptions": [{"assumption_id": "a1", "statement": "CKD assumption"}]}
    llm = FakeLLM([extraction_for(actions), attachments_for(actions),
                   {"assignments": [{"predicate_id": "p3", "domain": "medication"}]}, one,
                   {"assignments": [{"predicate_id": "p4", "domain": "condition"}]}, two])
    updated = apply_task_delta(base, CohortTaskParser(llm, parsing_mode="graphene", correction_mode="robust_delta", localized_strategy="mode_aware")
                               .interpret_correction(base, "add insulin and CKD").task_delta)
    assert {a.statement for a in updated.assumptions} == {"Original assumption", "Insulin assumption", "CKD assumption"}
    assert len({a.assumption_id for a in updated.assumptions}) == 3


def test_attachment_can_request_clarification_without_compilation():
    actions = [add_action("insulin")]
    llm = FakeLLM([extraction_for(actions), {"requires_clarification": True, "actions": [],
                  "clarification_question": "Should insulin replace metformin or be required as well?"}])
    analysis = CohortTaskParser(llm, parsing_mode="diamond", correction_mode="robust_delta", localized_strategy="mode_aware").interpret_correction(pair_task(), "insulin too")
    assert analysis.intent.requires_clarification
    assert not analysis.task_delta.acts
    assert len(llm.calls) == 2


@pytest.mark.parametrize("strategy", ["mode_aware", "legacy"])
def test_specialized_patient_override_needs_no_semantic_calls(strategy):
    llm = FakeLLM([])
    analysis = CohortTaskParser(llm, correction_mode="robust_delta", localized_strategy=strategy)
    updated = apply_task_delta(pair_task(), analysis.interpret_correction(pair_task(), "include patient 42").task_delta)
    assert updated.context["individual_overrides"]["include"] == (42,)
    assert not llm.calls


@pytest.mark.parametrize("mode", list(ParsingMode))
def test_modes_preserve_new_local_handles_in_nested_logic_and_relations(mode):
    base = task([req("a", "condition", {"term": "diabetes"})], LogicExpression.predicate("a", "diabetes"))
    actions = [add_action("metformin"), add_action("insulin", "new_2", "new_1", "OR"),
               {"action": "ADD_RELATION", "local_ref": "new_r1", "relation": "BEFORE", "left": "p1", "right": "new_1"}]
    results = [compiled("p2", "medication", {"term": "metformin"}), compiled("p3", "medication", {"term": "insulin"})]
    if mode is ParsingMode.CARBON:
        responses = [{**interpreted(actions), "requirements": [
            {**result["requirements"][0], "requirement_id": f"new_{i}"} for i, result in enumerate(results, 1)
        ], "assumptions": []}]
    elif mode is ParsingMode.GRAPHITE:
        responses = [interpreted(actions), {"requirements": [result["requirements"][0] for result in results], "assumptions": []}]
    elif mode is ParsingMode.DIAMOND:
        responses = [extraction_for(actions), attachments_for(actions),
                     {"assignments": [{"predicate_id": p, "domain": "medication"} for p in ("p2", "p3")]},
                     {"requirements": [result["requirements"][0] for result in results], "assumptions": []}]
    else:
        responses = [extraction_for(actions), attachments_for(actions),
                     {"assignments": [{"predicate_id": "p2", "domain": "medication"}]}, results[0],
                     {"assignments": [{"predicate_id": "p3", "domain": "medication"}]}, results[1]]
    parser = CohortTaskParser(FakeLLM(responses), parsing_mode=mode, correction_mode="robust_delta", localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "require metformin after diabetes, or insulin as an alternative").task_delta)
    assert updated.logic.kind == "and"
    assert any(child.kind == "or" for child in updated.logic.children)
    refs, _ = predicate_ref_maps(updated)
    assert updated.context["relations"][0]["right_requirement_id"] == refs["p2"]
    assert next(r for r in updated.requirements if r.requirement_id == refs["p3"]).payload == {"term": "insulin"}


def test_setting_change_mid_correction_does_not_mix_pipelines():
    actions = [add_action("insulin")]

    class SwitchingLLM(FakeLLM):
        def generate_structured(self, **kwargs):
            if not self.calls:
                parser.configure_parsing(mode="carbon")
                parser.configure_localized_correction("legacy")
            return super().generate_structured(**kwargs)

    llm = SwitchingLLM([extraction_for(actions), attachments_for(actions),
                        {"assignments": [{"predicate_id": "p3", "domain": "medication"}]},
                        compiled("p3", "medication", {"term": "insulin"})])
    parser = CohortTaskParser(llm, parsing_mode="diamond", correction_mode="robust_delta", localized_strategy="mode_aware")
    parser.interpret_correction(pair_task(), "add insulin")
    assert [c["stage_label"] for c in llm.calls] == ["Correction intent extraction", "Correction attachment", "Table assignment", "Predicate compilation"]
    assert parser.parsing_mode is ParsingMode.CARBON  # applies to the next operation


def test_carbon_prompt_has_no_conflicting_do_not_compile_instruction():
    from prompts.cohort import CARBON_CORRECTION_SYSTEM_PROMPT
    assert "Do NOT compile clinical shorthand" not in CARBON_CORRECTION_SYSTEM_PROMPT
    assert "transaction-local handle" in CARBON_CORRECTION_SYSTEM_PROMPT


def test_add_excluded_requirement_is_attached_as_not():
    base = task(
        [req("diabetes", "condition", {"term": "diabetes"})],
        LogicExpression.predicate("diabetes", "diabetes"),
    )
    llm = FakeLLM([
        interpreted([{
            "action": "ADD", "local_ref": "new_1",
            "requirement_text": "patients with CKD", "excluded": True,
            "anchor": "ROOT", "operator": "AND", "placement": "WITH",
            "relation_memberships": [],
        }]),
        compiled("p2", "condition", {"term": "CKD"}),
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "also exclude CKD").task_delta)
    ckd = next(r for r in updated.requirements if r.payload.get("term") == "CKD")
    assert ckd.polarity == "exclude"
    assert updated.logic.kind == "and"
    assert any(child.kind == "not" for child in updated.logic.children)


def test_compound_add_can_reference_transaction_local_predicate_in_new_relation():
    base = task(
        [req("diabetes", "condition", {"term": "diabetes"})],
        LogicExpression.predicate("diabetes", "diabetes"),
    )
    llm = FakeLLM([
        interpreted([
            {
                "action": "ADD", "local_ref": "new_1",
                "requirement_text": "furosemide", "excluded": False,
                "anchor": "ROOT", "operator": "AND", "placement": "WITH",
                "relation_memberships": [],
            },
            {
                "action": "ADD_RELATION", "local_ref": "new_r1",
                "relation": "BEFORE", "left": "p1", "right": "new_1",
                "max_gap_value": 30, "max_gap_unit": "day",
                "same_admission": True,
            },
        ]),
        compiled("p2", "medication", {"term": "furosemide"}),
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "add furosemide within 30 days after diabetes").task_delta)
    refs, _ = predicate_ref_maps(updated)
    relation = updated.context["relations"][0]
    assert relation["relation_id"] == "r1"
    assert relation["left_requirement_id"] == refs["p1"]
    assert relation["right_requirement_id"] == refs["p2"]
    assert relation["max_gap_hours"] == 720
    assert relation["same_admission"] is True


def test_change_operator_costs_one_correction_call_and_no_compilation():
    base = task(
        [req("a", "condition", {"term": "A"}), req("b", "condition", {"term": "B"})],
        LogicExpression.or_(
            LogicExpression.predicate("a", "A"),
            LogicExpression.predicate("b", "B"),
            node_id="g1",
        ),
    )
    llm = FakeLLM([interpreted([{"action": "CHANGE_OPERATOR", "target": "g1", "operator": "AND"}])])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "require both").task_delta)
    assert updated.logic.kind == "and"
    assert len(llm.calls) == 1


def test_remove_predicate_deterministically_removes_dangling_relation():
    base = task(
        [req("a", "condition", {"term": "A"}), req("b", "medication", {"term": "B"})],
        LogicExpression.and_(
            LogicExpression.predicate("a", "A"),
            LogicExpression.predicate("b", "B"),
            node_id="g1",
        ),
        relations=[{
            "relation_id": "r1", "relation": "before",
            "left_requirement_id": "a", "right_requirement_id": "b",
            "max_gap_hours": 4, "same_admission": False,
        }],
    )
    llm = FakeLLM([interpreted([{"action": "REMOVE", "target": "p2"}])])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "remove B").task_delta)
    assert len(updated.requirements) == 1
    assert updated.context["relations"] == []
    assert updated.logic.kind == "predicate"
    assert len(llm.calls) == 1


def test_set_excluded_is_idempotent_not_toggle():
    base = task(
        [req("a", "condition", {"term": "A"})],
        LogicExpression.predicate("a", "A"),
    )
    llm = FakeLLM([interpreted([{"action": "SET_EXCLUDED", "target": "p1", "excluded": True}])])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    delta = parser.interpret_correction(base, "exclude A").task_delta
    once = apply_task_delta(base, delta)
    twice = apply_task_delta(once, delta)
    assert once.requirements[0].polarity == twice.requirements[0].polarity == "exclude"
    assert once.logic.kind == twice.logic.kind == "not"


def test_modify_relation_can_change_type_endpoints_and_gap_in_place():
    base = task(
        [req("a", "condition", {"term": "A"}), req("b", "medication", {"term": "B"})],
        LogicExpression.and_(
            LogicExpression.predicate("a", "A"),
            LogicExpression.predicate("b", "B"),
            node_id="g1",
        ),
        relations=[{
            "relation_id": "r1", "relation": "before",
            "left_requirement_id": "a", "right_requirement_id": "b",
            "max_gap_hours": 24, "same_admission": False,
        }],
    )
    llm = FakeLLM([interpreted([{
        "action": "MODIFY_RELATION", "target": "r1", "relation": "AFTER",
        "left": "p2", "right": "p1", "max_gap_value": 7,
        "max_gap_unit": "day", "same_admission": True,
    }])])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "actually B should be after A within 7 days").task_delta)
    refs, _ = predicate_ref_maps(updated)
    relation = updated.context["relations"][0]
    assert relation["relation_id"] == "r1"
    assert relation["relation"] == "after"
    assert relation["left_requirement_id"] == refs["p2"]
    assert relation["right_requirement_id"] == refs["p1"]
    assert relation["max_gap_hours"] == 168
    assert relation["same_admission"] is True


def test_add_alternative_around_existing_predicate_anchor():
    base = task(
        [req("a", "condition", {"term": "A"}), req("b", "medication", {"term": "metformin"})],
        LogicExpression.and_(
            LogicExpression.predicate("a", "A"),
            LogicExpression.predicate("b", "metformin"),
            node_id="g1",
        ),
    )
    llm = FakeLLM([
        interpreted([{
            "action": "ADD", "local_ref": "new_1",
            "requirement_text": "insulin", "excluded": False,
            "anchor": "p2", "operator": "OR", "placement": "WITH",
            "relation_memberships": [],
        }]),
        compiled("p3", "medication", {"term": "insulin"}),
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "insulin can also satisfy the medication requirement").task_delta)
    assert updated.logic.kind == "and"
    assert any(child.kind == "or" for child in updated.logic.children)


def test_noop_semantic_action_is_rejected_for_explicit_fallback_confirmation():
    base = task(
        [req("a", "condition", {"term": "A"})],
        LogicExpression.predicate("a", "A"),
    )
    llm = FakeLLM([interpreted([{"action": "SET_EXCLUDED", "target": "p1", "excluded": False}])])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    try:
        parser.interpret_correction(base, "change something")
        assert False, "expected fallback request"
    except CorrectionFallbackRequired as exc:
        assert "no semantic change" in exc.reason
    assert len(llm.calls) == 1


def test_two_new_requirements_can_cross_reference_local_handle_for_nested_logic():
    base = task(
        [req("a", "condition", {"term": "A"})],
        LogicExpression.predicate("a", "A"),
    )
    llm = FakeLLM([
        interpreted([
            {
                "action": "ADD", "local_ref": "new_1",
                "requirement_text": "metformin", "excluded": False,
                "anchor": "ROOT", "operator": "AND", "placement": "WITH",
                "relation_memberships": [],
            },
            {
                "action": "ADD", "local_ref": "new_2",
                "requirement_text": "insulin", "excluded": False,
                "anchor": "new_1", "operator": "OR", "placement": "WITH",
                "relation_memberships": [],
            },
        ]),
        {"requirements": [
            *compiled("p2", "medication", {"term": "metformin"})["requirements"],
            *compiled("p3", "medication", {"term": "insulin"})["requirements"],
        ], "assumptions": []},
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "require metformin or insulin as well").task_delta)
    assert updated.logic.kind == "and"
    assert any(child.kind == "or" for child in updated.logic.children)


def test_added_requirement_can_join_existing_same_admission_relation():
    base = task(
        [
            req("a", "procedure", {"term": "paracentesis"}),
            req("b", "medication", {"term": "furosemide"}),
        ],
        LogicExpression.and_(
            LogicExpression.predicate("a", "paracentesis"),
            LogicExpression.predicate("b", "furosemide"),
            node_id="g1",
        ),
        relations=[{
            "relation_id": "r1", "relation": "same_admission", "members": ["a", "b"]
        }],
    )
    llm = FakeLLM([
        interpreted([{
            "action": "ADD", "local_ref": "new_1",
            "requirement_text": "lactulose", "excluded": False,
            "anchor": "ROOT", "operator": "AND", "placement": "WITH",
            "relation_memberships": ["r1"],
        }]),
        compiled("p3", "medication", {"term": "lactulose"}),
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "also require lactulose during that admission").task_delta)
    refs, _ = predicate_ref_maps(updated)
    assert refs["p3"] in updated.context["relations"][0]["members"]


def test_adult_addition_wraps_existing_or_root_and_compiles_locally():
    base = task(
        [req("a", "condition", {"term": "diabetes"}), req("b", "condition", {"term": "hypertension"})],
        LogicExpression.or_(
            LogicExpression.predicate("a", "diabetes"),
            LogicExpression.predicate("b", "hypertension"),
            node_id="g1",
        ),
    )
    llm = FakeLLM([
        interpreted([{
            "action": "ADD", "local_ref": "new_1",
            "requirement_text": "patients must be adults", "excluded": False,
            "anchor": "ROOT", "operator": "AND", "placement": "WITH",
            "relation_memberships": [],
        }]),
        compiled("p3", "patient", {"attribute": "age", "operator": ">=", "value": 18}),
    ])
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA, localized_strategy="mode_aware")
    updated = apply_task_delta(base, parser.interpret_correction(base, "patients must be adults").task_delta)
    assert updated.logic.kind == "and"
    assert any(child.kind == "or" and child.node_id == "g1" for child in updated.logic.children)
    assert any(r.payload.get("attribute") == "age" and r.payload.get("value") == 18 for r in updated.requirements)


@pytest.mark.parametrize("anchor,operator,expected_kind", [
    ("p2", "OR", "and"),
    ("ROOT", "OR", "or"),
])
def test_compact_v2_default_preserves_scope_and_untouched_requirements(anchor, operator, expected_kind):
    from prompts.correction import COMPACT_CORRECTION_PROMPT
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    actions = [add_action("insulin", anchor=anchor, operator=operator)]
    llm = FakeLLM([interpreted(actions), compiled("p3", "medication", {"term": "insulin"})])
    parser = CohortTaskParser(llm)
    assert parser.correction_mode is CorrectionMode.ROBUST_DELTA
    assert parser.localized_strategy == "compact"
    updated = apply_task_delta(base, parser.interpret_correction(base, "insulin counts too").task_delta)
    assert base.to_dict() == snapshot
    assert updated.requirements[:2] == base.requirements
    assert updated.logic.kind == expected_kind
    if anchor == "p2":
        assert updated.logic.children[0].predicate_id == "a"
        assert updated.logic.children[1].kind == "or"
    assert llm.calls[0]["system_prompt"] == COMPACT_CORRECTION_PROMPT
    assert len(llm.calls) == 2
    assert not llm.responses


def test_compact_v2_ambiguous_edit_preserves_current_state():
    base = pair_task()
    snapshot = deepcopy(base.to_dict())
    llm = FakeLLM([{
        "actions": [], "requires_clarification": True,
        "clarification_question": "Which requirement should change?",
    }])
    result = CohortTaskParser(llm).interpret_correction(base, "change that")
    assert result.intent.requires_clarification
    assert not result.task_delta.acts
    assert base.to_dict() == snapshot
    assert len(llm.calls) == 1
