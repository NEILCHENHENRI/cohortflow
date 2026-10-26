from __future__ import annotations

import json

import pytest

from agents.cohort.adapters.registry import FieldRegistry
from agents.cohort.adapters.models import FieldSpec
from agents.cohort.adapters.semantics import default_semantic_description, identifier_or_code_hint, semantic_field_name


def test_identifier_or_code_hint_uses_names_and_leading_zero_values():
    assert identifier_or_code_hint("subject_id", ["1", "2"])
    assert identifier_or_code_hint("custom_code", ["100", "200"])
    assert identifier_or_code_hint("account", ["001", "002", "10"])
    assert not identifier_or_code_hint("dose", ["1", "2", "3"])
    assert not identifier_or_code_hint("empty", ["", " "])


def test_semantic_names_and_descriptions_are_stable_and_database_agnostic():
    assert semantic_field_name("admissions.csv", "admission_type") == "admissions.admission_type"
    assert default_semantic_description("labs.parquet", "chart_time") == "chart time recorded in labs"
    assert semantic_field_name("ward_a.labs", "value") == "ward_a.labs.value"
    assert semantic_field_name("labs.CSV.GZ", "value") == "labs.value"


def test_registry_rejects_duplicate_semantic_names():
    field = FieldSpec("stock.value", "Quantity", "stock", "value", "value", "numeric")
    with pytest.raises(ValueError, match="unique"):
        FieldRegistry([field, field])


def test_registry_prioritizes_identifier_semantics_and_preserves_raw_units(tmp_path):
    catalog = [
        {"semantic_field": "t.code", "description": "Code", "table": "t", "column": "code", "is_numeric": True, "is_identifier_or_code": True},
        {"semantic_field": "t.value", "description": "Value", "table": "t", "column": "value", "is_numeric": True, "unit_column": "unit"},
    ]
    (tmp_path / "field_catalog.json").write_text(json.dumps(catalog))
    derived = {"table": "t", "source_column": "value", "target_column": "normalized", "kind": "numeric"}
    path = tmp_path / "construction_spec.json"
    path.write_text(json.dumps({"derived_fields": [derived]}))
    registry = FieldRegistry.from_adapter_dir(tmp_path)
    assert registry.get("t.code").value_kind == "identifier"
    assert registry.get("t.value").unit_column == "unit"
    path.write_text(json.dumps({"derived_fields": [derived, derived]}))
    with pytest.raises(ValueError, match="ambiguous derived"):
        FieldRegistry.from_adapter_dir(tmp_path)


def test_field_registry_builds_runtime_contract_from_catalog_and_construction_spec(tmp_path):
    catalog = [
        {
            "semantic_field": "labs.value",
            "description": "laboratory numeric result",
            "table": "labs",
            "column": "value",
            "is_numeric": True,
            "is_datetime": False,
            "is_identifier_or_code": False,
            "unit_column": "value_unit",
        },
        {
            "semantic_field": "labs.code",
            "description": "laboratory code",
            "table": "labs",
            "column": "code",
            "is_numeric": False,
            "is_datetime": False,
            "is_identifier_or_code": True,
            "unit_column": None,
        },
    ]
    construction = {
        "derived_fields": [
            {
                "table": "labs",
                "source_column": "value",
                "target_column": "value_norm",
                "unit_target_column": "value_unit_norm",
            }
        ]
    }
    (tmp_path / "field_catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "construction_spec.json").write_text(json.dumps(construction))

    registry = FieldRegistry.from_adapter_dir(tmp_path)
    value = registry.get("labs.value")
    assert value.value_kind == "numeric"
    assert value.value_column == "value_norm"
    assert value.unit_column == "value_unit_norm"
    assert value.source_unit_column == "value_unit"
    assert registry.get("labs.code").value_kind == "identifier"
    assert "labs.value" in registry
    assert {f.semantic_field for f in registry.all()} == {"labs.value", "labs.code"}
    with pytest.raises(KeyError, match="unknown semantic field"):
        registry.get("missing")


@pytest.fixture
def custom_bundle(tmp_path):
    from agents.cohort.adapters.constructor import adapt_dataset
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'patients.csv').write_text('patient_id,gender\n001,F\n002,M\n003,F\n')
    (source / 'labs.csv').write_text('patient_id,code,value,value_unit,visit_id,time\n001,A,1,g,v1,2026-01-01\n002,A,500,mg,v2,2026-01-02\n003,A,1,mg,v3,2026-01-03\n')
    mapping = {'patient': {'table': 'patients', 'columns': ['patient_id']}, 'tables': {
        'patients': {'patient_columns': ['patient_id']},
        'labs': {'patient_columns': ['patient_id'], 'admission_column': 'visit_id', 'event_time_column': 'time__norm'},
    }}
    root = tmp_path / 'adapted'
    result = adapt_dataset(source, root, mapping=mapping)
    assert result['status'] == 'ready', result
    return root


def _custom_filter(field='labs.value', value=600, unit='mg'):
    return {'filters': [{'field': field, 'operator': '>=', 'value': value, 'unit': unit}]}


def test_custom_pipeline_and_runtime_units_preserve_source_ids(custom_bundle):
    from agents.cohort.adapters.registry import CustomDataset
    dataset = CustomDataset(custom_bundle)
    assert dataset.evaluate_requirement('dataset:labs', _custom_filter()) == {1}
    assert dataset.patient_detail(1)['demographics'] == {'patient_id': '001'}
    assert dataset.all_patients() == {1, 2, 3}
    assert dataset.admissions_matching_requirement('dataset:labs', _custom_filter()) == {(1, 'v1')}
    assert 'laboratory' not in dataset.render_fields()  # no fabricated clinical meaning
    assert 'labs.value' in dataset.render_fields()
    assert dataset.registry.get('labs.value').observed_units == ('g', 'mg')
    log = json.loads((custom_bundle / 'adaptation_log.json').read_text())
    assert all(x['unmatched_rows'] == 0 for x in log['mapping_validation'])


def test_custom_unit_failures_stop_instead_of_returning_empty_cohort(custom_bundle, caplog):
    from agents.cohort.adapters.registry import CustomDataset
    from agents.cohort.adapters.semantics import UnitConversionError
    dataset = CustomDataset(custom_bundle)
    with pytest.raises(UnitConversionError, match='[Cc]linician'):
        dataset.evaluate_requirement('dataset:labs', _custom_filter(unit='mmol'))
    assert 'unresolved' in caplog.text
    with pytest.raises(UnitConversionError, match='missing'):
        dataset.evaluate_requirement('dataset:labs', {'filters': [{'field': 'labs.value', 'operator': '>', 'value': 10}]})
    with pytest.raises(ValueError, match='field not available'):
        dataset.evaluate_requirement('dataset:labs', _custom_filter(field='patients.gender'))
    with pytest.raises(ValueError, match='operator'):
        dataset.evaluate_requirement('dataset:labs', {'filters': [{'field': 'labs.value', 'operator': '; DROP TABLE labs', 'value': 1}]})


def test_custom_unmatched_identity_and_duplicate_parent_block_readiness(tmp_path):
    from agents.cohort.adapters.constructor import adapt_dataset
    source = tmp_path / 'src'; source.mkdir()
    (source / 'patients.csv').write_text('patient_id,gender\n001,F\n002,M\n')
    (source / 'labs.csv').write_text('patient_id,value\n999,10\n')
    result = adapt_dataset(source, tmp_path / 'out')
    assert result['database_created'] and result['status'] == 'needs_review'
    assert any('lack a validated' in x.get('message', '') for x in result['issues'])
    (source / 'patients.csv').write_text('patient_id,gender\n001,F\n001,M\n')
    result = adapt_dataset(source, tmp_path / 'other')
    assert result['status'] == 'needs_review'
    assert any('unique' in x.get('message', '') for x in result['issues'])


def test_custom_composite_keys_and_reviewed_join_chain(tmp_path):
    from agents.cohort.adapters.constructor import adapt_dataset
    from agents.cohort.adapters.registry import CustomDataset
    source = tmp_path / 'src'; source.mkdir()
    (source / 'people.csv').write_text('site,person_key,gender\na,001,F\nb,001,M\n')
    (source / 'visits.csv').write_text('visit_key,site,person_key\nx,a,001\ny,b,001\n')
    (source / 'observations.csv').write_text('encounter,value\nx,10\ny,20\n')
    mapping = {'patient': {'table': 'people', 'columns': ['site', 'person_key']}, 'tables': {
        'observations': {'patient_columns': ['site', 'person_key'], 'joins': [
            {'table': 'visits', 'left_columns': ['encounter'], 'right_columns': ['visit_key']}
        ]}
    }}
    root = tmp_path / 'out'
    assert adapt_dataset(source, root, mapping=mapping)['status'] == 'ready'
    dataset = CustomDataset(root)
    assert dataset.evaluate_requirement('dataset:observations', {'filters': [{'field': 'observations.value', 'operator': '>', 'value': 15}]}) == {2}
    assert dataset.patient_detail(2)['demographics'] == {'site': 'b', 'person_key': '001'}


@pytest.mark.parametrize('source,target,value,expected', [
    ('kg', 'kilogram', '9007199254740993', '9007199254740993'),
    ('kg', 'g', 1, '1000'), ('µg', 'mg', 1, '0.001'),
    ('μg', 'mg', 1, '0.001'), ('degF', 'degC', 32, '0'),
    ('mg/dL', 'g/L', 100, '1'), ('mL', 'L', 1, '.001'),
])
def test_shared_units(source, target, value, expected):
    from agents.cohort.adapters.semantics import UnitConverter
    from decimal import Decimal
    assert UnitConverter().convert(value, source, target) == Decimal(expected)


def test_units_do_not_casefold_si_or_guess_molar_conversions():
    from agents.cohort.adapters.semantics import UnitConverter, UnitConversionError
    from decimal import Decimal
    converter = UnitConverter()
    assert converter.convert(1, 'Mg', 'mg') == Decimal('1000000000')
    for a, b in [('mg/dL', 'mmol/L'), ('mystery', 'g'), ('mg{drugA}', 'mg{drugB}')]:
        with pytest.raises(UnitConversionError):
            converter.convert(1, a, b)
    assert UnitConverter({'local_mg': 'mg'}).convert(1000, 'local_mg', 'g') == 1


@pytest.mark.parametrize('mode', ['carbon', 'graphite', 'diamond', 'graphene'])
def test_custom_parser_and_real_cohort_execution_all_modes(custom_bundle, mode):
    from agents.cohort.adapters.registry import CustomDataset
    from agents.cohort.parser import CohortTaskParser
    from agents.cohort.agent import CohortAgent
    from agents.cohort.planner import CohortPlanner
    from framework.parsing import ParsingMode
    dataset = CustomDataset(custom_bundle)
    flat = {'predicates': [{'predicate_id': 'p1', 'text': 'value at least 600 mg'}], 'logic': 'p1',
        'ambiguities': [], 'requires_clarification': False, 'clarification_question': None, 'rationale': ''}
    compiled = {'requirements': [{'requirement_id': 'p1', 'kind': 'dataset:labs', 'payload': _custom_filter(), 'polarity': 'include', 'required': True}], 'assumptions': []}
    if mode == 'carbon':
        responses = [{**flat, **compiled}]
    elif mode == 'graphite':
        responses = [flat, compiled]
    else:
        responses = [{k: v for k, v in flat.items() if k != 'logic'}, {k: v for k, v in flat.items() if k != 'predicates'}, compiled]
    class LLM:
        calls = []
        def generate_structured(self, **kwargs):
            self.calls.append(kwargs)
            return responses.pop(0)
    llm = LLM()
    parser = CohortTaskParser(llm, dataset=dataset, parsing_mode=ParsingMode(mode))
    agent = CohortAgent(parser, None, dataset, planner=CohortPlanner(dataset, dataset.domains()))
    session = agent.create_session()
    session.initialize(parser.parse_task('Find patients with a value of at least 600 mg'))
    session.execute()
    assert session.artifacts.get('cohort_patient_ids').value == {1}
    assert 'labs.value' in llm.calls[-1]['user_prompt']
    assert 'condition' not in json.dumps(llm.calls[-1]['schema'])


def test_custom_long_table_unit_scope_and_temporal_relations(custom_bundle):
    import duckdb
    from agents.cohort.adapters.registry import CustomDataset
    with duckdb.connect(str(custom_bundle / 'dataset.duckdb')) as con:
        con.execute("INSERT INTO labs SELECT patient_id, 'B', '1', 'mmol', visit_id, time, 1, 'mmol', time__norm FROM labs LIMIT 1")
    dataset = CustomDataset(custom_bundle)
    payload = _custom_filter()
    payload['filters'].append({'field': 'labs.code', 'operator': '=', 'value': 'A'})
    assert dataset.evaluate_requirement('dataset:labs', payload) == {1}
    assert dataset.patients_satisfying_relation(relation='before', left_kind='dataset:labs', left_payload=payload,
        right_kind='dataset:labs', right_payload=payload) == set()  # same event isn't before itself


@pytest.mark.parametrize('mode', ['carbon', 'graphite', 'diamond', 'graphene'])
def test_custom_localized_modify_uses_dataset_schema(custom_bundle, mode):
    from agents.cohort.adapters.registry import CustomDataset
    from agents.cohort.parser import CohortTaskParser
    from agents.cohort.agent import CohortAgent
    from agents.cohort.planner import CohortPlanner
    from framework.task import TaskSpec, Requirement, Objective
    from framework.logic import LogicExpression
    dataset = CustomDataset(custom_bundle)
    task = TaskSpec('cohort_curation', Objective('cohort', 'Find patients'),
        requirements=(Requirement('p1', 'dataset:labs', _custom_filter()),),
        logic=LogicExpression.predicate('p1', 'value at least 600 mg'))
    action = {'action': 'MODIFY', 'target': 'p1', 'requirement_text': 'value at least 400 mg'}
    interpretation = {'actions': [action], 'requires_clarification': False, 'clarification_question': None, 'clarification_rationale': None}
    compiled = {'requirements': [{'requirement_id': 'p1', 'kind': 'dataset:labs', 'payload': _custom_filter(value=400), 'polarity': 'include', 'required': True}], 'assumptions': []}
    if mode == 'carbon':
        responses = [{**interpretation, **compiled}]
    elif mode == 'graphite':
        responses = [interpretation, compiled]
    else:
        extraction = {'intents': [{'intent_id': 'c1', 'action': 'MODIFY', 'instruction': 'lower threshold', 'requirement_text': action['requirement_text']}],
            'requires_clarification': False, 'clarification_question': None, 'clarification_rationale': None}
        responses = [extraction, {**interpretation, 'actions': [{**action, 'intent_id': 'c1'}]}, compiled]
    class LLM:
        def generate_structured(self, **kwargs):
            return responses.pop(0)
    parser = CohortTaskParser(LLM(), dataset=dataset, parsing_mode=mode, correction_mode='robust_delta', localized_strategy='mode_aware')
    session = CohortAgent(parser, None, dataset, planner=CohortPlanner(dataset, dataset.domains())).create_session()
    session.initialize(task); session.execute()
    analysis = session.analyze_correction('Lower the threshold to 400 mg')
    session.stage_correction(analysis, session.plan_correction(analysis))
    session.execute_staged()
    assert session.artifacts.get('cohort_patient_ids').value == {1, 2}
    assert not responses


@pytest.mark.parametrize('count', [30, 31])
@pytest.mark.parametrize('mode', ['diamond', 'graphene'])
def test_custom_field_threshold_uses_active_visible_catalog(custom_bundle, count, mode):
    import duckdb
    from agents.cohort.adapters.registry import CustomDataset
    from agents.cohort.parser import CohortTaskParser
    catalog_path = custom_bundle / 'field_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    visible = sum(x['included_in_prompt'] for x in catalog)
    with duckdb.connect(str(custom_bundle / 'dataset.duckdb')) as con:
        for i in range(count - visible + 1):
            column = f'extra_{i}'
            con.execute(f'ALTER TABLE patients ADD COLUMN {column} INTEGER DEFAULT 0')
            catalog.append({'semantic_field': f'patients.{column}', 'description': 'extra numeric attribute',
                'table': 'patients', 'column': column, 'is_numeric': True,
                'included_in_prompt': i < count - visible})
    catalog_path.write_text(json.dumps(catalog))
    dataset = CustomDataset(custom_bundle)
    assert len(dataset.fields()) == count
    flat = {'predicates': [{'predicate_id': 'p1', 'text': 'value at least 600 mg'}], 'logic': 'p1',
        'ambiguities': [], 'requires_clarification': False, 'clarification_question': None, 'rationale': ''}
    responses = [{k: v for k, v in flat.items() if k != 'logic'}, {k: v for k, v in flat.items() if k != 'predicates'}]
    if count > 30:
        responses.append({'assignments': [{'predicate_id': 'p1', 'domain': 'dataset:labs'}]})
    responses.append({'requirements': [{'requirement_id': 'p1', 'kind': 'dataset:labs', 'payload': _custom_filter(), 'polarity': 'include', 'required': True}], 'assumptions': []})
    class LLM:
        def __init__(self): self.calls = []
        def generate_structured(self, **kwargs):
            self.calls.append(kwargs)
            return responses.pop(0)
    llm = LLM()
    CohortTaskParser(llm, dataset=dataset, parsing_mode=mode).parse_task('Find patients with value at least 600 mg')
    assert len(llm.calls) == (4 if count > 30 else 3)
    if count > 30:
        assert 'patients.extra' not in llm.calls[-1]['user_prompt']
    assert 'labs.value' in llm.calls[-1]['user_prompt']
