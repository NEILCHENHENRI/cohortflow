from pathlib import Path

import pytest

from infrastructure.config import InfrastructureConfig, ProjectPaths
from infrastructure.factory import build_local_cohort_agent


def test_factory_validates_missing_databases(tmp_path):
    paths = ProjectPaths.from_project_root(tmp_path)
    config = InfrastructureConfig(paths=paths)

    with pytest.raises(FileNotFoundError):
        build_local_cohort_agent(config)


def test_custom_factory_uses_adapted_contract_without_mimic_services(tmp_path, monkeypatch):
    from agents.cohort.adapters.constructor import adapt_dataset
    from infrastructure.config import AppMode
    import infrastructure.factory as factory
    source = tmp_path / 'patients.csv'
    source.write_text('patient_id,gender\n001,F\n')
    bundle = tmp_path / 'bundle'
    adapt_dataset(source, bundle)
    initialized = []
    class LLM:
        def __init__(self, config): initialized.append(config.model)
        def ensure_ready(self, **kwargs): pass
    monkeypatch.setattr(factory, 'OllamaClient', LLM)
    def forbidden(*args, **kwargs): raise AssertionError('Custom must not initialize MIMIC services')
    monkeypatch.setattr(factory, 'MimicHospConceptResolver', forbidden)
    monkeypatch.setattr(factory, 'BGEHybridNoteRetriever', forbidden)
    config = InfrastructureConfig(
        paths=ProjectPaths(tmp_path, bundle / 'dataset.duckdb', None, tmp_path),
        mode=AppMode.CUSTOM, adapter_directory=bundle,
    )
    stack = factory.build_local_cohort_stack(config)
    assert stack.agent.repository is stack.agent.parser.dataset
    assert stack.agent.planner.supported_kinds == ('dataset:patients',)
    assert initialized == [config.ollama_model]
    assert stack.agent._manual_editor.dataset is stack.agent.repository


def test_invalid_custom_bundle_is_checked_before_model_readiness(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import infrastructure.factory as factory
    from infrastructure.config import AppMode
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'dataset.duckdb').touch()
    model = Mock(side_effect=AssertionError('Model must not be contacted for an invalid bundle'))
    monkeypatch.setattr(factory, 'OllamaClient', model)
    config = InfrastructureConfig(paths=ProjectPaths(tmp_path, bundle / 'dataset.duckdb', None, tmp_path),
        mode=AppMode.CUSTOM, adapter_directory=bundle)
    with pytest.raises(FileNotFoundError, match='field_catalog'):
        factory.build_local_cohort_stack(config)
    model.assert_not_called()
