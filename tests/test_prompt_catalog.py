from pathlib import Path

from prompts.adapter import ADAPTER_CONVERSION_SYSTEM, ADAPTER_FIELD_SYSTEM
from prompts.cohort import (
    CORRECTION_INTERPRETER_SYSTEM_PROMPT,
    PREDICATE_COMPILER_SYSTEM_PROMPT,
)
from prompts.notes import NOTE_ADJUDICATION_SYSTEM
from prompts.repair import REPAIR_ANALYSIS_SYSTEM, REPAIR_PATCH_SYSTEM
from prompts.resolution import SEMANTIC_RESOLUTION_SYSTEM_PROMPT
from prompts.terminology import MEDICAL_TERMINOLOGY_EXPANSION_SYSTEM


def test_prompt_catalog_contains_critical_contracts():
    assert "semantic decisions only" in CORRECTION_INTERPRETER_SYSTEM_PROMPT
    assert "transaction-local handle" in CORRECTION_INTERPRETER_SYSTEM_PROMPT
    assert "SET_EXCLUDED" in CORRECTION_INTERPRETER_SYSTEM_PROMPT
    assert "Boolean relationships are already fixed" in PREDICATE_COMPILER_SYSTEM_PROMPT
    assert "Identifiers and codes are NOT numeric" in ADAPTER_FIELD_SYSTEM
    assert "TRY_CAST" in ADAPTER_CONVERSION_SYSTEM
    assert "interchangeable" in MEDICAL_TERMINOLOGY_EXPANSION_SYSTEM
    assert "Use only the supplied excerpts" in NOTE_ADJUDICATION_SYSTEM
    assert "Never invent an option ID" in SEMANTIC_RESOLUTION_SYSTEM_PROMPT
    assert "failure analyst" in REPAIR_ANALYSIS_SYSTEM
    assert "smallest source patch" in REPAIR_PATCH_SYSTEM


def test_llm_system_prompts_are_not_redeclared_in_runtime_modules():
    runtime_files = [
        "agents/cohort/parser.py",
        "agents/cohort/adapters/llm_inspector.py",
        "infrastructure/mimic_hosp/terminology.py",
        "infrastructure/mimic_hosp/note_retrieval.py",
        "infrastructure/llm/recovery.py",
        "infrastructure/repair/local.py",
    ]
    forbidden = ("_SYSTEM_PROMPT =", "_SYSTEM =", "SYSTEM_PROMPT =", "HUMAN_REPLY_PROMPT =")
    for filename in runtime_files:
        text = Path(filename).read_text()
        assert not any(token in text for token in forbidden), filename
