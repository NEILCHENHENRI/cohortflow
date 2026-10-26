from __future__ import annotations

from dataclasses import dataclass

from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from agents.cohort.parser import CohortTaskParser
from agents.cohort.planner import CohortPlanner
from framework.session import SteerableSession
from infrastructure.config import AppMode, InfrastructureConfig
from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.demo import ensure_demo_databases
from infrastructure.llm.ollama import OllamaClient, OllamaConfig
from infrastructure.llm.recovery import LLMSemanticExceptionResolver
from framework.repair import RepairManager, RepairPolicy
from infrastructure.repair.local import LocalRepairBackend
from infrastructure.mimic_hosp.capabilities import inspect_mimic_hosp_capabilities
from infrastructure.mimic_hosp.concepts import MimicHospConceptResolver
from infrastructure.mimic_hosp.terminology import LLMMedicalTerminologyExpander
from infrastructure.mimic_hosp.repository import MimicHospCohortRepository
from infrastructure.mimic_hosp.note_retrieval import (
    BGEHybridNoteRetriever, HybridNoteMatcher, NoteRetrievalConfig,
)
from infrastructure.omop.capabilities import inspect_omop_capabilities
from infrastructure.omop.concepts import OMOPConceptResolver
from infrastructure.omop.repository import DuckDBCohortRepository
from interfaces.controller import AgentInteractionController


@dataclass(frozen=True, slots=True)
class LocalCohortStack:
    agent: CohortAgent
    session: SteerableSession
    interaction_controller: AgentInteractionController
    llm: OllamaClient
    semantic_resolver: LLMSemanticExceptionResolver | None
    repair_manager: RepairManager


def build_local_cohort_stack(config: InfrastructureConfig, *, max_workers: int = 1):
    if config.mode is AppMode.DEMO:
        ensure_demo_databases(config.paths.clinical_database.parent)
    config.paths.validate_databases()
    dataset = None
    if config.mode is AppMode.CUSTOM:
        from agents.cohort.adapters.registry import CustomDataset
        dataset = CustomDataset(config.adapter_directory)
        if dataset.database.config.database_path.resolve() != config.paths.clinical_database.resolve():
            raise ValueError("Custom database must belong to the selected adapter bundle")
    llm = OllamaClient(OllamaConfig(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        think=config.ollama_think,
        timeout_seconds=config.ollama_timeout_seconds,
    ))
    clinical = DuckDBConnectionFactory(DuckDBConfig(
        config.paths.clinical_database, read_only=True, threads=config.duckdb_threads
    ))
    # The native Ollama adapter implements the same generate_structured()
    # protocol used by the parser and semantic recovery layer. Keeping one
    # adapter avoids duplicate model wrappers, duplicate loading, and stale
    # Outlines-specific imports.
    llm.ensure_ready(preload=config.ollama_preload)
    semantic = (
        LLMSemanticExceptionResolver(
            llm, auto_accept_confidence=config.semantic_recovery_confidence
        ) if config.semantic_recovery_enabled else None
    )

    if config.mode is AppMode.CUSTOM:
        resolver = None
        repository = capabilities = dataset
        note_matcher = None
    elif config.mode is AppMode.DEMO:
        # Demo mode is synthetic, self-contained, and stays OMOP-shaped: a
        # clinical database plus a separate standardized vocabulary database.
        vocabulary = DuckDBConnectionFactory(DuckDBConfig(
            config.paths.vocabulary_database, read_only=True, threads=config.duckdb_threads
        ))
        resolver = OMOPConceptResolver(vocabulary, exception_resolver=semantic)
        repository = DuckDBCohortRepository(clinical)
        capabilities = inspect_omop_capabilities(clinical)
        note_matcher = None
    else:
        # Local mode reads the raw MIMIC-IV hosp-module tables directly. There
        # is no separate vocabulary database to connect.
        terminology_expander = None
        if config.terminology_expansion_enabled:
            terminology_llm = OllamaClient(OllamaConfig(
                model=config.terminology_expansion_model,
                base_url=config.ollama_base_url,
                think=False,
                timeout_seconds=config.ollama_timeout_seconds,
            ))
            # Verify the fallback model exists up front without loading it into
            # memory. Generation remains lazy and only happens after zero lexical candidates.
            terminology_llm.ensure_ready(preload=False)
            terminology_expander = LLMMedicalTerminologyExpander(terminology_llm)
        resolver = MimicHospConceptResolver(
            clinical,
            exception_resolver=semantic,
            terminology_expander=terminology_expander,
        )
        repository = MimicHospCohortRepository(clinical)
        capabilities = inspect_mimic_hosp_capabilities(clinical)
        note_config = NoteRetrievalConfig()
        note_matcher = HybridNoteMatcher(
            BGEHybridNoteRetriever(
                clinical,
                cache_directory=config.paths.data_directory / "note_index",
                config=note_config,
            ),
            llm,
            think=True,
            batch_size=note_config.adjudication_batch_size,
        )

    agent = CohortAgent(
        parser=CohortTaskParser(llm, reuse_unchanged_predicates=True, dataset=dataset),
        resolver=resolver,
        repository=repository,
        note_matcher=note_matcher,
        max_workers=max_workers,
        planner=(CohortPlanner(capabilities=capabilities, supported_kinds=dataset.domains()) if dataset else CohortPlanner(capabilities=capabilities)),
    )
    session = agent.create_session()
    repair_manager = RepairManager(
        repository_root=str(config.paths.project_root),
        local_backend=LocalRepairBackend(llm),
        policy=RepairPolicy(),
    )
    controller = AgentInteractionController(
        session, CohortInteractionAdapter(), semantic, repair_manager
    )
    return LocalCohortStack(agent, session, controller, llm, semantic, repair_manager)


def build_local_cohort_agent(config: InfrastructureConfig, *, max_workers: int = 1):
    return build_local_cohort_stack(config, max_workers=max_workers).agent
