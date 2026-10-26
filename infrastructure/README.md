# CohortFlow infrastructure

Concrete services assembled by `factory.py`: Ollama, read-only DuckDB repositories, concept resolution, note retrieval, and repair backends. Configuration and paths live in `config.py`.

## Backends

| Mode | Implementation | Data and resolution |
| --- | --- | --- |
| Local (default) | `mimic_hosp/` | Raw MIMIC tables. ICD dictionary text and recorded drug names |
| Demo | `omop/` | Synthetic OMOP-shaped clinical data and a separate concept vocabulary |
| Custom | `agents/cohort/adapters/` | Adapted DuckDB, catalog fields and reviewed patient/event mappings |

Prepared backends implement the cohort's resolver/repository protocols. Custom executes catalog fields directly and has no MIMIC concept-resolution step. Capability inspection tells the planner which criteria the selected data supports. Queries use fresh DuckDB connections rather than a shared global connection.

MIMIC payloads are compiled deterministically by `compile_requirement_sql()`. Cross-requirement relations operate on a common `subject_id`, `hadm_id`, `event_time` representation from `event_sql()`. Age calculations use MIMIC's `anchor_age`/`anchor_year` scheme. See [data layout](../data/README.md) for tables.

## Models and note evidence

`llm/ollama.py` implements `StructuredLLM` through Ollama `/api/chat`, using the `format` JSON schema and optional `think` setting. Defaults are `qwen3:4b` at `http://localhost:11434`. In local mode, `medgemma1.5:4b` is checked at startup without preloading and used only when lexical terminology search finds no candidates.

Note requirements carry a semantic `claim`. After user confirmation, local matching uses section/paragraph chunks → per-patient BM25 and BGE dense candidate retrieval → cross-encoder reranking with a fixed relevance cutoff → patient-level LLM adjudication (`include`, `exclude`, `uncertain`). Models default to `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-base`. Derived data is cached under `data/note_index/`. Adjudication thinking defaults on.

Note evidence uses the union of BM25 and BGE dense candidates selected separately for each patient (up to 5,000 passages per method per patient by default). A cross-encoder reranks this union and applies a configurable fixed sigmoid-score cutoff (default 0.3). There is no shared cohort-wide top-K or retained-fraction cap, so adding or removing candidate patients does not change another patient's retained evidence. Unchanged note requirements can reuse cached patient decisions. This guarantee assumes unchanged source data, models, and settings, and a custom retriever must provide the same independence. All passages retained after reranking are grouped by patient and included in the LLM adjudication prompt, without an additional per-patient passage cap.

## Programmatic composition

```python
from infrastructure.config import AppMode, InfrastructureConfig, ProjectPaths
from infrastructure.factory import build_local_cohort_stack

config = InfrastructureConfig(
    paths=ProjectPaths.from_project_root(".", mode=AppMode.DEMO),
    mode=AppMode.DEMO,
)
stack = build_local_cohort_stack(config)
result = stack.interaction_controller.handle_message("Find patients with diabetes")
```

Use the controller for user-facing requests so confirmation and continuation gates are honored. Direct agent/session methods support lower-level integration and tests. The factory creates demo databases when needed and checks model readiness.

Custom mode is selected with `AppMode.CUSTOM` and `adapter_directory`. Its clinical path must point to that bundle's `dataset.duckdb`. `agents.cohort.adapters.registry.CustomDataset` supplies the field contract, validated patient/event mapping and parameterized comparisons. Factory wiring reuses the cohort parser, planner, operators and presentation. It does not initialize MIMIC terminology models or note indexes. Prepared MIMIC and demo factory branches are unchanged. See [Custom datasets](../data/README.md).

## Experimental source repair

Repair is off by default. It diagnoses failures, attempts bounded recovery, and verifies the outcome. Clear transient failures use deterministic rules. Other failures use local `qwen3:4b` analysis. Local → Cloud optionally escalates to OpenAI `gpt-5.4`. Credentials remain in the active process or session.

Retryable operators without declared side effects may retry at most twice on detached inputs. The task must remain unchanged, and the result must satisfy its output contract before execution continues. Missing input or access requires user attention. Unsupported operations stop.

Source patches are applied in isolated Git worktrees that snapshot the checkout, including untracked source files. The live checkout is not patched. Validation enforces file and line budgets, protects tests, rejects symlink traversal, and checks compilation and tests against the same baseline snapshot. New failures, missing tests, and new skips are rejected. Worktree isolation is not an operating-system sandbox.

An optional agent replay hook checks whether the original failure occurs in the baseline and succeeds after the patch. Without a replay hook, that failure remains unverified even if patch checks pass.

A validated patch can be saved on a `codex/repair/...` branch or discarded. Adopting a patch requires review, a process restart, and a fresh session. There is no automatic adoption or hot reload.

Tests use temporary databases and mocked HTTP responses. See the [root test command](../README.md#validation).

## External resources

Packages, downloaded models, and datasets retain their own licenses. Prepare real clinical data separately under its applicable access conditions. The synthetic demo runs without MIMIC access. Optional cloud repair sends context to the selected provider. Use it only with content you are authorized to share.
