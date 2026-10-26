# Cohort agent

The `cohort/` package owns clinical semantics: eligibility parsing, correction, planning, execution operators, and presentation. Shared lifecycle and task-state mechanics live in [framework](../framework/README.md).

## Components

| Component | Responsibility |
| --- | --- |
| `agent.py` | Assemble the cohort services and operator registry |
| `parser.py` | Interpret requests and corrections. Validate structured output |
| `schema.py`, `structured.py`, `field_catalog.py` | Response contracts and field meanings |
| `planner.py`, `operators.py` | Build and execute capability-checked graphs |
| `manual_edit.py` | Compile validated manual changes |
| `interaction.py`, `presentation.py` | User messages and result views |
| `adapters/` | Custom dataset inspection, construction, mappings, and execution |
| `note_policy.py` | Note-decline behavior |

Prepared backends support patient, admission, condition, procedure, laboratory, medication, and note criteria according to the available dataset capabilities.

## Initial parsing

| Strategy | Stages |
| --- | --- |
| Carbon | Combined interpretation and compilation |
| Graphite (default) | Boolean structure, then batch predicate compilation |
| Diamond | Predicate extraction, relationships, then batch compilation |
| Graphene | Predicate extraction, relationships, then individual compilation |

Diamond and Graphene select domains when the available field catalog exceeds 30 fields. Validation preserves predicate identity and rejects missing, rewritten, or invented predicates between stages. Thinking settings apply per pass.

## Correction

The default cohort parser and UI use **Robust Delta with compact TaskDelta (v2)**. The interpreter selects existing predicate, group, and relation references and describes new or changed criteria. Code assigns persistent IDs, compiles changed criteria, and applies typed edits transactionally. Unchanged requirements remain intact. The prompt lives in `prompts/correction.py`.

Compact correction uses one interpretation stage, regardless of the initial parsing strategy. Changed predicates then use the selected compilation strategy. The default Graphite path batches them. Clarification stops before application. An unsafe correction preserves the current task and offers explicit Full Regeneration rather than silently restarting.

Advanced localized options remain available:

- `compact` (default): compact v2 action interpretation.
- `mode_aware`: interpretation stages follow Carbon/Graphite/Diamond/Graphene.
- `legacy` (Original localized steering, v1): original action interpretation and per-predicate compilation. This is a localized steering strategy, not the Full Regeneration restart baseline.

Select these through `localized_strategy` or `configure_localized_correction(...)`. **Full Regeneration** merges a correction into a complete request and reparses it. Select `CorrectionMode.FULL_REGENERATION` explicitly. Specialized patient facts and retrieval-scope changes are routed separately before ordinary cohort corrections.

## Extension

See [framework extension points](../framework/README.md#extension-points) and [custom dataset setup](../data/README.md#custom-dataset-onboarding).
