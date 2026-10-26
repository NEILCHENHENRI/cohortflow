# CohortFlow framework

Shared task state, execution, steering, and reusable domain components. `SteerableSession` coordinates initialization, execution, staged revisions, human continuation, and commit/discard.

## State and execution contracts

- **Tasks:** `task.py` defines `TaskSpec` and `TaskDelta`. Applying a delta returns revised state. `logic.py` represents AND/OR/NOT expressions.
- **Structural edits:** `task_delta.py` applies typed edits over stable references and validates relations. The existing `before`, `after`, and `same_admission` vocabulary is retained for serialized-task compatibility.
- **Graphs and artifacts:** `graph.py` provides deterministic ordering and semantic fingerprints. Artifacts carry provenance. `reconcile.py` reuses completed outputs only when fingerprints match, outputs exist, and parents are reusable. Session cache keys also include transitive semantic lineage.
- **Steering:** lifecycle modules and operators expose cooperative checkpoints. `pending_interaction.py` records what needs a response and where execution resumes.
- **Repair:** `failure.py` and `repair/` capture failures and validate source patches in isolated worktrees. Validation, rather than model confidence, controls acceptance.

Native and LangGraph runtimes share operators, artifacts, and failure handling. LangGraph uses an in-memory checkpointer. Semantic invalidation and reuse remain framework-owned. Both backends currently execute graph operators sequentially. Parallel steering permits correction interpretation to overlap an active action.

## Extension points

| Parent / contract | What a domain supplies |
| --- | --- |
| `DomainAgent` | Parsing, correction interpretation, graph/input construction, operators, and summaries |
| `ComposedDomainAgent` | `parser`, `planner`, `_registry`, `summarize()`, and optional `_manual_editor` |
| `StructuredTaskParser` | Prompts/converters, semantic orchestration, retry guidance, and standalone module metadata |
| `StructuredManualEditProvider` | Form layout, payload extraction/validation, and optional predicate display text |

For component-based agents, call `_bind_parser_modules()` after constructing the parser. `TaskParser` and `TaskPlanner` protocols define component interfaces. `create_session()` and delegation are inherited. Other designs can implement `DomainAgent` directly.

Parser subclasses use `_generate_with_validation()` for timed LLM calls and retries, `_thinking_for()` for temporary per-pass settings, and `_run_module()` for lifecycle execution. Override `_validation_repair_rules()` and `_standalone_module_spec()` when needed. `LogicParseResult` is the shared Boolean-interpretation result.

Manual-editor subclasses implement `build_form()`, `_payload_from_submission()`, and `_validate_payload()`. Compilation collects issues without modifying the source and prunes removed predicates when the submitted Boolean expression is unchanged.

Clinical prompts, payload validation, and result interpretation belong to the domain. Some session APIs still explicitly support cohort note confirmation. Relation names also retain clinical terminology. These are compatibility boundaries, not a claim that every existing API is domain-neutral.

Software tests, including non-clinical inventory fixtures, exercise the inherited components. Run the suite from the repository root using the [root test command](../README.md#validation).

Agents may override `assess_failure()` for deterministic domain guidance and `repair_replay()` to supply bounded isolated baseline/candidate verification. Both are optional. The default supplies no replay. `RecoveryHooks` are trusted host code, never model-provided authorization. `RepairOutcome.verification` separates patch checks from evidence about the original failed operation. Controller-managed sessions connect repair to both execution backends. Direct session callers can set `session.repair_manager` explicitly. Code patches require review, restart, and a fresh session to avoid reuse of pre-patch computations.
