from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Callable

from framework.agent import ProgressCallback, StructuredLLM
from agents.cohort.schema import (
    ASSUMPTION_SCHEMA,
    _requirement_from_payload,
    requirement_schema,
    _flat_logic_result_schema,
    _predicate_extraction_schema,
    _relationship_schema,
    _carbon_schema,
    _merged_request_schema,
    _correction_actions_schema,
    _predicate_compilation_schema,
    _correction_extraction_schema,
    _correction_attachment_schema,
    _carbon_correction_schema,
)
from agents.cohort.field_catalog import (
    render_semantic_fields, semantic_field_count, supported_domains,
)
from framework.correction import (
    ArtifactMutation,
    CorrectionAnalysis,
    CorrectionFallbackRequired,
    CorrectionIntent,
    CorrectionMode,
    CorrectionScope,
    CorrectionTarget,
    MutationKind,
)
from framework.errors import DomainValidationError
from framework.logic import LogicExpression, PredicateText, assign_logic_node_ids, parse_boolean_expression
from framework.steering import SteeringYield, steering_checkpoint
from framework.timing import measure
from framework.module import AgentModuleRole, AgentModuleRunner, AgentModuleSpec
from framework.task import Assumption, Deliverable, Objective, Requirement, TaskDelta, TaskSpec
from framework.task_delta import (
    AddPredicateAct, AddRelationAct, AttachmentSpec, ChangeGroupOperatorAct,
    MovePredicateAct, RemovePredicateAct, RemoveRelationAct, ReplaceLogicSubtreeAct,
    ReplacePredicateAct, SetPredicateNegationAct, SetResolutionAct,
    UpdateAggregationAct, UpdatePredicateFieldsAct, UpdateRelationAct,
    normalize_relations, predicate_ref_maps,
)


from framework.parsing import (
    ParsingMode, DEFAULT_PASS_THINKING, temporary_llm_thinking, StructuredTaskParser,
    LogicParseResult,
)
from prompts.correction import COMPACT_CORRECTION_PROMPT
from prompts.cohort import (
    PREDICATE_EXTRACTION_SYSTEM_PROMPT,
    RELATIONSHIP_PARSE_SYSTEM_PROMPT,
    CARBON_SYSTEM_PROMPT,
    LOGIC_PARSE_SYSTEM_PROMPT,
    MERGE_CORRECTION_SYSTEM_PROMPT,
    CORRECTION_INTERPRETER_SYSTEM_PROMPT,
    TABLE_ASSIGNMENT_SYSTEM_PROMPT,
    PREDICATE_COMPILER_SYSTEM_PROMPT,
    CORRECTION_EXTRACTION_SYSTEM_PROMPT,
    CORRECTION_ATTACHMENT_SYSTEM_PROMPT,
    CARBON_CORRECTION_SYSTEM_PROMPT,
)


TABLE_SELECTION_FIELD_THRESHOLD = 30


def _table_assignment_schema(domains=None) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["assignments"],
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["predicate_id", "domain"],
                    "properties": {
                        "predicate_id": {"type": "string"},
                        "domain": {"type": "string", "enum": list(domains or supported_domains())},
                    },
                },
            }
        },
    }


PREDICATE_COMPILATION_SCHEMA: dict[str, Any] = _predicate_compilation_schema()




@dataclass(frozen=True, slots=True)
class SpecializedCorrectionPatch:
    residual_message: str
    context_updates: dict[str, Any]
    mutations: tuple[ArtifactMutation, ...]
    rationale: str = ""
    clarification_question: str | None = None


def _extract_specialized_corrections(
    current_task: TaskSpec, message: str
) -> SpecializedCorrectionPatch:
    """Extract correction acts whose target is not the cohort definition.

    This intentionally handles only high-confidence surface forms
    deterministically. Anything left in ``residual_message`` goes through the
    existing global LLM merge/reparse path, preserving its quality guarantees.
    """
    residual = message
    mutations: list[ArtifactMutation] = []
    context_updates: dict[str, Any] = {}
    rationale: list[str] = []
    clarification_question: str | None = None

    # Explicit individual patient overrides. This does not match general
    # criteria such as "exclude patients with cancer" because an ID is required.
    existing_overrides = current_task.context.get("individual_overrides", {})
    include_ids = set(existing_overrides.get("include", ()))
    exclude_ids = set(existing_overrides.get("exclude", ()))
    patient_pattern = re.compile(
        r"\b(include|exclude)\s+patients?\s+(?:id\s*)?#?"
        r"((?:\d+)(?:\s*(?:,|and)\s*\d+)*)",
        flags=re.IGNORECASE,
    )
    for match in list(patient_pattern.finditer(residual)):
        action = match.group(1).casefold()
        ids = {int(x) for x in re.findall(r"\d+", match.group(2))}
        if action == "include":
            include_ids.update(ids)
            exclude_ids.difference_update(ids)
            kind = MutationKind.OVERRIDE_RESULT
        else:
            exclude_ids.update(ids)
            include_ids.difference_update(ids)
            kind = MutationKind.OVERRIDE_RESULT
        mutations.append(ArtifactMutation(
            kind=kind,
            target=CorrectionTarget(
                "individual_patient_override",
                ",".join(str(x) for x in sorted(ids)),
                f"Force {action} selected patient(s).",
            ),
            description=f"Force {action} selected patient(s).",
            payload={"action": action, "patient_ids": sorted(ids)},
        ))
        residual = residual.replace(match.group(0), " ", 1)
    if mutations and (include_ids or exclude_ids):
        context_updates["individual_overrides"] = {
            "include": tuple(sorted(include_ids)),
            "exclude": tuple(sorted(exclude_ids)),
        }
        rationale.append("Applied explicit patient-level inclusion/exclusion override.")

    # Session-local fact corrections for an already-defined requirement.
    fact_overrides = list(current_task.context.get("fact_overrides", ()))
    fact_tail = (
        r"([^.;\n]+?)(?="
        r"\s+(?:and|but|then)\s+"
        r"(?:make|change|set|include|exclude|require|remove|add|only|restrict)\b"
        r"|[.;\n]|$)"
    )
    fact_patterns = (
        (False, re.compile(
            r"\bpatient\s+#?(\d+)\s+(?:does\s+not|doesn't|did\s+not|didn't)\s+"
            r"(?:actually\s+)?have\s+" + fact_tail, re.IGNORECASE
        )),
        (False, re.compile(
            r"\bpatient\s+#?(\d+)\s+has\s+no\s+" + fact_tail,
            re.IGNORECASE
        )),
        (True, re.compile(
            r"\bpatient\s+#?(\d+)\s+(?:actually\s+)?(?:does\s+)?have\s+" + fact_tail,
            re.IGNORECASE
        )),
    )
    for present, pattern in fact_patterns:
        for match in list(pattern.finditer(residual)):
            patient_id = int(match.group(1))
            term = match.group(2).strip(" ,")
            requirement_id = _match_requirement_for_fact(current_task, term)
            override = {
                "patient_id": patient_id,
                "term": term,
                "present": present,
                "requirement_id": requirement_id,
            }
            if requirement_id is None:
                clarification_question = (
                    f"I can apply the fact correction for patient {patient_id}, "
                    f"but I cannot safely map {term!r} to an existing cohort "
                    "criterion. Which criterion should it correct?"
                )
                mutations.append(ArtifactMutation(
                    kind=MutationKind.REQUEST_INFORMATION,
                    target=CorrectionTarget(
                        "patient_fact", None,
                        f"Clarify a fact correction for patient {patient_id}.",
                    ),
                    description=f"Clarify a fact correction for patient {patient_id}.",
                    payload=override,
                ))
            else:
                # Latest explicit correction wins for the same patient/requirement.
                fact_overrides = [
                    item for item in fact_overrides
                    if not (
                        int(item.get("patient_id", -1)) == patient_id
                        and item.get("requirement_id") == requirement_id
                    )
                ]
                fact_overrides.append(override)
                mutations.append(ArtifactMutation(
                    kind=MutationKind.UPDATE_VALUE,
                    target=CorrectionTarget(
                        "patient_fact", requirement_id,
                        f"Correct a fact for patient {patient_id}.",
                    ),
                    description=f"Correct a fact for patient {patient_id}.",
                    payload=override,
                ))
            residual = residual.replace(match.group(0), " ", 1)
    if fact_overrides != list(current_task.context.get("fact_overrides", ())):
        context_updates["fact_overrides"] = tuple(fact_overrides)
        rationale.append("Applied session-local patient fact correction.")

    # Retrieval-scope directives are first-class state. They deliberately
    # invalidate requirement retrieval fingerprints even when the Boolean cohort
    # definition is unchanged. Concrete repositories may inspect these directives
    # as their retrieval implementations become scope-aware.
    retrieval = dict(current_task.context.get("retrieval_scope", {}))
    directives = list(retrieval.get("directives", ()))
    retrieval_pattern = re.compile(
        r"(?:^|[.;\n])\s*((?:only\s+use|restrict\s+(?:the\s+)?retrieval\s+to|"
        r"search\s+only|retrieve\s+only|do\s+not\s+retrieve\s+from|"
        r"don't\s+retrieve\s+from)[^.;\n]+)",
        re.IGNORECASE,
    )
    for match in list(retrieval_pattern.finditer(residual)):
        directive = match.group(1).strip()
        if directive and directive.casefold() not in {x.casefold() for x in directives}:
            directives.append(directive)
        mutations.append(ArtifactMutation(
            kind=MutationKind.UPDATE_VALUE,
            target=CorrectionTarget("retrieval_scope", "cohort", "Change retrieval scope."),
            description="Change retrieval scope.",
            payload={"directive": directive},
        ))
        residual = residual.replace(match.group(0), " ", 1)
    if directives != list(retrieval.get("directives", ())):
        context_updates["retrieval_scope"] = {"directives": tuple(directives)}
        rationale.append("Updated retrieval-scope directives.")

    residual = re.sub(r"\s+", " ", residual).strip(" ,.:.\n")
    if mutations:
        residual = re.sub(
            r"^(?:and|also|then|but)\s+", "", residual, flags=re.IGNORECASE
        )
    return SpecializedCorrectionPatch(
        residual_message=residual,
        context_updates=context_updates,
        mutations=tuple(mutations),
        rationale=" ".join(rationale),
        clarification_question=clarification_question,
    )


def _match_requirement_for_fact(task: TaskSpec, term: str) -> str | None:
    needle = _slug(term)
    candidates: list[tuple[int, str]] = []
    for requirement in task.requirements:
        value = requirement.payload.get("term")
        if not isinstance(value, str):
            continue
        normalized = _slug(value)
        if needle == normalized:
            return requirement.requirement_id
        if needle in normalized or normalized in needle:
            candidates.append((abs(len(needle) - len(normalized)), requirement.requirement_id))
    return min(candidates)[1] if candidates else None


class CohortTaskParser(StructuredTaskParser):
    """Cohort semantics on top of the shared structured-generation lifecycle.

    Owns parsing strategies, clinical prompts, and correction interpretation.
    Full Regeneration merges and reparses a complete request. Robust Delta
    compiles localized actions before deterministic structural validation.
    """

    def __init__(self, llm: StructuredLLM, *, localized_strategy: str = "compact", dataset=None, **kwargs):
        kwargs.setdefault("correction_mode", CorrectionMode.ROBUST_DELTA)
        super().__init__(llm, **kwargs)
        self.configure_localized_correction(localized_strategy)
        self.dataset = dataset

    def _domains(self):
        return self.dataset.domains() if self.dataset else supported_domains()

    def _fields(self, domains=None):
        return self.dataset.render_fields(domains) if self.dataset else render_semantic_fields(domains)

    def _requirement(self, item):
        return self.dataset.requirement_from_payload(item) if self.dataset else _requirement_from_payload(item)

    def _response_schema(self, schema, domains=None):
        if self.dataset is not None:
            schema["properties"]["requirements"]["items"] = self.dataset.requirement_schema(domains)
        return schema

    def _compiler_prompt(self):
        if self.dataset is None:
            return PREDICATE_COMPILER_SYSTEM_PROMPT
        return (
            "Compile each fixed predicate into the supplied dataset schema. Preserve every predicate ID exactly once. "
            "Each payload contains filters combined with AND on the SAME row of its selected table. "
            "Use exact catalog field names, descriptions and units. Copy the user's numeric value as a decimal JSON string "
            "and copy the unit literally, preserving all digits. "
            "code performs conversion. Never convert units yourself or generate SQL. Do not infer a missing unit. "
            "No clinical MIMIC fields or concept resolution are available. Never drop or approximate unsupported criteria. "
            "Keep Boolean negation in logic (normally polarity=include). Return required=true. "
            "Unsupported/ambiguous criteria must be clarified, never represented by a weaker filter."
        )

    def configure_localized_correction(self, strategy: str) -> None:
        """Select compact v2, mode-aware, or legacy localized correction."""
        if strategy not in {"compact", "mode_aware", "legacy"}:
            raise ValueError("localized strategy must be compact, mode_aware or legacy")
        self.localized_strategy = strategy


    def _validation_repair_rules(self, label: str) -> str:
        repair_rules = ""
        if label == "Predicate compilation":
            repair_rules = (
                "\nRepair rules:"
                "\n- Return exactly one requirement for each supplied predicate_id, each exactly once."
                "\n- Preserve supplied IDs verbatim. Do not duplicate, omit, rename, or add IDs."
                "\n- Use only payload fields allowed by the chosen requirement kind."
                "\n- polarity must be exactly include or exclude."
                "\n- Do not encode cross-predicate ordering as start_time/end_time fields."
            )
        elif label in {"Initial cohort scheme", "Relationship parsing"}:
            repair_rules = (
                "\nRepair rules:"
                "\n- Every predicate_id must be unique."
                "\n- Logic may contain only declared predicate IDs, AND, OR, NOT, and parentheses."
                "\n- Never emit temporal arithmetic/comparison syntax in the logic string."
                "\n- Keep a time qualifier inside the relevant natural-language predicate instead of creating a temporal-only anchor."
            )
        elif label == "Correction interpretation":
            repair_rules = (
                "\nRepair rules:"
                "\n- Use only explicitly listed existing refs and valid transaction-local refs."
                "\n- Express ordering between existing predicates with relation actions."
                "\n- Prefer the smallest local edit and leave unrelated structure untouched."
                "\n- If meaning is genuinely ambiguous, request clarification."
            )
        return repair_rules

    def _standalone_module_spec(self, name: str) -> AgentModuleSpec:
        role = {
            "semantic.initial_logic_parse": AgentModuleRole.INTERPRET,
            "semantic.relationship_parse": AgentModuleRole.INTERPRET,
            "semantic.correction_interpret": AgentModuleRole.INTERPRET,
            "semantic.correction_extraction": AgentModuleRole.INTERPRET,
            "semantic.correction_attachment": AgentModuleRole.INTERPRET,
            "semantic.correction_merge": AgentModuleRole.INTERPRET,
            "semantic.logic_reparse": AgentModuleRole.INTERPRET,
            "semantic.table_selection": AgentModuleRole.INTERPRET,
            "semantic.predicate_compilation": AgentModuleRole.COMPILE,
            "semantic.canonicalization": AgentModuleRole.VALIDATE,
            "semantic.structured_diff": AgentModuleRole.DIFF,
        }.get(name, AgentModuleRole.OTHER)
        return AgentModuleSpec(
            name=name, role=role, description=f"Cohort semantic module {name}.",
            category="semantic",
        )

    def parse_task(self, request: str, on_progress: ProgressCallback | None = None) -> TaskSpec:
        if not request.strip():
            raise ValueError("request cannot be empty")
        parsed, requirements, compiled_assumptions = self._parse_and_compile_request(
            request.strip(), on_progress=on_progress, initial=True
        )
        canonical_logic, requirements = self._run_module(
            "semantic.canonicalization",
            lambda: _canonicalize_compiled_scheme(parsed.logic, requirements),
            on_progress=on_progress,
        )
        return TaskSpec(
            domain="cohort_curation",
            objective=Objective("cohort_retrieval", request.strip()),
            deliverables=(Deliverable(
                "patients", "patient_set", "Patients matching the cohort specification",
                True, "json",
            ),),
            requirements=requirements,
            logic=canonical_logic,
            assumptions=_merge_assumptions(compiled_assumptions, parsed.ambiguities),
            metadata={
                "logic_requires_clarification": parsed.requires_clarification,
                "logic_clarification_question": parsed.clarification_question,
                "logic_clarification_rationale": parsed.rationale if parsed.requires_clarification else None,
            },
            original_request=request,
        )

    def interpret_correction(
        self, current_task: TaskSpec, message: str,
        on_progress: ProgressCallback | None = None,
    ) -> CorrectionAnalysis:
        # Route correction acts that target session-local facts, individual
        # patient overrides, or retrieval scope before the normal full-state
        # cohort rewrite. A single message may contain any of these plus an
        # ordinary cohort-definition correction.
        specialized = _extract_specialized_corrections(current_task, message)
        residual = specialized.residual_message.strip()

        if residual:
            # Select the configured correction strategy. Robust Delta preserves
            # untouched structure and requests confirmation if it needs fallback.
            if self.correction_mode is CorrectionMode.FULL_REGENERATION:
                delta, parsed = self._compile_revision(
                    current_task, residual, on_progress=on_progress
                )
            else:
                delta, parsed = self._compile_delta_revision(
                    current_task, residual, on_progress=on_progress
                )
        else:
            parsed = LogicParseResult(
                logic=current_task.logic,
                predicates=tuple(
                    PredicateText(x.predicate_id or "", x.text or "")
                    for x in current_task.logic.leaves()
                ),
                ambiguities=(),
                requires_clarification=False,
                clarification_question=None,
                rationale="Applied scoped correction without rewriting cohort logic.",
            )
            delta = TaskDelta(rationale=parsed.rationale)

        if specialized.context_updates:
            delta = replace(
                delta,
                context_updates={
                    **delta.context_updates,
                    **specialized.context_updates,
                },
                rationale=(
                    delta.rationale
                    + (", " if delta.rationale else "")
                    + specialized.rationale
                ).strip(", "),
            )

        mutations = (
            *_correction_mutations(current_task, delta),
            *specialized.mutations,
        )
        has_global = bool(residual)
        has_scoped = bool(specialized.mutations)
        scope = (
            CorrectionScope.MIXED if has_global and has_scoped
            else CorrectionScope.GLOBAL if has_global
            else CorrectionScope.LOCAL
        )
        return CorrectionAnalysis(
            intent=CorrectionIntent(
                user_text=message,
                summary=(
                    "Apply scoped correction acts and regenerate the cohort definition where needed."
                    if has_scoped
                    else "Merge correction into the current cohort and reparse it as a complete request."
                ),
                desired_effect=parsed.logic.render(),
                scope=scope,
                mutations=tuple(mutations),
                confidence=(
                    0.5 if specialized.clarification_question else parsed.confidence
                ),
                requires_clarification=(
                    parsed.requires_clarification
                    or specialized.clarification_question is not None
                ),
                clarification_question=(
                    specialized.clarification_question
                    or parsed.clarification_question
                ),
                rationale=delta.rationale or parsed.rationale,
            ),
            task_delta=delta,
        )

    def _parse_and_compile_request(
        self, request: str, *, on_progress: ProgressCallback | None, initial: bool = False,
    ) -> tuple[LogicParseResult, tuple[Requirement, ...], tuple[Assumption, ...]]:
        mode = self.parsing_mode
        if mode is ParsingMode.CARBON:
            with self._thinking_for("carbon"):
                parsed, requirements, assumptions = self._run_module(
                    "semantic.initial_logic_parse" if initial else "semantic.logic_reparse",
                    lambda: self._generate_with_validation(
                        system_prompt=CARBON_SYSTEM_PROMPT + ("\n" + self._compiler_prompt() if self.dataset else ""),
                        user_prompt=request + ("\n\n" + self._fields() if self.dataset else ""), schema=self._response_schema(_carbon_schema()),
                        converter=lambda item: _carbon_from_payload(item, requirement_converter=self._requirement), label="Carbon one-pass parsing", on_progress=on_progress,
                    ), on_progress=on_progress,
                )
            return parsed, requirements, assumptions

        if mode is ParsingMode.GRAPHITE:
            with self._thinking_for("relationships"):
                parsed = self._run_module(
                    "semantic.initial_logic_parse" if initial else "semantic.logic_reparse",
                    lambda: self._parse_initial_request(request, on_progress=on_progress), on_progress=on_progress,
                )
        else:
            with self._thinking_for("extraction"):
                extracted = self._run_module(
                    "semantic.predicate_extraction",
                    lambda: self._extract_predicates(request, on_progress=on_progress), on_progress=on_progress,
                )
            with self._thinking_for("relationships"):
                parsed = self._run_module(
                    "semantic.relationship_parse",
                    lambda: self._attach_relationships(request, extracted, on_progress=on_progress),
                    on_progress=on_progress,
                )

        requirements, assumptions = self._compile_for_mode(parsed.logic.leaves(), on_progress=on_progress, mode=mode)
        return parsed, requirements, assumptions

    def _compile_for_mode(
        self, leaves: tuple[LogicExpression, ...], *, on_progress: ProgressCallback | None,
        mode: ParsingMode | None = None,
    ) -> tuple[tuple[Requirement, ...], tuple[Assumption, ...]]:
        """Shared initial/localized compilation. Never revises the fixed predicates."""
        if not leaves:
            return (), ()
        mode = mode or self.parsing_mode
        needs_table_narrowing = (len(self.dataset.fields()) if self.dataset else semantic_field_count()) > TABLE_SELECTION_FIELD_THRESHOLD

        if mode is ParsingMode.DIAMOND and needs_table_narrowing:
            with self._thinking_for("table_selection"):
                assignments = self._run_module(
                    "semantic.table_selection",
                    lambda: self._assign_tables(leaves, on_progress=on_progress),
                    on_progress=on_progress,
                )
            selected_domains = tuple(dict.fromkeys(assignments[x.predicate_id or ""] for x in leaves))
            with self._thinking_for("compilation"):
                requirements, assumptions = self._run_module(
                    "semantic.predicate_compilation",
                    lambda: self._compile_predicates(
                        leaves, on_progress=on_progress, allowed_domains=selected_domains,
                        domains_by_id=assignments
                    ),
                    on_progress=on_progress,
                )
            return requirements, assumptions

        if mode is ParsingMode.GRAPHENE and needs_table_narrowing:
            requirements: list[Requirement] = []
            assumptions: list[Assumption] = []
            for leaf in leaves:
                with self._thinking_for("table_selection"):
                    assignment = self._run_module(
                        "semantic.table_selection",
                        lambda leaf=leaf: self._assign_tables((leaf,), on_progress=on_progress),
                        on_progress=on_progress,
                    )[leaf.predicate_id or ""]
                with self._thinking_for("compilation"):
                    compiled, local_assumptions = self._run_module(
                        "semantic.predicate_compilation",
                        lambda leaf=leaf, assignment=assignment: self._compile_predicates(
                            (leaf,), on_progress=on_progress, allowed_domains=(assignment,),
                            domains_by_id={leaf.predicate_id: assignment}
                        ),
                        on_progress=on_progress,
                    )
                requirements.extend(compiled)
                assumptions.extend(local_assumptions)
            return tuple(requirements), _preserve_compilation_assumptions((), assumptions)

        with self._thinking_for("compilation"):
            if mode is ParsingMode.GRAPHENE:
                compile_fn = lambda: self._compile_predicates_individually(leaves, on_progress=on_progress)
            else:
                compile_fn = lambda: self._compile_predicates(leaves, on_progress=on_progress)
            requirements, assumptions = self._run_module(
                "semantic.predicate_compilation", compile_fn, on_progress=on_progress
            )
        return requirements, assumptions

    def _extract_predicates(
        self, request: str, *, on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        return self._generate_with_validation(
            system_prompt=PREDICATE_EXTRACTION_SYSTEM_PROMPT, user_prompt=request,
            schema=_predicate_extraction_schema(), converter=_predicate_extraction_from_payload,
            label="Predicate extraction", on_progress=on_progress,
        )

    def _attach_relationships(
        self, request: str, extracted: dict[str, Any], *, on_progress: ProgressCallback | None,
    ) -> LogicParseResult:
        predicates = extracted["predicates"]
        prompt = (
            "FIXED PREDICATES (IDs and text are read-only):\n"
            + _predicate_bullets(predicates)
            + "\n\nORIGINAL REQUEST (relationship context only):\n" + request
        )
        return self._generate_with_validation(
            system_prompt=RELATIONSHIP_PARSE_SYSTEM_PROMPT, user_prompt=prompt, schema=_relationship_schema(),
            converter=lambda payload: _relationship_from_payload(payload, predicates, extracted),
            label="Relationship parsing", on_progress=on_progress,
        )

    def _assign_tables(
        self, leaves: tuple[LogicExpression, ...], *, on_progress: ProgressCallback | None,
    ) -> dict[str, str]:
        payload = [{"predicate_id": x.predicate_id, "text": x.text} for x in leaves]
        expected_ids = tuple(x.predicate_id or "" for x in leaves)
        result = self._generate_with_validation(
            system_prompt=TABLE_ASSIGNMENT_SYSTEM_PROMPT,
            user_prompt=(
                "PREDICATES TO ASSIGN:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
                + "\n\nAVAILABLE DOMAINS:\n"
                + (self.dataset.render_domains() if self.dataset else "\n".join(f"- {domain}" for domain in self._domains()))
            ),
            schema=_table_assignment_schema(self._domains()),
            converter=lambda item: _table_assignments_from_payload(item, expected_ids, domains=self._domains() if self.dataset else None),
            label="Table assignment",
            on_progress=on_progress,
        )
        return result

    def _compile_predicates_individually(
        self, leaves: tuple[LogicExpression, ...], *, on_progress: ProgressCallback | None,
    ) -> tuple[tuple[Requirement, ...], tuple[Assumption, ...]]:
        requirements: list[Requirement] = []
        assumptions: list[Assumption] = []
        for leaf in leaves:
            compiled, local_assumptions = self._compile_predicates((leaf,), on_progress=on_progress)
            requirements.extend(compiled)
            assumptions.extend(local_assumptions)
        return tuple(requirements), _preserve_compilation_assumptions((), assumptions)

    def _compile_delta_revision(
        self, current_task: TaskSpec, message: str, *, on_progress: ProgressCallback | None,
    ) -> tuple[TaskDelta, LogicParseResult]:
        """Interpret one localized semantic correction, then edit structure deterministically.

        The LLM never allocates persistent refs or emits TaskDelta acts. It selects
        existing refs, describes new/changed requirement text, and chooses semantic
        attachment. The selected localized strategy compiles only new/changed
        requirements before deterministic graph surgery.
        """
        if current_task.domain != "cohort_curation":
            raise ValueError("current task is not a cohort-curation task")
        if current_task.logic is None:
            raise DomainValidationError("Current cohort task has no canonical Boolean scheme.")

        normalized = _task_with_structure_refs(current_task)
        # A UI setting change during an LLM call must not mix two pipelines.
        strategy, mode = self.localized_strategy, self.parsing_mode
        refs = _structure_ref_catalog(normalized)
        prompt = (
            "CURRENT STRUCTURED COHORT (READ ONLY):\n"
            + render_structured_task(normalized)
            + "\n\nALLOWED EXISTING REFS:\n"
            + json.dumps(refs, indent=2, ensure_ascii=False)
            + "\n\nLATEST CORRECTION:\n"
            + message.strip()
        )
        try:
            if strategy in {"compact", "mode_aware"}:
                interpreted = self._interpret_localized(prompt, refs, on_progress=on_progress, mode=mode)
            else:
                with self._thinking_for("correction_delta"):
                    interpreted = self._run_module(
                        "semantic.correction_interpret",
                        lambda: self._generate_with_validation(
                            system_prompt=CORRECTION_INTERPRETER_SYSTEM_PROMPT,
                            user_prompt=prompt,
                            schema=_correction_actions_schema(
                                predicate_refs=refs["predicates"],
                                group_refs=refs["groups"],
                                relation_refs=refs["relations"],
                            ),
                            converter=lambda payload: _correction_actions_from_payload(
                                payload, refs
                            ),
                            label="Correction interpretation",
                            on_progress=on_progress,
                            retry_limit=1,
                        ),
                        on_progress=on_progress,
                    )
        except SteeringYield:
            raise
        except Exception as exc:
            raise CorrectionFallbackRequired(
                "Robust correction interpretation failed. Full Regeneration is available with your confirmation.",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        if interpreted["requires_clarification"]:
            question = interpreted.get("clarification_question") or "Could you clarify what should change?"
            clarification_rationale = (
                interpreted.get("clarification_rationale")
                or "The correction has materially different plausible interpretations."
            )
            parsed = LogicParseResult(
                logic=normalized.logic,
                predicates=tuple(
                    PredicateText(x.predicate_id or "", x.text or "")
                    for x in normalized.logic.leaves()
                ),
                ambiguities=(question,),
                requires_clarification=True,
                clarification_question=question,
                rationale=clarification_rationale,
            )
            return TaskDelta(rationale=parsed.rationale), parsed

        try:
            acts, assumptions = self._compile_correction_actions(
                normalized, interpreted["actions"], on_progress=on_progress,
                compiled_changes=interpreted.get("compiled_changes"),
                compiled_assumptions=interpreted.get("compiled_assumptions", ()),
                strategy=strategy, mode=mode,
            )
            if strategy in {"compact", "mode_aware"}:
                assumptions = _preserve_compilation_assumptions(normalized.assumptions, assumptions)
            delta = TaskDelta(
                acts=acts,
                assumption_updates=assumptions,
                rationale="Applied localized semantic correction.",
            )
            from framework.task import apply_task_delta
            proposed = apply_task_delta(normalized, delta)
            if _cohort_semantic_signature(proposed) == _cohort_semantic_signature(normalized):
                raise DomainValidationError("correction produced no semantic change")
        except (CorrectionFallbackRequired, SteeringYield):
            raise
        except Exception as exc:
            raise CorrectionFallbackRequired(
                "The localized correction could not be applied safely. Full Regeneration is available with your confirmation.",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        logic = proposed.logic
        if logic is None:  # pragma: no cover - apply_task_delta already guards this
            raise CorrectionFallbackRequired("correction produced empty logic")
        parsed = LogicParseResult(
            logic=logic,
            predicates=tuple(
                PredicateText(x.predicate_id or "", x.text or "") for x in logic.leaves()
            ),
            ambiguities=(),
            requires_clarification=False,
            clarification_question=None,
            rationale=delta.rationale,
        )
        return delta, parsed

    def _interpret_localized(self, prompt, refs, *, on_progress, mode: ParsingMode):
        if self.localized_strategy == "compact":
            with self._thinking_for("correction_delta"):
                return self._run_module(
                    "semantic.correction_interpret",
                    lambda: self._generate_with_validation(
                        system_prompt=COMPACT_CORRECTION_PROMPT,
                        user_prompt=prompt,
                        schema=_correction_actions_schema(
                            predicate_refs=refs["predicates"],
                            group_refs=refs["groups"],
                            relation_refs=refs["relations"],
                        ),
                        converter=lambda raw: _checked_correction_actions(raw, refs),
                        label="Correction interpretation", on_progress=on_progress,
                        retry_limit=1,
                    ), on_progress=on_progress,
                )

        if mode is ParsingMode.CARBON:
            with self._thinking_for("carbon"):
                return self._run_module(
                    "semantic.correction_interpret",
                    lambda: self._generate_with_validation(
                        system_prompt=(CARBON_CORRECTION_SYSTEM_PROMPT.replace(PREDICATE_COMPILER_SYSTEM_PROMPT, self._compiler_prompt())),
                        user_prompt=prompt + "\n\n" + self._fields(),
                        schema=self._response_schema(_carbon_correction_schema(refs)),
                        converter=lambda item: _carbon_correction_from_payload(item, refs, requirement_converter=self._requirement),
                        label="Carbon localized correction", on_progress=on_progress,
                    ), on_progress=on_progress,
                )

        if mode in {ParsingMode.DIAMOND, ParsingMode.GRAPHENE}:
            with self._thinking_for("extraction"):
                extracted = self._run_module(
                    "semantic.correction_extraction",
                    lambda: self._generate_with_validation(
                        system_prompt=CORRECTION_EXTRACTION_SYSTEM_PROMPT,
                        user_prompt=prompt, schema=_correction_extraction_schema(),
                        converter=_correction_intents_from_payload,
                        label="Correction intent extraction", on_progress=on_progress,
                    ), on_progress=on_progress,
                )
            if extracted["requires_clarification"]:
                return extracted
            intents = extracted["intents"]
            fixed = "\n".join("- " + json.dumps(item, ensure_ascii=False) for item in intents)
            with self._thinking_for("relationships"):
                return self._run_module(
                    "semantic.correction_attachment",
                    lambda: self._generate_with_validation(
                        system_prompt=CORRECTION_ATTACHMENT_SYSTEM_PROMPT,
                        user_prompt="FIXED CHANGE INTENTS (read-only):\n" + fixed + "\n\n" + prompt,
                        schema=_correction_attachment_schema(refs, intents),
                        converter=lambda item: _correction_attachments_from_payload(item, refs, intents),
                        label="Correction attachment", on_progress=on_progress,
                    ), on_progress=on_progress,
                )

        with self._thinking_for("correction_delta"):
            return self._run_module(
                "semantic.correction_interpret",
                lambda: self._generate_with_validation(
                    system_prompt=CORRECTION_INTERPRETER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    schema=_correction_actions_schema(
                        predicate_refs=refs["predicates"], group_refs=refs["groups"], relation_refs=refs["relations"],
                    ),
                    converter=lambda item: _checked_correction_actions(item, refs),
                    label="Correction interpretation", on_progress=on_progress,
                    retry_limit=1,
                ), on_progress=on_progress,
            )

    def _compile_correction_actions(
        self,
        task: TaskSpec,
        actions: tuple[dict[str, Any], ...],
        *,
        on_progress: ProgressCallback | None,
        compiled_changes: dict[str, Requirement] | None = None,
        compiled_assumptions: tuple[Assumption, ...] = (),
        strategy: str | None = None,
        mode: ParsingMode | None = None,
    ) -> tuple[tuple[Any, ...], tuple[Assumption, ...]]:
        """Compile localized predicate text and translate semantic actions to typed acts."""
        strategy = strategy or self.localized_strategy
        pred_to_id, _ = predicate_ref_maps(task)
        req_by_id = {r.requirement_id: r for r in task.requirements}
        group_kinds = _logic_group_kinds(task.logic)
        relations = {r["relation_id"]: r for r in normalize_relations(task.context.get("relations", ())) }

        # Transaction-local refs are resolved by code before TaskDelta is built.
        next_p = _next_numeric_ref("p", tuple(pred_to_id))
        next_r = _next_numeric_ref("r", tuple(relations))
        local_predicates: dict[str, str] = {}
        local_relations: dict[str, str] = {}
        for action in actions:
            if action["action"] == "ADD":
                local_predicates[action["local_ref"]] = f"p{next_p}"
                next_p += 1
            elif action["action"] == "ADD_RELATION":
                local_relations[action["local_ref"]] = f"r{next_r}"
                next_r += 1

        def pred_ref(value: str) -> str:
            return local_predicates.get(value, value)

        def relation_ref(value: str) -> str:
            return local_relations.get(value, value)

        # New strategies compile a validated batch (or per-leaf narrowed calls)
        # before producing any acts. Legacy retains the original interleaved loop.
        if strategy in {"compact", "mode_aware"} and compiled_changes is None:
            leaves = tuple(
                LogicExpression.predicate(
                    local_predicates[action["local_ref"]] if action["action"] == "ADD" else action["target"],
                    action["requirement_text"],
                )
                for action in actions if action["action"] in {"ADD", "MODIFY"}
            )
            requirements, compiled_assumptions = self._compile_for_mode(leaves, on_progress=on_progress, mode=mode)
            by_id = {item.requirement_id: item for item in requirements}
            compiled_changes = {
                action["local_ref"] if action["action"] == "ADD" else action["target"]:
                    by_id[local_predicates[action["local_ref"]] if action["action"] == "ADD" else action["target"]]
                for action in actions if action["action"] in {"ADD", "MODIFY"}
            }

        acts: list[Any] = []
        assumptions: list[Assumption] = list(compiled_assumptions)
        polarity_by_ref = {ref: req_by_id[rid].polarity for ref, rid in pred_to_id.items()}
        for action in actions:
            kind = action["action"]
            if kind == "ADD":
                new_ref = local_predicates[action["local_ref"]]
                if compiled_changes is None:
                    compiled, local_assumptions = self._compile_predicates(
                        (LogicExpression.predicate(new_ref, action["requirement_text"]),),
                        on_progress=on_progress,
                    )
                    assumptions.extend(local_assumptions)
                    changed = compiled[0]
                else:
                    changed = compiled_changes[action["local_ref"]]
                requirement = replace(
                    changed,
                    requirement_id=new_ref,
                    polarity="exclude" if action["excluded"] else "include",
                )
                attachment = _semantic_add_attachment(
                    anchor=pred_ref(action["anchor"]),
                    operator=action["operator"],
                )
                extras = tuple(
                    AttachmentSpec(relation_ref(rid), "relation_member")
                    for rid in action.get("relation_memberships", ())
                )
                acts.append(AddPredicateAct(
                    requirement, (attachment, *extras),
                    display_text=action["requirement_text"],
                ))

            elif kind == "MODIFY":
                target = action["target"]
                rid = pred_to_id[target]
                old = req_by_id[rid]
                if compiled_changes is None:
                    compiled, local_assumptions = self._compile_predicates(
                        (LogicExpression.predicate(target, action["requirement_text"]),),
                        on_progress=on_progress,
                    )
                    assumptions.extend(local_assumptions)
                    changed = compiled[0]
                else:
                    changed = compiled_changes[target]
                replacement = replace(
                    changed, requirement_id=target,
                    polarity=polarity_by_ref[target] if strategy in {"compact", "mode_aware"} else old.polarity,
                )
                acts.append(ReplacePredicateAct(
                    target, replacement, display_text=action["requirement_text"]
                ))

            elif kind == "REMOVE":
                acts.append(RemovePredicateAct(action["target"]))

            elif kind == "MOVE":
                acts.append(MovePredicateAct(
                    action["target"],
                    _semantic_attachment(
                        anchor=pred_ref(action["anchor"]),
                        operator=action["operator"],
                        placement=action["placement"],
                        task=task,
                        group_kinds=group_kinds,
                    ),
                ))

            elif kind == "SET_EXCLUDED":
                acts.append(SetPredicateNegationAct(action["target"], action["excluded"]))
                polarity_by_ref[action["target"]] = "exclude" if action["excluded"] else "include"

            elif kind == "CHANGE_OPERATOR":
                acts.append(ChangeGroupOperatorAct(
                    action["target"], action["operator"].casefold()
                ))

            elif kind == "ADD_RELATION":
                relation = _relation_from_correction_action(
                    action,
                    relation_id=local_relations[action["local_ref"]],
                    pred_ref=pred_ref,
                )
                acts.append(AddRelationAct(relation))

            elif kind == "MODIFY_RELATION":
                target = action["target"]
                if target not in relations:
                    raise DomainValidationError(f"unknown relation ref: {target}")
                changes = _relation_changes_from_correction_action(
                    action, current=relations[target], pred_ref=pred_ref
                )
                acts.append(UpdateRelationAct(target, changes))

            elif kind == "REMOVE_RELATION":
                acts.append(RemoveRelationAct(action["target"]))

            else:  # pragma: no cover - converter/schema own the vocabulary
                raise DomainValidationError(f"unsupported correction action: {kind}")

        return tuple(acts), tuple(assumptions)

    def _compile_revision(
        self, current_task: TaskSpec, message: str, *, on_progress: ProgressCallback | None,
    ) -> tuple[TaskDelta, LogicParseResult]:
        if current_task.domain != "cohort_curation":
            raise ValueError("current task is not a cohort-curation task")
        if not message.strip():
            raise ValueError("correction cannot be empty")
        if current_task.logic is None:
            raise DomainValidationError("Current cohort task has no canonical Boolean scheme.")

        old_scheme = _render_logic_natural(current_task.logic)
        merge_prompt = (
            "CURRENT COHORT LOGIC:\n" + old_scheme
            + "\n\nLATEST CORRECTION:\n" + message.strip()
        )
        with self._thinking_for("correction_merge"):
            merged_request = self._run_module(
                "semantic.correction_merge",
                lambda: self._generate_with_validation(
                    system_prompt=MERGE_CORRECTION_SYSTEM_PROMPT,
                    user_prompt=merge_prompt,
                    schema=_merged_request_schema(),
                    converter=_merged_request_from_payload,
                    label="Correction merge",
                    on_progress=on_progress,
                ),
                on_progress=on_progress,
            )
        _emit_semantic_trace(on_progress, "merged_request", merged_request)

        parsed, new_requirements, compiled_assumptions = self._parse_and_compile_request(
            merged_request, on_progress=on_progress, initial=False
        )
        _emit_semantic_trace(
            on_progress, "parsed_logic", _render_logic_natural(parsed.logic)
        )
        canonical_logic, new_requirements = self._run_module(
            "semantic.canonicalization",
            lambda: _canonicalize_compiled_scheme(parsed.logic, new_requirements),
            on_progress=on_progress,
        )
        parsed = LogicParseResult(
            logic=canonical_logic,
            predicates=tuple(
                PredicateText(x.predicate_id or "", x.text or "")
                for x in canonical_logic.leaves()
            ),
            ambiguities=parsed.ambiguities,
            requires_clarification=parsed.requires_clarification,
            clarification_question=parsed.clarification_question,
            rationale=parsed.rationale,
        )
        delta = self._run_module(
            "semantic.structured_diff",
            lambda: _diff_structured_task(
                current_task,
                new_requirements,
                canonical_logic,
                _merge_assumptions(compiled_assumptions, parsed.ambiguities),
                rationale=parsed.rationale or merged_request,
            ),
            on_progress=on_progress,
        )
        return delta, parsed

    def _parse_initial_request(
        self, request: str, *, on_progress: ProgressCallback | None,
    ) -> LogicParseResult:
        """Parse initial cohort semantics once. Deterministic validation guards structure."""
        return self._parse_flat_logic(
            user_prompt=request,
            system_prompt=LOGIC_PARSE_SYSTEM_PROMPT,
            label="Initial cohort scheme",
            on_progress=on_progress,
        )

    def _parse_flat_logic(
        self, *, user_prompt: str, system_prompt: str, label: str,
        on_progress: ProgressCallback | None,
    ) -> LogicParseResult:
        parsed = self._generate_with_validation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=_flat_logic_result_schema(),
            converter=_logic_parse_result_from_flat_payload,
            label=label,
            on_progress=on_progress,
        )
        return _atomize_explicit_age_ranges(parsed)

    def _compile_predicates_with_reuse(
        self,
        current_task: TaskSpec,
        leaves: tuple[LogicExpression, ...],
        *,
        on_progress: ProgressCallback | None,
    ) -> tuple[tuple[Requirement, ...], tuple[Assumption, ...]]:
        """Compile only leaves whose semantics are not text-identical to old ones.

        The complete corrected request is still globally merged and reparsed first.
        Reuse happens only after that quality-preserving step, and only for a
        one-to-one normalized-text match. Ambiguous/duplicate matches are compiled
        normally rather than guessed.
        """
        if not leaves:
            return (), ()
        if current_task.logic is None:
            return self._compile_predicates(leaves, on_progress=on_progress)

        old_requirements = {item.requirement_id: item for item in current_task.requirements}
        old_by_text: dict[str, list[LogicExpression]] = {}
        for old_leaf in current_task.logic.leaves():
            old_by_text.setdefault(_normalized_leaf_text(old_leaf.text or ""), []).append(old_leaf)
        new_counts: dict[str, int] = {}
        for leaf in leaves:
            key = _normalized_leaf_text(leaf.text or "")
            new_counts[key] = new_counts.get(key, 0) + 1

        reused: dict[str, Requirement] = {}
        to_compile: list[LogicExpression] = []
        for leaf in leaves:
            key = _normalized_leaf_text(leaf.text or "")
            candidates = old_by_text.get(key, ())
            old_leaf = candidates[0] if len(candidates) == 1 and new_counts.get(key) == 1 else None
            old_requirement = (
                old_requirements.get(old_leaf.predicate_id or "")
                if old_leaf is not None
                else None
            )
            if old_requirement is None:
                to_compile.append(leaf)
                continue
            reused[leaf.predicate_id or ""] = replace(
                old_requirement, requirement_id=leaf.predicate_id or ""
            )
            if on_progress is not None:
                on_progress(
                    "predicate_compilation_reused",
                    json.dumps(
                        {
                            "predicate_id": leaf.predicate_id,
                            "text": leaf.text,
                            "kind": old_requirement.kind,
                        },
                        ensure_ascii=False,
                    ),
                )

        compiled, assumptions = self._compile_predicates(
            tuple(to_compile), on_progress=on_progress
        )
        compiled_by_id = {item.requirement_id: item for item in compiled}
        ordered = tuple(
            reused.get(leaf.predicate_id or "")
            or compiled_by_id[leaf.predicate_id or ""]
            for leaf in leaves
        )
        return ordered, assumptions

    def _compile_predicates(
        self, leaves: tuple[LogicExpression, ...], *, on_progress: ProgressCallback | None,
        allowed_domains: tuple[str, ...] | None = None,
        domains_by_id: dict[str, str] | None = None,
    ) -> tuple[tuple[Requirement, ...], tuple[Assumption, ...]]:
        if not leaves:
            return (), ()
        payload = [{"predicate_id": x.predicate_id, "text": x.text} for x in leaves]
        expected_ids = tuple(x.predicate_id or "" for x in leaves)
        field_context = self._fields(allowed_domains)
        return self._generate_with_validation(
            system_prompt=self._compiler_prompt(),
            user_prompt=(
                "PREDICATES TO COMPILE:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
                + "\n\n" + field_context
                + ("\n\nFIXED DOMAIN ASSIGNMENTS:\n" + json.dumps(domains_by_id)
                   if domains_by_id is not None else "")
            ),
            schema=self._response_schema(_predicate_compilation_schema(None if self.dataset else allowed_domains), allowed_domains),
            converter=lambda item: _compiled_predicates_from_payload(
                item, expected_ids=expected_ids, domains_by_id=domains_by_id, requirement_converter=self._requirement,
            ),
            label="Predicate compilation",
            on_progress=on_progress,
        )


def _emit_semantic_trace(
    callback: ProgressCallback | None,
    scope: str,
    payload: str,
) -> None:
    """Expose semantic correction boundaries without mutating them.

    These diagnostic events show the merged natural-language request and its
    reparsed Boolean meaning. Execution code never reads them.
    """
    if callback is not None:
        callback(f"semantic_trace:{scope}", payload)


def _predicate_bullets(predicates) -> str:
    # JSON quoting keeps embedded newlines and delimiter-like text inside one item.
    return "\n".join(
        f"- {p.predicate_id}: {json.dumps(p.text, ensure_ascii=False)}" for p in predicates
    )


def _predicate_extraction_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("predicates")
    if not isinstance(raw, list) or not raw:
        raise TypeError("predicates must be a non-empty list")
    if any(not isinstance(x, dict) or not isinstance(x.get("predicate_id"), str)
           or not isinstance(x.get("text"), str) for x in raw):
        raise TypeError("predicates require string IDs and text")
    predicates = tuple(PredicateText(str(x.get("predicate_id", "")), str(x.get("text", ""))) for x in raw)
    if any(not p.predicate_id.strip() or not p.text.strip() for p in predicates):
        raise ValueError("predicate IDs and text must be non-empty")
    if len({p.predicate_id for p in predicates}) != len(predicates):
        raise ValueError("predicate IDs must be unique")
    question = payload.get("clarification_question")
    needs = bool(payload.get("requires_clarification", False))
    if needs and not isinstance(question, str):
        raise ValueError("clarification_question is required when clarification is needed")
    return {
        "predicates": predicates,
        "ambiguities": tuple(str(x).strip() for x in payload.get("ambiguities", []) if str(x).strip()),
        "requires_clarification": needs,
        "clarification_question": question.strip() if isinstance(question, str) else None,
        "rationale": str(payload.get("rationale", "")).strip(),
    }


def _relationship_from_payload(
    payload: dict[str, Any], predicates: tuple[PredicateText, ...], extracted: dict[str, Any],
) -> LogicParseResult:
    if "predicates" in payload:
        raise DomainValidationError("relationship pass must not return or rewrite predicates")
    combined = {
        "predicates": [{"predicate_id": p.predicate_id, "text": p.text} for p in predicates],
        "logic": payload.get("logic"),
        "ambiguities": list(extracted.get("ambiguities", ())) + list(payload.get("ambiguities", [])),
        "requires_clarification": bool(extracted.get("requires_clarification")) or bool(payload.get("requires_clarification")),
        "clarification_question": payload.get("clarification_question") or extracted.get("clarification_question"),
        "rationale": str(payload.get("rationale", "")).strip() or str(extracted.get("rationale", "")).strip(),
    }
    return _logic_parse_result_from_flat_payload(combined)


def _carbon_from_payload(payload: dict[str, Any], *, requirement_converter=_requirement_from_payload):
    parsed = _logic_parse_result_from_flat_payload(payload)
    expected_ids = tuple(p.predicate_id for p in parsed.predicates)
    requirements, assumptions = _compiled_predicates_from_payload(payload, expected_ids=expected_ids, requirement_converter=requirement_converter)
    return parsed, requirements, assumptions


def _logic_parse_result_from_flat_payload(payload: dict[str, Any]) -> LogicParseResult:
    raw_predicates = payload.get("predicates")
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise TypeError("predicates must be a non-empty list")
    predicates = tuple(
        PredicateText(str(item.get("predicate_id", "")), str(item.get("text", "")))
        for item in raw_predicates
        if isinstance(item, dict)
    )
    if len(predicates) != len(raw_predicates):
        raise TypeError("each predicate must be an object")
    if len({x.predicate_id for x in predicates}) != len(predicates):
        raise ValueError("predicate IDs must be unique")
    raw_logic = payload.get("logic")
    if not isinstance(raw_logic, str):
        raise TypeError("logic must be a string")
    logic = parse_boolean_expression(raw_logic, predicates)
    ambiguities = payload.get("ambiguities", [])
    if not isinstance(ambiguities, list) or not all(isinstance(x, str) for x in ambiguities):
        raise TypeError("ambiguities must be a list of strings")
    question = payload.get("clarification_question")
    if question is not None and not isinstance(question, str):
        raise TypeError("clarification_question must be text or null")
    needs = bool(payload.get("requires_clarification", False))
    if needs and not (question or "").strip():
        raise ValueError("clarification_question is required when clarification is needed")
    return LogicParseResult(
        logic=logic,
        predicates=predicates,
        ambiguities=tuple(x.strip() for x in ambiguities if x.strip()),
        requires_clarification=needs,
        clarification_question=None if question is None else question.strip(),
        rationale=str(payload.get("rationale", "")).strip(),
    )


_AGE_RANGE_PATTERNS = (
    re.compile(
        r"\b(?:patients?\s+)?aged?\s+(?P<low>\d+(?:\.\d+)?)\s+"
        r"(?:to|through|-)\s+(?P<high>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bage\s+between\s+(?P<low>\d+(?:\.\d+)?)\s+and\s+"
        r"(?P<high>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
)


def _atomize_explicit_age_ranges(parsed: LogicParseResult) -> LogicParseResult:
    """Split a clearly bounded age phrase into two atomic leaves.

    The structured patient schema represents a closed age interval as two
    independent requirements (>= lower, <= upper). Keeping a bounded range as
    one parser leaf forces the compiler either to duplicate the temporary ID or
    to lose one bound. This deterministic normalization only touches explicit
    numeric age intervals and leaves every other predicate unchanged.
    """
    used = set(parsed.logic.predicate_ids())
    next_index = 1
    for ref in used:
        match = re.fullmatch(r"p(\d+)", ref)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)

    def next_ref() -> str:
        nonlocal next_index
        while f"p{next_index}" in used:
            next_index += 1
        value = f"p{next_index}"
        used.add(value)
        next_index += 1
        return value

    changed = False

    def visit(node: LogicExpression) -> LogicExpression:
        nonlocal changed
        if node.kind == "predicate":
            text = node.text or ""
            match = None
            for pattern in _AGE_RANGE_PATTERNS:
                match = pattern.search(text)
                if match:
                    break
            if match is None:
                return node

            low = match.group("low")
            high = match.group("high")
            if float(low) > float(high):
                return node

            changed = True
            lower = LogicExpression.predicate(
                node.predicate_id or "",
                f"age at least {low} years",
            )
            upper = LogicExpression.predicate(
                next_ref(),
                f"age at most {high} years",
            )
            return LogicExpression.and_(lower, upper)

        children = tuple(visit(child) for child in node.children)

        # Flatten only same-operator groups created by the normalization so
        # ordinary AND/OR structure is preserved rather than needlessly nested.
        if node.kind in {"and", "or"}:
            flattened = []
            for child in children:
                if child.kind == node.kind and child.node_id is None:
                    flattened.extend(child.children)
                else:
                    flattened.append(child)
            children = tuple(flattened)

        return LogicExpression(node.kind, children=children, node_id=node.node_id)

    logic = visit(parsed.logic)
    if not changed:
        return parsed

    return LogicParseResult(
        logic=logic,
        predicates=tuple(
            PredicateText(leaf.predicate_id or "", leaf.text or "")
            for leaf in logic.leaves()
        ),
        ambiguities=parsed.ambiguities,
        requires_clarification=parsed.requires_clarification,
        clarification_question=parsed.clarification_question,
        rationale=parsed.rationale,
    )


def _merged_request_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("merged_request")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("merged_request must be non-empty text")
    return value.strip()


def _table_assignments_from_payload(
    payload: dict[str, Any], expected_ids: tuple[str, ...], *, domains=None,
) -> dict[str, str]:
    raw = payload.get("assignments")
    if not isinstance(raw, list):
        raise TypeError("assignments must be a list")
    result: dict[str, str] = {}
    allowed = set(domains or supported_domains())
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("each assignment must be an object")
        predicate_id = str(item.get("predicate_id", "")).strip()
        domain = str(item.get("domain", "")).strip()
        if domains is None:
            domain = domain.casefold()
        if predicate_id in result:
            raise DomainValidationError(f"duplicate table assignment for {predicate_id!r}")
        if domain not in allowed:
            raise DomainValidationError(f"unsupported assigned domain {domain!r}")
        result[predicate_id] = domain
    if set(result) != set(expected_ids) or len(result) != len(expected_ids):
        raise DomainValidationError(
            "table assignment must return exactly the supplied predicate IDs"
        )
    return result


def _compiled_predicates_from_payload(
    payload: dict[str, Any], *, expected_ids: tuple[str, ...],
    domains_by_id: dict[str, str] | None = None,
    requirement_converter=_requirement_from_payload,
) -> tuple[tuple[Requirement, ...], tuple[Assumption, ...]]:
    raw = payload.get("requirements")
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise DomainValidationError("requirements must be an array of objects")
    requirements = tuple(requirement_converter(x) for x in raw)
    actual_ids = tuple(x.requirement_id for x in requirements)
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise DomainValidationError(
            "predicate compiler must return exactly the supplied predicate IDs. "
            f"expected {sorted(expected_ids)}, received {sorted(actual_ids)}"
        )
    by_id = {x.requirement_id: x for x in requirements}
    if domains_by_id is not None:
        for item in requirements:
            if item.kind != domains_by_id[item.requirement_id]:
                raise DomainValidationError(f"compiler changed assigned domain for {item.requirement_id}")
    raw_assumptions = payload.get("assumptions", [])
    if not isinstance(raw_assumptions, list) or any(not isinstance(item, dict) for item in raw_assumptions):
        raise DomainValidationError("assumptions must be an array of objects")
    assumptions = tuple(
        Assumption(x["assumption_id"], x["statement"], x.get("status", "unresolved"))
        for x in raw_assumptions
    )
    return tuple(by_id[x] for x in expected_ids), assumptions


def _preserve_compilation_assumptions(existing, incoming) -> tuple[Assumption, ...]:
    """Independent compiler calls must not overwrite each other's local IDs."""
    reserved = {item.assumption_id: item for item in existing}
    result = []
    for item in incoming:
        original_id = item.assumption_id
        suffix = 2
        while item.assumption_id in reserved and reserved[item.assumption_id] != item:
            item = replace(item, assumption_id=f"{original_id}_{suffix}")
            suffix += 1
        if item.assumption_id not in reserved:
            result.append(item)
            reserved[item.assumption_id] = item
    return tuple(result)


def _render_logic_natural(logic: LogicExpression) -> str:
    """Render Boolean state using predicate meaning rather than internal IDs.

    This is the representation shown to the correction merger/verifier. It keeps
    the language model focused on eligibility semantics while internal IDs remain
    purely an execution concern.
    """
    if logic.kind == "predicate":
        return logic.text or logic.predicate_id or ""
    if logic.kind == "not":
        child = logic.children[0]
        rendered = _render_logic_natural(child)
        return f"NOT ({rendered})"
    joiner = f" {logic.kind.upper()} "
    return "(" + joiner.join(_render_logic_natural(child) for child in logic.children) + ")"


def _logic_scheme_payload(logic: LogicExpression) -> dict[str, Any]:
    return {
        "predicates": [
            {"predicate_id": x.predicate_id, "text": x.text}
            for x in logic.leaves()
        ],
        "logic": logic.render_boolean(),
    }


def _normalized_leaf_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _canonical_value(value: Any) -> Any:
    """Normalize execution-relevant structured fields for stable identity.

    LLM-local predicate IDs and surface wording are deliberately excluded. Text
    values are normalized for case and whitespace so harmless compiler drift does
    not create a new task identity.
    """
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def _requirement_signature(requirement: Requirement) -> str:
    """Canonical structured identity, independent of an LLM predicate ID."""
    return json.dumps(
        {
            "kind": requirement.kind.casefold().strip(),
            "payload": _canonical_value(requirement.payload),
            "polarity": requirement.polarity.casefold().strip(),
            "required": bool(requirement.required),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_requirement_id(requirement: Requirement, *, suffix: str = "") -> str:
    signature = _requirement_signature(requirement)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
    term = requirement.payload.get("term") if isinstance(requirement.payload, dict) else None
    if isinstance(term, str) and term.strip():
        stem = _slug(term)[:28]
    else:
        stem = _slug(requirement.kind)[:28] or "criterion"
    base = f"{stem}_{digest}"
    return base if not suffix else f"{base}_{suffix}"


def _canonicalize_compiled_scheme(
    logic: LogicExpression,
    requirements: tuple[Requirement, ...],
) -> tuple[LogicExpression, tuple[Requirement, ...]]:
    """Discard parse-local LLM IDs and assign deterministic structured IDs.

    The LLM's predicate IDs are syntax-local references only. After compilation,
    all downstream state uses IDs derived from structured requirement semantics.
    This removes cross-turn ID reconciliation entirely.
    """
    by_temp_id = {item.requirement_id: item for item in requirements}
    if len(by_temp_id) != len(requirements):
        raise DomainValidationError("predicate compiler produced duplicate temporary IDs")

    # Group exact structured duplicates. They are unusual but must never create
    # duplicate canonical IDs. Stable suffixes are based only on local occurrence.
    by_signature: dict[str, list[Requirement]] = {}
    for item in requirements:
        by_signature.setdefault(_requirement_signature(item), []).append(item)

    mapping: dict[str, str] = {}
    canonical: list[Requirement] = []
    for signature in sorted(by_signature):
        group = sorted(by_signature[signature], key=lambda item: item.requirement_id)
        for index, item in enumerate(group, start=1):
            suffix = "" if len(group) == 1 else str(index)
            canonical_id = _canonical_requirement_id(item, suffix=suffix)
            mapping[item.requirement_id] = canonical_id
            canonical.append(Requirement(
                requirement_id=canonical_id,
                kind=item.kind,
                payload=item.payload,
                polarity=item.polarity,
                required=item.required,
            ))

    canonical_logic = assign_logic_node_ids(logic.remap_predicate_ids(mapping))
    if len({item.requirement_id for item in canonical}) != len(canonical):
        raise DomainValidationError("canonical requirement identity collision")
    return canonical_logic, tuple(sorted(canonical, key=lambda item: item.requirement_id))

def _diff_structured_task(
    current_task: TaskSpec,
    new_requirements: tuple[Requirement, ...],
    new_logic: LogicExpression,
    assumptions: tuple[Assumption, ...],
    *, rationale: str,
) -> TaskDelta:
    old = {x.requirement_id: x for x in current_task.requirements}
    new = {x.requirement_id: x for x in new_requirements}
    removed = tuple(sorted(set(old) - set(new)))
    added = tuple(new[x] for x in sorted(set(new) - set(old)))
    replaced = tuple(
        new[x] for x in sorted(set(old) & set(new))
        if _requirement_signature(old[x]) != _requirement_signature(new[x])
    )
    return TaskDelta(
        add_requirements=added,
        remove_requirement_ids=removed,
        replace_requirements=replaced,
        logic=new_logic,
        assumption_updates=assumptions,
        metadata_updates={
            "logic_requires_clarification": False,
            "logic_clarification_question": None,
        },
        rationale=rationale,
    )


def _merge_assumptions(
    existing: tuple[Assumption, ...], ambiguities: tuple[str, ...],
) -> tuple[Assumption, ...]:
    by_id = {x.assumption_id: x for x in existing}
    for index, statement in enumerate(ambiguities, start=1):
        base = _slug(statement)[:48] or f"ambiguity_{index}"
        assumption_id = f"logic_{base}"
        suffix = 2
        while assumption_id in by_id:
            assumption_id = f"logic_{base}_{suffix}"
            suffix += 1
        by_id[assumption_id] = Assumption(assumption_id, statement, "unresolved")
    return tuple(by_id.values())


def _task_with_structure_refs(task: TaskSpec) -> TaskSpec:
    logic = assign_logic_node_ids(task.logic) if task.logic is not None else None
    pred_to_id, _ = predicate_ref_maps(TaskSpec(
        domain=task.domain,
        objective=task.objective,
        deliverables=task.deliverables,
        requirements=task.requirements,
        logic=logic,
        context=task.context,
        assumptions=task.assumptions,
        metadata=task.metadata,
        original_request=task.original_request,
    ))
    context = dict(task.context)
    structure_refs = dict(context.get("structure_refs", {}))
    structure_refs["predicates"] = pred_to_id
    context["structure_refs"] = structure_refs
    if "relations" in context:
        context["relations"] = normalize_relations(context.get("relations", ()))
    return TaskSpec(
        domain=task.domain,
        objective=task.objective,
        deliverables=task.deliverables,
        requirements=task.requirements,
        logic=logic,
        context=context,
        assumptions=task.assumptions,
        metadata=task.metadata,
        original_request=task.original_request,
    )


def render_structured_task(task: TaskSpec) -> str:
    pred_to_id, id_to_pred = predicate_ref_maps(task)
    req_by_id = {r.requirement_id: r for r in task.requirements}
    predicates = []
    for pref, rid in sorted(pred_to_id.items(), key=lambda item: int(item[0][1:]) if item[0][1:].isdigit() else 999999):
        req = req_by_id.get(rid)
        if req is None:
            continue
        predicates.append({
            "ref": pref,
            "kind": req.kind,
            "payload": req.payload,
            "polarity": req.polarity,
        })

    def render_logic(node: LogicExpression) -> Any:
        if node.kind == "predicate":
            return id_to_pred.get(node.predicate_id or "", node.predicate_id)
        return {
            "ref": node.node_id,
            "operator": node.kind.upper(),
            "children": [render_logic(child) for child in node.children],
        }

    relations = []
    for rel in normalize_relations(task.context.get("relations", ())):
        item = dict(rel)
        for key in ("left_requirement_id", "right_requirement_id"):
            if key in item:
                item[key] = id_to_pred.get(str(item[key]), item[key])
        if isinstance(item.get("members"), list):
            converted = []
            for member in item["members"]:
                if isinstance(member, str):
                    converted.append(id_to_pred.get(member, member))
                elif isinstance(member, dict):
                    m = dict(member)
                    if m.get("kind") == "predicate":
                        rid = str(m.get("requirement_id", ""))
                        m["requirement_id"] = id_to_pred.get(rid, rid)
                    elif m.get("kind") == "or":
                        m["requirement_ids"] = [id_to_pred.get(str(x), x) for x in m.get("requirement_ids", [])]
                    converted.append(m)
                else:
                    converted.append(member)
            item["members"] = converted
        relations.append(item)
    return json.dumps({"predicates": predicates, "logic": render_logic(task.logic), "relations": relations}, indent=2, ensure_ascii=False)


def _structure_ref_catalog(task: TaskSpec) -> dict[str, tuple[str, ...]]:
    pred_to_id, _ = predicate_ref_maps(task)
    groups = tuple(sorted(_logic_group_kinds(task.logic)))
    relations = tuple(
        str(r["relation_id"]) for r in normalize_relations(task.context.get("relations", ()))
    )
    return {
        "predicates": tuple(sorted(pred_to_id, key=_ref_sort_key)),
        "groups": tuple(sorted(groups, key=_ref_sort_key)),
        "relations": tuple(sorted(relations, key=_ref_sort_key)),
    }


def _logic_group_kinds(logic: LogicExpression | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if logic is None:
        return result
    def walk(node: LogicExpression) -> None:
        if node.kind != "predicate" and node.node_id:
            result[node.node_id] = node.kind
        for child in node.children:
            walk(child)
    walk(assign_logic_node_ids(logic))
    return result


def _ref_sort_key(value: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([a-zA-Z_]+)(\d+)", value)
    return (match.group(1), int(match.group(2)), value) if match else (value, 10**9, value)


def _next_numeric_ref(prefix: str, refs: tuple[str, ...]) -> int:
    numbers = [int(x[len(prefix):]) for x in refs if re.fullmatch(re.escape(prefix) + r"\d+", x)]
    return max(numbers, default=0) + 1


def _checked_correction_actions(payload, refs):
    """Validate stage boundaries independently of the model's JSON schema support."""
    if not isinstance(payload, dict) or type(payload.get("requires_clarification")) is not bool:
        raise DomainValidationError("requires_clarification must be a boolean")
    if not payload["requires_clarification"]:
        raw = payload.get("actions")
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise DomainValidationError("actions must be objects")
        modified = set()
        for item in raw:
            action = item.get("action")
            if action in {"ADD", "MODIFY"}:
                text = item.get("requirement_text")
                if not isinstance(text, str) or not text.strip():
                    raise DomainValidationError("ADD/MODIFY requires non-empty requirement_text")
            if action == "MODIFY":
                if item.get("target") in modified:
                    raise DomainValidationError("multiple MODIFY actions target the same predicate. Combine the changes")
                modified.add(item.get("target"))
            for field in ("excluded", "same_admission", "clear_max_gap"):
                if field in item and type(item[field]) is not bool:
                    raise DomainValidationError(f"{field} must be a boolean")
            if action == "SET_EXCLUDED" and "excluded" not in item:
                raise DomainValidationError("SET_EXCLUDED requires excluded")
            if action == "CHANGE_OPERATOR" and item.get("operator") not in {"AND", "OR"}:
                raise DomainValidationError("group operator must be AND or OR")
        removed = {item.get("target") for item in raw if item.get("action") == "REMOVE"}
        if removed & modified:
            raise DomainValidationError("cannot REMOVE and MODIFY the same predicate")
    return _correction_actions_from_payload(payload, refs)


def _correction_intents_from_payload(payload):
    if not isinstance(payload, dict) or type(payload.get("requires_clarification")) is not bool:
        raise DomainValidationError("requires_clarification must be a boolean")
    if payload["requires_clarification"]:
        return _checked_correction_actions(payload, {"predicates": (), "groups": (), "relations": ()})
    intents = payload.get("intents")
    if not isinstance(intents, list) or not intents:
        raise DomainValidationError("correction extraction requires non-empty intents")
    allowed = {"ADD", "MODIFY", "REMOVE", "MOVE", "SET_EXCLUDED", "CHANGE_OPERATOR",
               "ADD_RELATION", "MODIFY_RELATION", "REMOVE_RELATION"}
    seen = set()
    for item in intents:
        if not isinstance(item, dict) or set(item) != {"intent_id", "action", "instruction", "requirement_text"}:
            raise DomainValidationError("intent must contain only ID, action, instruction and requirement_text")
        ref = item["intent_id"]
        if not isinstance(ref, str) or not re.fullmatch(r"c[1-9]\d*", ref) or ref in seen:
            raise DomainValidationError("intent IDs must be unique c1/c2/... references")
        seen.add(ref)
        if item["action"] not in allowed or not isinstance(item["instruction"], str) or not item["instruction"].strip():
            raise DomainValidationError("intent requires a supported action and non-empty instruction")
        text = item["requirement_text"]
        if item["action"] in {"ADD", "MODIFY"}:
            if not isinstance(text, str) or not text.strip():
                raise DomainValidationError("ADD/MODIFY intent requires requirement_text")
        elif text is not None:
            raise DomainValidationError("structural intents must not introduce requirement_text")
    return {"intents": tuple(dict(item) for item in intents), "requires_clarification": False}


def _correction_attachments_from_payload(payload, refs, intents):
    if payload.get("requires_clarification") is True:
        return _checked_correction_actions(payload, refs)
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != len(intents):
        raise DomainValidationError("attachment must preserve every extracted intent exactly once")
    clean = []
    for action, intent in zip(actions, intents):
        if not isinstance(action, dict) or action.get("intent_id") != intent["intent_id"] or action.get("action") != intent["action"]:
            raise DomainValidationError("attachment changed intent IDs, order or action types")
        if intent["action"] in {"ADD", "MODIFY"} and action.get("requirement_text") != intent["requirement_text"]:
            raise DomainValidationError("attachment rewrote fixed requirement_text")
        clean.append({key: value for key, value in action.items() if key != "intent_id"})
    return _checked_correction_actions({**payload, "actions": clean}, refs)


def _carbon_correction_from_payload(payload, refs, *, requirement_converter=_requirement_from_payload):
    interpreted = _checked_correction_actions(payload, refs)
    expected = tuple(
        item["local_ref"] if item["action"] == "ADD" else item["target"]
        for item in interpreted["actions"] if item["action"] in {"ADD", "MODIFY"}
    )
    requirements, assumptions = _compiled_predicates_from_payload(payload, expected_ids=expected, requirement_converter=requirement_converter)
    if not expected and assumptions:
        raise DomainValidationError("structural-only/clarification responses must not add assumptions")
    return {**interpreted, "compiled_changes": {r.requirement_id: r for r in requirements},
            "compiled_assumptions": assumptions}


def _correction_actions_from_payload(
    payload: dict[str, Any], refs: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    clarify = bool(payload.get("requires_clarification", False))
    question = payload.get("clarification_question")
    if clarify:
        if not isinstance(question, str) or not question.strip():
            raise DomainValidationError("clarification requires a question")
        rationale = payload.get("clarification_rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise DomainValidationError("clarification rationale must be text")
        return {
            "actions": (),
            "requires_clarification": True,
            "clarification_question": question.strip(),
            "clarification_rationale": (rationale or "").strip() or None,
        }

    raw = payload.get("actions")
    if not isinstance(raw, list) or not raw:
        raise DomainValidationError("correction interpretation requires at least one action")
    p_refs, g_refs, r_refs = map(set, (refs["predicates"], refs["groups"], refs["relations"]))
    local_p: set[str] = set()
    local_r: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise DomainValidationError("each correction action must be an object")
        action = str(item.get("action", "")).strip().upper()
        if action == "ADD":
            local = str(item.get("local_ref", ""))
            if not re.fullmatch(r"new_[1-9]\d*", local) or local in local_p:
                raise DomainValidationError(f"invalid or duplicate local predicate ref: {local}")
            local_p.add(local)
        elif action == "ADD_RELATION":
            local = str(item.get("local_ref", ""))
            if not re.fullmatch(r"new_r[1-9]\d*", local) or local in local_r:
                raise DomainValidationError(f"invalid or duplicate local relation ref: {local}")
            local_r.add(local)

    normalized: list[dict[str, Any]] = []
    # Local predicate anchors become valid only after their ADD has appeared.
    # This prevents self-anchors and future-local anchors while still allowing
    # a later ADD in the same correction to refer to an earlier new predicate.
    available_local_p: set[str] = set()
    for item in raw:
        action = str(item.get("action", "")).strip().upper()
        data = dict(item)
        data["action"] = action
        if action == "ADD":
            # Inclusion is the semantic default. Some structured-generation
            # paths omit an explicit false boolean. Normalize that omission
            # here instead of failing later with KeyError('excluded').
            data["excluded"] = bool(data.get("excluded", False))
            _validate_add_attachment_fields(data, p_refs | available_local_p, g_refs)
            for rid in data.get("relation_memberships", ()): 
                if rid not in r_refs:
                    raise DomainValidationError(f"unknown relation ref: {rid}")
            available_local_p.add(str(data["local_ref"]))
        elif action in {"MODIFY", "REMOVE", "SET_EXCLUDED", "MOVE"}:
            target = str(data.get("target", ""))
            if target not in p_refs:
                raise DomainValidationError(f"unknown predicate ref: {target}")
            if action == "MOVE":
                _validate_attachment_fields(data, p_refs, g_refs)
        elif action == "CHANGE_OPERATOR":
            if str(data.get("target", "")) not in g_refs:
                raise DomainValidationError(f"unknown group ref: {data.get('target')}")
        elif action == "ADD_RELATION":
            pass
        elif action in {"MODIFY_RELATION", "REMOVE_RELATION"}:
            if str(data.get("target", "")) not in r_refs:
                raise DomainValidationError(f"unknown relation ref: {data.get('target')}")
        else:
            raise DomainValidationError(f"unsupported correction action: {action}")
        normalized.append(data)

    endpoint_refs = p_refs | local_p
    for data in normalized:
        if data["action"] not in {"ADD_RELATION", "MODIFY_RELATION"}:
            continue
        for key in ("left", "right"):
            value = data.get(key)
            if value is not None and value not in endpoint_refs:
                raise DomainValidationError(f"unknown relation endpoint ref: {value}")
        for value in data.get("members", ()):
            if value not in endpoint_refs:
                raise DomainValidationError(f"unknown relation member ref: {value}")
    return {
        "actions": tuple(normalized),
        "requires_clarification": False,
        "clarification_question": None,
        "clarification_rationale": None,
    }


def _validate_add_attachment_fields(data: dict[str, Any], p_refs: set[str], g_refs: set[str]) -> None:
    """Validate the minimal ADD attachment interface: anchor + Boolean operator."""
    anchor = str(data.get("anchor", ""))
    if anchor != "ROOT" and anchor not in p_refs and anchor not in g_refs:
        raise DomainValidationError(f"unknown attachment anchor: {anchor}")
    if str(data.get("operator", "")).upper() not in {"AND", "OR"}:
        raise DomainValidationError("attachment operator must be AND or OR")


def _validate_attachment_fields(data: dict[str, Any], p_refs: set[str], g_refs: set[str]) -> None:
    """Placement-aware attachment validation retained for MOVE."""
    anchor = str(data.get("anchor", ""))
    if anchor != "ROOT" and anchor not in p_refs and anchor not in g_refs:
        raise DomainValidationError(f"unknown attachment anchor: {anchor}")
    if str(data.get("operator", "")).upper() not in {"AND", "OR"}:
        raise DomainValidationError("attachment operator must be AND or OR")
    placement = str(data.get("placement", "")).upper()
    if placement not in {"WITH", "INSIDE"}:
        raise DomainValidationError("attachment placement must be WITH or INSIDE")
    if placement == "INSIDE" and anchor in p_refs:
        raise DomainValidationError("INSIDE requires ROOT or a Boolean group anchor")


def _semantic_add_attachment(*, anchor: str, operator: str) -> AttachmentSpec:
    """Translate ADD anchor/operator directly into the existing graph attachment."""
    op = operator.casefold()
    if op not in {"and", "or"}:
        raise DomainValidationError("attachment operator must be AND or OR")
    normalized_anchor = "root" if anchor == "ROOT" else anchor
    return AttachmentSpec(
        normalized_anchor,
        "and_child" if op == "and" else "or_child",
    )


def _semantic_attachment(
    *, anchor: str, operator: str, placement: str, task: TaskSpec,
    group_kinds: dict[str, str],
) -> AttachmentSpec:
    op = operator.casefold()
    placement = placement.upper()
    normalized_anchor = "root" if anchor == "ROOT" else anchor
    if placement == "INSIDE":
        if anchor == "ROOT":
            root = assign_logic_node_ids(task.logic)
            if root.kind != op:
                raise DomainValidationError(
                    f"cannot attach {operator} INSIDE ROOT because ROOT is {root.kind.upper()}"
                )
        elif anchor.startswith("g"):
            if group_kinds.get(anchor) != op:
                raise DomainValidationError(
                    f"cannot attach {operator} INSIDE {anchor}. It is {group_kinds.get(anchor, 'unknown').upper()}"
                )
        else:
            raise DomainValidationError("INSIDE requires a Boolean group")
    return AttachmentSpec(normalized_anchor, "and_child" if op == "and" else "or_child")


def _gap_hours(value: Any, unit: Any) -> int:
    if value is None or unit is None:
        raise DomainValidationError("max gap requires both value and unit")
    number = float(value)
    factor = {"hour": 1, "day": 24, "week": 168}.get(str(unit).casefold())
    if factor is None:
        raise DomainValidationError(f"unsupported gap unit: {unit}")
    hours = number * factor
    if hours < 0 or abs(hours - round(hours)) > 1e-9:
        raise DomainValidationError("max gap must convert to a whole non-negative number of hours")
    return int(round(hours))


def _relation_from_correction_action(action: dict[str, Any], *, relation_id: str, pred_ref) -> dict[str, Any]:
    kind = str(action.get("relation", "")).casefold()
    if kind in {"before", "after"}:
        if not action.get("left") or not action.get("right"):
            raise DomainValidationError(f"{kind} relation requires left and right endpoints")
        result: dict[str, Any] = {
            "relation_id": relation_id,
            "relation": kind,
            "left_requirement_id": pred_ref(str(action["left"])),
            "right_requirement_id": pred_ref(str(action["right"])),
            "same_admission": bool(action.get("same_admission", False)),
        }
        if action.get("max_gap_value") is not None or action.get("max_gap_unit") is not None:
            result["max_gap_hours"] = _gap_hours(action.get("max_gap_value"), action.get("max_gap_unit"))
        return result
    if kind == "same_admission":
        members = tuple(action.get("members") or ())
        if len(members) < 2:
            raise DomainValidationError("same_admission relation requires at least two members")
        return {
            "relation_id": relation_id,
            "relation": "same_admission",
            "members": [pred_ref(str(x)) for x in members],
        }
    raise DomainValidationError(f"unsupported relation type: {kind}")


def _relation_changes_from_correction_action(action: dict[str, Any], *, current: dict[str, Any], pred_ref) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if action.get("relation") is not None:
        changes["relation"] = str(action["relation"]).casefold()
    effective_kind = changes.get("relation", str(current.get("relation", "")).casefold())
    if effective_kind in {"before", "after"}:
        if action.get("left") is not None:
            changes["left_requirement_id"] = pred_ref(str(action["left"]))
        if action.get("right") is not None:
            changes["right_requirement_id"] = pred_ref(str(action["right"]))
        if action.get("members") is not None:
            raise DomainValidationError("before/after relation cannot use members")
    elif effective_kind == "same_admission":
        if action.get("members") is not None:
            changes["members"] = [pred_ref(str(x)) for x in action["members"]]
        if action.get("left") is not None or action.get("right") is not None:
            raise DomainValidationError("same_admission relation uses members, not left/right")
    else:
        raise DomainValidationError(f"unsupported relation type: {effective_kind}")
    if action.get("clear_max_gap"):
        changes["max_gap_hours"] = None
    elif action.get("max_gap_value") is not None or action.get("max_gap_unit") is not None:
        changes["max_gap_hours"] = _gap_hours(action.get("max_gap_value"), action.get("max_gap_unit"))
    if action.get("same_admission") is not None:
        changes["same_admission"] = bool(action["same_admission"])
    return changes


def _cohort_semantic_signature(task: TaskSpec) -> str:
    return json.dumps(
        {
            "requirements": sorted(
                (_requirement_signature(r) for r in task.requirements)
            ),
            "logic": task.logic.to_dict() if task.logic is not None else None,
            "relations": normalize_relations(task.context.get("relations", ())),
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )

def _correction_mutations(current_task: TaskSpec, delta: TaskDelta) -> tuple[ArtifactMutation, ...]:
    mutations: list[ArtifactMutation] = []
    for act in delta.acts:
        target = getattr(act, "target_ref", None)
        name = type(act).__name__
        kind = MutationKind.UPDATE_VALUE
        if name.startswith("Add"):
            kind = MutationKind.ADD_CONSTRAINT
        elif name.startswith("Remove"):
            kind = MutationKind.REMOVE_CONSTRAINT
        mutations.append(ArtifactMutation(
            kind,
            CorrectionTarget("structured_ref", target, name),
            name,
        ))
    if delta.logic is not None and delta.logic != current_task.logic:
        mutations.append(ArtifactMutation(
            MutationKind.UPDATE_VALUE,
            CorrectionTarget("eligibility_logic", "root", "Revise Boolean eligibility structure."),
            "Revise Boolean eligibility structure.",
            {"logic": delta.logic.to_dict()},
        ))
    for item in delta.add_requirements:
        mutations.append(ArtifactMutation(
            MutationKind.ADD_CONSTRAINT,
            CorrectionTarget("requirement", item.requirement_id, "Add cohort predicate."),
            "Add cohort predicate.",
        ))
    for item in delta.replace_requirements:
        mutations.append(ArtifactMutation(
            MutationKind.UPDATE_VALUE,
            CorrectionTarget("requirement", item.requirement_id, "Update cohort predicate semantics."),
            "Update cohort predicate semantics.",
        ))
    for item_id in delta.remove_requirement_ids:
        mutations.append(ArtifactMutation(
            MutationKind.REMOVE_CONSTRAINT,
            CorrectionTarget("requirement", item_id, "Remove cohort predicate."),
            "Remove cohort predicate.",
        ))
    return tuple(mutations)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
