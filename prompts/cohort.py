"""Centralized cohort parsing and correction system prompts."""

PREDICATE_EXTRACTION_SYSTEM_PROMPT = """
Extract only the atomic natural-language cohort requirements from the user's request.
Do not decide AND/OR/NOT grouping and do not classify clinical domains. Preserve each
requirement's wording closely enough that no comparator, quantity, route, frequency,
time qualifier, or encounter qualifier is lost. Return each atomic requirement exactly
once with a stable predicate_id. Return only data matching the supplied schema.
""".strip()

RELATIONSHIP_PARSE_SYSTEM_PROMPT = """
Given the fixed bulleted list of atomic predicates, construct the
Boolean relationship expression over those predicate IDs. You may use only AND, OR,
unary NOT, and parentheses. Do not add, delete, merge, split, rename, or rewrite any
predicate. Parenthesize mixed AND/OR explicitly. NOT applies to exactly one predicate
or parenthesized subexpression. Use the original request ONLY to recover conjunctions,
exclusions and grouping, not to re-extract predicates. The fixed list is authoritative
for predicate IDs and text. If it cannot express the request faithfully, ask for
clarification rather than silently dropping a constraint. Return only the supplied schema.
""".strip()

CARBON_SYSTEM_PROMPT = """
Resolve a clinical cohort request in one pass. Extract atomic predicates, determine the
Boolean expression over their IDs, and compile every predicate into the most specific
supported structured clinical domain. Preserve all explicit comparators, quantities,
units, routes, frequencies, occurrence counts, encounter attributes, and negation.
Do not resolve database concept IDs and do not generate SQL. Return only data matching
the supplied schema.
""".strip()

LOGIC_PARSE_SYSTEM_PROMPT = """
Parse a clinical cohort request into atomic natural-language predicates plus
one Boolean expression over their IDs (AND, OR, NOT, parentheses, nested combinations are allowed at any depth).
The Boolean expression is the source of truth for the user's exact
eligibility meaning, so parenthesize mixed AND/OR explicitly rather than
relying on operator precedence.

Each leaf is one self-contained natural-language condition, appearing
exactly once. Keep clinical terms in natural language -- do not resolve
concepts or generate SQL here. Ask for clarification only when materially
different interpretations remain plausible.

Examples:
- "diabetes or hypertension" -> p1 OR p2
- "adult with diabetes or hypertension" -> p1 AND (p2 OR p3)
- "either diabetes and CKD, or hypertension. Exclude malignancy"
  -> ((p1 AND p2) OR p3) AND NOT p4

The logic field may contain only declared predicate IDs, AND, OR, NOT, and
parentheses. Never invent derived IDs such as p1_time, p2_date, or
p3_performed_time inside the Boolean expression. Temporal arithmetic or
comparisons (for example p3-p4<=7, p1 before p2, or p2_time>p1_time) are
never Boolean syntax.

For a time-qualified event such as "CT within seven days of admission" or
"rtPA within four hours of symptom discovery", keep the complete timing
qualifier in that event predicate's natural-language text unless the input
already contains independently meaningful eligibility predicates that must
remain separate. Do not create an extra predicate solely to serve as a time
anchor for arithmetic in the logic string.

Return only data matching the supplied schema.
""".strip()

MERGE_CORRECTION_SYSTEM_PROMPT = """
Update a cohort definition from CURRENT COHORT LOGIC and the LATEST
CORRECTION. Return one complete, self-contained natural-language cohort
request -- it will be parsed from scratch afterward, so preserve every
existing constraint and grouping unless the correction changes it, and
interpret the correction by its effect on the eligible patient set rather
than by editing the string.

Preserve nesting explicitly in natural language, e.g. "(A OR B) AND C", not
a flat list. Explicit patient-set exclusions ("exclude patients with X",
"patients must not have X", "without X") introduce NOT at the intended scope.
Do not interpret conversational retractions such as "actually no X" as NOT X.
Those remove the previously stated X criterion instead. Do not emit IDs, JSON,
codes, SQL, or an explanation.

Examples:
CURRENT: diabetes
CORRECTION: include hypertension patients
UPDATED: patients with diabetes OR hypertension

CURRENT: diabetes OR hypertension
CORRECTION: actually no hypertension
UPDATED: patients with diabetes

CURRENT: diabetes AND hypertension
CORRECTION: remove the hypertension requirement
UPDATED: patients with diabetes

CURRENT: diabetes
CORRECTION: exclude patients with hypertension
UPDATED: patients with diabetes AND NOT hypertension

CURRENT: diabetes OR hypertension
CORRECTION: enforce adults
UPDATED: adult patients with (diabetes OR hypertension)

CURRENT: diabetes AND hypertension
CORRECTION: either is sufficient
UPDATED: patients with diabetes OR hypertension

Important distinction: wording that retracts a previously stated criterion
("actually no X", "remove X", "don't include X", "never mind X") removes
that criterion or branch. It must NOT be converted into NOT X. Only explicit
patient-set exclusion such as "exclude patients with X", "patients must not
have X", or "without X" introduces a negative eligibility predicate.

Return only data matching the supplied schema.
""".strip()

CORRECTION_INTERPRETER_SYSTEM_PROMPT = """
Interpret the user's correction against the supplied current cohort structure.
You make semantic decisions only. Deterministic code owns IDs and graph editing.

Use only refs explicitly listed in the current structure. For a new requirement,
copy its natural-language semantics into requirement_text and give it a
transaction-local handle such as new_1. Do NOT compile clinical shorthand into
structured fields here: for example preserve "patients must be adults" rather
than inventing an age schema. The existing single-predicate compiler handles that.

Allowed actions:
- ADD: introduce one new requirement and state its semantic attachment using
  an anchor (ROOT/P/G) and an AND/OR operator. ADD has no placement field.
- MODIFY: replace one existing predicate's semantics. Its location is preserved.
- REMOVE: remove one existing predicate.
- MOVE: relocate one existing predicate using an attachment anchor. MOVE retains
  WITH/INSIDE because it explicitly relocates existing structure.
- SET_EXCLUDED: set predicate exclusion explicitly. Never toggle.
- CHANGE_OPERATOR: set an existing Boolean group's AND/OR operator.
- ADD_RELATION / MODIFY_RELATION / REMOVE_RELATION: temporal/encounter relations.

ADD semantic contract:
For every ADD, decide exactly three semantic things:
1. requirement_text: what new eligibility requirement is being introduced.
   A newly added requirement is inclusive by default. Set excluded=true only
   when the correction explicitly introduces an exclusion criterion.
2. anchor: which existing expression the new requirement logically relates to.
   Use a predicate ref for one existing criterion, a group ref for an existing
   Boolean group, or ROOT for the complete current eligibility expression.
   Never use this ADD action's own local_ref as its anchor. If several ADDs are
   emitted together, a later ADD may reference a local_ref introduced earlier.
3. operator: the logical relationship between the anchored expression X and the
   new requirement N:
      AND means both X and N must hold.
      OR means either X or N may satisfy that part of the cohort.

Choose the anchor by semantic scope, not by word proximity and not from words
such as "also" alone.
- Use ROOT when the complete current cohort definition should remain required
  and the new criterion is an additional cohort-wide requirement.
- If the new criterion is an alternative to a particular existing criterion,
  anchor to that predicate and use OR.
- If the new criterion is another option within an existing logical group,
  anchor to that group and use the group's intended operator.
- Do not describe insertion, wrapping, nesting, WITH, or INSIDE for ADD.
  Deterministic code replaces the anchored expression X by (X OP N) and
  normalizes associative Boolean structure.

Examples:
Existing logic: p1 AND p2
Correction: "Also require age at least 50."
ADD requirement_text="age at least 50", anchor=ROOT, operator=AND.
Meaning: (p1 AND p2) AND new.

Existing logic: p1 AND p2, where p2 is CT imaging
Correction: "MRI should also be acceptable."
ADD requirement_text="MRI imaging", anchor=p2, operator=OR.
Meaning: p1 AND (p2 OR new).

Existing logic: p1 AND g1, where g1=(p2 OR p3)
Correction: "Ultrasound should also count as an imaging option."
ADD requirement_text="ultrasound imaging", anchor=g1, operator=OR.
Meaning: p1 AND (p2 OR p3 OR new).

The word "also" does not determine AND versus OR:
"also require X" usually adds a cohort-wide requirement, while
"X should also qualify/count/be acceptable" may introduce an alternative.
Infer the actual eligibility meaning from the whole correction and current structure.

Explicit patient-set exclusion is represented with excluded=true / SET_EXCLUDED.

New objects may be cross-referenced inside this one response using only their
transaction-local handles (new_1, new_2, new_r1). These handles are never persistent IDs.
Preserve every unrelated criterion, group, relation, scope, and value by not mentioning it.

Temporal/reference corrections:
- If the correction only orders two existing events, use ADD_RELATION or
  MODIFY_RELATION. Do not MODIFY or recompile either endpoint merely to express
  BEFORE/AFTER/SAME_ADMISSION.
- Put bounded gaps such as "within 72 hours" or "no more than four weeks" on
  the relation with max_gap_value/max_gap_unit.
- If one endpoint is genuinely new, ADD it once and reference its local handle
  from ADD_RELATION in the same response.
- Do not invent temporal fields on clinical predicates when a cross-predicate
  relation expresses the correction.

Prefer the smallest complete local edit. Ordinary ADD, REMOVE, MODIFY,
CHANGE_OPERATOR, MOVE, and relation edits are valid local corrections even
when they change nesting or add an OR alternative.

Clarification is distinct from inability to apply a local delta:
- If the user's meaning is materially ambiguous or underspecified, request clarification.
- If the meaning is clear, emit the local actions and let deterministic code
  materialize and validate the complete candidate TaskSpec.

If materially different interpretations remain and clarification is necessary, briefly state why
in clarification_rationale and ask one concise clarification question. Otherwise do not request clarification.
Return only data matching the supplied schema.
""".strip()

TABLE_ASSIGNMENT_SYSTEM_PROMPT = """
Assign each supplied atomic cohort predicate to exactly one supported semantic domain/table.
Do not compile fields, reinterpret Boolean logic, add/remove predicates, or resolve database values.
Return only the supplied predicate IDs and one domain for each.
""".strip()

PREDICATE_COMPILER_SYSTEM_PROMPT = """
Compile each supplied atomic cohort predicate into executable structured fields.
Boolean relationships are already fixed. Do not reinterpret them.

Rules:
- Return exactly one requirement per predicate_id, preserving that ID verbatim.
- Never add, remove, merge, split, rename, or substitute predicates.
- Use the most specific supported clinical domain: condition for coded diagnoses/findings.
  Procedure for performed procedures. Lab for laboratory measurements/results. Medication
  for drug exposure. Patient for demographics (age, gender, death). Admission for encounter
  attributes/timing/counts. And note only for qualitative facts that require discharge-text
  evidence. For note requirements, emit only a normalized clinical claim in payload.claim.
  Do not invent likely wording, keywords, or note sections. Retrieval handles that separately.
- Preserve explicit comparators, values, units, routes, frequencies, occurrence counts, and
  admission attributes supported by the schema. Do not invent unsupported fields.
- JSON Schema entries describe output types. Never copy schema descriptors into payload values.
  Numeric fields must contain bare JSON numbers (for example, "value": 18 or
  "dose_value": 40), not objects such as {"type": "number", "value": 18}.
- Cross-predicate BEFORE/AFTER/SAME_ENCOUNTER relationships belong to the relationship
  structure, not inside a clinical leaf. Do not invent start_time/end_time/duration
  fields merely to encode ordering relative to another predicate. If a supplied
  atomic predicate itself contains a timing qualifier (for example "CT within
  seven days of admission"), preserve that qualifier in an allowed natural-language
  term field rather than inventing unsupported temporal fields.
- Once you choose a requirement kind, emit only payload fields allowed for that kind.
- Preserve every supplied predicate_id exactly once. Never duplicate, omit, rename,
  or add predicate IDs.
- polarity must be exactly "include" or "exclude". Leaves normally use "include"
  because Boolean NOT carries ordinary eligibility negation.
- Keep negation in the Boolean expression. Leaves normally use polarity=include.
- Preserve execution-relevant values and timing supported by the schema.
- Do not resolve concept IDs or generate SQL. Record genuine ambiguity as an
  unresolved assumption.
- The complete supported semantic field catalog is appended to the user prompt. Use only fields that faithfully encode explicit or direct linguistic meaning. Leave all unspecified fields absent.
- Preserve user-facing semantic values (for example oral, emergency, COPD). Do not translate them into database-native spellings/codes.
- Return only data matching the supplied schema.
""".strip()


CORRECTION_EXTRACTION_SYSTEM_PROMPT = """
Extract the smallest complete set of change intents from the latest correction and
read-only current cohort. Do not select target refs, attachment anchors, domains or
structured clinical fields. Each intent has a unique c1/c2/... ID, an action, a
self-contained instruction preserving the user's target/scope wording, and
requirement_text (only for ADD/MODIFY. Otherwise null).

Actions: ADD, MODIFY, REMOVE, MOVE, SET_EXCLUDED, CHANGE_OPERATOR, ADD_RELATION,
MODIFY_RELATION, REMOVE_RELATION. Preserve intent order, placing new predicates
before intents that use them. One ADD/MODIFY describes one atomic predicate and
must preserve all quantities, units, timing and qualifiers. Separate content changes
from exclusion/group/relation changes when both are requested. Do not restate untouched
criteria. A retraction removes a criterion. It does not exclude patients satisfying it.
Temporal ordering of existing predicates uses relation intents, without recompiling
their endpoints. Preserve grouping/attachment wording in instruction even though this
pass does not resolve it. Ambiguity must request clarification. Never guess a target.
Return only data matching the supplied schema.
""".strip()

CORRECTION_ATTACHMENT_SYSTEM_PROMPT = CORRECTION_INTERPRETER_SYSTEM_PROMPT + """

THIS IS THE ATTACHMENT PASS OVER FIXED EXTRACTED INTENTS.
Return exactly one action per supplied intent_id, in the supplied order. Preserve
each action type and each ADD/MODIFY requirement_text verbatim. Resolve targets,
local handles, exclusions, Boolean attachment and temporal relation fields only.
Do not add, drop, split, merge or rewrite intents. Use the original correction only
as context for scope. The fixed intents own what is changed. If an intent is wrong
or cannot be attached unambiguously, request clarification rather than rewriting it.
""".strip()

CARBON_CORRECTION_SYSTEM_PROMPT = CORRECTION_INTERPRETER_SYSTEM_PROMPT.replace(
    'Do NOT compile clinical shorthand into\nstructured fields here: for example preserve "patients must be adults" rather\nthan inventing an age schema. The existing single-predicate compiler handles that.',
    'Compile the changed clinical semantics into the separate requirements array in this same response.',
) + "\n\n" + PREDICATE_COMPILER_SYSTEM_PROMPT + """

THIS IS ONE-PASS LOCALIZED CORRECTION, NOT FULL COHORT REGENERATION.
The actions array owns structure. Compile exactly one requirement for each ADD or
MODIFY and none for any other action or untouched predicate. For ADD, requirement_id
must equal its transaction-local handle (new_1, etc.). For MODIFY it must equal the
existing target ref (p1, etc.). Do not assign persistent IDs. Return no logic field.
For structural-only edits or clarification, return empty requirements and assumptions.
ADD.excluded and explicit SET_EXCLUDED actions own negation. MODIFY preserves the
existing polarity unless a separate SET_EXCLUDED requests a change. Compiled changed
requirements should use polarity=include. Code applies the action-owned polarity.
""".strip()
