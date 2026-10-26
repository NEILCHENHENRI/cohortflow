"""Compact TaskDelta correction prompt (v2)."""

COMPACT_CORRECTION_PROMPT = """
Translate the correction into the smallest complete edit of the current cohort.
Use only listed predicate/group/relation refs and new_1, new_2, ... handles for new
predicates. Code assigns persistent IDs and applies edits. Keep each clinical
predicate atomic and in natural language. Do not put AND/OR inside requirement_text.

ADD introduces a criterion. MODIFY replaces an existing criterion in place.
REMOVE retracts it. SET_EXCLUDED explicitly sets negation. CHANGE_OPERATOR changes
a group. MOVE relocates a criterion using WITH/INSIDE. ADD has no placement field.
ADD_RELATION/MODIFY_RELATION/REMOVE_RELATION edit event relationships separately.

For ADD, anchor is the expression X replaced by (X operator NEW).
Use a predicate/group for a local alternative, ROOT for a whole-cohort change.
Preserve every unrelated criterion. "Also" alone does not determine scope/operator.
"Count too" retains the existing option: ADD with OR, not MODIFY.
"Require" adds AND. "Never mind X" removes X. "exclude X" uses excluded=true.

Examples, current ROOT=(adult AND diagnosis AND CT), CT=p3:
"MRI counts too" -> ADD MRI, anchor=p3, operator=OR.
"MRI patients qualify regardless of other criteria" -> ADD MRI, anchor=ROOT, operator=OR.
"Also require MRI after diagnosis" -> ADD MRI, anchor=ROOT, operator=AND,
  plus AFTER relation left=the new MRI handle, right=the diagnosis ref.

Preserve timing/units/qualifiers. A relation connects event predicate refs, never
relation refs. Add a new endpoint before referencing its handle. Put cross-event
time gaps on the relation. Do not recompile unchanged endpoints.
New ADDs may anchor to earlier new handles, never themselves or future handles.
Request clarification only for materially different plausible meanings.
Return the supplied schema. For ADD, use relation_memberships=[] when none apply.
""".strip()
