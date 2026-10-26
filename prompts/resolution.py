"""Centralized LLM system prompts."""

SEMANTIC_RESOLUTION_SYSTEM_PROMPT = """
Resolve a narrow semantic mismatch using only the supplied input and options.
Never invent an option ID. Do not silently choose a more specific clinical
meaning. Ask the human when meaning could materially change.
""".strip()

HUMAN_RESOLUTION_REPLY_SYSTEM_PROMPT = """
Interpret a user's reply to a pending option-selection question.

A single reply may simultaneously answer the pending question, clarify why a
choice is intended, modify any other part of the task, and request information.
The reply may also make the pending question obsolete. If the user retracts,
removes, replaces, or negates the task criterion that caused this clarification
(e.g. pending question is about hypertension and the user says "actually no
hypertension", "remove hypertension", "don't include hypertension", or
"replace hypertension with CKD"), DO NOT force an option selection. Return
action="abort", put the complete task-changing instruction in residual_message,
and include modify_task plus the best matching change act. This means "the
question is no longer relevant", not "abort the whole conversation".
Treat clauses such as "option 2, and make age at least 60" as two acts: answer
the pending question and preserve the new correction verbatim in
residual_message. Select a candidate only when it is uniquely identified from
the supplied options. Preserve every remaining instruction in residual_message.
Never discard clauses before or after the answer. If a phrase matches
multiple options and the question only allows one answer, ask a narrower
question rather than selecting the first.

If the pending question allows selecting more than one option (response_mode
is "select_many"), populate candidate_ids with every option the reply
selects. Set candidate_id to the first entry of candidate_ids for backward
compatibility. If only one option is selected, candidate_ids may hold that
single id. Never invent an option ID.

A reply may name what to EXCLUDE rather than what to select (e.g.
"everything except the ketoacidosis one", "all but the first two"). In that
case candidate_ids must be every option EXCEPT the ones named -- invert the
selection yourself. Never return only the named/excluded options.
""".strip()
