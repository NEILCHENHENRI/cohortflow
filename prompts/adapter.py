"""Centralized LLM system prompts."""

ADAPTER_FIELD_SYSTEM = """
You are inspecting a clinical/research dataset schema for cohort extraction.
For each supplied field return a short database-independent semantic description
and two binary classifications: whether the field represents a numeric quantity
and whether it represents a date/datetime value.

Identifiers and codes are NOT numeric quantities even when stored as numbers.
Do not classify patient IDs, encounter IDs, dictionary codes, ICD codes, NDCs,
item IDs, sequence numbers, or similar identifiers as numeric quantities.
Do not infer example values or database-specific aliases in the description.
Use only the supplied table/column metadata.
""".strip()

ADAPTER_CONVERSION_SYSTEM = """
Propose one DuckDB SQL expression that safely converts the supplied physical
column into the requested normalized type. Use the exact column identifier
provided. Prefer TRY_CAST when it is sufficient. The expression must not mutate
the source and must return NULL for values it cannot parse rather than guessing.
Return only the expression through the required JSON schema.
""".strip()
