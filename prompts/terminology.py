"""Centralized LLM system prompts."""

MEDICAL_TERMINOLOGY_EXPANSION_SYSTEM = """
Expand one clinical term only for terminology retrieval against a clinical database.
Return clinically interchangeable surface forms, not merely related concepts.
Allowed forms include abbreviation/full-name pairs, common clinical synonyms,
lay/medical equivalents, legacy or historical terminology, spelling/punctuation
variants, standard alternate phrasing, common acronym expansions, brand/generic
medication equivalents, and closely equivalent coding-language phrasing.
Do NOT add symptoms, causes, complications, risk factors, differential diagnoses,
broader/narrower diseases that are not interchangeable, or speculative associations.
Keep the list compact (at most 8 terms) and preserve the intended clinical meaning.
""".strip()
