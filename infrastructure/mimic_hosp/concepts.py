from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from framework.agent import ProgressCallback

from framework.resolution import (
    InMemoryResolutionStore,
    ResolutionAction,
    ResolutionKey,
    ResolutionOption,
    ResolutionRecord,
    ResolutionRequest,
    ResolutionRequired,
    ResolutionStore,
    ResponseMode,
    SemanticResolver,
)
from infrastructure.database.duckdb import DuckDBConnectionFactory




class TerminologyExpander(Protocol):
    def expand(
        self, term: str, *, domain: str, on_progress: ProgressCallback | None = None,
    ) -> tuple[str, ...]: ...

@dataclass(frozen=True, slots=True)
class MimicHospConceptEntry:
    """One resolved code (or drug name) within a concept."""

    code: str
    display_name: str
    vocabulary_id: str  # ICD9CM | ICD10CM | ICD9PROC | ICD10PCS | d_labitems | prescriptions
    display_identifier: str | None = None

    @property
    def source_tables(self) -> tuple[str, ...]:
        """Which raw MIMIC-IV hosp tables actually backed this fact.

        Computed from vocabulary_id rather than stored, so it needs no
        changes to the resolution-store round trip (_entry_payload /
        _concept_payload / _concept_from_payload already serialize just
        the three fields above). Conditions resolve against a dictionary
        table (d_icd_diagnoses) but patient membership is computed against
        a separate fact table (diagnoses_icd). Medications have no
        dictionary at all -- prescriptions.drug is both.
        """
        if self.vocabulary_id in ("ICD9CM", "ICD10CM"):
            return ("d_icd_diagnoses", "diagnoses_icd")
        if self.vocabulary_id in ("ICD9PROC", "ICD10PCS"):
            return ("d_icd_procedures", "procedures_icd")
        if self.vocabulary_id == "d_labitems":
            return ("d_labitems", "labevents")
        if self.vocabulary_id == "prescriptions":
            return ("prescriptions",)
        return ()


@dataclass(frozen=True, slots=True)
class MimicHospConcept:
    """A resolved clinical term against raw MIMIC-IV hosp-module tables.

    There is no standardized vocabulary in this schema (unlike OMOP's
    ``concept`` table), and a single clinical idea is often split across
    several literal codes: ICD-9-CM and ICD-10-CM code the same condition
    differently, and a condition family (e.g. "type 2 diabetes") spans many
    distinct codes for its complication variants. A concept therefore holds
    one or more :class:`MimicHospConceptEntry` values, combined with OR when
    used to query the repository.

    ``concept_id``/``concept_name``/``domain_id``/``vocabulary_id`` are
    derived summary properties (not stored fields) that deliberately mirror
    :class:`infrastructure.omop.concepts.OMOPConcept`'s attribute names, so
    backend-agnostic cohort code (evidence rendering, presentation views)
    that duck-types against those attributes works unchanged regardless of
    which backend resolved the term.
    """

    entries: tuple[MimicHospConceptEntry, ...]
    domain_id: str  # Condition | Procedure | Measurement | Drug
    resolution_source: str = "exact"
    original_term: str | None = None

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a concept must have at least one entry")

    @property
    def concept_id(self) -> str:
        return "+".join(entry.code for entry in self.entries)

    @property
    def concept_name(self) -> str:
        if len(self.entries) == 1:
            return self.entries[0].display_name
        names = ", ".join(entry.display_name for entry in self.entries)
        return f"{len(self.entries)} codes ({names})"

    @property
    def vocabulary_id(self) -> str:
        vocabularies = sorted({entry.vocabulary_id for entry in self.entries})
        return "+".join(vocabularies)

    @property
    def condition_codes(self) -> tuple[tuple[str, int], ...]:
        """(icd_code, icd_version) pairs, OR-combined, for condition concepts."""
        return tuple(
            (entry.code, 9 if entry.vocabulary_id == "ICD9CM" else 10)
            for entry in self.entries
        )


    @property
    def procedure_codes(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (entry.code, 9 if entry.vocabulary_id == "ICD9PROC" else 10)
            for entry in self.entries
        )

    @property
    def lab_itemids(self) -> tuple[int, ...]:
        return tuple(int(entry.code) for entry in self.entries)
    @property
    def medication_names(self) -> tuple[str, ...]:
        """Drug names, OR-combined, for medication concepts."""
        return tuple(entry.code for entry in self.entries)


class MimicHospConceptResolver:
    """Resolves cohort terms against raw MIMIC-IV hosp tables.

    Resolution order for a term:

    1. Exact title/name match, searched across *both* ICD-9 and ICD-10 (no
       version filter) for conditions, or exact drug name for medications.
       A single exact match is accepted automatically.
    2. Otherwise, a token-based search (normalized for common ICD-9-CM vs.
       ICD-10-CM phrasing differences, e.g. "type II" vs. "type 2", and
       ranked by an IDF-weighted token-overlap score) produces a candidate
       list. The human — or an LLM, if configured — may select *one or more*
       candidates. Multiple selections are combined into a single OR-matched
       concept (see :class:`MimicHospConcept`).
    """

    def __init__(
        self,
        clinical: DuckDBConnectionFactory,
        exception_resolver: SemanticResolver | None = None,
        resolution_store: ResolutionStore | None = None,
        *,
        scope_id: str = "session",
        terminology_expander: TerminologyExpander | None = None,
    ) -> None:
        self._clinical = clinical
        self._exception_resolver = exception_resolver
        self._resolution_store = resolution_store or InMemoryResolutionStore()
        self._scope_id = scope_id
        self._terminology_expander = terminology_expander
        if not clinical.table_exists("d_icd_diagnoses"):
            raise ValueError(
                "clinical database does not contain a d_icd_diagnoses table"
            )
        if not clinical.table_exists("d_icd_diagnoses"):
            raise ValueError("clinical database does not contain a d_icd_diagnoses table")
        if not clinical.table_exists("prescriptions"):
            raise ValueError("clinical database does not contain a prescriptions table")
        self._d_icd_diagnoses = clinical.qualified_table_name("d_icd_diagnoses")
        self._d_icd_procedures = (
            clinical.qualified_table_name("d_icd_procedures")
            if clinical.table_exists("d_icd_procedures") else None
        )
        self._d_labitems = (
            clinical.qualified_table_name("d_labitems")
            if clinical.table_exists("d_labitems") else None
        )
        self._prescriptions = clinical.qualified_table_name("prescriptions")
        self._prescription_columns = {
            str(row[0]).casefold()
            for row in clinical.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE lower(table_name) = 'prescriptions'"""
            )
        }
        self._condition_index: _TokenIndex | None = None
        self._procedure_index: _TokenIndex | None = None
        self._lab_index: _TokenIndex | None = None
        self._medication_index: _TokenIndex | None = None

    @property
    def resolution_store(self) -> ResolutionStore:
        return self._resolution_store

    def resolve_condition(
        self, term: str, *, on_progress: ProgressCallback | None = None,
    ) -> MimicHospConcept:
        return self._resolve(term, domain="condition", on_progress=on_progress)

    def resolve_procedure(
        self, term: str, *, on_progress: ProgressCallback | None = None,
    ) -> MimicHospConcept:
        if self._d_icd_procedures is None:
            raise LookupError("d_icd_procedures is unavailable")
        return self._resolve(term, domain="procedure", on_progress=on_progress)

    def resolve_lab(
        self, term: str, *, on_progress: ProgressCallback | None = None,
    ) -> MimicHospConcept:
        if self._d_labitems is None:
            raise LookupError("d_labitems is unavailable")
        return self._resolve(term, domain="lab", on_progress=on_progress)

    def resolve_medication(
        self, term: str, *, on_progress: ProgressCallback | None = None,
    ) -> MimicHospConcept:
        return self._resolve(term, domain="medication", on_progress=on_progress)

    def _resolve(
        self, term: str, *, domain: str, on_progress: ProgressCallback | None = None,
    ) -> MimicHospConcept:
        cleaned = term.strip()
        if not cleaned:
            raise ValueError("concept term cannot be empty")
        key = self._key(cleaned, domain)
        record = self._resolution_store.get(key)
        if record is not None:
            return _concept_from_payload(record.option_payload, record.source, cleaned)

        exact = self._exact_matches(cleaned, domain)
        if len(exact) == 1:
            return _with_metadata(exact[0], "exact", cleaned)

        candidates = self._search(cleaned, domain, limit=8)
        if not candidates and self._terminology_expander is not None:
            _emit_progress(on_progress, "terminology_expansion",
                           f"No lexical candidates for {cleaned!r}. Trying medical terminology expansion...")
            candidates = self._search_expansions(
                cleaned, domain, limit=8, on_progress=on_progress,
            )
        if not candidates:
            raise LookupError(f"no {domain} term found for {cleaned!r}")

        request = ResolutionRequest(
            resolver_id="mimic_hosp_concept",
            error_code="no_exact_concept_match",
            message=f"No exact {domain} match was found for {cleaned!r}.",
            recoverable=True,
            response_mode=ResponseMode.SELECT_MANY,
            input_payload={
                "term": cleaned,
                "domain": domain,
                "resolution_key": {
                    "resolver_id": key.resolver_id,
                    "input_fingerprint": key.input_fingerprint,
                    "scope_id": key.scope_id,
                },
            },
            options=tuple(
                ResolutionOption(
                    _candidate_option_id(candidate),
                    _candidate_label(candidate),
                    _entry_payload(candidate),
                )
                for candidate in candidates
            ),
            context={
                "policy": (
                    "The human may select one or more options. Combine "
                    "multiple selections with OR. Never invent an option "
                    "not present in the candidate list."
                )
            },
        )

        if self._exception_resolver is None:
            raise ResolutionRequired(
                request,
                question=self._candidate_question(cleaned, candidates),
                rationale="No semantic resolver is configured.",
            )

        decision = self._exception_resolver.resolve(request)
        if decision.action is ResolutionAction.SELECT_OPTION:
            selected = tuple(
                candidate for candidate in candidates
                if _candidate_option_id(candidate) in decision.selected_option_ids
            )
            if not selected:
                raise LookupError(
                    f"semantic resolution selected no known option for {cleaned!r}"
                )
            concept = _combine(selected, "llm", cleaned)
            self._store_selection(key, concept, "llm", decision.rationale)
            return concept
        if decision.action is ResolutionAction.ASK_HUMAN:
            raise ResolutionRequired(
                request,
                question=decision.question or self._candidate_question(cleaned, candidates),
                rationale=decision.rationale,
            )
        raise LookupError(
            f"semantic resolution failed for {cleaned!r}: {decision.rationale}"
        )

    def record_human_selection(
        self,
        *,
        pending_context: dict[str, Any],
        options: tuple[ResolutionOption, ...],
        rationale: str,
    ) -> None:
        if not options:
            raise ValueError("at least one selected option is required")
        key_payload = pending_context["input_payload"]["resolution_key"]
        key = ResolutionKey(
            key_payload["resolver_id"],
            key_payload["input_fingerprint"],
            key_payload.get("scope_id", self._scope_id),
        )
        combined_payload = _combine_payloads([option.payload for option in options])
        combined_label = ", ".join(option.label for option in options)
        self._resolution_store.put(ResolutionRecord(
            key=key,
            option_id="+".join(option.option_id for option in options),
            option_label=combined_label,
            option_payload=combined_payload,
            source="human",
            rationale=rationale,
        ))

    def _exact_matches(self, term: str, domain: str) -> tuple[MimicHospConcept, ...]:
        if domain == "condition":
            rows = self._clinical.execute(
                f"""SELECT icd_code, icd_version, long_title FROM {self._d_icd_diagnoses}
                WHERE lower(long_title) = lower(?) ORDER BY icd_version, icd_code LIMIT 3""", [term]
            )
            return tuple(_condition_from_row(row) for row in rows)
        if domain == "procedure":
            rows = self._clinical.execute(
                f"""SELECT icd_code, icd_version, long_title FROM {self._d_icd_procedures}
                WHERE lower(long_title) = lower(?) ORDER BY icd_version, icd_code LIMIT 3""", [term]
            )
            return tuple(_procedure_from_row(row) for row in rows)
        if domain == "lab":
            rows = self._clinical.execute(
                f"""SELECT itemid, label, fluid, category FROM {self._d_labitems}
                WHERE lower(label) = lower(?) ORDER BY itemid LIMIT 8""", [term]
            )
            return tuple(_lab_from_row(row) for row in rows)
        rows = self._clinical.execute(
            f"""
            SELECT {self._medication_projection()}
            FROM {self._prescriptions}
            WHERE lower(drug) = lower(?)
            GROUP BY lower(drug)
            LIMIT 3
            """,
            [term],
        )
        return tuple(_medication_from_row(row) for row in rows)

    def _search(
        self, term: str, domain: str, *, limit: int = 8,
    ) -> tuple[MimicHospConcept, ...]:
        index = self._get_index(domain)
        results = index.search(term, limit=limit)
        if domain == "condition":
            return tuple(_condition_from_row(row) for row, _score in results)
        if domain == "procedure":
            return tuple(_procedure_from_row(row) for row, _score in results)
        if domain == "lab":
            return tuple(_lab_from_row(row) for row, _score in results)
        return tuple(_medication_from_row(row) for row, _score in results)

    def _search_expansions(
        self, term: str, domain: str, *, limit: int,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[MimicHospConcept, ...]:
        try:
            alternatives = self._terminology_expander.expand(  # type: ignore[union-attr]
                term, domain=domain, on_progress=on_progress,
            )
        except Exception as exc:
            _emit_progress(
                on_progress,
                "terminology_expansion_failed",
                f"Medical terminology expansion failed for {term!r}: {exc}",
            )
            raise LookupError(
                f"medical terminology expansion failed for {term!r}: {exc}"
            ) from exc
        _emit_progress(
            on_progress,
            "terminology_expansion",
            "Expanded " + repr(term) + " to: " + (", ".join(alternatives) or "(no alternatives)"),
        )
        by_key: dict[tuple[str, str, str], MimicHospConcept] = {}
        original_key = " ".join(term.casefold().split())
        for alternative in alternatives:
            if " ".join(alternative.casefold().split()) == original_key:
                continue
            exact = self._exact_matches(alternative, domain)
            found = exact if exact else self._search(alternative, domain, limit=limit)
            for concept in found:
                entry = concept.entries[0]
                by_key.setdefault(
                    (entry.vocabulary_id, entry.code, entry.display_identifier or ""), concept
                )
                if len(by_key) >= limit:
                    break
            if len(by_key) >= limit:
                break
        _emit_progress(
            on_progress,
            "terminology_expansion",
            f"Recovered {len(by_key)} {domain} candidate(s) after terminology expansion.",
        )
        return tuple(by_key.values())

    def _medication_projection(self) -> str:
        def value(column: str, *, zero_is_null: bool = False) -> str:
            if column.casefold() not in self._prescription_columns:
                return "NULL"
            rendered = f"TRY_CAST({column} AS VARCHAR)"
            if zero_is_null:
                return f"min(NULLIF({rendered}, '0'))"
            return f"min(NULLIF({rendered}, ''))"

        return ", ".join((
            "min(drug)",
            value("ndc", zero_is_null=True),
            value("gsn"),
            value("formulary_drug_cd"),
        ))

    def _get_index(self, domain: str) -> "_TokenIndex":
        if domain == "condition":
            if self._condition_index is None:
                rows = self._clinical.execute(
                    f"SELECT icd_code, icd_version, long_title FROM {self._d_icd_diagnoses}"
                )
                self._condition_index = _TokenIndex(
                    rows, text_column=2,
                )
            return self._condition_index
        if domain == "procedure":
            if self._procedure_index is None:
                rows = self._clinical.execute(
                    f"SELECT icd_code, icd_version, long_title FROM {self._d_icd_procedures}"
                )
                self._procedure_index = _TokenIndex(rows, text_column=2)
            return self._procedure_index
        if domain == "lab":
            if self._lab_index is None:
                rows = self._clinical.execute(
                    f"SELECT itemid, label, fluid, category FROM {self._d_labitems}"
                )
                self._lab_index = _TokenIndex(rows, text_column=1)
            return self._lab_index
        if self._medication_index is None:
            rows = self._clinical.execute(
                f"""
                SELECT {self._medication_projection()}
                FROM {self._prescriptions}
                WHERE drug IS NOT NULL
                GROUP BY lower(drug)
                """
            )
            self._medication_index = _TokenIndex(rows, text_column=0)
        return self._medication_index

    def _key(self, term: str, domain: str) -> ResolutionKey:
        return ResolutionKey(
            "mimic_hosp_concept",
            f"{domain}:{_normalize_key(term)}",
            self._scope_id,
        )

    def _store_selection(
        self,
        key: ResolutionKey,
        concept: MimicHospConcept,
        source: str,
        rationale: str,
    ) -> None:
        self._resolution_store.put(ResolutionRecord(
            key,
            concept.concept_id,
            concept.concept_name,
            _concept_payload(concept),
            source,
            rationale,
        ))

    @staticmethod
    def _candidate_question(term: str, candidates: tuple[MimicHospConcept, ...]) -> str:
        choices = ", ".join(
            f"{index}. {_candidate_label(candidate)}"
            for index, candidate in enumerate(candidates, 1)
        )
        return (
            f"I found several matches for {term!r}: {choices}. "
            "Which one(s) apply? You may pick more than one."
        )


class _TokenIndex:
    """A small in-process inverted index with IDF-weighted token overlap.

    Built once per resolver instance (paid once per session, not per
    lookup) and reused for every ambiguous/no-exact-match search in that
    session. This is a dependency-free stand-in for a real full-text-search
    engine (e.g. DuckDB's ``fts`` extension): no network fetch, no external
    index format, safe to run in network-restricted deployments. It does
    not attempt clinical synonym expansion (e.g. "MI" <-> "myocardial
    infarction") -- only surface-form tokenization and rare-token weighting.
    """

    def __init__(self, rows: list[tuple[Any, ...]], *, text_column: int) -> None:
        self._rows = rows
        self._row_tokens: list[frozenset[str]] = []
        self._postings: dict[str, set[int]] = defaultdict(set)
        for index, row in enumerate(rows):
            tokens = frozenset(_tokenize(str(row[text_column])))
            self._row_tokens.append(tokens)
            for token in tokens:
                self._postings[token].add(index)
        total_docs = max(len(rows), 1)
        self._idf: dict[str, float] = {
            token: math.log(1 + total_docs / len(matches))
            for token, matches in self._postings.items()
        }

    def search(
        self, term: str, *, limit: int,
    ) -> list[tuple[tuple[Any, ...], float]]:
        query_tokens = _tokenize(term)
        if not query_tokens:
            return []
        candidate_indices: set[int] = set()
        for token in query_tokens:
            candidate_indices |= self._postings.get(token, set())
        if not candidate_indices:
            return []

        scored: list[tuple[int, float]] = []
        for row_index in candidate_indices:
            row_tokens = self._row_tokens[row_index]
            overlap = [token for token in query_tokens if token in row_tokens]
            if not overlap:
                continue
            score = sum(self._idf.get(token, 0.0) for token in overlap)
            # Prefer more concise titles among equally-scored candidates.
            score -= 0.001 * len(row_tokens)
            scored.append((row_index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            (self._rows[row_index], score)
            for row_index, score in scored[:limit]
        ]


_ROMAN_NUMERAL_PATTERNS = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\btype\s+iii\b", "type 3"),
        (r"\btype\s+ii\b", "type 2"),
        (r"\btype\s+i\b", "type 1"),
    )
)

_STOPWORDS = frozenset({
    "with", "without", "of", "the", "in", "and", "or", "not", "other",
    "unspecified", "specified", "due", "to", "a", "an", "as", "for", "on",
})


def _fold_roman_numerals(text: str) -> str:
    result = text
    for pattern, replacement in _ROMAN_NUMERAL_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _tokenize(text: str) -> list[str]:
    folded = _fold_roman_numerals(text.casefold())
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded)
    return [token for token in cleaned.split() if token not in _STOPWORDS]


def _candidate_label(concept: MimicHospConcept) -> str:
    entry = concept.entries[0]
    if entry.vocabulary_id in {"ICD9CM", "ICD10CM", "ICD9PROC", "ICD10PCS"}:
        return f"{entry.display_name} ({entry.code})"
    if entry.vocabulary_id == "d_labitems":
        return f"{entry.display_name} (itemid: {entry.code})"
    if entry.vocabulary_id == "prescriptions":
        suffix = f" ({entry.display_identifier})" if entry.display_identifier else ""
        return f"{entry.display_name}{suffix}"
    return entry.display_name


def _candidate_option_id(concept: MimicHospConcept) -> str:
    entry = concept.entries[0]
    if entry.vocabulary_id == "prescriptions" and entry.display_identifier:
        return f"prescriptions:{entry.code}:{entry.display_identifier}"
    return concept.concept_id


def _entry_payload(concept: MimicHospConcept) -> dict[str, Any]:
    entry = concept.entries[0]
    return {
        "code": entry.code,
        "display_name": entry.display_name,
        "vocabulary_id": entry.vocabulary_id,
        "display_identifier": entry.display_identifier,
        "domain_id": concept.domain_id,
    }


def _combine_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "domain_id": payloads[0]["domain_id"],
        "entries": [
            {
                "code": payload["code"],
                "display_name": payload["display_name"],
                "vocabulary_id": payload["vocabulary_id"],
                "display_identifier": payload.get("display_identifier"),
            }
            for payload in payloads
        ],
    }


def _concept_payload(concept: MimicHospConcept) -> dict[str, Any]:
    return {
        "domain_id": concept.domain_id,
        "entries": [
            {
                "code": entry.code,
                "display_name": entry.display_name,
                "vocabulary_id": entry.vocabulary_id,
                "display_identifier": entry.display_identifier,
            }
            for entry in concept.entries
        ],
    }


def _combine(
    candidates: tuple[MimicHospConcept, ...], source: str, original_term: str,
) -> MimicHospConcept:
    entries = tuple(candidate.entries[0] for candidate in candidates)
    return MimicHospConcept(
        entries=entries,
        domain_id=candidates[0].domain_id,
        resolution_source=source,
        original_term=original_term,
    )


def _condition_from_row(row: tuple[Any, ...]) -> MimicHospConcept:
    icd_code, icd_version, long_title = str(row[0]), int(row[1]), str(row[2])
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=icd_code,
            display_name=long_title,
            vocabulary_id="ICD9CM" if icd_version == 9 else "ICD10CM",
        ),),
        domain_id="Condition",
    )


def _procedure_from_row(row: tuple[Any, ...]) -> MimicHospConcept:
    code, version, title = str(row[0]), int(row[1]), str(row[2])
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=code, display_name=title,
            vocabulary_id="ICD9PROC" if version == 9 else "ICD10PCS",
        ),),
        domain_id="Procedure",
    )


def _lab_from_row(row: tuple[Any, ...]) -> MimicHospConcept:
    itemid, label, fluid, category = int(row[0]), str(row[1]), str(row[2]), str(row[3])
    display = f"{label} [{fluid}, {category}]"
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=str(itemid), display_name=display, vocabulary_id="d_labitems",
        ),),
        domain_id="Measurement",
    )


def _medication_from_row(row: tuple[Any, ...]) -> MimicHospConcept:
    drug = str(row[0])
    ndc = _clean_identifier(row[1] if len(row) > 1 else None)
    gsn = _clean_identifier(row[2] if len(row) > 2 else None)
    formulary = _clean_identifier(row[3] if len(row) > 3 else None)
    if ndc:
        display_identifier = f"NDC: {ndc}"
    elif gsn:
        display_identifier = f"GSN: {gsn}"
    elif formulary:
        display_identifier = f"Formulary: {formulary}"
    else:
        display_identifier = None
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=drug, display_name=drug, vocabulary_id="prescriptions",
            display_identifier=display_identifier,
        ),),
        domain_id="Drug",
    )


def _concept_from_payload(
    payload: dict[str, Any], source: str, original_term: str,
) -> MimicHospConcept:
    entries = tuple(
        MimicHospConceptEntry(
            code=str(item["code"]),
            display_name=str(item["display_name"]),
            vocabulary_id=str(item["vocabulary_id"]),
            display_identifier=(
                None if item.get("display_identifier") is None
                else str(item["display_identifier"])
            ),
        )
        for item in payload["entries"]
    )
    return MimicHospConcept(
        entries=entries,
        domain_id=str(payload["domain_id"]),
        resolution_source=source,
        original_term=original_term,
    )


def _with_metadata(
    concept: MimicHospConcept, source: str, original_term: str,
) -> MimicHospConcept:
    return MimicHospConcept(
        concept.entries, concept.domain_id, source, original_term,
    )


def _clean_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"none", "nan", "null", "0"}:
        return None
    return text


def _emit_progress(
    on_progress: ProgressCallback | None, event: str, message: str,
) -> None:
    if on_progress is not None:
        on_progress(event, message)


def _normalize_key(value: str) -> str:
    return " ".join(_tokenize(value))
