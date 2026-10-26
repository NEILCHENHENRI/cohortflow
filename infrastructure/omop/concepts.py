from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from framework.resolution import (
    InMemoryResolutionStore,
    ResolutionAction,
    ResolutionKey,
    ResolutionOption,
    ResolutionRecord,
    ResolutionRequest,
    ResolutionRequired,
    ResolutionStore,
    SemanticResolver,
)
from infrastructure.database.duckdb import DuckDBConnectionFactory


@dataclass(frozen=True, slots=True)
class OMOPConcept:
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str | None = None
    standard_concept: str | None = None
    concept_code: str | None = None
    invalid_reason: str | None = None
    resolution_source: str = "exact"
    original_term: str | None = None


class OMOPConceptResolver:
    def __init__(
        self,
        vocabulary: DuckDBConnectionFactory,
        exception_resolver: SemanticResolver | None = None,
        resolution_store: ResolutionStore | None = None,
        *,
        scope_id: str = "session",
    ) -> None:
        self._vocabulary = vocabulary
        self._exception_resolver = exception_resolver
        self._resolution_store = resolution_store or InMemoryResolutionStore()
        self._scope_id = scope_id
        if not vocabulary.table_exists("concept"):
            raise ValueError("vocabulary database does not contain a concept table")

    @property
    def resolution_store(self) -> ResolutionStore:
        return self._resolution_store

    def resolve_condition(self, term: str) -> OMOPConcept:
        return self.resolve_exact(term, domain_id="Condition", standard_only=True)

    def resolve_medication(self, term: str) -> OMOPConcept:
        return self.resolve_exact(term, domain_id="Drug", standard_only=True)

    def resolve_exact(
        self,
        term: str,
        *,
        domain_id: str | None = None,
        standard_only: bool = True,
    ) -> OMOPConcept:
        cleaned = term.strip()
        if not cleaned:
            raise ValueError("concept term cannot be empty")
        domain = domain_id or "Any"
        key = self._key(cleaned, domain)
        record = self._resolution_store.get(key)
        if record is not None:
            return _with_metadata(
                self.lookup_by_id(int(record.option_id)),
                record.source,
                cleaned,
            )

        rows = self._exact_rows(cleaned, domain_id, standard_only)
        if rows:
            return _with_metadata(_concept_from_row(rows[0]), "exact", cleaned)

        candidates = self.search(
            cleaned,
            domain_id=domain_id,
            limit=5,
            standard_only=standard_only,
        )
        if not candidates:
            raise LookupError(f"no OMOP concept found for {cleaned!r}")

        request = ResolutionRequest(
            resolver_id="omop_concept",
            error_code="no_exact_concept_match",
            message=f"No exact OMOP concept was found for {cleaned!r}.",
            recoverable=True,
            input_payload={
                "term": cleaned,
                "domain_id": domain_id,
                "standard_only": standard_only,
                "resolution_key": {
                    "resolver_id": key.resolver_id,
                    "input_fingerprint": key.input_fingerprint,
                    "scope_id": key.scope_id,
                },
            },
            options=tuple(
                ResolutionOption(
                    str(candidate.concept_id),
                    candidate.concept_name,
                    {
                        "concept_id": candidate.concept_id,
                        "domain_id": candidate.domain_id,
                        "vocabulary_id": candidate.vocabulary_id,
                        "concept_code": candidate.concept_code,
                        "standard_concept": candidate.standard_concept,
                    },
                )
                for candidate in candidates
            ),
            context={
                "policy": (
                    "Prefer the general concept for a general phrase. "
                    "never infer an unmentioned complication."
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
            selected = next(
                item for item in candidates
                if str(item.concept_id) == decision.option_id
            )
            self._store_selection(key, selected, "llm", decision.rationale)
            return _with_metadata(selected, "llm", cleaned)
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
        if len(options) != 1:
            raise ValueError(
                "OMOP concepts are canonical and singular. Exactly one "
                f"selection is required, got {len(options)}"
            )
        selected = options[0]
        key_payload = pending_context["input_payload"]["resolution_key"]
        key = ResolutionKey(
            key_payload["resolver_id"],
            key_payload["input_fingerprint"],
            key_payload.get("scope_id", self._scope_id),
        )
        self._resolution_store.put(ResolutionRecord(
            key=key,
            option_id=selected.option_id,
            option_label=selected.label,
            option_payload=selected.payload,
            source="human",
            rationale=rationale,
        ))

    def lookup_by_id(self, concept_id: int) -> OMOPConcept:
        rows = self._vocabulary.execute(
            """
            SELECT TRY_CAST(concept_id AS BIGINT), concept_name, domain_id,
                   vocabulary_id, concept_class_id, standard_concept,
                   concept_code, invalid_reason
            FROM concept
            WHERE TRY_CAST(concept_id AS BIGINT) = ?
            LIMIT 1
            """,
            [concept_id],
        )
        if not rows:
            raise LookupError(f"unknown OMOP concept ID: {concept_id}")
        return _concept_from_row(rows[0])

    def search(
        self,
        term: str,
        *,
        domain_id: str | None = None,
        limit: int = 10,
        standard_only: bool = True,
    ) -> tuple[OMOPConcept, ...]:
        cleaned = term.strip()
        if not cleaned:
            return ()
        clauses = ["lower(concept_name) LIKE lower(?)", "invalid_reason IS NULL"]
        params: list[Any] = [f"%{cleaned}%"]
        if domain_id is not None:
            clauses.append("domain_id = ?")
            params.append(domain_id)
        if standard_only:
            clauses.append("standard_concept = 'S'")
        rows = self._vocabulary.execute(
            f"""
            SELECT TRY_CAST(concept_id AS BIGINT), concept_name, domain_id,
                   vocabulary_id, concept_class_id, standard_concept,
                   concept_code, invalid_reason
            FROM concept
            WHERE {' AND '.join(clauses)}
            ORDER BY
                CASE WHEN lower(concept_name) = lower(?) THEN 0
                     WHEN lower(concept_name) LIKE lower(?) THEN 1 ELSE 2 END,
                length(concept_name), concept_name
            LIMIT ?
            """,
            [*params, cleaned, f"{cleaned}%", limit],
        )
        return tuple(_concept_from_row(row) for row in rows)

    def _exact_rows(self, term: str, domain_id: str | None, standard_only: bool):
        clauses = ["lower(concept_name) = lower(?)", "invalid_reason IS NULL"]
        params: list[Any] = [term]
        if domain_id is not None:
            clauses.append("domain_id = ?")
            params.append(domain_id)
        if standard_only:
            clauses.append("standard_concept = 'S'")
        return self._vocabulary.execute(
            f"""
            SELECT TRY_CAST(concept_id AS BIGINT), concept_name, domain_id,
                   vocabulary_id, concept_class_id, standard_concept,
                   concept_code, invalid_reason
            FROM concept
            WHERE {' AND '.join(clauses)}
            ORDER BY TRY_CAST(concept_id AS BIGINT)
            LIMIT 2
            """,
            params,
        )

    def _key(self, term: str, domain: str) -> ResolutionKey:
        return ResolutionKey(
            "omop_concept",
            f"{domain}:{_normalize(term)}",
            self._scope_id,
        )

    def _store_selection(
        self,
        key: ResolutionKey,
        concept: OMOPConcept,
        source: str,
        rationale: str,
    ) -> None:
        self._resolution_store.put(ResolutionRecord(
            key,
            str(concept.concept_id),
            concept.concept_name,
            {"concept_id": concept.concept_id, "domain_id": concept.domain_id},
            source,
            rationale,
        ))

    @staticmethod
    def _candidate_question(term: str, candidates: tuple[OMOPConcept, ...]) -> str:
        choices = ", ".join(
            f"{index}. {candidate.concept_name}"
            for index, candidate in enumerate(candidates, 1)
        )
        return f"I found several OMOP concepts for {term!r}: {choices}. Which one?"


def _concept_from_row(row: tuple[Any, ...]) -> OMOPConcept:
    return OMOPConcept(
        int(row[0]), str(row[1]), str(row[2]), str(row[3]),
        _optional(row[4]), _optional(row[5]), _optional(row[6]), _optional(row[7]),
    )


def _with_metadata(concept: OMOPConcept, source: str, original_term: str) -> OMOPConcept:
    return OMOPConcept(
        concept.concept_id, concept.concept_name, concept.domain_id,
        concept.vocabulary_id, concept.concept_class_id,
        concept.standard_concept, concept.concept_code,
        concept.invalid_reason, source, original_term,
    )


def _optional(value: Any) -> str | None:
    return None if value is None else str(value)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())
