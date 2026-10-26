from __future__ import annotations

from framework.parsing import temporary_llm_thinking

from prompts.notes import NOTE_ADJUDICATION_SYSTEM

from dataclasses import dataclass
import hashlib
import heapq
import inspect
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Protocol


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _progress_bar(
    iterable=None, *, total=None, initial: int = 0, desc: str = "", unit: str = "it"
):
    """Best-effort terminal progress without making tqdm a hard dependency."""
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable if iterable is not None else _NullProgress(total=total)
    return tqdm(
        iterable, total=total, initial=max(0, int(initial)),
        desc=desc, unit=unit, dynamic_ncols=True
    )


class _NullProgress:
    def __init__(self, *, total=None):
        self.total = total

    def update(self, amount=1):
        return None

    def close(self):
        return None


from framework.operator import (
    emit_current_operator_progress,
    measure_current_operator_component,
    raise_if_current_operator_boundary_stop_requested,
    raise_if_current_operator_checkpoint_cancelled,
)


@dataclass(frozen=True, slots=True)
class NoteChunk:
    chunk_id: int
    subject_id: int
    hadm_id: int | None
    note_id: str
    section: str
    text: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class NoteMatchDecision:
    subject_id: int
    decision: str
    reason: str
    supporting_chunk_ids: tuple[int, ...] = ()
    supporting_admission_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "decision": self.decision,
            "reason": self.reason,
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "supporting_admission_ids": list(self.supporting_admission_ids),
        }


@dataclass(frozen=True, slots=True)
class NoteRetrievalConfig:
    encoder_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    # Limits apply independently within each patient, never across the cohort.
    sparse_top_k: int = 5000
    dense_top_k: int = 5000
    relevance_threshold: float = 0.3
    adjudication_batch_size: int = 6
    max_chars_per_chunk: int = 1800
    overlap_chars: int = 200
    rerank_batch_size: int = 256
    index_batch_size: int = 1024


class NoteLLM(Protocol):
    think: bool

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        on_progress=None,
        stage_label: str | None = None,
    ) -> dict[str, Any]: ...


class NoteRetriever(Protocol):
    def retrieve(
        self,
        claim: str,
        *,
        candidate_subject_ids: set[int] | None = None,
        on_progress=None,
    ) -> list[NoteChunk]: ...


class HybridNoteMatcher:
    """High-recall note retrieval followed by patient-level LLM adjudication.

    Retrieval is deliberately separated from interpretation. The parser emits a
    claim only. Sparse+dense retrieval finds plausible evidence, the reranker
    reduces noise, and one structured LLM call adjudicates a small batch of
    patients. Thinking is enabled for this final semantic pass by default.
    """

    def __init__(
        self,
        retriever: NoteRetriever,
        llm: NoteLLM,
        *,
        think: bool = True,
        batch_size: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.think = bool(think)
        self.batch_size = max(1, int(batch_size))
        # Exact-result cache is useful for repeated identical calls.  The
        # patient cache below is the correction-aware unit of reuse: an
        # unchanged canonical note requirement can keep prior patient verdicts
        # even when surrounding cohort logic changes its candidate set.
        self._match_cache: dict[str, tuple[set[int], set[tuple[int, int]]]] = {}
        self._decision_cache: dict[str, dict[int, NoteMatchDecision]] = {}

    def patients_with_note(
        self,
        payload: dict[str, Any],
        *,
        candidate_subject_ids: set[int] | None = None,
    ) -> set[int]:
        patients, _ = self._match(payload, candidate_subject_ids=candidate_subject_ids)
        return set(patients)

    def admissions_with_note(self, payload: dict[str, Any]) -> set[tuple[int, int]]:
        _, admissions = self._match(payload)
        return set(admissions)

    def _match(
        self,
        payload: dict[str, Any],
        *,
        candidate_subject_ids: set[int] | None = None,
    ) -> tuple[set[int], set[tuple[int, int]]]:
        claim = str(payload.get("claim") or payload.get("query") or "").strip()
        if not claim:
            raise ValueError("note requirement requires a non-empty claim")

        requirement_key = self._requirement_cache_key(payload)
        cache_key = json.dumps(
            {
                "requirement_key": requirement_key,
                "candidate_subject_ids": None if candidate_subject_ids is None else sorted(candidate_subject_ids),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        cached = self._match_cache.get(cache_key)
        if cached is not None:
            return set(cached[0]), set(cached[1])

        # Candidate-scoped calls reuse patient-level LLM verdicts. The built-in
        # retriever uses a fixed cutoff and per-patient candidate limits, so each
        # patient's evidence is independent of the surrounding candidate pool.
        # Custom retrievers must preserve this property to support exact reuse.
        decisions = self._decision_cache.setdefault(requirement_key, {})
        if candidate_subject_ids is not None:
            candidates = {int(pid) for pid in candidate_subject_ids}
            missing = candidates.difference(decisions)
            if missing:
                self._evaluate_patients(
                    claim,
                    payload,
                    missing,
                    decisions,
                )
            result = self._result_from_decisions(decisions, candidates)
            self._match_cache[cache_key] = result
            return set(result[0]), set(result[1])

        # Unscoped calls are retained for admission-level helpers and tests.
        # We can reuse an exact result, but cannot safely infer the complete set
        # of patients with no retained evidence from a corpus-wide retrieval.
        self._evaluate_patients(claim, payload, None, decisions)
        result = self._result_from_decisions(decisions, set(decisions))
        self._match_cache[cache_key] = result
        return set(result[0]), set(result[1])

    def decision_artifacts(
        self,
        payload: dict[str, Any],
        *,
        candidate_subject_ids: set[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Return preserved patient-level note provenance for an evaluated requirement."""
        key = self._requirement_cache_key(payload)
        decisions = self._decision_cache.get(key, {})
        allowed = None if candidate_subject_ids is None else {int(pid) for pid in candidate_subject_ids}
        return {
            pid: decision.to_dict()
            for pid, decision in decisions.items()
            if allowed is None or pid in allowed
        }

    def _requirement_cache_key(self, payload: dict[str, Any]) -> str:
        config = getattr(self.retriever, "config", None)
        retrieval_policy = None
        if config is not None and hasattr(config, "__dataclass_fields__"):
            retrieval_policy = {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
                if name not in {"rerank_batch_size", "index_batch_size"}
            }
        identity = {
            "payload": payload,
            "retrieval_policy": retrieval_policy,
            "think": self.think,
            "adjudication_policy": "all_retained_passages_v1",
        }
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _evaluate_patients(
        self,
        claim: str,
        payload: dict[str, Any],
        candidate_subject_ids: set[int] | None,
        decisions: dict[int, NoteMatchDecision],
    ) -> None:
        emit_current_operator_progress("note_stage", "Preparing note retrieval...")
        emit_current_operator_progress("note_deterministic_started", claim)
        try:
            with measure_current_operator_component("note_retrieval"):
                try:
                    parameters = inspect.signature(self.retriever.retrieve).parameters
                except (TypeError, ValueError):
                    parameters = {}
                if "candidate_subject_ids" in parameters:
                    chunks = self.retriever.retrieve(
                        claim,
                        candidate_subject_ids=candidate_subject_ids,
                    )
                else:
                    chunks = self.retriever.retrieve(claim)
                    if candidate_subject_ids is not None:
                        chunks = [c for c in chunks if c.subject_id in candidate_subject_ids]
        finally:
            emit_current_operator_progress("note_deterministic_finished", claim)
        # A correction that arrived during resumable retrieval owns the next
        # semantic boundary. In Parallel mode this waits for its preanalysis so
        # the deterministic stage can finish when still reusable, but prevents
        # old-state LLM adjudication from starting afterward.
        raise_if_current_operator_boundary_stop_requested()
        emit_current_operator_progress(
            "note_stage",
            f"Retrieved and relevance-filtered {len(chunks)} candidate note chunks.",
        )

        by_patient: dict[int, list[NoteChunk]] = {}
        for chunk in chunks:
            by_patient.setdefault(chunk.subject_id, []).append(chunk)
        for patient_chunks in by_patient.values():
            patient_chunks.sort(key=lambda item: item.score, reverse=True)

        patient_ids = sorted(pid for pid in by_patient if pid not in decisions)
        if not patient_ids:
            return
        emit_current_operator_progress(
            "note_stage",
            f"Grouped evidence for {len(patient_ids)} new candidate patients. Starting LLM adjudication...",
        )
        uncertain = 0
        with temporary_llm_thinking(self.llm, self.think):
            for offset in range(0, len(patient_ids), self.batch_size):
                batch_ids = patient_ids[offset: offset + self.batch_size]
                with measure_current_operator_component("note_llm_adjudication"):
                    result = self.llm.generate_structured(
                        system_prompt=NOTE_ADJUDICATION_SYSTEM,
                        user_prompt=_adjudication_prompt(
                            claim,
                            payload,
                            batch_ids,
                            by_patient,
                        ),
                        schema=_adjudication_schema(batch_ids),
                        stage_label="Note evidence adjudication",
                        on_progress=_operator_progress_adapter,
                    )
                seen: set[int] = set()
                for item in result.get("decisions", []):
                    pid = int(item["subject_id"])
                    if pid not in batch_ids or pid in seen:
                        continue
                    seen.add(pid)
                    label = str(item["decision"]).casefold()
                    supporting_ids = tuple(dict.fromkeys(
                        int(chunk_id) for chunk_id in item.get("supporting_chunk_ids", ())
                    ))
                    selected = [
                        chunk for chunk in by_patient.get(pid, ())
                        if chunk.chunk_id in supporting_ids
                    ]
                    if label == "include" and not selected:
                        selected = by_patient.get(pid, ())
                    admissions = tuple(sorted({
                        int(chunk.hadm_id)
                        for chunk in selected
                        if chunk.hadm_id is not None
                    }))
                    decisions[pid] = NoteMatchDecision(
                        subject_id=pid,
                        decision=label,
                        reason=str(item.get("reason") or "").strip(),
                        supporting_chunk_ids=supporting_ids,
                        supporting_admission_ids=admissions,
                    )
                    if label == "uncertain":
                        uncertain += 1
                # Structured generation should cover every patient in the batch.
                # If a provider violates that contract, do not silently mark the
                # patient as included/excluded. Preserve an explicit uncertain
                # state so it can be audited.
                for pid in set(batch_ids).difference(seen):
                    decisions[pid] = NoteMatchDecision(
                        subject_id=pid,
                        decision="uncertain",
                        reason="LLM adjudication returned no decision for this patient.",
                    )
                    uncertain += 1
                emit_current_operator_progress(
                    "note_stage",
                    f"Adjudicated {min(offset + len(batch_ids), len(patient_ids))}/{len(patient_ids)} new candidate patients.",
                )
        included_count = sum(1 for pid in patient_ids if decisions[pid].decision == "include")
        emit_current_operator_progress(
            "note_stage",
            f"Note matching update complete: {included_count} newly included, {uncertain} uncertain.",
        )

    @staticmethod
    def _result_from_decisions(
        decisions: dict[int, NoteMatchDecision],
        patient_ids: set[int],
    ) -> tuple[set[int], set[tuple[int, int]]]:
        included: set[int] = set()
        admissions: set[tuple[int, int]] = set()
        for pid in patient_ids:
            decision = decisions.get(pid)
            if decision is None or decision.decision != "include":
                continue
            included.add(pid)
            admissions.update((pid, hadm_id) for hadm_id in decision.supporting_admission_ids)
        return included, admissions


class BGEHybridNoteRetriever:
    """Persistent note index with patient-scoped BM25 and BGE retrieval plus cross-encoder reranking.

    Index construction is lazy and intentionally expensive. It occurs only when
    a note criterion is actually confirmed by the user. The clinical DuckDB is
    read-only. All derived note artifacts live under ``cache_directory``.
    """

    def __init__(
        self,
        clinical,
        *,
        cache_directory: str | Path,
        config: NoteRetrievalConfig | None = None,
    ) -> None:
        self.clinical = clinical
        self.config = config or NoteRetrievalConfig()
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._discharge = clinical.qualified_table_name("discharge")
        self.sqlite_path = self.cache_directory / "note_index.sqlite3"
        self.embedding_path = self.cache_directory / "bge_embeddings.npy"
        self.meta_path = self.cache_directory / "meta.json"
        self.build_state_path = self.cache_directory / "note_index_build.json"
        self.rerank_cache_path = self.cache_directory / "note_rerank_cache.sqlite3"
        self._query_embeddings: dict[tuple[str, str], Any] = {}
        self._encoder = None
        self._reranker = None

    def retrieve(self, claim: str, *, candidate_subject_ids: set[int] | None = None,
                 on_progress=None) -> list[NoteChunk]:
        del on_progress
        self._ensure_index()
        if candidate_subject_ids is None:
            with sqlite3.connect(self.sqlite_path) as con:
                patients = {int(row[0]) for row in con.execute(
                    "SELECT DISTINCT subject_id FROM chunks"
                )}
        else:
            patients = {int(pid) for pid in candidate_subject_ids}
        sparse_limit = int(self.config.sparse_top_k)
        dense_limit = int(self.config.dense_top_k)
        if sparse_limit < 1 or dense_limit < 1:
            raise ValueError("note retrieval candidate limits must be positive")
        candidate_ids: set[int] = set()
        # Each patient's sparse and dense candidates are selected independently.
        # Changing the surrounding cohort cannot crowd out existing evidence.
        for pid in sorted(patients):
            allowed = self._candidate_chunk_ids({pid})
            if not allowed:
                continue
            with measure_current_operator_component("note_bm25"):
                sparse = self._sparse_search(
                    claim, min(sparse_limit, len(allowed)), allowed_chunk_ids=allowed
                )
            with measure_current_operator_component("note_dense_retrieval"):
                dense = self._dense_search(
                    claim, min(dense_limit, len(allowed)), allowed_chunk_ids=allowed
                )
            candidate_ids.update(cid for cid, _ in sparse)
            candidate_ids.update(cid for cid, _ in dense)
        if not candidate_ids:
            return []
        with measure_current_operator_component("note_rerank"):
            return self._rerank(claim, self._fetch_chunks(sorted(candidate_ids)))

    def _ensure_index(self) -> None:
        if self._cache_is_current():
            return
        emit_current_operator_progress(
            "note_stage",
            "Building discharge-note chunk index. This first run can take a long time. Later note queries reuse it.",
        )
        self._build_chunk_index()
        self._build_dense_embeddings()
        fingerprint = self._source_fingerprint()
        self.meta_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "encoder_model": self.config.encoder_model,
            "max_chars_per_chunk": int(self.config.max_chars_per_chunk),
            "overlap_chars": int(self.config.overlap_chars),
        }, indent=2))
        emit_current_operator_progress("note_stage", "Note index build complete.")

    def _cache_is_current(self) -> bool:
        if not (self.sqlite_path.exists() and self.embedding_path.exists() and self.meta_path.exists()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except Exception:
            return False
        # Older caches predate explicit chunk-policy metadata. They were built
        # with these same defaults, so accept them without forcing a million-
        # chunk rebuild. Non-default policies still invalidate safely.
        cached_max_chars = meta.get("max_chars_per_chunk", 1800)
        cached_overlap = meta.get("overlap_chars", 200)
        return (
            meta.get("fingerprint") == self._source_fingerprint()
            and meta.get("encoder_model") == self.config.encoder_model
            and cached_max_chars == int(self.config.max_chars_per_chunk)
            and cached_overlap == int(self.config.overlap_chars)
        )

    def _source_fingerprint(self) -> str:
        rows = self.clinical.execute(
            f"SELECT COUNT(*), COALESCE(MAX(CAST(storetime AS VARCHAR)), '') FROM {self._discharge}"
        )
        raw = repr(rows[0] if rows else (0, ""))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _index_build_identity(self) -> dict[str, Any]:
        # Chunk identity is corpus/text-splitting specific. Encoder choice does
        # not invalidate already chunked/BM25-indexed notes.
        return {
            "fingerprint": self._source_fingerprint(),
            "max_chars_per_chunk": int(self.config.max_chars_per_chunk),
            "overlap_chars": int(self.config.overlap_chars),
        }

    def _read_build_state(self) -> dict[str, Any] | None:
        if not self.build_state_path.exists():
            return None
        try:
            state = json.loads(self.build_state_path.read_text())
        except Exception:
            return None
        if state.get("identity") != self._index_build_identity():
            return None
        return state

    def _write_build_state(self, **updates: Any) -> dict[str, Any]:
        # Do not re-query the source fingerprint after every batch. The identity
        # was validated when this build/resume invocation began.
        try:
            state = json.loads(self.build_state_path.read_text())
        except Exception:
            state = {
                "identity": self._index_build_identity(),
                "stage": "chunking",
                "processed_notes": 0,
                "next_chunk_id": 1,
                "embedded_chunks": 0,
                "encoder_model": self.config.encoder_model,
            }
        state.update(updates)
        tmp = self.build_state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        tmp.replace(self.build_state_path)
        return state

    def _reset_partial_index(self) -> None:
        for path in (
            self.sqlite_path, self.embedding_path, self.build_state_path,
            Path(str(self.sqlite_path) + "-wal"), Path(str(self.sqlite_path) + "-shm"),
        ):
            if path.exists():
                path.unlink()

    def _build_chunk_index(self) -> None:
        state = self._read_build_state()
        if state is None or not self.sqlite_path.exists():
            self._reset_partial_index()
            connection = sqlite3.connect(self.sqlite_path)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute(
                    "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL, hadm_id INTEGER, note_id TEXT NOT NULL, section TEXT NOT NULL, text TEXT NOT NULL)"
                )
                try:
                    connection.execute(
                        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='chunk_id', tokenize='unicode61')"
                    )
                except sqlite3.OperationalError as exc:
                    raise RuntimeError("SQLite FTS5 is required for BM25 note retrieval") from exc
                connection.commit()
            finally:
                connection.close()
            state = self._write_build_state(
                stage="chunking", processed_notes=0, next_chunk_id=1,
                embedded_chunks=0, encoder_model=self.config.encoder_model
            )

        if state.get("stage") in {"embedding", "complete"}:
            return

        note_count = int(self.clinical.execute(
            f"SELECT COUNT(*) FROM {self._discharge} WHERE text IS NOT NULL AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL"
        )[0][0])
        processed_notes = int(state.get("processed_notes", 0))
        next_id = int(state.get("next_chunk_id", 1))
        batch_size = max(1, int(self.config.index_batch_size))
        emit_current_operator_progress(
            "note_stage",
            f"Chunking {note_count:,} discharge summaries (resuming at {processed_notes:,})...",
        )
        connection = sqlite3.connect(self.sqlite_path)
        try:
            with self.clinical.connect() as source:
                cursor = source.execute(
                    f"SELECT note_id, TRY_CAST(subject_id AS BIGINT), TRY_CAST(hadm_id AS BIGINT), text "
                    f"FROM {self._discharge} "
                    "WHERE text IS NOT NULL AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL "
                    f"ORDER BY note_id LIMIT {max(0, note_count - processed_notes)} OFFSET {processed_notes}"
                )
                next_ui_update = ((processed_notes // 25_000) + 1) * 25_000
                bar = _progress_bar(
                    total=note_count, initial=processed_notes,
                    desc="Chunking discharge notes", unit="note"
                )
                try:
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        inserts = []
                        for note_id, subject_id, hadm_id, text in rows:
                            for section, chunk_text in split_discharge_note(
                                str(text),
                                max_chars=self.config.max_chars_per_chunk,
                                overlap_chars=self.config.overlap_chars,
                            ):
                                inserts.append((
                                    next_id, int(subject_id),
                                    None if hadm_id is None else int(hadm_id),
                                    str(note_id), section, chunk_text,
                                ))
                                next_id += 1
                        connection.executemany(
                            "INSERT INTO chunks(chunk_id,subject_id,hadm_id,note_id,section,text) VALUES(?,?,?,?,?,?)",
                            inserts,
                        )
                        connection.commit()
                        processed_notes += len(rows)
                        self._write_build_state(
                            stage="chunking",
                            processed_notes=processed_notes,
                            next_chunk_id=next_id,
                            embedded_chunks=0,
                        )
                        bar.update(len(rows))
                        if processed_notes >= next_ui_update:
                            emit_current_operator_progress(
                                "note_index_progress",
                                f"Chunked {processed_notes:,}/{note_count:,} notes ({next_id - 1:,} chunks).",
                            )
                            next_ui_update += 25_000
                        # Safe interruption point: the SQLite rows and progress
                        # checkpoint for this batch have both been committed.
                        raise_if_current_operator_checkpoint_cancelled()
                finally:
                    bar.close()
            emit_current_operator_progress(
                "note_stage",
                f"Chunking complete: {next_id - 1:,} chunks. Finalizing BM25 index...",
            )
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            connection.commit()
            self._write_build_state(
                stage="embedding",
                processed_notes=processed_notes,
                next_chunk_id=next_id,
                embedded_chunks=int(state.get("embedded_chunks", 0)),
            )
            emit_current_operator_progress(
                "note_stage", "BM25 index finalized. Preparing dense embeddings..."
            )
        finally:
            connection.close()

    def _build_dense_embeddings(self) -> None:
        import numpy as np

        state = self._read_build_state()
        if state is None:
            raise RuntimeError("note index build state is missing")
        if state.get("encoder_model") != self.config.encoder_model:
            if self.embedding_path.exists():
                self.embedding_path.unlink()
            state = self._write_build_state(
                stage="embedding", embedded_chunks=0,
                encoder_model=self.config.encoder_model,
            )
        encoder = self._get_encoder()
        connection = sqlite3.connect(self.sqlite_path)
        try:
            count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if count == 0:
                raise RuntimeError("discharge note index contains no chunks")
            written = min(int(state.get("embedded_chunks", 0)), count)
            emit_current_operator_progress("note_stage", "Loading BGE encoder for note embeddings...")
            sample = connection.execute(
                "SELECT text FROM chunks ORDER BY chunk_id LIMIT 1"
            ).fetchone()[0]
            sample_embedding = encoder.encode(
                [sample], normalize_embeddings=True, show_progress_bar=False
            )
            dim = int(sample_embedding.shape[1])
            if written > 0 and self.embedding_path.exists():
                mmap = np.load(self.embedding_path, mmap_mode="r+")
                if mmap.shape != (count, dim):
                    written = 0
                    mmap = np.lib.format.open_memmap(
                        self.embedding_path, mode="w+", dtype="float16", shape=(count, dim)
                    )
            else:
                written = 0
                mmap = np.lib.format.open_memmap(
                    self.embedding_path, mode="w+", dtype="float16", shape=(count, dim)
                )
            cursor = connection.execute(
                "SELECT chunk_id,text FROM chunks WHERE chunk_id > ? ORDER BY chunk_id",
                (written,),
            )
            emit_current_operator_progress(
                "note_stage",
                f"Embedding {count:,} note chunks with BGE (resuming at {written:,})...",
            )
            next_ui_update = max(2_048, count // 20)
            next_ui_mark = ((written // next_ui_update) + 1) * next_ui_update
            batch_size = max(1, int(self.config.index_batch_size))
            bar = _progress_bar(
                total=count, initial=written, desc="Embedding note chunks", unit="chunk"
            )
            try:
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    texts = [row[1] for row in rows]
                    vectors = encoder.encode(
                        texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32
                    )
                    mmap[written:written + len(rows)] = vectors.astype("float16")
                    written += len(rows)
                    mmap.flush()
                    self._write_build_state(
                        stage="embedding", embedded_chunks=written
                    )
                    bar.update(len(rows))
                    if written >= next_ui_mark:
                        emit_current_operator_progress(
                            "note_index_progress",
                            f"Embedded {written:,}/{count:,} chunks.",
                        )
                        next_ui_mark += next_ui_update
                    raise_if_current_operator_checkpoint_cancelled()
            finally:
                bar.close()
            mmap.flush()
            self._write_build_state(stage="complete", embedded_chunks=written)
            emit_current_operator_progress(
                "note_stage", f"Dense embedding build complete: {written:,} chunks."
            )
        finally:
            connection.close()

    def _sparse_search(
        self,
        claim: str,
        top_k: int,
        *,
        allowed_chunk_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        query = _fts_query(claim)
        if not query:
            return []
        connection = sqlite3.connect(self.sqlite_path)
        try:
            if allowed_chunk_ids is None:
                rows = connection.execute(
                    "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?",
                    (query, int(top_k)),
                ).fetchall()
            else:
                connection.execute(
                    "CREATE TEMP TABLE allowed_chunks(chunk_id INTEGER PRIMARY KEY)"
                )
                connection.executemany(
                    "INSERT INTO allowed_chunks(chunk_id) VALUES(?)",
                    ((int(chunk_id),) for chunk_id in allowed_chunk_ids),
                )
                rows = connection.execute(
                    "SELECT chunks_fts.rowid, bm25(chunks_fts) "
                    "FROM chunks_fts JOIN allowed_chunks ON allowed_chunks.chunk_id = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?",
                    (query, int(top_k)),
                ).fetchall()
            return [(int(rowid), float(-score)) for rowid, score in rows]
        finally:
            connection.close()

    def _dense_search(
        self,
        claim: str,
        top_k: int,
        *,
        allowed_chunk_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        import numpy as np
        query_key = (self.config.encoder_model, claim)
        query = self._query_embeddings.get(query_key)
        if query is None:
            encoder = self._get_encoder()
            query = encoder.encode([claim], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
            self._query_embeddings[query_key] = query
        matrix = np.load(self.embedding_path, mmap_mode="r")
        top: list[tuple[float, int]] = []
        block = 8192
        allowed_by_block: dict[int, list[int]] | None = None
        if allowed_chunk_ids is not None:
            allowed_by_block = {}
            for chunk_id in allowed_chunk_ids:
                zero_based = int(chunk_id) - 1
                allowed_by_block.setdefault(zero_based // block, []).append(
                    zero_based % block
                )
        total_blocks = max(1, (matrix.shape[0] + block - 1) // block)
        emit_current_operator_progress(
            "note_stage", f"Dense retrieval scanning {matrix.shape[0]:,} cached note embeddings..."
        )
        bar = _progress_bar(total=total_blocks, desc="Dense note retrieval", unit="block")
        next_ui_block = max(1, total_blocks // 4)
        completed_blocks = 0
        try:
            for block_index, start in enumerate(range(0, matrix.shape[0], block)):
                if allowed_by_block is None:
                    rows = np.arange(start, min(start + block, matrix.shape[0]))
                else:
                    rows = np.asarray([start + i for i in allowed_by_block.get(block_index, ())], dtype="int64")
                if len(rows):
                    values = np.asarray(matrix[rows], dtype="float32") @ query
                    keep = min(top_k, len(values))
                    indices = np.argpartition(values, -keep)[-keep:]
                    for local_idx in indices:
                        chunk_id = int(rows[local_idx]) + 1
                        score = float(values[local_idx])
                        item = (score, chunk_id)
                        if len(top) < top_k:
                            heapq.heappush(top, item)
                        elif score > top[0][0]:
                            heapq.heapreplace(top, item)
                completed_blocks += 1
                bar.update(1)
                if completed_blocks % next_ui_block == 0 and completed_blocks < total_blocks:
                    emit_current_operator_progress(
                        "note_index_progress",
                        f"Dense retrieval {completed_blocks}/{total_blocks} blocks complete.",
                    )
        finally:
            bar.close()
        return [(chunk_id, score) for score, chunk_id in sorted(top, reverse=True)]

    def _fetch_chunks(self, chunk_ids: list[int]) -> list[NoteChunk]:
        connection = sqlite3.connect(self.sqlite_path)
        try:
            result: list[NoteChunk] = []
            for offset in range(0, len(chunk_ids), 500):
                ids = chunk_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"SELECT chunk_id,subject_id,hadm_id,note_id,section,text FROM chunks WHERE chunk_id IN ({placeholders})",
                    ids,
                ).fetchall()
                result.extend(NoteChunk(int(a), int(b), None if c is None else int(c), str(d), str(e), str(f)) for a,b,c,d,e,f in rows)
            by_id = {chunk.chunk_id: chunk for chunk in result}
            return [by_id[cid] for cid in chunk_ids if cid in by_id]
        finally:
            connection.close()

    def _rerank_cache_key(self, claim: str) -> str:
        try:
            corpus = self._index_build_identity()
        except (AttributeError, TypeError):
            # Standalone/test retrievers without a clinical execute() method do
            # not have a persistent corpus identity. Their cache is local only.
            corpus = "standalone"
        encoded = json.dumps(
            {"claim": claim, "reranker_model": self.config.reranker_model, "corpus": corpus},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_rerank_scores(
        self, requirement_key: str, chunk_ids: list[int]
    ) -> dict[int, float]:
        if not chunk_ids or not self.rerank_cache_path.exists():
            return {}
        connection = sqlite3.connect(self.rerank_cache_path)
        try:
            result: dict[int, float] = {}
            for offset in range(0, len(chunk_ids), 800):
                batch = chunk_ids[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                try:
                    rows = connection.execute(
                        f"SELECT chunk_id, score FROM rerank_scores "
                        f"WHERE requirement_key=? AND chunk_id IN ({placeholders})",
                        (requirement_key, *batch),
                    ).fetchall()
                except sqlite3.OperationalError:
                    return {}
                result.update((int(chunk_id), float(score)) for chunk_id, score in rows)
            return result
        finally:
            connection.close()

    def _store_rerank_scores(
        self, requirement_key: str, scores: list[tuple[int, float]]
    ) -> None:
        if not scores:
            return
        connection = sqlite3.connect(self.rerank_cache_path)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rerank_scores ("
                "requirement_key TEXT NOT NULL, chunk_id INTEGER NOT NULL, "
                "score REAL NOT NULL, PRIMARY KEY(requirement_key, chunk_id))"
            )
            connection.executemany(
                "INSERT OR REPLACE INTO rerank_scores(requirement_key,chunk_id,score) "
                "VALUES(?,?,?)",
                ((requirement_key, int(chunk_id), float(score)) for chunk_id, score in scores),
            )
            connection.commit()
        finally:
            connection.close()

    def _rerank(self, claim: str, chunks: list[NoteChunk]) -> list[NoteChunk]:
        if not chunks:
            return []
        emit_current_operator_progress(
            "note_stage", "Loading BGE reranker and scoring candidate passages..."
        )
        requirement_key = self._rerank_cache_key(claim)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        score_by_id = self._load_rerank_scores(requirement_key, chunk_ids)
        missing = [chunk for chunk in chunks if chunk.chunk_id not in score_by_id]
        batch_size = max(1, int(self.config.rerank_batch_size))
        bar = _progress_bar(
            total=len(chunks), initial=len(chunks) - len(missing),
            desc="BGE relevance scoring", unit="chunk"
        )
        try:
            if missing:
                reranker = self._get_reranker()
                for offset in range(0, len(missing), batch_size):
                    batch = missing[offset:offset + batch_size]
                    pairs = [[claim, chunk.text] for chunk in batch]
                    raw_scores = reranker.predict(pairs, show_progress_bar=False)
                    completed = [
                        (chunk.chunk_id, _sigmoid(float(score)))
                        for chunk, score in zip(batch, raw_scores)
                    ]
                    # Persist before checking for steering, so a stopped batch is
                    # a real resume point and tqdm can restart from this count.
                    self._store_rerank_scores(requirement_key, completed)
                    score_by_id.update(completed)
                    bar.update(len(completed))
                    emit_current_operator_progress(
                        "note_index_progress",
                        f"BGE relevance scoring {len(score_by_id):,}/{len(chunks):,} chunks complete.",
                    )
                    raise_if_current_operator_checkpoint_cancelled()
        finally:
            bar.close()

        ranked = sorted(
            (
                NoteChunk(
                    c.chunk_id, c.subject_id, c.hadm_id, c.note_id, c.section,
                    c.text, float(score_by_id[c.chunk_id]),
                )
                for c in chunks
                if c.chunk_id in score_by_id
            ),
            key=lambda item: (-item.score, item.chunk_id),
        )
        if not ranked:
            return []
        cutoff = float(self.config.relevance_threshold)
        if not 0 <= cutoff <= 1:
            raise ValueError("note relevance threshold must be between 0 and 1")
        retained = [chunk for chunk in ranked if chunk.score >= cutoff]
        emit_current_operator_progress(
            "note_stage",
            f"Relevance cutoff {cutoff:.3f}: retained {len(retained):,}/{len(ranked):,} chunks.",
        )
        return retained

    def _candidate_chunk_ids(
        self,
        candidate_subject_ids: set[int] | None,
    ) -> set[int] | None:
        if candidate_subject_ids is None:
            return None
        if not candidate_subject_ids:
            return set()
        connection = sqlite3.connect(self.sqlite_path)
        try:
            result: set[int] = set()
            ids = sorted(int(value) for value in candidate_subject_ids)
            for offset in range(0, len(ids), 800):
                batch = ids[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT chunk_id FROM chunks WHERE subject_id IN ({placeholders})",
                    batch,
                ).fetchall()
                result.update(int(row[0]) for row in rows)
            return result
        finally:
            connection.close()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError("Note dense retrieval requires sentence-transformers") from exc
            self._encoder = SentenceTransformer(self.config.encoder_model)
        return self._encoder

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError("Note reranking requires sentence-transformers") from exc
            self._reranker = CrossEncoder(self.config.reranker_model)
        return self._reranker




def _adjudication_schema(patient_ids: list[int]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(patient_ids),
                "maxItems": len(patient_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subject_id", "decision", "supporting_chunk_ids", "reason"],
                    "properties": {
                        "subject_id": {"type": "integer", "enum": patient_ids},
                        "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
                        "supporting_chunk_ids": {"type": "array", "items": {"type": "integer"}},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def _adjudication_prompt(claim: str, payload: dict[str, Any], patient_ids: list[int], by_patient: dict[int, list[NoteChunk]]) -> str:
    count_min = int(payload.get("count_min") or 1)
    blocks = []
    for pid in patient_ids:
        evidence = []
        for chunk in by_patient[pid]:
            evidence.append({
                "chunk_id": chunk.chunk_id,
                "hadm_id": chunk.hadm_id,
                "note_id": chunk.note_id,
                "section": chunk.section,
                "text": chunk.text,
            })
        blocks.append({"subject_id": pid, "evidence": evidence})
    return (
        f"CLAIM: {claim}\n"
        f"INCLUSION RULE: include only if the claim is supported in at least {count_min} distinct admission(s).\n"
        "Evaluate every patient below independently.\n\n"
        + json.dumps(blocks, ensure_ascii=False, indent=2)
    )


def _operator_progress_adapter(event_type: str, payload: str) -> None:
    if event_type == "thinking":
        return
    emit_current_operator_progress(f"note_llm_{event_type}", payload)


_HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /&()'\-]{2,80})\s*:\s*$")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")


def split_discharge_note(text: str, *, max_chars: int = 1800, overlap_chars: int = 200) -> Iterable[tuple[str, str]]:
    section = "Unsectioned"
    buffer: list[str] = []

    def flush() -> Iterable[tuple[str, str]]:
        joined = "\n".join(buffer).strip()
        if not joined:
            return ()
        chunks = []
        start = 0
        while start < len(joined):
            end = min(len(joined), start + max_chars)
            if end < len(joined):
                split_at = joined.rfind("\n", start, end)
                if split_at > start + max_chars // 2:
                    end = split_at
            piece = joined[start:end].strip()
            if piece:
                chunks.append((section, piece))
            if end >= len(joined):
                break
            start = max(start + 1, end - overlap_chars)
        return tuple(chunks)

    for raw_line in text.splitlines():
        match = _HEADING.match(raw_line)
        if match:
            yield from flush()
            buffer.clear()
            section = match.group(1).strip()
            continue
        if not raw_line.strip() and buffer and sum(len(line) for line in buffer) >= max_chars:
            yield from flush()
            buffer.clear()
            continue
        buffer.append(raw_line)
    yield from flush()


def _fts_query(text: str) -> str:
    tokens = [match.group(0).casefold() for match in _TOKEN.finditer(text)]
    stop = {"the", "a", "an", "of", "to", "and", "or", "with", "was", "is", "patient", "patients"}
    useful = [token for token in tokens if token not in stop and len(token) > 1]
    # OR favors recall. Semantic dense retrieval covers paraphrases.
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in dict.fromkeys(useful))
