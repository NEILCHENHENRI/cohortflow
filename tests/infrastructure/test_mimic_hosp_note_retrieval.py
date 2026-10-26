from infrastructure.mimic_hosp.note_retrieval import (
    HybridNoteMatcher,
    NoteChunk,
    split_discharge_note,
)


class Retriever:
    def retrieve(self, claim, *, on_progress=None):
        assert claim == "patient was non-adherent to prescribed medication"
        return [
            NoteChunk(1, 10, 100, "n1", "Hospital Course", "She stopped taking Lasix at home.", 0.9),
            NoteChunk(2, 10, 100, "n1", "HPI", "Medication adherence was poor.", 0.8),
            NoteChunk(3, 20, 200, "n2", "Hospital Course", "Patient reports taking all medications.", 0.7),
        ]


class LLM:
    def __init__(self):
        self.think = False
        self.think_seen = []
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        self.think_seen.append(self.think)
        assert "CLAIM: patient was non-adherent" in kwargs["user_prompt"]
        return {
            "decisions": [
                {"subject_id": 10, "decision": "include", "supporting_chunk_ids": [1], "reason": "Explicitly stopped medication."},
                {"subject_id": 20, "decision": "exclude", "supporting_chunk_ids": [3], "reason": "Evidence contradicts claim."},
            ]
        }


def test_patient_level_adjudication_uses_thinking_by_default_and_restores_llm_state():
    llm = LLM()
    matcher = HybridNoteMatcher(Retriever(), llm, think=True, batch_size=6)

    patients = matcher.patients_with_note({
        "claim": "patient was non-adherent to prescribed medication"
    })

    assert patients == {10}
    assert llm.calls == 1
    assert llm.think_seen == [True]
    assert llm.think is False


def test_note_adjudication_thinking_can_be_disabled():
    llm = LLM()
    matcher = HybridNoteMatcher(Retriever(), llm, think=False)
    matcher.patients_with_note({"claim": "patient was non-adherent to prescribed medication"})
    assert llm.think_seen == [False]


def test_discharge_chunking_preserves_section_labels():
    text = """History of Present Illness:\nPatient stopped taking medication.\n\nBrief Hospital Course:\nRestarted therapy and improved.\n"""
    chunks = list(split_discharge_note(text, max_chars=200))
    assert chunks == [
        ("History of Present Illness", "Patient stopped taking medication."),
        ("Brief Hospital Course", "Restarted therapy and improved."),
    ]


def test_note_matcher_exposes_supporting_admissions_without_second_llm_call():
    llm = LLM()
    matcher = HybridNoteMatcher(Retriever(), llm, think=True, batch_size=6)
    payload = {"claim": "patient was non-adherent to prescribed medication"}
    assert matcher.patients_with_note(payload) == {10}
    assert matcher.admissions_with_note(payload) == {(10, 100)}
    assert llm.calls == 1


def test_fixed_relevance_filter_has_no_population_retention_cap(tmp_path):
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever, NoteRetrievalConfig

    class Clinical:
        def qualified_table_name(self, name):
            return name

    class Reranker:
        def predict(self, pairs, show_progress_bar=True):
            # Sigmoid scores are roughly .95, .90, .80, .60, .50.
            return [2.944, 2.197, 1.386, 0.405, 0.0]

    retriever = BGEHybridNoteRetriever(
        Clinical(),
        cache_directory=tmp_path / "note-cache",
        config=NoteRetrievalConfig(relevance_threshold=0.75),
    )
    retriever._reranker = Reranker()
    chunks = [
        NoteChunk(i, i, None, str(i), "section", f"chunk {i}")
        for i in range(1, 6)
    ]

    kept = retriever._rerank("claim", chunks)

    # Every passage above the fixed cutoff survives, with no fraction cap.
    assert [chunk.chunk_id for chunk in kept] == [1, 2, 3]
    assert all(0.0 <= chunk.score <= 1.0 for chunk in kept)

class CandidateAwareRetriever:
    def __init__(self):
        self.calls = []
        self._chunks = {
            10: NoteChunk(10, 10, 100, "n10", "HPI", "non-adherent to medication", 0.9),
            20: NoteChunk(20, 20, 200, "n20", "HPI", "reports medication adherence", 0.8),
            30: NoteChunk(30, 30, 300, "n30", "HPI", "stopped medication at home", 0.85),
        }

    def retrieve(self, claim, *, candidate_subject_ids=None, on_progress=None):
        selected = None if candidate_subject_ids is None else set(candidate_subject_ids)
        self.calls.append((claim, selected))
        ids = set(self._chunks) if selected is None else selected
        # Patient 40 intentionally has no retained evidence.
        return [self._chunks[pid] for pid in sorted(ids) if pid in self._chunks]


class CandidateAwareLLM:
    def __init__(self):
        self.think = False
        self.calls = []

    def generate_structured(self, **kwargs):
        import json
        prompt = kwargs["user_prompt"]
        evidence_json = prompt.split("\n\n", 1)[1]
        patients = json.loads(evidence_json)
        ids = [int(item["subject_id"]) for item in patients]
        self.calls.append(ids)
        decisions = []
        for pid in ids:
            include = pid in {10, 30}
            decisions.append({
                "subject_id": pid,
                "decision": "include" if include else "exclude",
                "supporting_chunk_ids": [pid],
                "reason": f"reason for {pid}",
            })
        return {"decisions": decisions}


def test_note_reuse_candidate_shrink_does_not_repeat_retrieval_or_llm():
    retriever = CandidateAwareRetriever()
    llm = CandidateAwareLLM()
    matcher = HybridNoteMatcher(retriever, llm)
    payload = {"claim": "medication non-adherence"}

    assert matcher.patients_with_note(payload, candidate_subject_ids={10, 20}) == {10}
    assert len(retriever.calls) == 1
    assert len(llm.calls) == 1

    assert matcher.patients_with_note(payload, candidate_subject_ids={10}) == {10}
    assert len(retriever.calls) == 1
    assert len(llm.calls) == 1


def test_note_reuse_candidate_expand_processes_only_new_patients():
    retriever = CandidateAwareRetriever()
    llm = CandidateAwareLLM()
    matcher = HybridNoteMatcher(retriever, llm)
    payload = {"claim": "medication non-adherence"}

    assert matcher.patients_with_note(payload, candidate_subject_ids={10, 20}) == {10}
    assert matcher.patients_with_note(payload, candidate_subject_ids={10, 20, 30, 40}) == {10, 30}

    assert retriever.calls == [
        ("medication non-adherence", {10, 20}),
        ("medication non-adherence", {30, 40}),
    ]
    assert llm.calls == [[10, 20], [30]]


def test_note_requirement_change_invalidates_requirement_specific_reuse():
    retriever = CandidateAwareRetriever()
    llm = CandidateAwareLLM()
    matcher = HybridNoteMatcher(retriever, llm)

    matcher.patients_with_note(
        {"claim": "medication non-adherence"},
        candidate_subject_ids={10, 20},
    )
    matcher.patients_with_note(
        {"claim": "medication non-adherence during admission"},
        candidate_subject_ids={10, 20},
    )

    assert len(retriever.calls) == 2
    assert len(llm.calls) == 2


def test_note_decision_artifacts_preserve_reason_and_supporting_chunks():
    retriever = CandidateAwareRetriever()
    llm = CandidateAwareLLM()
    matcher = HybridNoteMatcher(retriever, llm)
    payload = {"claim": "medication non-adherence"}

    matcher.patients_with_note(payload, candidate_subject_ids={10, 20, 40})
    artifacts = matcher.decision_artifacts(payload, candidate_subject_ids={10, 20, 40})

    assert artifacts[10]["decision"] == "include"
    assert artifacts[10]["reason"] == "reason for 10"
    assert artifacts[10]["supporting_chunk_ids"] == [10]
    assert artifacts[10]["supporting_admission_ids"] == [100]
    # Patient 40 was filtered before LLM adjudication.  That filter outcome is
    # candidate-pool dependent, so it is intentionally not persisted as a
    # reusable patient-level semantic verdict.
    assert 40 not in artifacts


def test_prefilter_rejection_is_not_reused_across_changed_candidate_pool():
    retriever = CandidateAwareRetriever()
    llm = CandidateAwareLLM()
    matcher = HybridNoteMatcher(retriever, llm)
    payload = {"claim": "medication non-adherence"}

    assert matcher.patients_with_note(payload, candidate_subject_ids={40}) == set()
    assert matcher.patients_with_note(payload, candidate_subject_ids={30, 40}) == {30}

    # Custom retriever prefilter misses are reconsidered after a pool change.
    # Only actual LLM verdicts are cached by this matcher.
    assert retriever.calls == [
        ("medication non-adherence", {40}),
        ("medication non-adherence", {30, 40}),
    ]
    assert llm.calls == [[30]]


def test_rerank_checkpoint_persists_completed_batches_and_resumes_tqdm(tmp_path, monkeypatch):
    import threading
    from framework.operator import OperatorContext, operator_context_scope
    import infrastructure.mimic_hosp.note_retrieval as module
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever, NoteRetrievalConfig

    class Clinical:
        def qualified_table_name(self, name):
            return name
        def execute(self, sql):
            return [(5, "2026-01-01")]

    stop = threading.Event()

    class FirstReranker:
        def __init__(self):
            self.calls = []
        def predict(self, pairs, show_progress_bar=False):
            self.calls.append([pair[1] for pair in pairs])
            stop.set()
            return [2.0] * len(pairs)

    config = NoteRetrievalConfig(rerank_batch_size=2, relevance_threshold=0.0)
    first = BGEHybridNoteRetriever(Clinical(), cache_directory=tmp_path, config=config)
    first._reranker = FirstReranker()
    chunks = [NoteChunk(i, i, None, str(i), "s", f"chunk {i}") for i in range(1, 6)]
    context = OperatorContext(
        1, threading.Event(), "note", checkpoint_stop_event=stop
    )
    try:
        with operator_context_scope(context):
            first._rerank("claim", chunks)
    except InterruptedError:
        pass
    else:
        raise AssertionError("expected checkpoint cancellation after first rerank batch")

    key = first._rerank_cache_key("claim")
    assert set(first._load_rerank_scores(key, [1, 2, 3, 4, 5])) == {1, 2}

    seen_initial = []
    real_progress = module._progress_bar
    def capture_progress(*args, **kwargs):
        seen_initial.append(kwargs.get("initial", 0))
        return real_progress(*args, **kwargs)
    monkeypatch.setattr(module, "_progress_bar", capture_progress)

    class SecondReranker:
        def __init__(self):
            self.calls = []
        def predict(self, pairs, show_progress_bar=False):
            self.calls.append([pair[1] for pair in pairs])
            return [1.0] * len(pairs)

    stop.clear()
    second = BGEHybridNoteRetriever(Clinical(), cache_directory=tmp_path, config=config)
    second._reranker = SecondReranker()
    context = OperatorContext(
        1, threading.Event(), "note", checkpoint_stop_event=stop
    )
    with operator_context_scope(context):
        second._rerank("claim", chunks)

    assert seen_initial[-1] == 2
    rescored = [text for batch in second._reranker.calls for text in batch]
    assert rescored == ["chunk 3", "chunk 4", "chunk 5"]


def test_chunk_index_checkpoint_resumes_without_duplicate_notes(tmp_path, monkeypatch):
    import json
    import re
    import infrastructure.mimic_hosp.note_retrieval as module
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever, NoteRetrievalConfig

    rows = [
        ("n1", 1, 11, "HPI:\\none"),
        ("n2", 2, 22, "HPI:\\ntwo"),
        ("n3", 3, 33, "HPI:\\nthree"),
    ]

    class Cursor:
        def __init__(self, values):
            self.values = list(values)
            self.pos = 0
        def fetchmany(self, size):
            batch = self.values[self.pos:self.pos + size]
            self.pos += len(batch)
            return batch

    class Source:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def execute(self, sql):
            match = re.search(r"OFFSET (\d+)", sql)
            offset = int(match.group(1)) if match else 0
            return Cursor(rows[offset:])

    class Clinical:
        def qualified_table_name(self, name):
            return name
        def execute(self, sql):
            if "MAX(CAST(storetime" in sql:
                return [(len(rows), "2026-01-01")]
            if "COUNT(*)" in sql:
                return [(len(rows),)]
            raise AssertionError(sql)
        def connect(self):
            return Source()

    config = NoteRetrievalConfig(index_batch_size=2)
    retriever = BGEHybridNoteRetriever(Clinical(), cache_directory=tmp_path, config=config)
    calls = 0
    def stop_after_first_batch():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("steer")
    monkeypatch.setattr(module, "raise_if_current_operator_checkpoint_cancelled", stop_after_first_batch)

    try:
        retriever._build_chunk_index()
    except InterruptedError:
        pass
    state = json.loads(retriever.build_state_path.read_text())
    assert state["processed_notes"] == 2

    seen_initial = []
    real_progress = module._progress_bar
    def capture_progress(*args, **kwargs):
        seen_initial.append(kwargs.get("initial", 0))
        return real_progress(*args, **kwargs)
    monkeypatch.setattr(module, "_progress_bar", capture_progress)
    monkeypatch.setattr(module, "raise_if_current_operator_checkpoint_cancelled", lambda: None)
    retriever._build_chunk_index()

    assert seen_initial[-1] == 2
    import sqlite3
    con = sqlite3.connect(retriever.sqlite_path)
    try:
        assert con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        assert [r[0] for r in con.execute("SELECT note_id FROM chunks ORDER BY chunk_id")] == ["n1", "n2", "n3"]
    finally:
        con.close()


def test_population_changes_preserve_hybrid_evidence_without_shared_top_k(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever, NoteRetrievalConfig

    class Clinical:
        def qualified_table_name(self, name):
            return name

    retriever = BGEHybridNoteRetriever(
        Clinical(), cache_directory=tmp_path,
        config=NoteRetrievalConfig(relevance_threshold=0.7),
    )
    # Patient 20 has much stronger passages. They must not raise patient 10's
    # cutoff or crowd out any of its six above-threshold passages.
    chunks = [
        NoteChunk(i, 10 if i <= 6 else 20, None, str(i), "HPI", str(i))
        for i in range(1, 17)
    ]
    monkeypatch.setattr(retriever, "_ensure_index", lambda: None)
    monkeypatch.setattr(retriever, "_candidate_chunk_ids",
                        lambda pids: {c.chunk_id for c in chunks if c.subject_id in pids})
    monkeypatch.setattr(retriever, "_fetch_chunks",
                        lambda ids: [c for c in chunks if c.chunk_id in ids])
    def dense(claim, count, *, allowed_chunk_ids):
        assert count == len(allowed_chunk_ids)
        return [(i, 1.0) for i in sorted(allowed_chunk_ids)]
    monkeypatch.setattr(retriever, "_dense_search", dense)
    monkeypatch.setattr(retriever, "_sparse_search",
                        lambda *args, **kwargs: [])
    calls = []
    def predict(pairs, **kwargs):
        calls.extend(int(text) for _, text in pairs)
        return [1.0 if int(text) <= 6 else 8.0 for _, text in pairs]
    retriever._reranker = SimpleNamespace(predict=predict)
    alone = retriever.retrieve("claim", candidate_subject_ids={10})
    expanded = retriever.retrieve("claim", candidate_subject_ids={10, 20})
    shrunk = retriever.retrieve("claim", candidate_subject_ids={10})
    expected = {c.chunk_id for c in alone}
    assert expected == set(range(1, 7))
    assert {c.chunk_id for c in expanded if c.subject_id == 10} == expected
    assert {c.chunk_id for c in shrunk} == expected
    assert len(expanded) == 16
    assert sorted(calls) == list(range(1, 17))
    retriever.config = replace(retriever.config, relevance_threshold=0.9)
    assert {c.subject_id for c in retriever.retrieve("claim", candidate_subject_ids={10, 20})} == {20}
    assert len(calls) == 16


def test_dense_scoring_covers_all_scoped_passages(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever

    class Clinical:
        def qualified_table_name(self, name):
            return name

    retriever = BGEHybridNoteRetriever(Clinical(), cache_directory=tmp_path)
    matrix = np.zeros((8200, 2), dtype="float32")
    matrix[:, 0] = np.linspace(0.0, 1.0, len(matrix))
    np.save(retriever.embedding_path, matrix)
    retriever._encoder = SimpleNamespace(
        encode=lambda *args, **kwargs: np.array([[1.0, 0.0]], dtype="float32")
    )
    allowed = {1, 2, 8192, 8193, 8200}
    result = retriever._dense_search("claim", len(allowed), allowed_chunk_ids=allowed)
    assert {cid for cid, score in result} == allowed
    assert [cid for cid, score in result] == [8200, 8193, 8192, 2, 1]


def test_hybrid_union_uses_separate_patient_limits_and_handles_unscoped_calls(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever, NoteRetrievalConfig

    class Clinical:
        def qualified_table_name(self, name):
            return name

    retriever = BGEHybridNoteRetriever(
        Clinical(), cache_directory=tmp_path,
        config=NoteRetrievalConfig(sparse_top_k=1, dense_top_k=1, relevance_threshold=0.7),
    )
    chunks = [
        NoteChunk(i, 10 if i <= 3 else 20, None, str(i), "HPI", str(i))
        for i in range(1, 7)
    ]
    with sqlite3.connect(retriever.sqlite_path) as con:
        con.execute("CREATE TABLE chunks(subject_id INTEGER)")
        con.executemany("INSERT INTO chunks VALUES(?)", [(10,), (20,)])
    monkeypatch.setattr(retriever, "_ensure_index", lambda: None)
    monkeypatch.setattr(retriever, "_candidate_chunk_ids",
                        lambda pids: {c.chunk_id for c in chunks if c.subject_id in pids})
    monkeypatch.setattr(retriever, "_fetch_chunks",
                        lambda ids: [c for c in chunks if c.chunk_id in ids])
    searches = []
    def sparse(claim, count, *, allowed_chunk_ids):
        searches.append(("sparse", count, frozenset(allowed_chunk_ids)))
        return [(min(allowed_chunk_ids), 1.0)]
    def dense(claim, count, *, allowed_chunk_ids):
        searches.append(("dense", count, frozenset(allowed_chunk_ids)))
        return [(max(allowed_chunk_ids), 1.0)]
    monkeypatch.setattr(retriever, "_sparse_search", sparse)
    monkeypatch.setattr(retriever, "_dense_search", dense)
    retriever._reranker = SimpleNamespace(
        predict=lambda pairs, **kwargs: [1.0 if int(text) <= 3 else 8.0 for _, text in pairs]
    )
    assert {c.chunk_id for c in retriever.retrieve("claim", candidate_subject_ids={10})} == {1, 3}
    expanded = retriever.retrieve("claim", candidate_subject_ids={10, 20})
    assert {c.chunk_id for c in expanded} == {1, 3, 4, 6}
    assert {c.chunk_id for c in retriever.retrieve("claim")} == {1, 3, 4, 6}
    assert retriever.retrieve("claim", candidate_subject_ids=set()) == []
    assert all(count == 1 and ids in ({1, 2, 3}, {4, 5, 6}) for _, count, ids in searches)


def test_patient_scoped_bm25_returns_lexical_candidates(tmp_path):
    import sqlite3
    from infrastructure.mimic_hosp.note_retrieval import BGEHybridNoteRetriever

    class Clinical:
        def qualified_table_name(self, name):
            return name

    retriever = BGEHybridNoteRetriever(Clinical(), cache_directory=tmp_path)
    with sqlite3.connect(retriever.sqlite_path) as con:
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text)")
        con.executemany("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", [
            (1, "medication adherence"), (2, "unrelated condition"),
            (3, "medication medication medication"),
        ])
    assert [cid for cid, _ in retriever._sparse_search(
        "medication", 1, allowed_chunk_ids={1, 2}
    )] == [1]
    assert [cid for cid, _ in retriever._sparse_search(
        "medication", 1, allowed_chunk_ids={3}
    )] == [3]


def test_all_retained_passages_reach_llm_including_evidence_beyond_five():
    import json

    class ManyPassages:
        def retrieve(self, claim, *, candidate_subject_ids=None):
            return [
                NoteChunk(i, 10, 100 + i, f"n{i}", "HPI", f"retained evidence {i}", 1 - i / 100)
                for i in range(1, 9)
            ]

    class InspectEvidence:
        think = False

        def generate_structured(self, **kwargs):
            blocks = json.loads(kwargs["user_prompt"].split("\n\n", 1)[1])
            evidence = blocks[0]["evidence"]
            assert [c["chunk_id"] for c in evidence] == list(range(1, 9))
            assert [c["text"] for c in evidence] == [
                f"retained evidence {i}" for i in range(1, 9)
            ]
            return {"decisions": [{
                "subject_id": 10, "decision": "include",
                "supporting_chunk_ids": [8], "reason": "Supported by the last passage.",
            }]}

    matcher = HybridNoteMatcher(ManyPassages(), InspectEvidence())
    payload = {"claim": "event"}
    assert matcher.patients_with_note(payload, candidate_subject_ids={10}) == {10}
    assert matcher.decision_artifacts(payload)[10]["supporting_chunk_ids"] == [8]
    assert matcher.admissions_with_note(payload) == {(10, 108)}
