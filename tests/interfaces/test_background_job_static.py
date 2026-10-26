from __future__ import annotations

from interfaces.background import BackgroundInteractionJob, _ProgressSink, _QueuedCorrection


class DummySession:
    def __init__(self):
        self.reasons = []
        self.state = None
        self.pending_initial_interaction = None
    def request_stop(self, *, reason):
        self.reasons.append(reason)
    def clear_stop_request(self):
        pass


class DummyController:
    def __init__(self):
        self.session = DummySession()
        self.latest_timing_report = None


def test_background_job_queue_stop_and_snapshot_are_deterministic_without_worker_thread():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "initial", max_events=3)
    assert job.submit_correction("   ") is False
    assert job.submit_correction("change potassium") is True
    assert controller.session.reasons == ["correction"]
    assert job.queue_depth == 1
    assert job.should_yield() is False
    snapshot = job.snapshot()
    assert snapshot.running and not snapshot.done
    assert snapshot.steering_requested
    assert snapshot.queue_depth == 1
    assert snapshot.active_text == "initial"

    assert job.request_stop() is True
    assert controller.session.reasons[-1] == "user_stop"
    assert job.snapshot().stop_requested is True


def test_progress_sink_forwards_events_and_steering_probe():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "x")
    sink = _ProgressSink(job)
    sink("stage", "hello")
    assert job.snapshot().events[-1].payload == "hello"
    assert sink.should_yield() is False

def test_background_job_fifo_helpers_keep_corrections_individual():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "initial")
    first = _QueuedCorrection("first")
    second = _QueuedCorrection("second")
    job._pending.extend([first, second])
    job._steering_requested = True
    assert job._pop_next() is first
    assert job.queue_depth == 1
    assert job._pop_next() is second
    assert job.queue_depth == 0
    assert job._pop_next() is None


def test_background_parks_fifo_corrections_on_initial_parse_clarification():
    from framework.pending_interaction import PendingInteraction

    controller = DummyController()
    controller.session.pending_initial_interaction = PendingInteraction.initial_parse_clarification(
        original_message="ambiguous", question="clarify", rationale="unclear"
    )
    job = BackgroundInteractionJob(controller, "initial")
    job._pending.extend([_QueuedCorrection("c1"), _QueuedCorrection("c2")])
    job._steering_requested = True

    assert job._park_queue_on_spec_clarification() is True
    assert controller.session.pending_initial_interaction.deferred_corrections == ("c1", "c2")
    assert job.queue_depth == 0


def test_background_job_combines_timing_reports_without_double_counting_interaction_total():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "x")
    job._timing_reports = [
        {
            "spans": [{"name": "interaction.total"}, {"name": "parse"}],
            "summary": {"llm_generation_ms": 10, "semantic_stage_ms": 12, "executed_node_count": 2},
            "graph_nodes": [{"node_id": "n1"}], "graph_source": "native",
        },
        {
            "spans": [{"name": "correction"}],
            "summary": {"llm_generation_ms": 5, "semantic_stage_ms": 7, "reused_node_count": 1},
            "graph_nodes": [{"node_id": "n2"}], "graph_source": "native",
        },
    ]
    report = job._combined_timing_report()
    assert report is not None
    assert [s["name"] for s in report["spans"]] == ["parse", "correction"]
    assert report["summary"]["llm_generation_ms"] == 15
    assert report["summary"]["semantic_stage_ms"] == 19
    assert report["summary"]["executed_node_count"] == 2
    assert report["summary"]["reused_node_count"] == 1
    assert report["graph_nodes"] == [{"node_id": "n2"}]


def test_llm_activity_collapses_stream_and_persists_structured_result():
    import json
    from interfaces.background import ProgressEvent, llm_activity_from_events

    meta = {"call_id": "c1", "stage": "Correction delta", "model": "qwen3:4b", "think": False}
    events = (
        ProgressEvent(1, "llm_call_started", json.dumps(meta), "t", 100.0),
        ProgressEvent(2, "llm_output", json.dumps({**meta, "text": '{"acts":'}), "t", 200.0),
        ProgressEvent(3, "llm_output", json.dumps({**meta, "text": '{"acts":[]}'}), "t", 300.0),
        ProgressEvent(4, "llm_structured", json.dumps({**meta, "text": '{\n  "acts": []\n}'}), "t", 350.0),
        ProgressEvent(5, "llm_call_finished", json.dumps({**meta, "elapsed_seconds": 0.25}), "t", 360.0),
    )

    calls = llm_activity_from_events(events)
    assert len(calls) == 1
    assert calls[0]["status"] == "complete"
    assert calls[0]["raw_output"] == '{"acts":[]}'
    assert '"acts": []' in calls[0]["structured_output"]
    assert calls[0]["elapsed_seconds"] == 0.25


def test_existing_task_makes_correction_semantics_atomic_for_midrun_steering():
    controller = DummyController()
    controller.session.state = object()
    job = BackgroundInteractionJob(controller, "change age to 65")

    assert job.submit_correction("also require hypertension") is True
    # Newer intent is queued, but an in-flight Robust-Delta interpretation is
    # allowed to finish so stable-ref targeting is not replaced by a synthetic
    # concatenated correction prompt. The runtime stop flag handles graph yield.
    assert job.should_yield() is False
    assert job.queue_depth == 1
    assert "queued until the current TaskSpec is semantically up to date" in job.snapshot().events[-1].payload


def test_initial_request_queues_until_stable_task_exists():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "find diabetes patients")
    assert job.submit_correction("also require hypertension") is True
    assert job.should_yield() is False
    assert "queued until the current TaskSpec is semantically up to date" in job.snapshot().events[-1].payload


def test_parallel_mode_preinterprets_only_while_execution_llm_is_active():
    import json
    import time
    from framework.runtime import ExecutionMode

    class Runtime:
        execution_mode = ExecutionMode.PARALLEL

    class Session(DummySession):
        def __init__(self):
            super().__init__()
            self.state = object()
            self.runtime = Runtime()
            self.analyzed = []
        def analyze_correction(self, text, on_progress=None):
            self.analyzed.append(text)
            return {"analysis": text}

    controller = DummyController()
    controller.session = Session()
    job = BackgroundInteractionJob(controller, "initial")
    job._execution_started = True
    job._on_progress(
        "llm_call_started",
        json.dumps({"call_id": "active", "stage": "resolver", "model": "x"}),
    )
    assert job.submit_correction("also require CKD")
    queued = job._pending[0]
    assert queued.done is not None
    assert queued.done.wait(1.0)
    assert controller.session.analyzed == ["also require CKD"]
    assert queued.analysis == {"analysis": "also require CKD"}


def test_initial_semantics_never_yield_to_a_newer_correction_before_spec_materializes():
    controller = DummyController()
    job = BackgroundInteractionJob(controller, "find diabetes")
    assert job.submit_correction("also require CKD")
    assert job.should_yield() is False
    assert job.queue_depth == 1


def test_parallel_note_batching_waits_for_correction_analysis_and_stops_only_if_note_changes():
    from types import SimpleNamespace
    from framework.runtime import ExecutionMode
    from framework.task import Objective, Requirement, TaskDelta, TaskSpec

    note = Requirement("note1", "note", {"claim": "worsening dyspnea"})
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "find patients"),
        requirements=(note,),
    )

    class Runtime:
        execution_mode = ExecutionMode.PARALLEL

    class Session:
        def __init__(self):
            self.state = SimpleNamespace(task=task, staged=None)
            self.runtime = Runtime()
            self.pending_initial_interaction = None
            self.begun = 0
            self.finished = []
            self.stop_calls = []
        def begin_parallel_checkpoint(self):
            self.begun += 1
        def finish_parallel_checkpoint(self, *, cancel_current):
            self.finished.append(cancel_current)
        def request_stop(self, *, reason, checkpoint_current=False):
            self.stop_calls.append((reason, checkpoint_current))
        def analyze_correction(self, text, on_progress=None):
            changed = Requirement("note1", "note", {"claim": "dyspnea during admission"})
            return SimpleNamespace(task_delta=TaskDelta(replace_requirements=(changed,)))

    controller = DummyController()
    controller.session = Session()
    job = BackgroundInteractionJob(controller, "initial")
    job._execution_started = True
    job._active_note_requirement_id = "note1"
    job._active_note_deterministic = True

    assert job.submit_correction("make the note requirement admission-specific")
    queued = job._pending[0]
    assert queued.done is not None and queued.done.wait(1.0)
    assert controller.session.begun == 1
    assert controller.session.finished == [True]
    assert controller.session.stop_calls == [("correction", False)]


def test_parallel_note_batching_keeps_compatible_note_scoring_running():
    from types import SimpleNamespace
    from framework.runtime import ExecutionMode
    from framework.task import Objective, Requirement, TaskDelta, TaskSpec

    note = Requirement("note1", "note", {"claim": "worsening dyspnea"})
    structured = Requirement("age", "patient", {"age_min": 50})
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "find patients"),
        requirements=(note, structured),
    )

    class Runtime:
        execution_mode = ExecutionMode.PARALLEL

    class Session:
        def __init__(self):
            self.state = SimpleNamespace(task=task, staged=None)
            self.runtime = Runtime()
            self.pending_initial_interaction = None
            self.finished = []
        def begin_parallel_checkpoint(self):
            pass
        def finish_parallel_checkpoint(self, *, cancel_current):
            self.finished.append(cancel_current)
        def request_stop(self, *, reason, checkpoint_current=False):
            pass
        def analyze_correction(self, text, on_progress=None):
            changed_age = Requirement("age", "patient", {"age_min": 60})
            return SimpleNamespace(task_delta=TaskDelta(replace_requirements=(changed_age,)))

    controller = DummyController()
    controller.session = Session()
    job = BackgroundInteractionJob(controller, "initial")
    job._execution_started = True
    job._active_note_requirement_id = "note1"
    job._active_note_deterministic = True

    assert job.submit_correction("change age to 60")
    queued = job._pending[0]
    assert queued.done is not None and queued.done.wait(1.0)
    assert controller.session.finished == [False]
