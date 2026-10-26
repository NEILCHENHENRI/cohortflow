from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time

from framework.steering import SteeringYield
from interfaces.controller import AgentInteractionController
from presentation.models import InteractionResult
from framework.runtime import ExecutionMode
from framework.task import apply_task_delta


@dataclass(slots=True)
class _QueuedCorrection:
    text: str
    analysis: object | None = None
    error: BaseException | None = None
    done: threading.Event | None = None
    parallel_note_requirement_id: str | None = None
    parallel_base_task: object | None = None
    parallel_base_note_signature: str | None = None


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    sequence: int
    event_type: str
    payload: str
    timestamp: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class BackgroundInteractionSnapshot:
    running: bool
    done: bool
    stop_requested: bool
    steering_requested: bool
    execution_started: bool
    elapsed_ms: float
    events: tuple[ProgressEvent, ...]
    error: str | None
    queue_depth: int
    active_text: str




def llm_activity_from_events(events: tuple[ProgressEvent, ...] | list[ProgressEvent]) -> list[dict]:
    """Collapse streamed LLM events into one auditable record per call."""
    import json

    calls: dict[str, dict] = {}
    order: list[str] = []
    for event in events:
        base_type = event.event_type.split(":", 1)[0]
        is_note_wrapped = event.event_type.startswith("note_llm_llm_")
        if base_type not in {
            "llm_call_started", "llm_thinking", "llm_output",
            "llm_structured", "llm_call_finished", "llm_call_failed",
        } and not is_note_wrapped:
            continue
        try:
            payload = json.loads(event.payload)
        except Exception:
            continue
        call_id = str(payload.get("call_id") or "")
        if not call_id:
            continue
        if call_id not in calls:
            order.append(call_id)
            calls[call_id] = {
                "call_id": call_id,
                "stage": payload.get("stage") or "LLM call",
                "model": payload.get("model") or "unknown",
                "think": bool(payload.get("think", False)),
                "status": "running",
                "started_ms": event.elapsed_ms,
                "elapsed_seconds": None,
                "thinking": "",
                "raw_output": "",
                "structured_output": "",
                "error": None,
            }
        call = calls[call_id]
        call["stage"] = payload.get("stage") or call["stage"]
        call["model"] = payload.get("model") or call["model"]
        call["think"] = bool(payload.get("think", call["think"]))
        kind = base_type if not is_note_wrapped else event.event_type.removeprefix("note_llm_")
        if kind == "llm_thinking":
            call["thinking"] = str(payload.get("text") or "")
        elif kind == "llm_output":
            call["raw_output"] = str(payload.get("text") or "")
        elif kind == "llm_structured":
            call["structured_output"] = str(payload.get("text") or "")
        elif kind == "llm_call_finished":
            call["status"] = "complete"
            call["elapsed_seconds"] = payload.get("elapsed_seconds")
        elif kind == "llm_call_failed":
            call["status"] = "failed"
            call["elapsed_seconds"] = payload.get("elapsed_seconds")
            call["error"] = str(payload.get("error") or "Unknown LLM failure")
            if payload.get("text") and not call["raw_output"]:
                call["raw_output"] = str(payload["text"])
    return [calls[call_id] for call_id in order]


class _ProgressSink:
    """Callable progress adapter that also exposes a steering probe.

    Cohort parser module boundaries call ``should_yield`` after an atomic stage
    finishes. Graph execution uses the session/runtime stop flag between nodes.
    """

    def __init__(self, job: "BackgroundInteractionJob") -> None:
        self._job = job

    def __call__(self, event_type: str, payload: str) -> None:
        self._job._on_progress(event_type, payload)

    def should_yield(self) -> bool:
        return self._job.should_yield()


class BackgroundInteractionJob:
    """Persistent single-worker interaction supervisor with a correction queue.

    Only one semantic/execution worker is active. New messages can arrive while
    it runs: they are queued, the current atomic module/node is allowed to finish,
    then obsolete work yields before another module starts. Completed graph
    artifacts remain in the session/staged artifact store and are available to
    normal graph reconciliation for the corrected task.
    """

    def __init__(
        self,
        controller: AgentInteractionController,
        text: str,
        *,
        max_events: int | None = None,
    ) -> None:
        self.controller = controller
        self.text = text
        # Keep the complete per-job event history. The UI renders it in a
        # fixed-height scrolling container, so retention no longer needs to be
        # coupled to display length. A caller may still opt into a maxlen.
        self._events: deque[ProgressEvent] = deque(maxlen=max_events)
        self._timing_reports: list[dict] = []
        self._pending: deque[_QueuedCorrection] = deque()
        self._lock = threading.RLock()
        self._result: InteractionResult | None = None
        self._error: str | None = None
        self._explicit_stop = False
        self._steering_requested = False
        self._execution_started = False
        self._started_at = time.perf_counter()
        self._finished_at: float | None = None
        self._sequence = 0
        self._active_text = text
        self._active_llm_calls: set[str] = set()
        self._active_note_requirement_id: str | None = None
        self._active_note_deterministic = False
        # Once a stable task already exists, the active message is a correction
        # (or a continuation against that task). Robust-Delta interpretation is
        # transaction-like: let that semantic revision finish, queue any newer
        # correction, and use the runtime stop flag to avoid obsolete graph work.
        # Only an initial request, which has no stable task to target yet, may be
        # interrupted at a semantic module boundary and combined with newer intent.
        self._processing_queued_revision = controller.session.state is not None
        self._progress = _ProgressSink(self)
        self._thread = threading.Thread(
            target=self._run,
            name="steerable-agent-interaction",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit_correction(self, text: str) -> bool:
        """Queue newer user intent while the current interaction is running."""
        cleaned = text.strip()
        if not cleaned:
            return False
        if cleaned.casefold() in {"stop", "cancel"}:
            return self.request_stop()
        with self._lock:
            if self._finished_at is not None or self._explicit_stop:
                return False
            queued = _QueuedCorrection(cleaned)
            self._pending.append(queued)
            self._steering_requested = True
            overlap = self._can_overlap_correction()
            note_overlap = bool(
                overlap
                and self._active_note_deterministic
                and self._active_note_requirement_id
            )
            if note_overlap:
                state = self.controller.session.state
                if state is not None:
                    base_task = state.staged.task if state.staged is not None else state.task
                    queued.parallel_note_requirement_id = self._active_note_requirement_id
                    queued.parallel_base_task = base_task
                    queued.parallel_base_note_signature = self._note_requirement_signature(
                        base_task, self._active_note_requirement_id
                    )
                    self.controller.session.begin_parallel_checkpoint()
            if overlap:
                queued.done = threading.Event()
                threading.Thread(
                    target=self._preinterpret_correction, args=(queued,),
                    name="steerable-agent-correction-interpret", daemon=True,
                ).start()
        # This is cooperative steering, not hard cancellation. Initial parsing
        # may yield at its next module boundary. An in-flight Robust-Delta
        # correction finishes semantic interpretation and is then stopped before
        # obsolete graph work proceeds.
        runtime = getattr(self.controller.session, "runtime", None)
        checkpoint_current = (
            getattr(runtime, "execution_mode", None) is ExecutionMode.SEQUENTIAL
        )
        try:
            self.controller.session.request_stop(
                reason="correction", checkpoint_current=checkpoint_current
            )
        except TypeError:
            # Lightweight test/adapter sessions may implement the older stop
            # signature. Production InteractionSession accepts checkpoint_current.
            self.controller.session.request_stop(reason="correction")
        behavior = (
            "interpreting in parallel with the active action"
            if overlap
            else "queued until the current TaskSpec is semantically up to date"
        )
        self._append(
            "correction_queued",
            f"Queued correction ({self.queue_depth} waiting). {behavior}.",
        )
        return True

    def request_stop(self) -> bool:
        """Stop after the current atomic module while preserving completed work."""
        with self._lock:
            if self._finished_at is not None:
                return False
            self._explicit_stop = True
        try:
            self.controller.session.request_stop(
                reason="user_stop", checkpoint_current=True
            )
        except TypeError:
            self.controller.session.request_stop(reason="user_stop")
        self._append(
            "control",
            "Stop requested. The current module may finish and completed work will be preserved.",
        )
        return True

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def should_yield(self) -> bool:
        with self._lock:
            # Never interrupt semantic interpretation before its TaskSpec has
            # materialized. Newer corrections remain FIFO-queued.
            return self._explicit_stop

    def _can_overlap_correction(self) -> bool:
        state = getattr(self.controller.session, "state", None)
        runtime = getattr(self.controller.session, "runtime", None)
        return bool(
            state is not None
            and self._execution_started
            and (self._active_llm_calls or self._active_note_deterministic)
            and getattr(runtime, "execution_mode", None) is ExecutionMode.PARALLEL
        )

    @staticmethod
    def _note_requirement_signature(task, requirement_id: str) -> str | None:
        import json

        for requirement in getattr(task, "requirements", ()):
            if requirement.requirement_id != requirement_id:
                continue
            if requirement.kind != "note":
                return None
            return json.dumps(
                {
                    "kind": requirement.kind,
                    "payload": requirement.payload,
                    "polarity": requirement.polarity,
                    "required": requirement.required,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
        return None

    def _preinterpret_correction(self, queued: _QueuedCorrection) -> None:
        cancel_checkpoint = False
        parallel_note = queued.parallel_note_requirement_id is not None
        try:
            queued.analysis = self.controller.session.analyze_correction(
                queued.text, on_progress=self._progress
            )
            if parallel_note and queued.parallel_base_task is not None:
                proposed = apply_task_delta(
                    queued.parallel_base_task, queued.analysis.task_delta
                )
                updated_signature = self._note_requirement_signature(
                    proposed, queued.parallel_note_requirement_id or ""
                )
                cancel_checkpoint = (
                    updated_signature != queued.parallel_base_note_signature
                )
        except BaseException as exc:
            queued.error = exc
        finally:
            if parallel_note:
                # If the active note requirement changed/vanished, stop its
                # resumable deterministic loop after the current batch. If it
                # survived, let the deterministic stage finish, then the normal
                # graph stop yields at the stage boundary before adjudication.
                self.controller.session.finish_parallel_checkpoint(
                    cancel_current=cancel_checkpoint
                )
            if queued.done is not None:
                queued.done.set()

    def snapshot(self) -> BackgroundInteractionSnapshot:
        with self._lock:
            finished_at = self._finished_at
            now = finished_at if finished_at is not None else time.perf_counter()
            return BackgroundInteractionSnapshot(
                running=finished_at is None,
                done=finished_at is not None,
                stop_requested=self._explicit_stop,
                steering_requested=self._steering_requested,
                execution_started=self._execution_started,
                elapsed_ms=(now - self._started_at) * 1000.0,
                events=tuple(self._events),
                error=self._error,
                queue_depth=len(self._pending),
                active_text=self._active_text,
            )

    def take_result(self) -> InteractionResult | None:
        with self._lock:
            if self._finished_at is None:
                return None
            result = self._result
            self._result = None
            return result

    def _capture_latest_timing(self, result: InteractionResult | None = None) -> None:
        report = None
        if result is not None and isinstance(result.diagnostics, dict):
            candidate = result.diagnostics.get("timing")
            if isinstance(candidate, dict):
                report = candidate
        if report is None:
            candidate = self.controller.latest_timing_report
            if isinstance(candidate, dict):
                report = candidate
        if report is None:
            return
        with self._lock:
            # The controller property can still point at the same report after a
            # successful call. Avoid duplicating it when capture is invoked from
            # adjacent control-flow paths.
            if self._timing_reports and self._timing_reports[-1] is report:
                return
            self._timing_reports.append(report)

    def _combined_timing_report(self) -> dict | None:
        with self._lock:
            reports = list(self._timing_reports)
            finished_at = self._finished_at
            now = finished_at if finished_at is not None else time.perf_counter()
            wall_ms = (now - self._started_at) * 1000.0
        if not reports:
            return None

        spans: list[dict] = []
        llm_ms = 0.0
        semantic_ms = 0.0
        graph_ms = 0.0
        graph_node_compute_ms = 0.0
        graph_internal_ms = 0.0
        graph_attempt_ms = 0.0
        reused_compute_ms = 0.0
        reused_count = 0
        executed_count = 0
        latest_graph_nodes: list[dict] = []
        latest_graph_source = None
        for report in reports:
            spans.extend(
                item for item in report.get("spans", [])
                if item.get("name") != "interaction.total"
            )
            summary = report.get("summary", {})
            llm_ms += float(summary.get("llm_generation_ms", 0.0) or 0.0)
            semantic_ms += float(summary.get("semantic_stage_ms", 0.0) or 0.0)
            graph_ms += float(summary.get("graph_execution_wall_ms", 0.0) or 0.0)
            graph_node_compute_ms += float(summary.get("graph_node_compute_sum_ms", 0.0) or 0.0)
            graph_internal_ms += float(summary.get("graph_internal_overhead_sum_ms", 0.0) or 0.0)
            graph_attempt_ms += float(summary.get("graph_node_attempt_sum_ms", 0.0) or 0.0)
            reused_compute_ms += float(summary.get("reused_node_compute_ms", 0.0) or 0.0)
            reused_count += int(summary.get("reused_node_count", 0) or 0)
            executed_count += int(summary.get("executed_node_count", 0) or 0)
            if report.get("graph_nodes"):
                latest_graph_nodes = list(report.get("graph_nodes", []))
                latest_graph_source = report.get("graph_source")

        return {
            "total_wall_ms": round(wall_ms, 3),
            "spans": spans,
            "graph_nodes": latest_graph_nodes,
            "graph_source": latest_graph_source,
            "summary": {
                # This is now the same clock used by the operation log, so a
                # multi-revision steered job cannot report 24 s total beside a
                # 44 s operation timeline.
                "interaction_wall_ms": round(wall_ms, 3),
                "semantic_stage_ms": round(semantic_ms, 3),
                "llm_generation_ms": round(llm_ms, 3),
                "graph_execution_wall_ms": round(graph_ms, 3),
                "graph_node_compute_sum_ms": round(graph_node_compute_ms, 3),
                "graph_internal_overhead_sum_ms": round(graph_internal_ms, 3),
                "graph_node_attempt_sum_ms": round(graph_attempt_ms, 3),
                "reused_node_count": reused_count,
                "reused_node_compute_ms": round(reused_compute_ms, 3),
                "executed_node_count": executed_count,
            },
            "note": (
                "End-to-end background-job timing across all yielded/corrected "
                "revisions. LLM and Graph are component times and can overlap "
                "with Total. Nested detail spans must not be summed."
            ),
        }

    def _run(self) -> None:
        current_text = self.text
        current_queued: _QueuedCorrection | None = None
        current_was_applied = False
        try:
            while True:
                with self._lock:
                    self._active_text = current_text
                    self._execution_started = False
                # A previous revision yielded. The new revision now owns the
                # runtime lane. Clear only the old boundary-yield signal.
                self.controller.session.clear_stop_request()
                with self._lock:
                    later_corrections_waiting = (
                        self._processing_queued_revision and bool(self._pending)
                    )
                if later_corrections_waiting:
                    # Finish interpreting this queued correction, but do not
                    # spend graph compute on it when a newer correction already
                    # waits behind it.
                    self.controller.session.request_stop(reason="correction")
                try:
                    if current_queued is not None and current_queued.done is not None:
                        current_queued.done.wait()
                        if current_queued.error is not None:
                            raise current_queued.error
                        result = self.controller.handle_preanalyzed_correction(
                            current_text, current_queued.analysis, on_progress=self._progress
                        )
                    else:
                        result = self.controller.handle_message(
                            current_text, on_progress=self._progress
                        )
                    self._capture_latest_timing(result)
                    current_was_applied = True
                except SteeringYield as exc:
                    self._capture_latest_timing()
                    current_was_applied = False
                    self._append(
                        "execution_yielded",
                        f"Yielded at {exc.checkpoint}. Newer input has priority.",
                    )
                    if self._explicit_stop and not self._pending:
                        self._result = InteractionResult(
                            notifications=("execution_stopped",),
                            diagnostics={"yield_checkpoint": exc.checkpoint},
                        )
                        break
                    next_text = self._pop_next()
                    if next_text is None:
                        # Defensive: a checkpoint can only raise when stop/new
                        # input is present, but avoid spinning if state changes.
                        self._result = InteractionResult(
                            notifications=("execution_stopped",),
                            diagnostics={"yield_checkpoint": exc.checkpoint},
                        )
                        break
                    current_queued = next_text
                    current_text = next_text.text
                    with self._lock:
                        self._processing_queued_revision = True
                    continue

                # Parsing/correction clarification means there is still no
                # semantically up-to-date TaskSpec for newer corrections to
                # target. Park the FIFO queue on that clarification instead of
                # misreading the next correction as the requested restatement.
                if self._park_queue_on_spec_clarification():
                    self._result = result
                    break

                # A semantic clarification may have been created by the
                # atomic operation that just finished. A correction that was
                # already queued before that clarification became visible cannot
                # causally be an answer to it. More importantly, it may remove or
                # replace the very requirement being clarified. Supersede the
                # stale pending question and interpret the queued correction
                # first. If the requirement is still present, the corrected graph
                # will naturally raise a fresh clarification for it.
                preexisting_correction = (
                    self._take_preexisting_correction_before_pending()
                )
                if preexisting_correction is not None:
                    self._append(
                        "delivery_deferred",
                        "A queued correction predates the clarification. Applying the correction before asking for a potentially obsolete resolution.",
                    )
                    current_queued = preexisting_correction
                    current_text = preexisting_correction.text
                    with self._lock:
                        self._processing_queued_revision = True
                    continue

                if self._explicit_stop and not self._pending:
                    self._result = result
                    break

                next_text = self._pop_next()
                if next_text is not None:
                    # Do not surface a final result that was already superseded.
                    # If the old run stopped mid-graph its completed artifacts are
                    # still in the session/staged store and reconciliation will
                    # reuse them on this next correction.
                    self._append(
                        "delivery_deferred",
                        "A newer correction is waiting. Suppressing the superseded result.",
                    )
                    current_queued = next_text
                    current_text = next_text.text
                    with self._lock:
                        self._processing_queued_revision = True
                    continue

                self._result = result
                break
        except Exception as exc:  # controller normally converts domain errors
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._finished_at = time.perf_counter()
            combined = self._combined_timing_report()
            if combined is not None and self._result is not None:
                # InteractionResult is frozen, but diagnostics is intentionally a
                # mutable mapping used for instrumentation. Replace only the
                # timing payload. Conversation semantics remain untouched.
                self._result.diagnostics["timing"] = combined


    def _park_queue_on_spec_clarification(self) -> bool:
        from framework.pending_interaction import PendingInteractionKind

        state = self.controller.session.state
        if state is None:
            pending = self.controller.session.pending_initial_interaction
            expected = PendingInteractionKind.INITIAL_PARSE_CLARIFICATION
        else:
            pending = state.pending_interaction
            expected = PendingInteractionKind.CORRECTION_CLARIFICATION
        if pending is None or pending.kind is not expected:
            return False

        with self._lock:
            if not self._pending:
                return False
            parked = tuple(item.text for item in self._pending)
            self._pending.clear()
            self._steering_requested = False
        pending.deferred_corrections = (
            *pending.deferred_corrections, *parked
        )
        self.controller.session.clear_stop_request()
        self._append(
            "correction_parked",
            f"Parked {len(parked)} correction(s) until the clarified request materializes an up-to-date TaskSpec.",
        )
        return True

    def _take_preexisting_correction_before_pending(self) -> _QueuedCorrection | None:
        """Return one correction that predates a newly-created clarification.

        Such a message cannot be an answer to the clarification because the
        question did not exist when the user sent it. It must therefore be
        interpreted as task intent first. This is also required for correctness:
        a replacement/removal correction may invalidate the requirement that
        triggered the clarification (for example, replacing diabetes with
        hypertension while diabetes concept resolution is still running).

        We process one queued correction at a time to preserve FIFO semantics.
        If the corrected task still needs the unresolved requirement, graph
        execution will raise a fresh pending resolution later. If the correction
        removes it, that stale clarification never reaches the user.
        """
        state = self.controller.session.state
        if state is None or state.pending_interaction is None:
            return None

        from framework.pending_interaction import PendingInteractionKind

        pending = state.pending_interaction
        if pending.kind is not PendingInteractionKind.SEMANTIC_RESOLUTION:
            return None

        with self._lock:
            if not self._pending:
                return None
            correction = self._pending.popleft()
            self._steering_requested = bool(self._pending)

        # The pending question belongs to the superseded execution. Keep all
        # completed graph/artifact work, but clear only the human-interaction
        # gate so correction analysis can diff against the latest task state.
        self.controller.session.supersede_pending_for_correction()
        self.controller.session.clear_stop_request()
        self._append(
            "pending_superseded",
            "A queued correction arrived before this clarification existed. The stale clarification was superseded and the correction is being interpreted first.",
        )
        return correction

    def _pop_next(self) -> _QueuedCorrection | None:
        with self._lock:
            if not self._pending:
                self._steering_requested = False
                return None
            value = self._pending.popleft()
            self._steering_requested = bool(self._pending)
            return value

    def _on_progress(self, event_type: str, payload: str) -> None:
        import json
        with self._lock:
            if event_type == "execution_started":
                self._execution_started = True
            if event_type == "requirement_context":
                try:
                    info = json.loads(payload)
                except Exception:
                    info = {}
                if info.get("category") == "note":
                    self._active_note_requirement_id = str(info.get("requirement_id") or "") or None
            elif event_type == "note_deterministic_started":
                self._active_note_deterministic = True
            elif event_type == "note_deterministic_finished":
                self._active_note_deterministic = False
            elif event_type in {"node_committed", "node_failed", "node_cancelled"}:
                try:
                    info = json.loads(payload)
                except Exception:
                    info = {}
                if (
                    self._active_note_requirement_id
                    and info.get("requirement_id") == self._active_note_requirement_id
                ):
                    self._active_note_requirement_id = None
                    self._active_note_deterministic = False
            base = event_type.split(":", 1)[0]
            wrapped = event_type.startswith("note_llm_llm_")
            kind = event_type.removeprefix("note_llm_") if wrapped else base
            if kind in {"llm_call_started", "llm_call_finished", "llm_call_failed"}:
                try:
                    call_id = str(json.loads(payload).get("call_id") or "")
                except Exception:
                    call_id = ""
                if call_id and kind == "llm_call_started":
                    self._active_llm_calls.add(call_id)
                elif call_id:
                    self._active_llm_calls.discard(call_id)
        self._append(event_type, payload)

    def _append(self, event_type: str, payload: str) -> None:
        with self._lock:
            self._sequence += 1
            self._events.append(
                ProgressEvent(
                    sequence=self._sequence,
                    event_type=event_type,
                    payload=payload,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    elapsed_ms=(time.perf_counter() - self._started_at) * 1000.0,
                )
            )
