import json
import pytest

from infrastructure.llm.ollama import (
    OllamaClient,
    OllamaConfig,
    OllamaError,
)

from urllib import request


class FakeHTTPResponse:
    def __init__(self, payload=None, lines=None):
        self._data = (
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else b""
        )
        self._lines = [
            json.dumps(line).encode("utf-8") + b"\n"
            for line in (lines or [])
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data

    def __iter__(self):
        return iter(self._lines)


def test_generate_structured_streams_schema_and_progress(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse(lines=[
            {
                "message": {
                    "role": "assistant",
                    "content": '{"answer":',
                },
                "done": False,
            },
            {
                "message": {
                    "role": "assistant",
                    "content": '"ok"}',
                },
                "done": True,
            },
        ])

    client = OllamaClient(
        OllamaConfig(
            model="qwen3:4b",
            think=False,
            timeout_seconds=9,
        )
    )
    monkeypatch.setattr(client._opener, "open", fake_urlopen)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    events = []
    result = client.generate_structured(
        system_prompt="Return structured data.",
        user_prompt="Hello",
        schema=schema,
        on_progress=lambda event_type, payload: events.append(
            (event_type, payload)
        ),
    )

    assert result == {"answer": "ok"}
    assert captured["url"].endswith("/api/chat")
    assert captured["timeout"] == 9
    assert captured["body"]["stream"] is True
    assert captured["body"]["think"] is False
    assert captured["body"]["format"] == schema
    assert any(
        event_type == "model_output" and payload == '{"answer":"ok"}'
        for event_type, payload in events
    )
    assert any(kind == "structured_output" for kind, _ in events)
    assert events[-1][0] == "llm_call_finished"
    lifecycle = {kind: payload for kind, payload in events if kind.startswith("llm_")}
    started = json.loads(lifecycle["llm_call_started"])
    finished = json.loads(lifecycle["llm_call_finished"])
    structured = json.loads(lifecycle["llm_structured"])
    assert started["call_id"] == finished["call_id"] == structured["call_id"]
    assert started["model"] == "qwen3:4b"
    assert started["think"] is False
    assert '"answer": "ok"' in structured["text"]


def test_generate_structured_can_cap_output_tokens_per_call(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse(lines=[
            {"message": {"role": "assistant", "content": '{"answer":"ok"}'}, "done": True},
        ])

    client = OllamaClient(OllamaConfig(model="qwen3:4b", think=False))
    monkeypatch.setattr(client._opener, "open", fake_urlopen)
    client.generate_structured(
        system_prompt="x", user_prompt="y",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        max_output_tokens=512,
    )
    assert captured["body"]["options"]["num_predict"] == 512


def test_generate_structured_surfaces_thinking_chunks_separately(monkeypatch):
    # Thinking and content are separate fields in Ollama's stream. Both
    # must reach the caller distinctly, or a thinking model goes silent
    # for its whole reasoning phase before any progress is visible.
    def fake_urlopen(req, timeout):
        return FakeHTTPResponse(lines=[
            {
                "message": {
                    "role": "assistant",
                    "thinking": "Let me consider the options",
                    "content": "",
                },
                "done": False,
            },
            {
                "message": {
                    "role": "assistant",
                    "thinking": "... and decide.",
                    "content": "",
                },
                "done": False,
            },
            {
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"ok"}',
                },
                "done": True,
            },
        ])

    client = OllamaClient(OllamaConfig(model="qwen3:4b", think=True))
    monkeypatch.setattr(client._opener, "open", fake_urlopen)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    events = []
    result = client.generate_structured(
        system_prompt="Return structured data.",
        user_prompt="Hello",
        schema=schema,
        on_progress=lambda event_type, payload: events.append(
            (event_type, payload)
        ),
    )

    assert result == {"answer": "ok"}
    thinking_events = [payload for kind, payload in events if kind == "thinking"]
    assert thinking_events, "no thinking progress events were emitted"
    assert thinking_events[-1] == "Let me consider the options... and decide."
    model_output_events = [payload for kind, payload in events if kind == "model_output"]
    assert model_output_events[-1] == '{"answer":"ok"}'


def test_think_is_mutable_after_construction_and_overrides_config(monkeypatch):
    # A UI toggle needs to flip thinking on/off on an already-built client
    # without reconstructing it (which would lose the warmed-up model).
    captured = {}

    def fake_urlopen(req, timeout):
        captured["think"] = json.loads(req.data.decode("utf-8"))["think"]
        return FakeHTTPResponse(lines=[
            {"message": {"role": "assistant", "content": "{}"}, "done": True},
        ])

    client = OllamaClient(OllamaConfig(model="qwen3:4b", think=False))
    monkeypatch.setattr(client._opener, "open", fake_urlopen)
    assert client.think is False

    client.think = True
    client.generate_structured(
        system_prompt="x", user_prompt="y", schema={"type": "object"},
    )
    assert captured["think"] is True


def test_client_disables_system_proxies(monkeypatch):
    captured = {}
    real_build_opener = request.build_opener

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return real_build_opener(*handlers)

    monkeypatch.setattr(
        "infrastructure.llm.ollama.request.build_opener",
        fake_build_opener,
    )

    OllamaClient()

    assert len(captured["handlers"]) == 1
    handler = captured["handlers"][0]

    assert isinstance(handler, request.ProxyHandler)
    assert handler.proxies == {}


def test_ensure_ready_preloads_exact_configured_model(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, json.loads(req.data.decode("utf-8")) if req.data else None))
        if req.full_url.endswith("/api/tags"):
            return FakeHTTPResponse(payload={"models": [{"name": "qwen3:4b"}]})
        if req.full_url.endswith("/api/generate"):
            return FakeHTTPResponse(payload={"response": "", "done": True})
        raise AssertionError(req.full_url)

    client = OllamaClient(OllamaConfig(model="qwen3:4b", think=False))
    monkeypatch.setattr(client._opener, "open", fake_urlopen)
    client.ensure_ready(preload=True)

    generate = [body for url, body in calls if url.endswith("/api/generate")]
    assert len(generate) == 1
    assert generate[0]["model"] == "qwen3:4b"
    assert generate[0]["prompt"] == ""
    assert generate[0]["keep_alive"] == "5m"


def test_model_availability_does_not_accept_wrong_tag(monkeypatch):
    client = OllamaClient(OllamaConfig(model="qwen3:4b"))
    monkeypatch.setattr(
        client._opener,
        "open",
        lambda req, timeout: FakeHTTPResponse(
            payload={"models": [{"name": "qwen3:8b"}]}
        ),
    )
    assert client.model_is_available() is False


def test_ensure_ready_reports_missing_model(monkeypatch):
    client = OllamaClient(OllamaConfig(model="qwen-test"))
    monkeypatch.setattr(client, "healthcheck", lambda: True)
    monkeypatch.setattr(client, "model_is_available", lambda: False)
    monkeypatch.setattr(client, "available_models", lambda: ("other:latest",))

    with pytest.raises(OllamaError, match="ollama pull qwen-test"):
        client.ensure_ready()


def test_parallel_thinking_scopes_do_not_leak_between_calls():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from framework.parsing import temporary_llm_thinking
    client = OllamaClient(OllamaConfig(think=False))
    barrier = Barrier(2)
    def call(enabled):
        with temporary_llm_thinking(client, enabled):
            barrier.wait(timeout=5)
            effective = client._effective_thinking()
            barrier.wait(timeout=5)
        return effective, client._effective_thinking()
    with ThreadPoolExecutor(max_workers=2) as pool:
        yes, no = pool.submit(call, True), pool.submit(call, False)
        assert yes.result() == (True, False)
        assert no.result() == (False, False)
    assert client.think is False
    with pytest.raises(ValueError), temporary_llm_thinking(client, True):
        raise ValueError("mock failed call")
    assert client._effective_thinking() is False
