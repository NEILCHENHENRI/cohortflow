from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Any
from uuid import uuid4
from urllib import error, request

from framework.agent import ProgressCallback


class OllamaError(RuntimeError):
    """Raised when a local Ollama request fails or returns invalid data."""


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    think: bool = False
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    keep_alive: str | int | None = "5m"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Ollama model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    content: str
    thinking: str | None
    model: str
    done_reason: str | None
    total_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count: int | None
    raw: dict[str, Any]


class OllamaClient:
    """Dependency-free adapter for Ollama's local `/api/chat` endpoint."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        # Mutable, independent of the frozen config: lets a UI toggle flip
        # thinking on/off per-session without rebuilding the client (and
        # therefore without losing the warmed-up model / keep_alive state).
        self.think: bool = self.config.think
        self._thinking_override = ContextVar("ollama_thinking", default=None)
        # Ollama is a local service. urllib inherits macOS system proxy
        # settings even when no proxy environment variables are set, which
        # can incorrectly route localhost requests through a proxy. Always
        # connect to Ollama directly.
        self._opener = request.build_opener(request.ProxyHandler({}))

    @contextmanager
    def thinking(self, enabled: bool):
        """Per-call-context override: parallel steering must not change note reasoning."""
        token = self._thinking_override.set(bool(enabled))
        try:
            yield
        finally:
            self._thinking_override.reset(token)

    def _effective_thinking(self):
        override = self._thinking_override.get()
        return self.think if override is None else override

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        on_progress: ProgressCallback | None = None,
        stage_label: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        grounded_prompt = (
            f"{user_prompt}\n\n"
            "Return JSON matching this schema exactly:\n"
            f"{schema_text}"
        )

        call_id = uuid4().hex
        call_meta = {
            "call_id": call_id,
            "stage": stage_label or "Structured generation",
            "model": self.config.model,
            "think": bool(self._effective_thinking()),
        }
        self._emit(
            on_progress,
            "llm_call_started",
            json.dumps(call_meta, ensure_ascii=False),
        )
        self._emit(
            on_progress,
            "stage",
            (
                f"{stage_label or 'Structured generation'} with Ollama "
                f"({self.config.model}, think={self._effective_thinking()})"
                + (
                    " -- reasoning through the request before answering, "
                    "this can take noticeably longer than a direct answer..."
                    if self._effective_thinking() else "..."
                )
            ),
        )

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        started = time.perf_counter()
        for kind, chunk in self.chat_stream(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": grounded_prompt},
            ],
            format_schema=schema,
            max_output_tokens=max_output_tokens,
        ):
            if kind == "thinking":
                thinking_parts.append(chunk)
                thinking_text = "".join(thinking_parts)
                self._emit(on_progress, "thinking", thinking_text)
                self._emit(
                    on_progress,
                    "llm_thinking",
                    json.dumps({**call_meta, "text": thinking_text}, ensure_ascii=False),
                )
            else:
                content_parts.append(chunk)
                output_text = "".join(content_parts)
                self._emit(on_progress, "model_output", output_text)
                self._emit(
                    on_progress,
                    "llm_output",
                    json.dumps({**call_meta, "text": output_text}, ensure_ascii=False),
                )

        content = "".join(content_parts).strip()
        elapsed = time.perf_counter() - started
        self._emit(
            on_progress,
            "stage",
            f"{stage_label or 'Structured generation'} finished in {elapsed:.2f}s. Parsing JSON...",
        )

        try:
            payload = self._decode_json_object(content)
        except Exception as exc:
            self._emit(
                on_progress,
                "llm_call_failed",
                json.dumps({
                    **call_meta,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": content,
                }, ensure_ascii=False),
            )
            raise
        structured_text = json.dumps(payload, indent=2, ensure_ascii=False)
        self._emit(on_progress, "structured_output", structured_text)
        self._emit(
            on_progress,
            "llm_structured",
            json.dumps({**call_meta, "text": structured_text}, ensure_ascii=False),
        )
        self._emit(
            on_progress,
            "llm_call_finished",
            json.dumps({**call_meta, "elapsed_seconds": round(elapsed, 3)}, ensure_ascii=False),
        )
        return payload

    def chat_stream(
        self,
        *,
        messages: list[dict[str, str]],
        format_schema: dict[str, Any] | str | None = None,
        think: bool | None = None,
        max_output_tokens: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yields (kind, text) chunks, kind is 'thinking' or 'content'.

        Ollama streams thinking and content as separate message fields.
        Both must be surfaced, not just content, or a thinking model goes
        silent for its entire reasoning phase before any visible output.
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "think": self._effective_thinking() if think is None else think,
            "options": {"temperature": self.config.temperature},
        }
        if max_output_tokens is not None:
            if max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be positive")
            body["options"]["num_predict"] = int(max_output_tokens)
        if format_schema is not None:
            body["format"] = format_schema
        if self.config.keep_alive is not None:
            body["keep_alive"] = self.config.keep_alive

        encoded = json.dumps(body).encode("utf-8")
        req = request.Request(
            self._url("/api/chat"),
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            },
            method="POST",
        )

        try:
            with self._opener.open(
                req,
                timeout=self.config.timeout_seconds,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError(
                            f"Ollama returned an invalid stream chunk: {line[:300]!r}"
                        ) from exc

                    if not isinstance(payload, dict):
                        raise OllamaError(
                            "Ollama stream chunk must be a JSON object"
                        )
                    if payload.get("error"):
                        raise OllamaError(str(payload["error"]))

                    message = payload.get("message", {})
                    if isinstance(message, dict):
                        thinking = message.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            yield ("thinking", thinking)
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            yield ("content", content)

                    if payload.get("done") is True:
                        break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise OllamaError(
                "Could not connect to Ollama at "
                f"{self.config.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama request timed out after "
                f"{self.config.timeout_seconds}s"
            ) from exc

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        format_schema: dict[str, Any] | str | None = None,
        think: bool | None = None,
    ) -> OllamaResponse:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": self._effective_thinking() if think is None else think,
            "options": {"temperature": self.config.temperature},
        }
        if format_schema is not None:
            body["format"] = format_schema
        if self.config.keep_alive is not None:
            body["keep_alive"] = self.config.keep_alive

        payload = self._post_json("/api/chat", body)
        message = payload.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response is missing a message object")

        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama response message has no text content")

        thinking = message.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            thinking = str(thinking)

        return OllamaResponse(
            content=content,
            thinking=thinking,
            model=str(payload.get("model", self.config.model)),
            done_reason=_optional_string(payload.get("done_reason")),
            total_duration_ns=_optional_int(payload.get("total_duration")),
            prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
            eval_count=_optional_int(payload.get("eval_count")),
            raw=payload,
        )

    def healthcheck(self) -> bool:
        try:
            payload = self._get_json("/api/tags")
        except OllamaError:
            return False
        return isinstance(payload.get("models"), list)

    def available_models(self) -> tuple[str, ...]:
        payload = self._get_json("/api/tags")
        models = payload.get("models", [])
        names = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])
        return tuple(sorted(names))

    def model_is_available(self) -> bool:
        target = self.config.model.strip()
        names = self.available_models()
        if ":" in target:
            return target in names
        return any(
            name == target or name.split(":", 1)[0] == target
            for name in names
        )

    def ensure_ready(self, *, preload: bool = True) -> None:
        """Fail early when Ollama/model setup is invalid and optionally warm it.

        This is intentionally called once while constructing the local stack,
        not once per generation.  It turns an opaque first-request hang into a
        clear connection/model error and keeps the configured model resident.
        """
        if not self.healthcheck():
            raise OllamaError(
                "Ollama is not reachable at "
                f"{self.config.base_url}. Start it with `ollama serve`."
            )
        if not self.model_is_available():
            available = ", ".join(self.available_models()) or "none"
            raise OllamaError(
                f"Ollama model {self.config.model!r} is not installed. "
                f"Available models: {available}. Pull it with "
                f"`ollama pull {self.config.model}`."
            )
        if preload:
            self.preload()

    def preload(self) -> None:
        """Load the configured model without asking it to generate text."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "prompt": "",
            "stream": False,
        }
        if self.config.keep_alive is not None:
            body["keep_alive"] = self.config.keep_alive
        self._post_json("/api/generate", body)

    def _post_json(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        req = request.Request(
            self._url(path),
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(req)

    def _get_json(self, path: str) -> dict[str, Any]:
        req = request.Request(self._url(path), method="GET")
        return self._open_json(req)

    def _open_json(self, req: request.Request) -> dict[str, Any]:
        try:
            with self._opener.open(
                req,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise OllamaError(
                "Could not connect to Ollama at "
                f"{self.config.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama request timed out after "
                f"{self.config.timeout_seconds}s"
            ) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"Ollama returned invalid JSON: {raw[:300]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise OllamaError("Ollama response must be a JSON object")
        return payload

    @staticmethod
    def _decode_json_object(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            # Some model templates may still place reasoning tags around an
            # otherwise valid JSON object. Extract only the outer JSON object.
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                raise OllamaError(
                    "Ollama returned no JSON object. Raw content: "
                    f"{content[:1000]!r}"
                )
            candidate = content[start:end + 1]
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise OllamaError(
                    "Ollama returned invalid structured JSON. Raw content: "
                    f"{content[:1000]!r}"
                ) from exc

        if not isinstance(payload, dict):
            raise OllamaError(
                "structured Ollama response must decode to a JSON object"
            )
        return payload

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        event_type: str,
        payload: str,
    ) -> None:
        if callback is not None:
            callback(event_type, payload)

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
