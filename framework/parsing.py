from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator
import json

from framework.agent import ProgressCallback, StructuredLLM
from framework.correction import CorrectionMode
from framework.errors import DomainValidationError
from framework.logic import LogicExpression, PredicateText
from framework.module import AgentModuleRole, AgentModuleRunner, AgentModuleSpec
from framework.steering import steering_checkpoint
from framework.timing import measure


class ParsingMode(str, Enum):
    """Generic staged semantic-parsing strategies for domain agents."""

    CARBON = "carbon"
    GRAPHITE = "graphite"
    DIAMOND = "diamond"
    GRAPHENE = "graphene"


DEFAULT_PASS_THINKING: dict[str, bool] = {
    "carbon": False,
    "extraction": False,
    "relationships": False,
    "compilation": False,
    "table_selection": False,
    "correction_merge": False,
    "correction_target": False,
    "correction_delta": False,
}


@contextmanager
def temporary_llm_thinking(llm, enabled: bool) -> Iterator[None]:
    """Temporarily set an LLM's optional ``think`` flag without leaking state."""
    scoped = getattr(llm, "thinking", None)
    if callable(scoped):
        with scoped(enabled):
            yield
        return
    if not hasattr(llm, "think"):
        yield
        return
    previous = getattr(llm, "think")
    setattr(llm, "think", bool(enabled))
    try:
        yield
    finally:
        setattr(llm, "think", previous)


@dataclass(frozen=True, slots=True)
class LogicParseResult:
    logic: LogicExpression
    predicates: tuple[PredicateText, ...]
    ambiguities: tuple[str, ...]
    requires_clarification: bool
    clarification_question: str | None
    rationale: str

    @property
    def confidence(self) -> float:
        return 0.5 if self.requires_clarification else 1.0


class StructuredTaskParser:
    """Shared configuration, lifecycle, and validated LLM calls for domain parsers.

    Subclasses supply prompts, converters, retry guidance and standalone module
    descriptions. This class does not decide task semantics or correction edits.
    """

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        structural_retry_limit: int = 1,
        reuse_unchanged_predicates: bool = False,
        parsing_mode: ParsingMode | str = ParsingMode.GRAPHITE,
        correction_mode: CorrectionMode | str = CorrectionMode.FULL_REGENERATION,
        thinking_by_pass: dict[str, bool] | None = None,
    ) -> None:
        if structural_retry_limit < 0:
            raise ValueError("structural_retry_limit cannot be negative")
        self._llm = llm
        self.structural_retry_limit = structural_retry_limit
        self.reuse_unchanged_predicates = bool(reuse_unchanged_predicates)
        self.parsing_mode = ParsingMode(parsing_mode)
        self.correction_mode = CorrectionMode(correction_mode)
        self.thinking_by_pass = dict(DEFAULT_PASS_THINKING)
        if thinking_by_pass:
            self.thinking_by_pass.update({k: bool(v) for k, v in thinking_by_pass.items()})
        self._module_runner = None

    def configure_correction(self, mode: CorrectionMode | str) -> None:
        self.correction_mode = CorrectionMode(mode)

    def configure_parsing(
        self,
        *,
        mode: ParsingMode | str | None = None,
        thinking_by_pass: dict[str, bool] | None = None,
    ) -> None:
        if mode is not None:
            self.parsing_mode = ParsingMode(mode)
        if thinking_by_pass is not None:
            for key, value in thinking_by_pass.items():
                if key not in DEFAULT_PASS_THINKING:
                    raise ValueError(f"unknown parsing pass {key!r}")
                self.thinking_by_pass[key] = bool(value)

    def _thinking_for(self, pass_name: str):
        return temporary_llm_thinking(
            self._llm, self.thinking_by_pass.get(pass_name, False)
        )

    def bind_module_runner(self, runner) -> None:
        """Bind the owning DomainAgent's generic lifecycle module runner."""
        self._module_runner = runner

    def _run_module(self, name: str, function, *, on_progress):
        if self._module_runner is not None:
            return self._module_runner(name, function, on_progress=on_progress)
        return AgentModuleRunner().run(
            self._standalone_module_spec(name), function, on_progress=on_progress
        )

    def _generate_with_validation(
        self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any],
        converter: Callable[[dict[str, Any]], Any], label: str,
        on_progress: ProgressCallback | None,
        retry_limit: int | None = None,
    ):
        prompt = user_prompt
        last_error: Exception | None = None
        effective_retry_limit = self.structural_retry_limit if retry_limit is None else max(0, int(retry_limit))
        for attempt in range(effective_retry_limit + 1):
            kwargs = {
                "system_prompt": system_prompt,
                "user_prompt": prompt,
                "schema": schema,
                "stage_label": label,
            }
            if on_progress is not None:
                kwargs["on_progress"] = on_progress
            try:
                with measure(
                    f"llm.{label}",
                    category="llm",
                    metadata={
                        "attempt": attempt + 1,
                        "system_prompt_chars": len(system_prompt),
                        "user_prompt_chars": len(prompt),
                    },
                ):
                    raw = self._llm.generate_structured(**kwargs)
                with measure(
                    f"validation.{label}",
                    category="validation",
                    metadata={"attempt": attempt + 1},
                ):
                    return converter(raw)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, DomainValidationError) as exc:
                last_error = exc
                if attempt >= effective_retry_limit:
                    break
                steering_checkpoint(
                    on_progress, f"semantic.validation_retry:{label}"
                )
                repair_rules = self._validation_repair_rules(label)
                prompt = (
                    user_prompt
                    + "\n\nTHE PREVIOUS OUTPUT FAILED DETERMINISTIC VALIDATION.\n"
                    + f"Validation error: {type(exc).__name__}: {exc}\n"
                    + repair_rules
                    + "\nRegenerate the complete object without changing the intended semantics."
                )
        raise DomainValidationError(
            f"{label} remained invalid after {effective_retry_limit + 1} attempt(s): {last_error}"
        )

    def _validation_repair_rules(self, label: str) -> str:
        """Domain-specific guidance appended verbatim on structural retries."""
        return ""

    def _standalone_module_spec(self, name: str) -> AgentModuleSpec:
        return AgentModuleSpec(
            name=name, role=AgentModuleRole.OTHER,
            description=f"Semantic module {name}.", category="semantic",
        )
