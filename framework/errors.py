from __future__ import annotations


class AgentError(RuntimeError):
    """Base error for expected agent/runtime failures."""


class DomainValidationError(AgentError):
    """A domain object violates the canonical agent contract."""


class UnsupportedCriterionError(AgentError):
    """The active backend cannot execute a requested criterion."""


class InfrastructureFailure(AgentError):
    """A non-semantic external-system failure."""


class PendingResolutionError(AgentError):
    """A new task message arrived while a clarification is pending."""
