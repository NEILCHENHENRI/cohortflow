from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class SectionKind(str, Enum):
    TEXT = "text"
    KEY_VALUE = "key_value"
    TABLE = "table"
    JSON = "json"
    METRICS = "metrics"
    WARNING = "warning"
    ENTITY_LIST = "entity_list"

@dataclass(frozen=True, slots=True)
class AgentMetadataView:
    agent_id: str
    display_name: str
    description: str
    domain: str
    icon: str = "🤖"

@dataclass(frozen=True, slots=True)
class MetricView:
    label: str
    value: str | int | float
    help: str | None = None

@dataclass(frozen=True, slots=True)
class SectionView:
    section_id: str
    title: str
    kind: SectionKind
    data: Any
    expanded: bool = False
    description: str | None = None

@dataclass(frozen=True, slots=True)
class UIActionView:
    action_id: str
    label: str
    description: str = ""
    enabled: bool = True
    requires_confirmation: bool = False
    style: str = "secondary"
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class GraphNodeView:
    node_id: str
    label: str
    status: str
    category: str
    subtitle: str = ""
    tooltip: str = ""
    execution_mode: str = "unchanged"
    duration_ms: float | None = None
    badges: tuple[str, ...] = ()
    selectable: bool = True
    group_id: str | None = None
    primary_metric: str | None = None
    detail_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class GraphEdgeView:
    source: str
    target: str
    label: str = ""
    edge_type: str = "dependency"
    changed: bool = False
    animated: bool = False

@dataclass(frozen=True, slots=True)
class GraphGroupView:
    group_id: str
    label: str
    node_ids: tuple[str, ...] = ()
    collapsed: bool = True

@dataclass(frozen=True, slots=True)
class GraphView:
    nodes: tuple[GraphNodeView, ...]
    edges: tuple[GraphEdgeView, ...]
    version: int
    title: str = "Cohort graph"
    legend: dict[str, str] = field(default_factory=dict)
    groups: tuple[GraphGroupView, ...] = ()
    description: str = ""

@dataclass(frozen=True, slots=True)
class EntityCollectionView:
    collection_id: str
    title: str
    entity_type: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    count: int
    id_field: str = "id"
    description: str = ""

@dataclass(frozen=True, slots=True)
class EntityDetailView:
    entity_id: str
    title: str
    subtitle: str = ""
    summary_fields: dict[str, Any] = field(default_factory=dict)
    sections: tuple[SectionView, ...] = ()
    actions: tuple[UIActionView, ...] = ()

@dataclass(frozen=True, slots=True)
class NodeDetailView:
    node_id: str
    title: str
    description: str
    status: str
    category: str
    metrics: tuple[MetricView, ...] = ()
    sections: tuple[SectionView, ...] = ()
    entity_collections: tuple[EntityCollectionView, ...] = ()
    actions: tuple[UIActionView, ...] = ()

@dataclass(frozen=True, slots=True)
class ConversationMessageView:
    message_id: str
    role: str
    content: str
    timestamp: str
    graph_version: int | None = None
    status: str = "complete"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class InteractionResult:
    messages: tuple[ConversationMessageView, ...] = ()
    graph_changed: bool = False
    graph_version: int | None = None
    selected_node_id: str | None = None
    notifications: tuple[str, ...] = ()
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ConceptEntryView:
    code: str
    display_name: str
    vocabulary_id: str
    source_tables: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class CriterionEvidenceView:
    requirement_id: str
    label: str
    kind: str
    polarity: str
    patient_count: int
    patient_ids: tuple[int, ...] = ()
    concept_id: str | None = None
    concept_name: str | None = None
    vocabulary_id: str | None = None
    resolution_source: str | None = None
    concept_entries: tuple[ConceptEntryView, ...] = ()

@dataclass(frozen=True, slots=True)
class ExecutionEvidenceView:
    operator: str
    final_count: int | None
    final_patient_ids: tuple[int, ...]
    criteria: tuple[CriterionEvidenceView, ...]
    membership_rows: tuple[dict[str, Any], ...] = ()
    status: str = "ready"
    status_message: str = ""
    summary_available: bool = False

@dataclass(frozen=True, slots=True)
class ExecutionStateView:
    version: int
    status: str
    node_counts: dict[str, int]
    artifact_ids: tuple[str, ...]
    pending_kind: str | None = None
    pending_question: str | None = None
    summary: dict[str, Any] | None = None

    @property
    def summary_available(self) -> bool:
        return self.summary is not None
