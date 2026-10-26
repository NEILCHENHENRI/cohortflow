from __future__ import annotations
from dataclasses import dataclass, field
from presentation.models import ConversationMessageView

@dataclass(slots=True)
class UIState:
    selected_node_id: str | None = None
    selected_collection_id: str | None = None
    selected_entity_id: str | None = None
    graph_version: int = 0
    graph_view_mode: str = "logic"
    messages: list[ConversationMessageView] = field(default_factory=list)
    notifications: list[str] = field(default_factory=list)
    def clear_entity_selection(self) -> None:
        self.selected_collection_id = None
        self.selected_entity_id = None
