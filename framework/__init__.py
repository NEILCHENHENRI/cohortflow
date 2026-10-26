from framework.agent import (
    ComposedDomainAgent, DomainAgent, ProgressCallback, StructuredLLM, TaskParser, TaskPlanner,
)
from framework.artifact import Artifact, ArtifactProvenance
from framework.artifact_store import ArtifactStore, InMemoryArtifactStore
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus
from framework.logic import (
    BooleanExpressionSyntaxError,
    LogicExpression,
    PredicateText,
    parse_boolean_expression,
)
from framework.interaction import InteractionAdapter, MessageRoute, MessageRouter
from framework.operator import OperatorContext, OperatorRegistry
from framework.reconcile import ReuseReport, reconcile_graphs
from framework.resolution import (
    InMemoryResolutionStore,
    PendingResolution,
    ResolutionAction,
    ResolutionDecision,
    ResolutionKey,
    ResolutionOption,
    ResolutionRecord,
    ResolutionRequest,
    ResolutionRequired,
    ResolutionStore,
    SemanticResolver,
)
from framework.execution_backend import ExecutionBackend
from framework.runtime import ExecutionMode, GraphRuntime, RuntimeResult
from framework.langgraph_runtime import LangGraphRuntime
from framework.session import (
    AgentState,
    SessionStatus,
    StagedExecution,
    SteerableSession,
)
from framework.task import TaskDelta, TaskSpec, apply_task_delta

from framework.correction import (
    ArtifactMutation,
    CorrectionAnalysis,
    CorrectionFallbackRequired,
    CorrectionMode,
    CorrectionIntent,
    CorrectionPlan,
    CorrectionPlanner,
    CorrectionScope,
    CorrectionTarget,
    MutationKind,
)

from framework.pending_interaction import (
    ContinuationKind,
    PendingInteraction,
    PendingInteractionKind,
)

from framework.interpretation import (
    GroundingIssue,
    GroundingIssueCode,
    GroundingResult,
    InteractionAct,
    PendingReplyInterpretation,
    SemanticReference,
)

from framework.manual_edit import (
    FALLBACK_COMMANDS,
    ManualEditChoice,
    ManualEditField,
    ManualEditForm,
    ManualEditIssue,
    ManualEditResult,
    ManualEditSection,
    ManualFieldType,
    StructuredManualEditProvider,
    is_fallback_command,
)


from framework.task_delta import (
    AttachmentSpec, AddPredicateAct, RemovePredicateAct, ReplacePredicateAct,
    UpdatePredicateFieldsAct, SetPredicateNegationAct, ChangeGroupOperatorAct,
    ReplaceLogicSubtreeAct, MovePredicateAct, AddRelationAct, RemoveRelationAct,
    UpdateRelationAct, UpdateAggregationAct, SetResolutionAct,
)

from framework.parsing import LogicParseResult, StructuredTaskParser
