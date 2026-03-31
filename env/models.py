"""
env/models.py
=============
All Pydantic types for MLOpsEnv.
Every field is documented — judges read models to understand your environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TaskID(str, Enum):
    DATA_TRIAGE  = "data_quality_triage"
    DEPLOYMENT   = "deployment_decision"
    INCIDENT     = "incident_cascade"


class ActionType(str, Enum):
    # ── Task 1: Data quality triage ──────────────────────────────────────────
    FIX_NULL        = "fix_null"        # Fill null with schema-appropriate value
    REMOVE_OUTLIER  = "remove_outlier"  # Drop record with statistical outlier
    CAST_TYPE       = "cast_type"       # Coerce field to expected schema type
    FLAG_DUPLICATE  = "flag_duplicate"  # Mark record as duplicate of another
    ACCEPT_RECORD   = "accept_record"   # Mark record as clean, no action needed

    # ── Task 2: Deployment decision ───────────────────────────────────────────
    DEPLOY_FULL     = "deploy_full"     # Full traffic switch to candidate
    DEPLOY_CANARY   = "deploy_canary"   # Gradual rollout (requires canary_pct param)
    ROLLBACK        = "rollback"        # Revert to champion model
    HOLD            = "hold"            # No deployment, monitor longer

    # ── Task 3: Incident cascade response ─────────────────────────────────────
    INVESTIGATE     = "investigate"     # Diagnose a component (reveals root cause)
    RESTART_SERVICE = "restart_service" # Restart a named service component
    REROUTE_TRAFFIC = "reroute_traffic" # Redirect traffic away from failing component
    ROLLBACK_MODEL  = "rollback_model"  # Revert model to previous checkpoint
    ESCALATE        = "escalate"        # Page on-call engineer (costs time budget)
    SILENCE_ALERT   = "silence_alert"   # Suppress alert without fixing (penalized)


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class DataIssueType(str, Enum):
    NULL_VALUE     = "null_value"
    TYPE_MISMATCH  = "type_mismatch"
    OUTLIER        = "outlier"
    DUPLICATE      = "duplicate"
    CLEAN          = "clean"


class Component(str, Enum):
    FEATURE_STORE    = "feature_store"
    MODEL_SERVING    = "model_serving"
    DATA_PIPELINE    = "data_pipeline"
    INFERENCE_CACHE  = "inference_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models (used inside Observation)
# ─────────────────────────────────────────────────────────────────────────────

class SystemMetrics(BaseModel):
    """Live system health snapshot visible to the agent."""
    model_config = {"protected_namespaces": ()}
    latency_p99_ms:  float = Field(description="99th-pct latency in ms")
    error_rate_pct:  float = Field(description="Current error rate %")
    throughput_rps:  float = Field(description="Requests per second")
    model_accuracy:  float = Field(description="Live model accuracy (0–1)")
    data_drift_score: float = Field(description="Feature drift (0=none, 1=full drift)")


class DataRecord(BaseModel):
    """A single data record awaiting triage (Task 1)."""
    record_id:       str
    fields:          dict[str, Any]
    schema_expected: dict[str, str]          # field → expected Python type string
    detected_issues: list[DataIssueType] = Field(default_factory=list)
    ground_truth_action: Optional[str] = Field(
        default=None,
        description="Correct action key — used only by grader, not shown to agent"
    )
    ground_truth_params: dict[str, Any] = Field(default_factory=dict)
    processed: bool = False


class Alert(BaseModel):
    """A firing system alert (Tasks 2 & 3)."""
    alert_id:          str
    severity:          Severity
    component:         Component
    message:           str
    triggered_at_step: int
    is_root_cause:     bool = Field(
        default=False,
        description="Internal flag — used by grader only"
    )
    acknowledged:  bool = False
    resolved:      bool = False


class ModelCandidate(BaseModel):
    """A model awaiting a deployment decision (Task 2)."""
    candidate_id:       str
    name:               str
    accuracy:           float
    latency_p99_ms:     float
    error_rate_pct:     float
    training_data_size: int
    is_champion:        bool = False


class SLARequirements(BaseModel):
    """Hard operational constraints the agent must respect."""
    max_latency_p99_ms: float  = 80.0
    max_error_rate_pct: float  = 0.5
    min_accuracy:       float  = 0.80
    min_throughput_rps: float  = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Observation  (what the agent sees every step)
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Full agent observation — returned by reset() and step()."""

    task_id:    TaskID
    step:       int = Field(description="Current step number (0-indexed)")
    max_steps:  int = Field(description="Episode terminates at this step")

    system_metrics:        SystemMetrics
    data_records:          list[DataRecord]    = Field(default_factory=list)
    alerts:                list[Alert]         = Field(default_factory=list)
    deployment_candidates: list[ModelCandidate] = Field(default_factory=list)

    context_history:   list[str]     = Field(
        default_factory=list,
        description="Last 5 action summaries — gives agent episodic memory"
    )
    available_actions: list[ActionType] = Field(default_factory=list)

    time_budget_remaining: float = Field(
        default=1.0,
        description="Normalized remaining time budget (1.0=full, 0.0=expired)"
    )
    sla_requirements: SLARequirements = Field(default_factory=SLARequirements)

    episode_score_so_far: float = Field(
        default=0.0,
        description="Cumulative score this episode — partial credit signal"
    )

    # Task-specific context string (human-readable prompt for the agent)
    task_context: str = Field(
        default="",
        description="Natural language description of current situation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action  (what the agent submits each step)
# ─────────────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent's chosen action for a single step."""

    action_type: ActionType
    target_id:   Optional[str] = Field(
        default=None,
        description="record_id, alert_id, or candidate_id this action targets"
    )
    parameters:  dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific params. Examples:\n"
            "  fix_null:      {'fill_value': 0.0, 'field': 'revenue'}\n"
            "  cast_type:     {'field': 'count', 'target_type': 'int'}\n"
            "  deploy_canary: {'canary_pct': 10, 'rollback_threshold_pct': 2.0}\n"
            "  restart_service: {'component': 'feature_store'}\n"
            "  reroute_traffic: {'from_component': 'feature_store', 'fallback': True}"
        )
    )
    reasoning: str = Field(
        default="",
        description="Agent's stated reasoning — graded in Hard task for causal accuracy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reward  (returned alongside Observation from step())
# ─────────────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Sub-scores explaining the reward — gives training signal at every dimension."""
    correctness:  float = Field(ge=0.0, le=1.0, description="Action was correct")
    efficiency:   float = Field(ge=0.0, le=1.0, description="Acted early / minimized waste")
    completeness: float = Field(ge=0.0, le=1.0, description="Coverage of all open issues")
    safety:       float = Field(ge=0.0, le=1.0, description="Did not worsen system state")


class Reward(BaseModel):
    """Full reward object — returned as part of StepResult."""
    score:     float = Field(ge=0.0, le=1.0, description="Final weighted score (0–1)")
    breakdown: RewardBreakdown
    feedback:  str  = Field(description="Human-readable explanation of score")
    done:      bool
    truncated: bool  = Field(description="True if episode ended due to step limit")


# ─────────────────────────────────────────────────────────────────────────────
# API response wrappers  (used by FastAPI server)
# ─────────────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Return type of step() — matches OpenEnv spec."""
    observation: Observation
    reward:      float
    done:        bool
    info:        dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Return type of reset() — matches OpenEnv spec."""
    observation: Observation


class StateResult(BaseModel):
    """Return type of state() — matches OpenEnv spec."""
    observation: Observation


# ─────────────────────────────────────────────────────────────────────────────
# Episode summary  (written to info dict at episode end)
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeResult(BaseModel):
    task_id:             TaskID
    total_score:         float
    steps_taken:         int
    reward_history:      list[float]
    breakdown_history:   list[dict[str, float]]
    final_state_summary: str