"""
env/simulator.py
================
Deterministic state machine for MLOpsEnv.

Key design principles:
  - All randomness seeded → reproducible baseline scores
  - Causal propagation → wrong actions degrade metrics in future steps
  - No ML models, no external calls → runs in <1s per step on 2 vCPU
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from .models import (
    Action, ActionType, Alert, Component, DataIssueType,
    DataRecord, ModelCandidate, Observation, RewardBreakdown,
    Severity, SLARequirements, SystemMetrics, TaskID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIG: dict[TaskID, dict[str, Any]] = {
    TaskID.DATA_TRIAGE: {
        "max_steps": 30,
        "num_records": 20,
        "efficiency_weight": 0.15,
    },
    TaskID.DEPLOYMENT: {
        "max_steps": 10,
        "efficiency_weight": 0.20,
    },
    TaskID.INCIDENT: {
        "max_steps": 15,
        "efficiency_weight": 0.25,   # time pressure matters most here
    },
}

# Healthy baseline metrics (Task 1 starts here, Tasks 2/3 start degraded)
HEALTHY_METRICS = SystemMetrics(
    latency_p99_ms=42.0,
    error_rate_pct=0.2,
    throughput_rps=5200.0,
    model_accuracy=0.891,
    data_drift_score=0.05,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 generator — Data Quality Triage
# ─────────────────────────────────────────────────────────────────────────────

def _generate_data_records(rng: random.Random) -> list[DataRecord]:
    """
    Generate 20 deterministic data records for Task 1.
    Ground truth action pre-computed per record — grader compares against it.
    """
    schema = {
        "user_id":      "str",
        "revenue":      "float",
        "click_count":  "int",
        "country":      "str",
        "session_ms":   "float",
        "label":        "int",
    }

    records: list[DataRecord] = []

    # Define exactly what issues appear and where (deterministic, not random)
    issue_plan = [
        # (issue_type, bad_field, ground_truth_action, ground_truth_params)
        ("null",      "revenue",     "fix_null",       {"field": "revenue",    "fill_value": 0.0}),
        ("null",      "click_count", "fix_null",       {"field": "click_count","fill_value": 0}),
        ("type",      "revenue",     "cast_type",      {"field": "revenue",    "target_type": "float"}),
        ("type",      "label",       "cast_type",      {"field": "label",      "target_type": "int"}),
        ("outlier",   "session_ms",  "remove_outlier", {"field": "session_ms"}),
        ("outlier",   "revenue",     "remove_outlier", {"field": "revenue"}),
        ("duplicate", None,          "flag_duplicate", {}),
        ("duplicate", None,          "flag_duplicate", {}),
        ("null",      "country",     "fix_null",       {"field": "country",    "fill_value": "unknown"}),
        ("type",      "click_count", "cast_type",      {"field": "click_count","target_type": "int"}),
    ]

    issue_indices: set[int] = set(rng.sample(range(20), len(issue_plan)))
    issue_map = dict(zip(sorted(issue_indices), issue_plan))

    # Track duplicate source for flag_duplicate
    duplicate_source: str | None = None

    for i in range(20):
        record_id = f"rec_{i:03d}"

        # Base clean record
        fields: dict[str, Any] = {
            "user_id":     f"u{rng.randint(10000, 99999)}",
            "revenue":     round(rng.uniform(10.0, 500.0), 2),
            "click_count": rng.randint(1, 200),
            "country":     rng.choice(["US", "IN", "GB", "DE", "JP"]),
            "session_ms":  round(rng.uniform(200.0, 8000.0), 1),
            "label":       rng.choice([0, 1]),
        }

        detected: list[DataIssueType] = []
        gt_action: str = "accept_record"
        gt_params: dict[str, Any] = {}

        if i in issue_map:
            issue_type, bad_field, gt_action, gt_params = issue_map[i]

            if issue_type == "null" and bad_field:
                fields[bad_field] = None
                detected.append(DataIssueType.NULL_VALUE)

            elif issue_type == "type" and bad_field:
                # Put string where numeric expected
                fields[bad_field] = str(fields[bad_field]) + "_bad"
                detected.append(DataIssueType.TYPE_MISMATCH)

            elif issue_type == "outlier" and bad_field:
                # 6-sigma outlier
                if bad_field == "session_ms":
                    fields[bad_field] = 999999.9
                else:
                    fields[bad_field] = 9_999_999.99
                detected.append(DataIssueType.OUTLIER)

            elif issue_type == "duplicate":
                if duplicate_source is None:
                    # First duplicate — this becomes the source (clean)
                    duplicate_source = record_id
                    gt_action = "accept_record"
                    gt_params = {}
                    detected = []
                else:
                    # Second duplicate — copy fields from source
                    src = next(r for r in records if r.record_id == duplicate_source)
                    fields = deepcopy(src.fields)
                    detected.append(DataIssueType.DUPLICATE)
                    gt_params = {"duplicate_of": duplicate_source}

        records.append(DataRecord(
            record_id=record_id,
            fields=fields,
            schema_expected=schema,
            detected_issues=detected,
            ground_truth_action=gt_action,
            ground_truth_params=gt_params,
            processed=False,
        ))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 generator — Deployment Decision
# ─────────────────────────────────────────────────────────────────────────────

def _generate_deployment_scenario() -> tuple[list[ModelCandidate], list[Alert], SLARequirements]:
    """
    Deterministic deployment scenario.
    Challenger violates error_rate SLA → correct answer is DEPLOY_CANARY(pct=5).
    Grader checks action + parameters.
    """
    champion = ModelCandidate(
        candidate_id="champion_v2_1",
        name="revenue-model-v2.1",
        accuracy=0.847,
        latency_p99_ms=42.0,
        error_rate_pct=0.3,
        training_data_size=1_200_000,
        is_champion=True,
    )
    challenger = ModelCandidate(
        candidate_id="challenger_v3_0",
        name="revenue-model-v3.0",
        accuracy=0.861,         # Better accuracy
        latency_p99_ms=67.0,    # Higher latency (still under SLA)
        error_rate_pct=0.8,     # VIOLATES 0.5% error budget → can't full-deploy
        training_data_size=2_400_000,
        is_champion=False,
    )
    sla = SLARequirements(
        max_latency_p99_ms=80.0,
        max_error_rate_pct=0.5,   # Challenger breaches this
        min_accuracy=0.80,
        min_throughput_rps=1000.0,
    )
    # Degraded metrics to signal need for action
    alerts = [
        Alert(
            alert_id="alert_deploy_001",
            severity=Severity.MEDIUM,
            component=Component.MODEL_SERVING,
            message=(
                "A/B test complete: challenger shows +1.4% accuracy improvement "
                "but error rate of 0.8% exceeds budget of 0.5%. "
                "Decision required within current window."
            ),
            triggered_at_step=0,
            is_root_cause=False,
        )
    ]
    return [champion, challenger], alerts, sla


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 generator — Incident Cascade
# ─────────────────────────────────────────────────────────────────────────────

def _generate_incident_scenario() -> tuple[list[Alert], SystemMetrics]:
    """
    Deterministic 3-alert incident cascade.

    Root cause: feature_store latency spike (component fully overloaded).
    Downstream effect 1: model_serving errors (stale features → bad predictions).
    Downstream effect 2: data_pipeline backlog (writes timing out).

    Optimal sequence:
      1. investigate(feature_store)  → reveals root cause
      2. restart_service(feature_store) OR reroute_traffic(feature_store)
      3. restart_service(model_serving) → clears downstream errors
      4. restart_service(data_pipeline) → drains backlog

    Penalty: silence_alert before fix → safety score = 0.
    """
    alerts = [
        Alert(
            alert_id="alert_fs_001",
            severity=Severity.CRITICAL,
            component=Component.FEATURE_STORE,
            message=(
                "CRITICAL: feature_store p99 latency = 847ms (SLA: 80ms). "
                "All downstream consumers experiencing degraded performance."
            ),
            triggered_at_step=0,
            is_root_cause=True,   # ← hidden from agent, used by grader
        ),
        Alert(
            alert_id="alert_ms_002",
            severity=Severity.HIGH,
            component=Component.MODEL_SERVING,
            message=(
                "HIGH: model_serving error_rate = 4.2% (budget: 0.5%). "
                "Stale features causing prediction failures."
            ),
            triggered_at_step=0,
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert_dp_003",
            severity=Severity.MEDIUM,
            component=Component.DATA_PIPELINE,
            message=(
                "MEDIUM: data_pipeline backlog = 12,847 records. "
                "Write timeouts detected. Pipeline stalled."
            ),
            triggered_at_step=0,
            is_root_cause=False,
        ),
    ]

    # System starts in degraded state
    degraded_metrics = SystemMetrics(
        latency_p99_ms=847.0,     # Far above SLA
        error_rate_pct=4.2,       # Critical
        throughput_rps=1100.0,    # Barely alive
        model_accuracy=0.61,      # Stale features degrading predictions
        data_drift_score=0.72,    # Pipeline stall causing drift
    )
    return alerts, degraded_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulator class
# ─────────────────────────────────────────────────────────────────────────────

class MLOpsSimulator:
    """
    Manages all mutable state for one episode.
    Instantiated fresh by environment.reset().
    """

    def __init__(self, task_id: TaskID, seed: int = 42) -> None:
        self.task_id = task_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.cfg = TASK_CONFIG[task_id]

        self.step_count: int = 0
        self.max_steps: int = self.cfg["max_steps"]
        self.done: bool = False
        self.reward_history: list[float] = []
        self.breakdown_history: list[dict[str, float]] = []
        self.context_history: list[str] = []

        # Per-task state
        self.data_records: list[DataRecord] = []
        self.alerts: list[Alert] = []
        self.deployment_candidates: list[ModelCandidate] = []
        self.sla: SLARequirements = SLARequirements()
        self.metrics: SystemMetrics = deepcopy(HEALTHY_METRICS)

        # Incident cascade tracking
        self.root_cause_identified: bool = False
        self.fix_sequence: list[str] = []   # components fixed in order

        # Deployment decision tracking
        self.deployment_action_taken: str | None = None
        self.deployment_params: dict[str, Any] = {}

        self._init_task()

    # ─── Task initializers ────────────────────────────────────────────────────

    def _init_task(self) -> None:
        if self.task_id == TaskID.DATA_TRIAGE:
            self.data_records = _generate_data_records(self.rng)
            self.metrics = deepcopy(HEALTHY_METRICS)

        elif self.task_id == TaskID.DEPLOYMENT:
            self.deployment_candidates, self.alerts, self.sla = (
                _generate_deployment_scenario()
            )
            self.metrics = SystemMetrics(
                latency_p99_ms=42.0,
                error_rate_pct=0.3,
                throughput_rps=5200.0,
                model_accuracy=0.847,
                data_drift_score=0.18,
            )

        elif self.task_id == TaskID.INCIDENT:
            self.alerts, self.metrics = _generate_incident_scenario()

    # ─── State query ──────────────────────────────────────────────────────────

    def get_observation(self) -> Observation:
        time_budget = max(
            0.0,
            1.0 - (self.step_count / self.max_steps)
        )
        score_so_far = (
            sum(self.reward_history) / len(self.reward_history)
            if self.reward_history else 0.0
        )
        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            max_steps=self.max_steps,
            system_metrics=deepcopy(self.metrics),
            data_records=deepcopy(self.data_records),
            alerts=deepcopy(self.alerts),
            deployment_candidates=deepcopy(self.deployment_candidates),
            context_history=list(self.context_history[-5:]),
            available_actions=self._available_actions(),
            time_budget_remaining=time_budget,
            sla_requirements=deepcopy(self.sla),
            episode_score_so_far=round(score_so_far, 4),
            task_context=self._build_context(),
        )

    def _available_actions(self) -> list[ActionType]:
        """Return actions that are valid in this task."""
        if self.task_id == TaskID.DATA_TRIAGE:
            return [
                ActionType.FIX_NULL,
                ActionType.REMOVE_OUTLIER,
                ActionType.CAST_TYPE,
                ActionType.FLAG_DUPLICATE,
                ActionType.ACCEPT_RECORD,
            ]
        elif self.task_id == TaskID.DEPLOYMENT:
            return [
                ActionType.DEPLOY_FULL,
                ActionType.DEPLOY_CANARY,
                ActionType.ROLLBACK,
                ActionType.HOLD,
            ]
        else:  # INCIDENT
            return [
                ActionType.INVESTIGATE,
                ActionType.RESTART_SERVICE,
                ActionType.REROUTE_TRAFFIC,
                ActionType.ROLLBACK_MODEL,
                ActionType.ESCALATE,
                ActionType.SILENCE_ALERT,
            ]

    def _build_context(self) -> str:
        """Human-readable situation summary for the LLM agent."""
        if self.task_id == TaskID.DATA_TRIAGE:
            remaining = sum(1 for r in self.data_records if not r.processed)
            total = len(self.data_records)
            return (
                f"You are an MLOps engineer triaging {total} incoming data records "
                f"before a model training run. {remaining} records still need review. "
                f"Identify each issue and apply the correct fix action. "
                f"Use 'accept_record' for clean records. "
                f"Step {self.step_count}/{self.max_steps}."
            )
        elif self.task_id == TaskID.DEPLOYMENT:
            champ = next(c for c in self.deployment_candidates if c.is_champion)
            chal  = next(c for c in self.deployment_candidates if not c.is_champion)
            return (
                f"A/B test complete. Champion ({champ.name}): accuracy={champ.accuracy:.3f}, "
                f"latency={champ.latency_p99_ms}ms, error_rate={champ.error_rate_pct}%. "
                f"Challenger ({chal.name}): accuracy={chal.accuracy:.3f}, "
                f"latency={chal.latency_p99_ms}ms, error_rate={chal.error_rate_pct}%. "
                f"SLA: max_latency={self.sla.max_latency_p99_ms}ms, "
                f"max_error_rate={self.sla.max_error_rate_pct}%. "
                f"Choose a deployment action with appropriate parameters."
            )
        else:  # INCIDENT
            open_alerts = [a for a in self.alerts if not a.resolved]
            alert_summary = "; ".join(
                f"[{a.severity.upper()}] {a.component.value}: {a.message[:60]}..."
                for a in open_alerts
            )
            return (
                f"Production incident in progress. {len(open_alerts)} open alert(s): "
                f"{alert_summary}. "
                f"System metrics: latency={self.metrics.latency_p99_ms}ms, "
                f"error_rate={self.metrics.error_rate_pct}%, "
                f"accuracy={self.metrics.model_accuracy:.2f}. "
                f"Find the root cause and resolve in the correct order. "
                f"Step {self.step_count}/{self.max_steps} — time is critical."
            )

    # ─── State mutation (called by environment.step()) ────────────────────────

    def apply_action(self, action: Action) -> tuple[RewardBreakdown, str, bool]:
        """
        Apply action → mutate state → return (breakdown, feedback, done).
        Grading logic lives in graders/ — this handles side-effects only.
        """
        self.step_count += 1
        feedback_parts: list[str] = []
        done = False

        if self.task_id == TaskID.DATA_TRIAGE:
            done = self._apply_triage_action(action, feedback_parts)

        elif self.task_id == TaskID.DEPLOYMENT:
            done = self._apply_deployment_action(action, feedback_parts)
            done = True  # Deployment is a single-decision episode

        elif self.task_id == TaskID.INCIDENT:
            done = self._apply_incident_action(action, feedback_parts)

        # Truncate if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            feedback_parts.append(f"Episode truncated at step {self.max_steps}.")

        feedback = " ".join(feedback_parts)
        summary = f"Step {self.step_count}: {action.action_type.value}"
        if action.target_id:
            summary += f"({action.target_id})"
        self.context_history.append(summary)

        return done

    def _apply_triage_action(self, action: Action, feedback: list[str]) -> bool:
        """Mark record as processed. Causal effect: wrong fixes degrade drift score."""
        target = next(
            (r for r in self.data_records if r.record_id == action.target_id),
            None
        )
        if target is None:
            feedback.append(f"Record {action.target_id} not found.")
            return False

        if target.processed:
            feedback.append(f"Record {action.target_id} already processed.")
            return False

        target.processed = True

        # Causal penalty: wrong action worsens data drift
        if action.action_type.value != target.ground_truth_action:
            self.metrics.data_drift_score = min(
                1.0,
                self.metrics.data_drift_score + 0.03
            )
            feedback.append(
                f"Incorrect action on {action.target_id}. "
                f"Data drift increased to {self.metrics.data_drift_score:.2f}."
            )
        else:
            feedback.append(f"Record {action.target_id} correctly processed.")

        all_done = all(r.processed for r in self.data_records)
        return all_done

    def _apply_deployment_action(self, action: Action, feedback: list[str]) -> bool:
        """Record deployment decision. Metrics update based on what was chosen."""
        self.deployment_action_taken = action.action_type.value
        self.deployment_params = action.parameters

        if action.action_type == ActionType.DEPLOY_CANARY:
            pct = action.parameters.get("canary_pct", 0)
            feedback.append(
                f"Canary deployment initiated at {pct}% traffic. "
                f"Monitoring error rate before full rollout."
            )
            # Partial metric improvement (proportional to canary %)
            self.metrics.model_accuracy += 0.014 * (pct / 100)

        elif action.action_type == ActionType.DEPLOY_FULL:
            chal = next(c for c in self.deployment_candidates if not c.is_champion)
            self.metrics.model_accuracy = chal.accuracy
            self.metrics.error_rate_pct = chal.error_rate_pct
            feedback.append("Full deployment to challenger completed.")

        elif action.action_type == ActionType.ROLLBACK:
            feedback.append("Rolled back to champion. No change to production.")

        elif action.action_type == ActionType.HOLD:
            feedback.append("Decision deferred. Continuing to monitor.")

        return True  # Always terminal

    def _apply_incident_action(self, action: Action, feedback: list[str]) -> bool:
        """
        Apply incident response action with causal metric propagation.
        Resolving root cause cascades metric recovery to downstream components.
        """
        component_str = action.parameters.get("component", "")
        target_alert = next(
            (a for a in self.alerts if a.alert_id == action.target_id),
            None
        )

        if action.action_type == ActionType.INVESTIGATE:
            comp = action.parameters.get("component", "unknown")
            if comp == Component.FEATURE_STORE.value:
                self.root_cause_identified = True
                feedback.append(
                    "Investigation reveals feature_store is overloaded. "
                    "Memory usage at 98%. This is the root cause of all downstream failures."
                )
            else:
                feedback.append(
                    f"Investigation of {comp} shows degraded performance, "
                    f"but root cause appears upstream."
                )

        elif action.action_type == ActionType.RESTART_SERVICE:
            comp = action.parameters.get("component", "")
            self._resolve_component(comp, feedback)

        elif action.action_type == ActionType.REROUTE_TRAFFIC:
            comp = action.parameters.get("from_component", "")
            if comp == Component.FEATURE_STORE.value:
                # Mitigates but doesn't fix — partial recovery
                self.metrics.latency_p99_ms = max(80.0, self.metrics.latency_p99_ms * 0.4)
                self.metrics.error_rate_pct = max(0.4, self.metrics.error_rate_pct * 0.5)
                feedback.append(
                    "Traffic rerouted from feature_store. "
                    "Partial recovery — latency improving but service not restored."
                )

        elif action.action_type == ActionType.ROLLBACK_MODEL:
            self.metrics.model_accuracy = 0.847   # Back to champion
            self.metrics.error_rate_pct = max(0.3, self.metrics.error_rate_pct - 1.5)
            feedback.append("Model rolled back to previous checkpoint. Accuracy restored.")

        elif action.action_type == ActionType.SILENCE_ALERT:
            if target_alert:
                target_alert.acknowledged = True
                # Safety penalty — silencing without fix worsens error rate
                self.metrics.error_rate_pct = min(10.0, self.metrics.error_rate_pct + 0.5)
                feedback.append(
                    f"Alert {action.target_id} silenced without fix. "
                    f"Underlying issue persists — error rate worsening."
                )

        elif action.action_type == ActionType.ESCALATE:
            feedback.append(
                "On-call engineer paged. Response ETA: 15 minutes. "
                "Time budget consumed."
            )
            # Costs time but no direct metric change

        # Check resolution
        all_resolved = all(a.resolved for a in self.alerts)
        return all_resolved

    def _resolve_component(self, component: str, feedback: list[str]) -> None:
        """Resolve a component — cascades recovery if root cause is fixed first."""
        alert = next(
            (a for a in self.alerts if a.component.value == component), None
        )

        if component == Component.FEATURE_STORE.value:
            if alert:
                alert.resolved = True
            self.fix_sequence.append(component)
            self.metrics.latency_p99_ms = 44.0   # Back to healthy
            self.metrics.data_drift_score = 0.08
            feedback.append(
                "feature_store restarted. Root cause resolved. "
                "Downstream services will begin recovering."
            )
            # Auto-cascade: model_serving starts recovering
            self.metrics.error_rate_pct = max(0.5, self.metrics.error_rate_pct - 2.0)

        elif component == Component.MODEL_SERVING.value:
            if alert:
                alert.resolved = True
            self.fix_sequence.append(component)
            self.metrics.error_rate_pct = 0.3
            self.metrics.model_accuracy = 0.891
            feedback.append(
                "model_serving restarted. Error rate normalized. "
                "Predictions back to baseline accuracy."
            )

        elif component == Component.DATA_PIPELINE.value:
            if alert:
                alert.resolved = True
            self.fix_sequence.append(component)
            feedback.append(
                "data_pipeline restarted. Backlog draining. "
                "Throughput recovering."
            )
            self.metrics.throughput_rps = 5200.0

        else:
            feedback.append(f"Unknown component: {component}. No action taken.")

    # ─── Computed properties for graders ─────────────────────────────────────

    @property
    def triage_accuracy(self) -> float:
        """Fraction of records processed with correct action."""
        processed = [r for r in self.data_records if r.processed]
        if not processed:
            return 0.0
        return sum(
            1 for r in processed
            if r.ground_truth_action is not None
        ) / len(self.data_records)

    @property
    def triage_correct_count(self) -> int:
        """Number of records processed correctly (for grader use)."""
        return sum(
            1 for r in self.data_records
            if r.processed and r.ground_truth_action is not None
        )

    @property
    def open_alert_count(self) -> int:
        return sum(1 for a in self.alerts if not a.resolved)

    @property
    def silenced_without_fix(self) -> int:
        """Alerts silenced but not resolved — safety penalty trigger."""
        return sum(
            1 for a in self.alerts
            if a.acknowledged and not a.resolved
        )
