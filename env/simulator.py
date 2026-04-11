"""
env/simulator.py
================
Deterministic state machine for MLOpsEnv.

Key design principles:
  - Seed controls episode randomness → reproducible per seed
  - Root cause randomized per episode → agents cannot memorize
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
        "efficiency_weight": 0.25,
    },
}

# All possible root causes — randomized per episode (Fix B)
ROOT_CAUSES = [
    Component.FEATURE_STORE.value,
    Component.MODEL_SERVING.value,
    Component.DATA_PIPELINE.value,
]

# Healthy baseline metrics
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
    """Generate data records — randomized via rng (seed-controlled)."""
    schema = {
        "user_id":      "str",
        "revenue":      "float",
        "click_count":  "int",
        "country":      "str",
        "session_ms":   "float",
        "label":        "int",
    }

    records: list[DataRecord] = []

    issue_plan = [
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

    duplicate_source: str | None = None

    for i in range(20):
        record_id = f"rec_{i:03d}"

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
                fields[bad_field] = str(fields[bad_field]) + "_bad"
                detected.append(DataIssueType.TYPE_MISMATCH)

            elif issue_type == "outlier" and bad_field:
                if bad_field == "session_ms":
                    fields[bad_field] = 999999.9
                else:
                    fields[bad_field] = 9_999_999.99
                detected.append(DataIssueType.OUTLIER)

            elif issue_type == "duplicate":
                if duplicate_source is None:
                    duplicate_source = record_id
                    gt_action = "accept_record"
                    gt_params = {}
                    detected = []
                else:
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

def _generate_deployment_scenario(rng: random.Random) -> tuple[list[ModelCandidate], list[Alert], SLARequirements]:
    """Randomized deployment scenario using rng."""
    # Vary metrics slightly per seed
    champ_acc   = round(rng.uniform(0.83, 0.86), 3)
    chal_acc    = round(champ_acc + rng.uniform(0.01, 0.02), 3)
    champ_err   = round(rng.uniform(0.2, 0.4), 2)
    chal_err    = round(rng.uniform(0.6, 1.0), 2)   # always breaches SLA
    chal_lat    = round(rng.uniform(55.0, 72.0), 1)

    champion = ModelCandidate(
        candidate_id="champion_v2_1",
        name="revenue-model-v2.1",
        accuracy=champ_acc,
        latency_p99_ms=42.0,
        error_rate_pct=champ_err,
        training_data_size=1_200_000,
        is_champion=True,
    )
    challenger = ModelCandidate(
        candidate_id="challenger_v3_0",
        name="revenue-model-v3.0",
        accuracy=chal_acc,
        latency_p99_ms=chal_lat,
        error_rate_pct=chal_err,
        training_data_size=2_400_000,
        is_champion=False,
    )
    sla = SLARequirements(
        max_latency_p99_ms=80.0,
        max_error_rate_pct=0.5,
        min_accuracy=0.80,
        min_throughput_rps=1000.0,
    )
    alerts = [
        Alert(
            alert_id="alert_deploy_001",
            severity=Severity.MEDIUM,
            component=Component.MODEL_SERVING,
            message=(
                f"A/B test complete: challenger shows +{round(chal_acc-champ_acc,3)*100:.1f}% "
                f"accuracy improvement but error rate of {chal_err}% exceeds budget of 0.5%. "
                f"Decision required within current window."
            ),
            triggered_at_step=0,
            is_root_cause=False,
        )
    ]
    return [champion, challenger], alerts, sla


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 generator — Incident Cascade (Fix B: randomized root cause)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_incident_scenario(rng: random.Random) -> tuple[list[Alert], SystemMetrics, str]:
    """
    Randomized 3-alert incident cascade.
    Root cause is randomly chosen per episode — agents cannot memorize.
    Returns (alerts, degraded_metrics, root_cause_component_str).
    """
    # Pick random root cause (Fix B)
    root_cause = rng.choice(ROOT_CAUSES)

    # Downstream effects — everything except root cause
    all_components = [
        Component.FEATURE_STORE.value,
        Component.MODEL_SERVING.value,
        Component.DATA_PIPELINE.value,
    ]
    downstreams = [c for c in all_components if c != root_cause][:2]

    severity_map = {
        Component.FEATURE_STORE.value:   Severity.CRITICAL,
        Component.MODEL_SERVING.value:   Severity.HIGH,
        Component.DATA_PIPELINE.value:   Severity.MEDIUM,
    }

    message_map = {
        Component.FEATURE_STORE.value:   "p99 latency = 847ms (SLA: 80ms). All downstream consumers degraded.",
        Component.MODEL_SERVING.value:   "error_rate = 4.2% (budget: 0.5%). Prediction failures detected.",
        Component.DATA_PIPELINE.value:   "backlog = 12,847 records. Write timeouts. Pipeline stalled.",
    }

    alerts = [
        Alert(
            alert_id=f"alert_{root_cause}_001",
            severity=severity_map.get(root_cause, Severity.CRITICAL),
            component=Component(root_cause),
            message=f"CRITICAL: {root_cause} {message_map.get(root_cause, 'degraded.')}",
            triggered_at_step=0,
            is_root_cause=True,
        ),
    ]

    for i, ds in enumerate(downstreams, 2):
        alerts.append(Alert(
            alert_id=f"alert_{ds}_{i:03d}",
            severity=severity_map.get(ds, Severity.MEDIUM),
            component=Component(ds),
            message=f"{ds.upper()}: {message_map.get(ds, 'degraded — likely downstream effect.')}",
            triggered_at_step=0,
            is_root_cause=False,
        ))

    degraded_metrics = SystemMetrics(
        latency_p99_ms=847.0,
        error_rate_pct=4.2,
        throughput_rps=1100.0,
        model_accuracy=0.61,
        data_drift_score=0.72,
    )
    return alerts, degraded_metrics, root_cause


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulator class
# ─────────────────────────────────────────────────────────────────────────────

class MLOpsSimulator:
    """
    Manages all mutable state for one episode.
    Instantiated fresh by environment.reset().
    """

    def __init__(self, task_id: TaskID, seed: int | None = None) -> None:
        self.task_id = task_id
        # Fix C: use random seed if none provided — different episode every time
        self.seed = seed if seed is not None else random.randint(0, 99999)
        self.rng = random.Random(self.seed)
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
        self.fix_sequence: list[str] = []
        self.root_cause: str = Component.FEATURE_STORE.value  # set in _init_task

        # Deployment decision tracking
        self.deployment_action_taken: str | None = None
        self.deployment_params: dict[str, Any] = {}
        self.deployment_phase: str = "strategy"  # "strategy" | "monitoring" | "terminal"
        self.canary_pct: float = 0.0             # current canary traffic %
        self.monitoring_step: int = 0            # steps spent in monitoring

        self._init_task()

    def _init_task(self) -> None:
        if self.task_id == TaskID.DATA_TRIAGE:
            self.data_records = _generate_data_records(self.rng)
            self.metrics = deepcopy(HEALTHY_METRICS)

        elif self.task_id == TaskID.DEPLOYMENT:
            self.deployment_candidates, self.alerts, self.sla = (
                _generate_deployment_scenario(self.rng)
            )
            champ = next(c for c in self.deployment_candidates if c.is_champion)
            self.metrics = SystemMetrics(
                latency_p99_ms=42.0,
                error_rate_pct=champ.error_rate_pct,
                throughput_rps=5200.0,
                model_accuracy=champ.accuracy,
                data_drift_score=0.18,
            )

        elif self.task_id == TaskID.INCIDENT:
            self.alerts, self.metrics, self.root_cause = _generate_incident_scenario(self.rng)

    # ─── Downstream component order for a given root cause ────────────────────

    def _get_fix_order(self) -> list[str]:
        """Returns the correct fix order: root_cause first, then downstreams."""
        all_components = [
            Component.FEATURE_STORE.value,
            Component.MODEL_SERVING.value,
            Component.DATA_PIPELINE.value,
        ]
        downstreams = [c for c in all_components if c != self.root_cause]
        return [self.root_cause] + downstreams

    # ─── State query ──────────────────────────────────────────────────────────

    def get_observation(self) -> Observation:
        time_budget = max(0.0, 1.0 - (self.step_count / self.max_steps))
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
        if self.task_id == TaskID.DATA_TRIAGE:
            return [
                ActionType.FIX_NULL, ActionType.REMOVE_OUTLIER,
                ActionType.CAST_TYPE, ActionType.FLAG_DUPLICATE,
                ActionType.ACCEPT_RECORD,
            ]
        elif self.task_id == TaskID.DEPLOYMENT:
            return [
                ActionType.DEPLOY_FULL, ActionType.DEPLOY_CANARY,
                ActionType.ROLLBACK, ActionType.HOLD,
            ]
        else:
            return [
                ActionType.INVESTIGATE, ActionType.RESTART_SERVICE,
                ActionType.REROUTE_TRAFFIC, ActionType.ROLLBACK_MODEL,
                ActionType.ESCALATE, ActionType.SILENCE_ALERT,
            ]

    def _build_context(self) -> str:
        if self.task_id == TaskID.DATA_TRIAGE:
            remaining = sum(1 for r in self.data_records if not r.processed)
            total = len(self.data_records)
            return (
                f"You are an MLOps engineer triaging {total} incoming data records "
                f"before a model training run. {remaining} records still need review. "
                f"Identify each issue type from the detected_issues field and apply the correct fix. "
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
        else:
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
                f"Investigate to identify root cause, then resolve components in causal order. "
                f"Step {self.step_count}/{self.max_steps}."
            )

    # ─── State mutation ───────────────────────────────────────────────────────

    def apply_action(self, action: Action) -> bool:
        self.step_count += 1
        feedback_parts: list[str] = []
        done = False

        if self.task_id == TaskID.DATA_TRIAGE:
            done = self._apply_triage_action(action, feedback_parts)
        elif self.task_id == TaskID.DEPLOYMENT:
            done = self._apply_deployment_action(action, feedback_parts)
        elif self.task_id == TaskID.INCIDENT:
            done = self._apply_incident_action(action, feedback_parts)

        if self.step_count >= self.max_steps:
            done = True
            feedback_parts.append(f"Episode truncated at step {self.max_steps}.")

        summary = f"Step {self.step_count}: {action.action_type.value}"
        if action.target_id:
            summary += f"({action.target_id})"
        self.context_history.append(summary)

        return done

    def _apply_triage_action(self, action: Action, feedback: list[str]) -> bool:
        target = next(
            (r for r in self.data_records if r.record_id == action.target_id), None
        )
        if target is None:
            feedback.append(f"Record {action.target_id} not found.")
            return False
        if target.processed:
            feedback.append(f"Record {action.target_id} already processed.")
            return False

        target.processed = True

        if action.action_type.value != target.ground_truth_action:
            self.metrics.data_drift_score = min(1.0, self.metrics.data_drift_score + 0.03)
            feedback.append(f"Incorrect action on {action.target_id}.")
        else:
            feedback.append(f"Record {action.target_id} correctly processed.")

        return all(r.processed for r in self.data_records)

    def _apply_deployment_action(self, action: Action, feedback: list[str]) -> bool:
        """
        Multi-step deployment (Fix F):
        Phase 1 — strategy selection (step 1)
        Phase 2 — monitoring with metric evolution (steps 2-N)
        Phase 3 — terminal decision (promote/rollback)
        """
        chal  = next(c for c in self.deployment_candidates if not c.is_champion)
        champ = next(c for c in self.deployment_candidates if c.is_champion)

        if self.deployment_phase == "strategy":
            # Step 1: Choose initial strategy
            self.deployment_action_taken = action.action_type.value
            self.deployment_params = action.parameters

            if action.action_type == ActionType.DEPLOY_CANARY:
                self.canary_pct = float(action.parameters.get("canary_pct", 10))
                self.deployment_phase = "monitoring"
                # Metrics: partial shift toward challenger proportional to canary %
                mix = self.canary_pct / 100.0
                self.metrics.error_rate_pct = (
                    champ.error_rate_pct * (1 - mix) + chal.error_rate_pct * mix
                )
                self.metrics.model_accuracy = (
                    champ.accuracy * (1 - mix) + chal.accuracy * mix
                )
                feedback.append(
                    f"Canary at {self.canary_pct}% initiated. "
                    f"Monitoring error_rate={self.metrics.error_rate_pct:.2f}%. "
                    f"Continue monitoring or promote/rollback."
                )
                return False  # Not done — monitoring phase begins

            elif action.action_type == ActionType.DEPLOY_FULL:
                self.metrics.model_accuracy = chal.accuracy
                self.metrics.error_rate_pct = chal.error_rate_pct
                self.deployment_phase = "terminal"
                feedback.append(
                    f"Full deployment. error_rate={chal.error_rate_pct}% "
                    f"(SLA={self.sla.max_error_rate_pct}%)."
                )
                return True

            elif action.action_type == ActionType.ROLLBACK:
                self.metrics.model_accuracy = champ.accuracy
                self.metrics.error_rate_pct = champ.error_rate_pct
                self.deployment_phase = "terminal"
                feedback.append("Rolled back to champion.")
                return True

            elif action.action_type == ActionType.HOLD:
                # Stay in strategy phase — efficiency decays
                feedback.append("Holding. Metrics unchanged. Decide soon.")
                return False

        elif self.deployment_phase == "monitoring":
            self.monitoring_step += 1
            # Metrics evolve: error_rate trends toward challenger rate each step
            self.metrics.error_rate_pct = min(
                chal.error_rate_pct,
                self.metrics.error_rate_pct + (chal.error_rate_pct - self.metrics.error_rate_pct) * 0.3
            )

            if action.action_type == ActionType.DEPLOY_FULL:
                # Promote canary to full
                self.metrics.model_accuracy = chal.accuracy
                self.metrics.error_rate_pct = chal.error_rate_pct
                self.deployment_phase = "terminal"
                self.deployment_action_taken = "deploy_full"
                feedback.append(
                    f"Canary promoted to full. "
                    f"Final error_rate={chal.error_rate_pct}%."
                )
                return True

            elif action.action_type == ActionType.ROLLBACK:
                self.metrics.model_accuracy = champ.accuracy
                self.metrics.error_rate_pct = champ.error_rate_pct
                self.deployment_phase = "terminal"
                self.deployment_action_taken = "rollback"
                feedback.append(
                    f"Rolled back after monitoring. "
                    f"error_rate was {self.metrics.error_rate_pct:.2f}%."
                )
                return True

            elif action.action_type == ActionType.HOLD:
                feedback.append(
                    f"Continuing to monitor. "
                    f"error_rate={self.metrics.error_rate_pct:.2f}% "
                    f"(SLA={self.sla.max_error_rate_pct}%)."
                )
                # After 3 monitoring steps, force terminal
                if self.monitoring_step >= 3:
                    self.deployment_phase = "terminal"
                    return True
                return False

            else:
                feedback.append(f"Invalid action {action.action_type.value} in monitoring phase.")
                return False

        return True

    def _apply_incident_action(self, action: Action, feedback: list[str]) -> bool:
        comp      = action.parameters.get("component", "")
        from_comp = action.parameters.get("from_component", "")

        if action.action_type == ActionType.INVESTIGATE:
            if comp == self.root_cause:
                self.root_cause_identified = True
                feedback.append(
                    f"Investigation reveals {comp} is overloaded — root cause confirmed. "
                    f"All downstream failures originate here."
                )
            else:
                feedback.append(
                    f"Investigation of {comp} shows degraded performance, "
                    f"but root cause appears to be upstream."
                )

        elif action.action_type == ActionType.RESTART_SERVICE:
            self._resolve_component(comp, feedback)

        elif action.action_type == ActionType.REROUTE_TRAFFIC:
            if from_comp == self.root_cause:
                self.metrics.latency_p99_ms = max(80.0, self.metrics.latency_p99_ms * 0.4)
                self.metrics.error_rate_pct = max(0.4, self.metrics.error_rate_pct * 0.5)
                feedback.append("Traffic rerouted. Partial recovery in progress.")

        elif action.action_type == ActionType.ROLLBACK_MODEL:
            self.metrics.model_accuracy = 0.847
            self.metrics.error_rate_pct = max(0.3, self.metrics.error_rate_pct - 1.5)
            feedback.append("Model rolled back.")

        elif action.action_type == ActionType.SILENCE_ALERT:
            alert = next((a for a in self.alerts if a.alert_id == action.target_id), None)
            if alert:
                alert.acknowledged = True
            self.metrics.error_rate_pct = min(10.0, self.metrics.error_rate_pct + 0.5)
            feedback.append("Alert silenced without fix — underlying issue persists.")

        elif action.action_type == ActionType.ESCALATE:
            feedback.append("On-call engineer paged.")

        return all(a.resolved for a in self.alerts)

    def _resolve_component(self, component: str, feedback: list[str]) -> None:
        alert = next((a for a in self.alerts if a.component.value == component), None)

        if component == self.root_cause:
            if alert:
                alert.resolved = True
            self.fix_sequence.append(component)
            self.metrics.latency_p99_ms   = 44.0
            self.metrics.data_drift_score = 0.08
            self.metrics.error_rate_pct   = max(0.5, self.metrics.error_rate_pct - 2.0)
            feedback.append(f"{component} restarted. Root cause resolved. Downstream recovering.")

        elif component in [Component.MODEL_SERVING.value, Component.DATA_PIPELINE.value]:
            if alert:
                alert.resolved = True
            self.fix_sequence.append(component)
            self.metrics.error_rate_pct  = 0.3
            self.metrics.model_accuracy  = 0.891
            self.metrics.throughput_rps  = 5200.0
            feedback.append(f"{component} restarted. Service recovered.")
        else:
            feedback.append(f"Unknown component: {component}.")

    # ─── Computed properties for graders ─────────────────────────────────────

    @property
    def open_alert_count(self) -> int:
        return sum(1 for a in self.alerts if not a.resolved)

    @property
    def silenced_without_fix(self) -> int:
        return sum(1 for a in self.alerts if a.acknowledged and not a.resolved)