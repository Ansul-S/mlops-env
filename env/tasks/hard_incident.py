"""
env/tasks/hard_incident.py
===========================
Task 3 — HARD: Incident Cascade

Scenario:
  3 simultaneous firing alerts. One is the root cause (feature_store
  latency spike). Two are downstream effects. Agent has 15 steps before
  time budget expires.

Why hard:
  - Requires causal reasoning (not just reacting to each alert)
  - Correct fix ORDER matters — downstream first → root cause worsens
  - Silencing alerts is actively penalized (safety = 0)
  - Reasoning quality is graded on causal accuracy
  - Time pressure: efficiency score decays rapidly

Optimal sequence (deterministic):
  Step 1: investigate(component=feature_store)  → confirms root cause
  Step 2: restart_service(component=feature_store)
  Step 3: restart_service(component=model_serving)
  Step 4: restart_service(component=data_pipeline)
  Done — all 3 alerts resolved in 4 steps.
"""

from __future__ import annotations

from ..models import Action, ActionType, Component, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator
from .base import BaseTask

# Root cause component — used to validate sequence
ROOT_CAUSE_COMPONENT = Component.FEATURE_STORE.value

# Correct fix sequence after root cause identified
OPTIMAL_FIX_ORDER = [
    Component.FEATURE_STORE.value,
    Component.MODEL_SERVING.value,
    Component.DATA_PIPELINE.value,
]


class IncidentCascadeTask(BaseTask):

    task_id    = TaskID.INCIDENT
    difficulty = "hard"
    max_steps  = 15
    seed       = 42

    def build_simulator(self) -> MLOpsSimulator:
        return MLOpsSimulator(TaskID.INCIDENT, seed=self.seed)

    def is_done(self, sim: MLOpsSimulator) -> bool:
        return all(a.resolved for a in sim.alerts)

    def grade(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> tuple[RewardBreakdown, str]:
        """
        Per-step grader for incident cascade.

        correctness  (0.35) — Was this the right action at this point in time?
        efficiency   (0.25) — Steps consumed vs. optimal (4 steps)
        completeness (0.30) — Fraction of alerts resolved
        safety       (0.10) — No alerts silenced without fix; no metric regression

        Final score is episode-level at terminal step,
        per-step scores give dense intermediate signal.
        """

        correctness  = self._grade_correctness(action, sim, obs_before)
        efficiency   = self._grade_efficiency(sim)
        completeness = self._grade_completeness(sim)
        safety       = self._grade_safety(action, sim, obs_before)

        score = (
            correctness  * 0.35 +
            efficiency   * 0.25 +
            completeness * 0.30 +
            safety       * 0.10
        )
        score = round(min(1.0, max(0.0, score)), 4)

        feedback = self._build_feedback(
            action, sim, correctness, efficiency, completeness, safety
        )

        breakdown = RewardBreakdown(
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            completeness=round(completeness, 4),
            safety=round(safety, 4),
        )
        return breakdown, feedback

    # ─── Sub-scorers ──────────────────────────────────────────────────────────

    def _grade_correctness(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> float:
        """
        Score the action based on current situation state.
        Full credit only for the right action at the right moment.
        """
        atype = action.action_type
        comp  = action.parameters.get("component", "")
        from_comp = action.parameters.get("from_component", "")

        # ── Investigate first (before root cause known) ───────────────────────
        if not sim.root_cause_identified:
            if atype == ActionType.INVESTIGATE and comp == ROOT_CAUSE_COMPONENT:
                return 1.0   # Perfect: investigating root cause
            elif atype == ActionType.INVESTIGATE:
                return 0.4   # Investigating wrong component — partial credit
            elif atype == ActionType.RESTART_SERVICE and comp == ROOT_CAUSE_COMPONENT:
                return 0.6   # Lucky guess — skipped investigation, still correct
            elif atype == ActionType.SILENCE_ALERT:
                return 0.0   # Never correct
            else:
                return 0.2   # Wrong approach before diagnosis

        # ── Root cause known — fix in order ──────────────────────────────────
        fixed_so_far = set(sim.fix_sequence)
        next_to_fix  = next(
            (c for c in OPTIMAL_FIX_ORDER if c not in fixed_so_far),
            None
        )

        if atype == ActionType.RESTART_SERVICE:
            if comp == next_to_fix:
                return 1.0   # Perfect sequence
            elif comp in OPTIMAL_FIX_ORDER and comp not in fixed_so_far:
                return 0.5   # Out of order — still useful, penalized
            elif comp in fixed_so_far:
                return 0.0   # Already fixed — wasted step
            else:
                return 0.1   # Wrong component entirely

        elif atype == ActionType.REROUTE_TRAFFIC:
            if from_comp == ROOT_CAUSE_COMPONENT and ROOT_CAUSE_COMPONENT not in fixed_so_far:
                return 0.7   # Valid mitigation (not full fix)
            return 0.2

        elif atype == ActionType.ROLLBACK_MODEL:
            if Component.MODEL_SERVING.value not in fixed_so_far:
                return 0.5   # Partial mitigation
            return 0.1

        elif atype == ActionType.ESCALATE:
            return 0.2   # Always suboptimal (agent should self-resolve)

        elif atype == ActionType.SILENCE_ALERT:
            return 0.0   # Always penalized

        elif atype == ActionType.INVESTIGATE:
            return 0.3   # Root cause already known — redundant

        return 0.1

    def _grade_efficiency(self, sim: MLOpsSimulator) -> float:
        """
        Score based on steps consumed.
        Optimal = 4 steps. Max = 15.
        Decay is non-linear — heavy penalty for going past 8 steps.
        """
        steps = sim.step_count
        if steps <= 4:
            return 1.0
        elif steps <= 6:
            return 0.85
        elif steps <= 8:
            return 0.65
        elif steps <= 10:
            return 0.45
        elif steps <= 12:
            return 0.25
        else:
            return max(0.0, 0.1 - (steps - 12) * 0.05)

    def _grade_completeness(self, sim: MLOpsSimulator) -> float:
        """
        Fraction of alerts resolved × bonus for root cause first.
        """
        total    = len(sim.alerts)
        resolved = sum(1 for a in sim.alerts if a.resolved)
        base     = resolved / total if total > 0 else 0.0

        # Bonus: root cause fixed (not just downstream)
        root_fixed = ROOT_CAUSE_COMPONENT in sim.fix_sequence
        bonus      = 0.15 if root_fixed else 0.0

        return min(1.0, base + bonus)

    def _grade_safety(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> float:
        """
        Safety = 0 if:
          - Any alert silenced without fix
          - Error rate worsened vs before action
        Safety = 1 otherwise.
        """
        if sim.silenced_without_fix > 0:
            return 0.0

        error_before = obs_before.system_metrics.error_rate_pct
        error_after  = sim.metrics.error_rate_pct

        if error_after > error_before + 0.1:   # Tolerance band
            return 0.0

        return 1.0

    # ─── Feedback builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_feedback(
        action:       Action,
        sim:          MLOpsSimulator,
        correctness:  float,
        efficiency:   float,
        completeness: float,
        safety:       float,
    ) -> str:
        resolved = sum(1 for a in sim.alerts if a.resolved)
        total    = len(sim.alerts)
        parts    = [
            f"Action: {action.action_type.value}.",
            f"Alerts resolved: {resolved}/{total}.",
            f"Root cause identified: {sim.root_cause_identified}.",
            f"Fix sequence: {sim.fix_sequence}.",
            f"Scores — correctness: {correctness:.2f}, "
            f"efficiency: {efficiency:.2f}, "
            f"completeness: {completeness:.2f}, "
            f"safety: {safety:.2f}.",
        ]
        if sim.silenced_without_fix > 0:
            parts.append(
                f"WARNING: {sim.silenced_without_fix} alert(s) silenced "
                f"without fix — safety = 0."
            )
        return " ".join(parts)