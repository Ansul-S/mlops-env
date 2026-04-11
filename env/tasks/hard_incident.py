"""
env/tasks/hard_incident.py
===========================
Task 3 — HARD: Incident Cascade

Root cause is randomized per episode (Fix B).
Grader uses pre-action state to correctly score the optimal sequence (Fix D).

Optimal sequence (4 steps):
  1. investigate(root_cause)       → confirm root cause
  2. restart_service(root_cause)   → fix root cause
  3. restart_service(downstream1)  → resolve downstream
  4. restart_service(downstream2)  → drain backlog

Optimal sequence should score >= 0.70 average.
"""

from __future__ import annotations

from ..models import Action, ActionType, Component, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator
from .base import BaseTask


class IncidentCascadeTask(BaseTask):

    task_id    = TaskID.INCIDENT
    difficulty = "hard"
    max_steps  = 15
    seed       = None   # None = random per episode

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
        Fix D: Grade using PRE-ACTION state to correctly score optimal sequence.

        The key insight: sim is post-action, so we must reconstruct pre-action state:
        - root_cause_identified_before: was root cause known BEFORE this action?
        - fix_sequence_before: what was fixed BEFORE this action?
        """

        atype  = action.action_type
        comp   = action.parameters.get("component", "")
        root   = sim.root_cause   # what the actual root cause is

        # ── Reconstruct pre-action state (Fix D) ─────────────────────────────
        # If this action JUST set root_cause_identified, it was False before
        if atype == ActionType.INVESTIGATE and comp == root and sim.root_cause_identified:
            root_known_before = False
        else:
            root_known_before = sim.root_cause_identified

        # If this action JUST added comp to fix_sequence, remove it for pre-action state
        if atype == ActionType.RESTART_SERVICE and comp in sim.fix_sequence:
            fixed_before = set(sim.fix_sequence) - {comp}
        else:
            fixed_before = set(sim.fix_sequence)

        # Get correct fix order for this episode's root cause
        fix_order = sim._get_fix_order()
        next_to_fix = next((c for c in fix_order if c not in fixed_before), None)

        correctness  = self._grade_correctness(atype, comp, root, root_known_before, fixed_before, next_to_fix, fix_order)
        efficiency   = self._grade_efficiency(sim)
        completeness = self._grade_completeness(sim)
        safety       = self._grade_safety(action, sim, obs_before)

        feedback = self._build_feedback(action, sim, correctness, efficiency, completeness, safety)

        breakdown = RewardBreakdown(
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            completeness=round(completeness, 4),
            safety=round(safety, 4),
        )
        return breakdown, feedback

    # ─── Correctness (Fix D: uses pre-action state) ───────────────────────────

    @staticmethod
    def _grade_correctness(
        atype: ActionType,
        comp: str,
        root: str,
        root_known_before: bool,
        fixed_before: set,
        next_to_fix: str | None,
        fix_order: list[str],
    ) -> float:

        if not root_known_before:
            # Pre-action: root cause unknown — should investigate
            if atype == ActionType.INVESTIGATE and comp == root:
                return 1.0   # Perfect: investigating actual root cause
            elif atype == ActionType.INVESTIGATE:
                return 0.4   # Wrong component — partial credit
            elif atype == ActionType.RESTART_SERVICE and comp == root:
                return 0.7   # Lucky guess — slightly penalized for skipping investigation
            elif atype == ActionType.SILENCE_ALERT:
                return 0.0
            else:
                return 0.2

        else:
            # Pre-action: root cause known — should fix in order
            if atype == ActionType.RESTART_SERVICE:
                if comp == next_to_fix:
                    return 1.0   # Perfect sequence
                elif comp in fix_order and comp not in fixed_before:
                    return 0.5   # Out of order — useful but penalized
                elif comp in fixed_before:
                    return 0.0   # Already fixed — wasted step
                else:
                    return 0.1   # Wrong component
            elif atype == ActionType.REROUTE_TRAFFIC:
                return 0.7 if comp == root and root not in fixed_before else 0.2
            elif atype == ActionType.ROLLBACK_MODEL:
                return 0.5
            elif atype == ActionType.INVESTIGATE:
                return 0.2   # Redundant after root cause known
            elif atype == ActionType.SILENCE_ALERT:
                return 0.0
            elif atype == ActionType.ESCALATE:
                return 0.2
            return 0.1

    def _grade_efficiency(self, sim: MLOpsSimulator) -> float:
        steps = sim.step_count
        if steps <= 4:   return 1.0
        elif steps <= 6: return 0.80
        elif steps <= 8: return 0.60
        elif steps <= 10: return 0.40
        elif steps <= 12: return 0.20
        else:            return max(0.0051, 0.1 - (steps - 12) * 0.03)

    def _grade_completeness(self, sim: MLOpsSimulator) -> float:
        total    = len(sim.alerts)
        resolved = sum(1 for a in sim.alerts if a.resolved)
        base     = resolved / total if total > 0 else 0.0
        # Bonus for fixing root cause (most important component)
        root_fixed = sim.root_cause in sim.fix_sequence
        bonus      = 0.10 if root_fixed else 0.0
        return min(0.9999, base + bonus)

    def _grade_safety(self, action: Action, sim: MLOpsSimulator, obs_before: Observation) -> float:
        if sim.silenced_without_fix > 0:
            return 0.0
        error_before = obs_before.system_metrics.error_rate_pct
        error_after  = sim.metrics.error_rate_pct
        if error_after > error_before + 0.1:
            return 0.0
        return 1.0

    @staticmethod
    def _build_feedback(action, sim, correctness, efficiency, completeness, safety) -> str:
        resolved = sum(1 for a in sim.alerts if a.resolved)
        total    = len(sim.alerts)
        return (
            f"Action: {action.action_type.value}. "
            f"Root cause: {sim.root_cause} (identified: {sim.root_cause_identified}). "
            f"Alerts resolved: {resolved}/{total}. "
            f"Fix sequence: {sim.fix_sequence}. "
            f"Scores: correct={correctness:.2f} eff={efficiency:.2f} "
            f"complete={completeness:.2f} safe={safety:.2f}."
        )