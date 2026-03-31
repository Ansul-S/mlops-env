"""
env/tasks/medium_deployment.py
================================
Task 2 — MEDIUM: Deployment Decision

Scenario:
  Champion vs. Challenger A/B test complete.
  Challenger has better accuracy but violates error_rate SLA.
  Agent must choose the correct deployment strategy + parameters.

Why medium:
  - Requires multi-constraint reasoning (accuracy vs. error rate vs. latency)
  - Single decision but parameters must be precise
  - Reasoning quality is graded (20% of score)

Correct answer (deterministic):
  action_type = deploy_canary
  canary_pct  = 5  (± 5 accepted)
  rollback_threshold_pct = 2.0 (anything ≤ error SLA is valid)
"""

from __future__ import annotations

from ..models import Action, ActionType, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator
from .base import BaseTask


# Ground-truth constraints (derived from SLARequirements in simulator)
OPTIMAL_ACTION        = ActionType.DEPLOY_CANARY
CANARY_PCT_TARGET     = 5       # % traffic to challenger
CANARY_PCT_tolerance  = 5       # ± accepted range
MAX_ERROR_RATE_SLA    = 0.5     # Challenger (0.8%) breaches this


class DeploymentDecisionTask(BaseTask):

    task_id    = TaskID.DEPLOYMENT
    difficulty = "medium"
    max_steps  = 10
    seed       = 42

    def build_simulator(self) -> MLOpsSimulator:
        return MLOpsSimulator(TaskID.DEPLOYMENT, seed=self.seed)

    def is_done(self, sim: MLOpsSimulator) -> bool:
        # Terminal after any deployment decision
        return sim.deployment_action_taken is not None

    def grade(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> tuple[RewardBreakdown, str]:
        """
        Grader for a single deployment decision.

        correctness  (0.50) — Did agent pick the right action type?
        efficiency   (0.10) — Did agent act without unnecessary delay?
        completeness (0.20) — Are parameters within acceptable range?
        safety       (0.20) — Does reasoning demonstrate understanding of SLA?
        """

        challenger = next(
            c for c in sim.deployment_candidates if not c.is_champion
        )

        # ── Correctness ───────────────────────────────────────────────────────
        action_scores = {
            ActionType.DEPLOY_CANARY: 1.0,   # Correct — limits blast radius
            ActionType.HOLD:          0.4,   # Partial — safe but no progress
            ActionType.ROLLBACK:      0.2,   # Wrong — no regression to roll back from
            ActionType.DEPLOY_FULL:   0.0,   # Wrong — challenger violates SLA
        }
        correctness = action_scores.get(action.action_type, 0.0)

        # ── Completeness (parameter quality) ─────────────────────────────────
        completeness = 0.0
        param_feedback = ""

        if action.action_type == ActionType.DEPLOY_CANARY:
            canary_pct = action.parameters.get("canary_pct")
            if canary_pct is None:
                completeness = 0.3
                param_feedback = "Missing canary_pct parameter."
            else:
                canary_pct = float(canary_pct)
                if 1 <= canary_pct <= (CANARY_PCT_TARGET + CANARY_PCT_tolerance):
                    completeness = 1.0
                    param_feedback = f"canary_pct={canary_pct}% is appropriate."
                elif canary_pct <= 20:
                    completeness = 0.7
                    param_feedback = f"canary_pct={canary_pct}% is higher than optimal but acceptable."
                else:
                    completeness = 0.2
                    param_feedback = f"canary_pct={canary_pct}% is too aggressive given error rate breach."

            rollback_thresh = action.parameters.get("rollback_threshold_pct")
            if rollback_thresh is not None and float(rollback_thresh) <= MAX_ERROR_RATE_SLA:
                completeness = min(1.0, completeness + 0.1)
                param_feedback += " Rollback threshold correctly set below SLA."

        elif action.action_type == ActionType.HOLD:
            completeness = 0.5   # Decision made, no parameters needed

        # ── Efficiency ────────────────────────────────────────────────────────
        # Task 2 is single-step — reward instant decisive action
        efficiency = max(0.0, 1.0 - (sim.step_count / self.max_steps))

        # ── Safety (reasoning quality) ────────────────────────────────────────
        safety = self._grade_reasoning(action.reasoning, challenger)

        # ── Weighted final score ──────────────────────────────────────────────
        score = (
            correctness  * 0.50 +
            efficiency   * 0.10 +
            completeness * 0.20 +
            safety       * 0.20
        )
        score = round(min(1.0, max(0.0, score)), 4)

        feedback = (
            f"Action: {action.action_type.value}. "
            f"Correctness: {correctness:.2f}. "
            f"{param_feedback} "
            f"Reasoning score: {safety:.2f}. "
            f"(Challenger error_rate={challenger.error_rate_pct}% "
            f"exceeds SLA={MAX_ERROR_RATE_SLA}% — full deploy would be unsafe.)"
        )

        breakdown = RewardBreakdown(
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            completeness=round(completeness, 4),
            safety=round(safety, 4),
        )
        return breakdown, feedback

    # ─── Reasoning quality grader ─────────────────────────────────────────────

    @staticmethod
    def _grade_reasoning(reasoning: str, challenger) -> float:
        """
        Score reasoning text on whether it identifies the correct tradeoffs.
        Deterministic keyword heuristic — no LLM calls.
        Score 0.0–1.0.
        """
        if not reasoning:
            return 0.2   # Minimal: action chosen without explanation

        reasoning_lower = reasoning.lower()
        signals: list[float] = []

        # Must mention the error rate violation
        if any(kw in reasoning_lower for kw in ["error rate", "error_rate", "0.8", "sla", "budget"]):
            signals.append(1.0)
        else:
            signals.append(0.0)

        # Should mention canary/gradual rollout rationale
        if any(kw in reasoning_lower for kw in ["canary", "gradual", "partial", "rollout", "risk"]):
            signals.append(1.0)
        else:
            signals.append(0.0)

        # Bonus: mentions accuracy improvement as the motivation
        if any(kw in reasoning_lower for kw in ["accuracy", "0.861", "improvement", "better"]):
            signals.append(1.0)
        else:
            signals.append(0.5)

        # Bonus: mentions rollback plan
        if any(kw in reasoning_lower for kw in ["rollback", "revert", "threshold", "monitor"]):
            signals.append(1.0)
        else:
            signals.append(0.5)

        return sum(signals) / len(signals)