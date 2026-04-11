"""
env/tasks/medium_deployment.py
================================
Task 2 — MEDIUM: Deployment Decision (Multi-Step)

Scenario:
  Champion vs Challenger A/B test complete.
  Challenger has better accuracy but violates error_rate SLA.

  MULTI-STEP FLOW (Fix F):
  Step 1:   Agent chooses strategy (canary/full/hold/rollback)
  Steps 2-4: Metrics evolve per step — agent monitors and decides to
             promote, rollback, or continue monitoring
  Terminal: Agent calls promote/rollback OR max_steps reached

  This transforms it from classification → genuine sequential decision-making.
"""

from __future__ import annotations

from ..models import Action, ActionType, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator
from .base import BaseTask

MAX_ERROR_RATE_SLA = 0.5


class DeploymentDecisionTask(BaseTask):

    task_id    = TaskID.DEPLOYMENT
    difficulty = "medium"
    max_steps  = 10
    seed       = None  # random per episode

    def build_simulator(self) -> MLOpsSimulator:
        return MLOpsSimulator(TaskID.DEPLOYMENT, seed=self.seed)

    def is_done(self, sim: MLOpsSimulator) -> bool:
        return sim.deployment_action_taken is not None and sim.deployment_phase == "terminal"

    def grade(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> tuple[RewardBreakdown, str]:
        """
        Multi-step grader.

        Phase 1 (step 1): Grade the initial deployment strategy choice.
        Phase 2 (steps 2+): Grade monitoring decisions (promote/rollback/continue).

        correctness  (0.45) — Right strategy or right monitoring decision?
        efficiency   (0.15) — Acted without unnecessary delay?
        completeness (0.25) — Parameters appropriate? Monitoring thorough?
        safety       (0.15) — System metrics within SLA after action?
        """
        challenger = next(
            c for c in sim.deployment_candidates if not c.is_champion
        )
        phase = getattr(sim, 'deployment_phase', 'strategy')

        if phase in ('strategy', 'initial') or sim.step_count == 1:
            correctness, completeness, param_fb = self._grade_strategy(action, challenger, sim)
        else:
            correctness, completeness, param_fb = self._grade_monitoring(action, sim, obs_before)

        efficiency = max(0.0051, 1.0 - (sim.step_count / self.max_steps))
        safety     = self._grade_safety(sim, obs_before)

        feedback = (
            f"Phase: {phase}. Action: {action.action_type.value}. "
            f"{param_fb} "
            f"Scores: correct={correctness:.2f} eff={efficiency:.2f} "
            f"complete={completeness:.2f} safe={safety:.2f}."
        )

        return RewardBreakdown(
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            completeness=round(completeness, 4),
            safety=round(safety, 4),
        ), feedback

    # ─── Strategy grader (Step 1) ─────────────────────────────────────────────

    @staticmethod
    def _grade_strategy(action: Action, challenger, sim: MLOpsSimulator):
        action_scores = {
            ActionType.DEPLOY_CANARY: 1.0,
            ActionType.HOLD:          0.5,
            ActionType.ROLLBACK:      0.2,
            ActionType.DEPLOY_FULL:   0.1,
        }
        correctness = action_scores.get(action.action_type, 0.0)

        completeness = 0.3
        param_fb     = ""

        if action.action_type == ActionType.DEPLOY_CANARY:
            pct = action.parameters.get("canary_pct")
            if pct is not None:
                pct = float(pct)
                if 1 <= pct <= 10:
                    completeness = 1.0
                    param_fb = f"canary_pct={pct}% appropriate for SLA-violating challenger."
                elif pct <= 20:
                    completeness = 0.7
                    param_fb = f"canary_pct={pct}% acceptable but higher than optimal."
                else:
                    completeness = 0.2
                    param_fb = f"canary_pct={pct}% too aggressive — challenger breaches error SLA."
            else:
                param_fb = "Missing canary_pct — defaulting to 10%."

            thresh = action.parameters.get("rollback_threshold_pct")
            if thresh is not None and float(thresh) <= MAX_ERROR_RATE_SLA:
                completeness = min(0.9999, completeness + 0.1)
                param_fb += " Rollback threshold set correctly."

        elif action.action_type == ActionType.HOLD:
            completeness = 0.5
            param_fb = "Hold strategy — safe but no improvement progress."

        elif action.action_type == ActionType.DEPLOY_FULL:
            completeness = 0.1
            param_fb = f"Full deploy with error_rate={challenger.error_rate_pct}% breaches SLA={MAX_ERROR_RATE_SLA}%."

        return correctness, completeness, param_fb

    # ─── Monitoring grader (Steps 2+) ─────────────────────────────────────────

    @staticmethod
    def _grade_monitoring(action: Action, sim: MLOpsSimulator, obs_before: Observation):
        """Grade promote/rollback/continue decisions during monitoring phase."""
        chal = next(c for c in sim.deployment_candidates if not c.is_champion)
        current_err = sim.metrics.error_rate_pct

        if action.action_type == ActionType.DEPLOY_FULL:
            # Promoting canary to full
            if current_err <= MAX_ERROR_RATE_SLA:
                correctness  = 1.0
                completeness = 1.0
                param_fb     = f"Correct promotion — error_rate={current_err:.2f}% within SLA."
            else:
                correctness  = 0.1
                completeness = 0.1
                param_fb     = f"Premature promotion — error_rate={current_err:.2f}% still above SLA={MAX_ERROR_RATE_SLA}%."

        elif action.action_type == ActionType.ROLLBACK:
            if current_err > MAX_ERROR_RATE_SLA:
                correctness  = 1.0
                completeness = 0.8
                param_fb     = f"Correct rollback — error_rate={current_err:.2f}% exceeds SLA."
            else:
                correctness  = 0.3
                completeness = 0.3
                param_fb     = "Unnecessary rollback — metrics were within SLA."

        elif action.action_type == ActionType.HOLD:
            # Continuing to monitor
            correctness  = 0.6
            completeness = 0.5
            param_fb     = "Continuing to monitor — valid if error rate still converging."

        else:
            correctness  = 0.1
            completeness = 0.1
            param_fb     = f"Unexpected action {action.action_type.value} in monitoring phase."

        return correctness, completeness, param_fb

    # ─── Safety grader ────────────────────────────────────────────────────────

    @staticmethod
    def _grade_safety(sim: MLOpsSimulator, obs_before: Observation) -> float:
        err_after  = sim.metrics.error_rate_pct
        err_before = obs_before.system_metrics.error_rate_pct
        # Penalize if error rate jumped significantly
        if err_after > err_before + 0.3:
            return 0.2
        if err_after > MAX_ERROR_RATE_SLA * 2:
            return 0.4
        return 1.0