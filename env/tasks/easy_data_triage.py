"""
env/tasks/easy_data_triage.py
==============================
Task 1 — EASY: Data Quality Triage

Scenario:
  20 incoming data records before a training run.
  Each has one issue (or is clean). Agent applies the correct
  fix action per record within 30 steps.

Why easy:
  - One action per record, no dependencies between records
  - Issues are clearly visible in the fields
  - No time pressure on metric cascades
"""

from __future__ import annotations

from ..models import Action, ActionType, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator
from .base import BaseTask


class DataQualityTriageTask(BaseTask):

    task_id    = TaskID.DATA_TRIAGE
    difficulty = "easy"
    max_steps  = 30
    seed       = 42

    def build_simulator(self) -> MLOpsSimulator:
        return MLOpsSimulator(TaskID.DATA_TRIAGE, seed=self.seed)

    def is_done(self, sim: MLOpsSimulator) -> bool:
        return all(r.processed for r in sim.data_records)

    def grade(
        self,
        action:     Action,
        sim:        MLOpsSimulator,
        obs_before: Observation,
    ) -> tuple[RewardBreakdown, str]:
        """
        Per-action grader.

        correctness  (0.6 weight) — Did agent apply the right action type?
        efficiency   (0.1 weight) — Is agent making progress (not re-processing)?
        completeness (0.2 weight) — What fraction of all records are now handled?
        safety       (0.1 weight) — Did action worsen drift score?

        Final score = weighted sum, normalized to 0–1.
        """

        # ── Find the targeted record (obs_before = pre-action snapshot) ─────────
        record = next(
            (r for r in obs_before.data_records if r.record_id == action.target_id),
            None,
        )

        # ── Correctness ───────────────────────────────────────────────────────
        if record is None:
            correctness = 0.0
            feedback = f"No record found with id='{action.target_id}'. Wasted step."
        elif record.processed and action.action_type != ActionType.ACCEPT_RECORD:
            # Already processed before this action — penalize re-work
            correctness = 0.0
            feedback = f"Record {action.target_id} was already processed before this step."
        else:
            gt = record.ground_truth_action
            applied = action.action_type.value

            if applied == gt:
                # Correct action type — check parameters for partial credit
                param_score = self._check_params(action, record.ground_truth_params)
                correctness = 0.6 + 0.4 * param_score   # up to 1.0
                feedback = (
                    f"Correct action '{applied}' on {action.target_id}. "
                    f"Param quality: {param_score:.2f}."
                )
            else:
                correctness = 0.0
                feedback = (
                    f"Wrong action '{applied}' on {action.target_id}. "
                    f"Expected '{gt}'."
                )

        # ── Efficiency ────────────────────────────────────────────────────────
        # Reward acting early in the episode
        steps_used = sim.step_count  # already incremented before grade()
        efficiency = max(0.0, 1.0 - (steps_used / self.max_steps))

        # ── Completeness ──────────────────────────────────────────────────────
        total    = len(sim.data_records)
        handled  = sum(1 for r in sim.data_records if r.processed)
        completeness = handled / total if total > 0 else 0.0

        # ── Safety ────────────────────────────────────────────────────────────
        drift_before = obs_before.system_metrics.data_drift_score
        drift_after  = sim.metrics.data_drift_score
        safety = 1.0 if drift_after <= drift_before + 1e-6 else 0.0

        # ── Weighted final score ──────────────────────────────────────────────
        score = (
            correctness  * 0.60 +
            efficiency   * 0.10 +
            completeness * 0.20 +
            safety       * 0.10
        )
        score = round(min(1.0, max(0.0, score)), 4)

        breakdown = RewardBreakdown(
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            completeness=round(completeness, 4),
            safety=round(safety, 4),
        )
        return breakdown, feedback

    # ─── Parameter quality checker ────────────────────────────────────────────

    @staticmethod
    def _check_params(action: Action, gt_params: dict) -> float:
        """
        Score parameter quality for actions that require them.
        Returns 0.0–1.0.
        """
        if not gt_params:
            # No params expected → full marks if none given
            return 1.0 if not action.parameters else 0.8

        score_parts: list[float] = []

        # fill_value check (fix_null)
        if "fill_value" in gt_params:
            given = action.parameters.get("fill_value")
            expected = gt_params["fill_value"]
            if given is None:
                score_parts.append(0.0)
            elif type(given) == type(expected):  # noqa: E721
                score_parts.append(1.0)
            else:
                score_parts.append(0.5)  # Partial — right intent, wrong type

        # target_type check (cast_type)
        if "target_type" in gt_params:
            given = action.parameters.get("target_type", "")
            score_parts.append(1.0 if given == gt_params["target_type"] else 0.0)

        # field check (present in most actions)
        if "field" in gt_params:
            given = action.parameters.get("field", "")
            score_parts.append(1.0 if given == gt_params["field"] else 0.0)

        return sum(score_parts) / len(score_parts) if score_parts else 1.0