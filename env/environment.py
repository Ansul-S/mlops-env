"""
env/environment.py
==================
MLOpsEnv — Core OpenEnv Interface

Implements the full OpenEnv spec:
  reset(task_id)  → ResetResult
  step(action)    → StepResult
  state()         → StateResult

Call order per step (enforced here):
  1. Validate action is in available_actions
  2. Capture obs_before (pre-action snapshot)
  3. sim.apply_action(action)   → mutates state + returns done
  4. task.grade(action, sim, obs_before) → RewardBreakdown
  5. Compute weighted score
  6. Build and return StepResult
"""

from __future__ import annotations

from typing import Any

from .models import (
    Action, ActionType, EpisodeResult, Observation,
    Reward, RewardBreakdown, ResetResult, StateResult,
    StepResult, TaskID,
)
from .simulator import MLOpsSimulator
from .tasks import TASK_REGISTRY
from .tasks.base import BaseTask


class MLOpsEnv:
    """
    Production ML pipeline operations environment.

    Three tasks of increasing difficulty:
      - data_quality_triage  (easy)   — fix 20 data records before training
      - deployment_decision  (medium) — choose safe deployment strategy under SLA
      - incident_cascade     (hard)   — diagnose + resolve a 3-alert incident

    Usage:
        env = MLOpsEnv()
        result = env.reset("data_quality_triage")
        obs = result.observation

        while not done:
            action = agent.act(obs)
            result = env.step(action)
            obs, reward, done = result.observation, result.reward, result.done
    """

    # Reward weights — consistent across all tasks
    WEIGHTS = {
        "correctness":  0.50,
        "efficiency":   0.15,
        "completeness": 0.25,
        "safety":       0.10,
    }

    def __init__(self) -> None:
        self._sim:  MLOpsSimulator | None = None
        self._task: BaseTask | None       = None
        self._task_id: TaskID | None      = None
        self._episode_rewards: list[float]           = []
        self._episode_breakdowns: list[dict[str, float]] = []
        self._done: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv spec — primary interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str | TaskID = TaskID.DATA_TRIAGE) -> ResetResult:
        """
        Reset environment for a new episode.

        Args:
            task_id: One of "data_quality_triage", "deployment_decision",
                     "incident_cascade" (or TaskID enum).

        Returns:
            ResetResult containing the initial Observation.
        """
        # Normalize task_id
        if isinstance(task_id, str):
            try:
                task_id = TaskID(task_id)
            except ValueError:
                valid = [t.value for t in TaskID]
                raise ValueError(
                    f"Unknown task_id '{task_id}'. Valid options: {valid}"
                )

        self._task_id = task_id
        task_cls      = TASK_REGISTRY[task_id]
        self._task    = task_cls()
        self._sim     = self._task.build_simulator()

        self._episode_rewards    = []
        self._episode_breakdowns = []
        self._done               = False

        return ResetResult(observation=self._sim.get_observation())

    def step(self, action: Action | dict) -> StepResult:
        """
        Execute one action in the environment.

        Args:
            action: Action model (or dict that will be coerced to Action).

        Returns:
            StepResult with (observation, reward, done, info).

        Raises:
            RuntimeError: If called before reset().
            ValueError:   If action_type not in available_actions.
        """
        self._require_reset()

        # Coerce dict → Action
        if isinstance(action, dict):
            action = Action(**action)

        # Validate action is legal in current task
        self._validate_action(action)

        if self._done:
            raise RuntimeError(
                "Episode already finished. Call reset() to start a new episode."
            )

        # ── 1. Snapshot pre-action observation ───────────────────────────────
        obs_before = self._sim.get_observation()

        # ── 2. Apply action → mutate simulator state ──────────────────────────
        done_from_sim = self._sim.apply_action(action)

        # ── 3. Grade the action ───────────────────────────────────────────────
        breakdown, feedback = self._task.grade(action, self._sim, obs_before)

        # ── 4. Compute weighted scalar score ─────────────────────────────────
        score = self._compute_score(breakdown)

        # ── 5. Determine episode termination ─────────────────────────────────
        task_done    = self._task.is_done(self._sim)
        truncated    = self._sim.step_count >= self._sim.max_steps
        done         = done_from_sim or task_done or truncated
        self._done   = done

        # ── 6. Record history ─────────────────────────────────────────────────
        self._episode_rewards.append(score)
        self._episode_breakdowns.append(breakdown.model_dump())

        # ── 7. Build info dict ────────────────────────────────────────────────
        info: dict[str, Any] = {
            "feedback":   feedback,
            "breakdown":  breakdown.model_dump(),
            "step":       self._sim.step_count,
            "truncated":  truncated,
        }

        if done:
            episode_result = self._build_episode_result()
            info["episode"] = episode_result.model_dump()

        # ── 8. Return result ──────────────────────────────────────────────────
        reward_obj = Reward(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            done=done,
            truncated=truncated,
        )

        return StepResult(
            observation=self._sim.get_observation(),
            reward=score,
            done=done,
            info=info,
        )

    def state(self) -> StateResult:
        """
        Return current environment state without advancing the episode.

        Returns:
            StateResult containing the current Observation (read-only snapshot).

        Raises:
            RuntimeError: If called before reset().
        """
        self._require_reset()
        return StateResult(observation=self._sim.get_observation())

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    def available_tasks(self) -> list[dict[str, Any]]:
        """Return metadata for all registered tasks."""
        return [
            {
                "task_id":    tid.value,
                "difficulty": cls().difficulty,
                "max_steps":  cls().max_steps,
            }
            for tid, cls in TASK_REGISTRY.items()
        ]

    def current_task_info(self) -> dict[str, Any] | None:
        """Return current task metadata, or None if not reset yet."""
        if self._task is None:
            return None
        return self._task.task_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_reset(self) -> None:
        if self._sim is None or self._task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset(task_id) first."
            )

    def _validate_action(self, action: Action) -> None:
        available = self._sim.get_observation().available_actions
        if action.action_type not in available:
            raise ValueError(
                f"Action '{action.action_type.value}' is not available in task "
                f"'{self._task_id.value}'. Available: {[a.value for a in available]}"
            )

    def _compute_score(self, breakdown: RewardBreakdown) -> float:
        """Weighted sum of sub-scores → scalar in [0, 1]."""
        score = (
            breakdown.correctness  * self.WEIGHTS["correctness"]  +
            breakdown.efficiency   * self.WEIGHTS["efficiency"]   +
            breakdown.completeness * self.WEIGHTS["completeness"] +
            breakdown.safety       * self.WEIGHTS["safety"]
        )
        # Clamp strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0
        # Clamp strictly so %.2f formatting never shows 0.00 or 1.00
        score = max(0.0051, min(0.9949, score))
        return round(score, 4)

    def _build_episode_result(self) -> EpisodeResult:
        total_score = (
            sum(self._episode_rewards) / len(self._episode_rewards)
            if self._episode_rewards else 0.0
        )
        resolved_alerts = sum(
            1 for a in self._sim.alerts if a.resolved
        ) if self._sim.alerts else 0

        processed_records = sum(
            1 for r in self._sim.data_records if r.processed
        ) if self._sim.data_records else 0

        summary_parts = [
            f"Task: {self._task_id.value}.",
            f"Steps: {self._sim.step_count}/{self._sim.max_steps}.",
            f"Avg score: {total_score:.4f}.",
        ]
        if self._sim.data_records:
            summary_parts.append(
                f"Records processed: {processed_records}/{len(self._sim.data_records)}."
            )
        if self._sim.alerts:
            summary_parts.append(
                f"Alerts resolved: {resolved_alerts}/{len(self._sim.alerts)}."
            )
        if self._sim.deployment_action_taken:
            summary_parts.append(
                f"Deployment action: {self._sim.deployment_action_taken}."
            )

        return EpisodeResult(
            task_id=self._task_id,
            total_score=round(total_score, 4),
            steps_taken=self._sim.step_count,
            reward_history=list(self._episode_rewards),
            breakdown_history=list(self._episode_breakdowns),
            final_state_summary=" ".join(summary_parts),
        )