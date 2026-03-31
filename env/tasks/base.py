"""
env/tasks/base.py
=================
Abstract base class every task must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from ..models import Action, Observation, RewardBreakdown, TaskID
from ..simulator import MLOpsSimulator


class BaseTask(ABC):

    task_id:    TaskID
    difficulty: str   # "easy" | "medium" | "hard"
    max_steps:  int
    seed:       int = 42

    @abstractmethod
    def build_simulator(self) -> MLOpsSimulator:
        """Return a fresh, seeded simulator for this task."""
        ...

    @abstractmethod
    def grade(
        self,
        action:    Action,
        sim:       MLOpsSimulator,
        obs_before: Observation,
    ) -> tuple[RewardBreakdown, str]:
        """
        Score a single action.
        Returns (RewardBreakdown, human-readable feedback string).
        Must be deterministic — no randomness, no external calls.
        """
        ...

    @abstractmethod
    def is_done(self, sim: MLOpsSimulator) -> bool:
        """Return True when the episode should terminate."""
        ...

    def task_summary(self) -> dict[str, Any]:
        return {
            "task_id":   self.task_id.value,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
        }