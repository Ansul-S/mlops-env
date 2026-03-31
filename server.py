"""
server.py
=========
FastAPI server for MLOpsEnv.

Endpoints (all required by OpenEnv validator):
  POST /reset        → ResetResult
  POST /step         → StepResult
  GET  /state        → StateResult
  GET  /tasks        → list of available tasks
  GET  /health       → health check

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 7860

The HuggingFace Space validator pings POST /reset and expects HTTP 200.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import MLOpsEnv
from env.models import Action, ResetResult, StateResult, StepResult, TaskID


# ─────────────────────────────────────────────────────────────────────────────
# Request bodies
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = TaskID.DATA_TRIAGE.value


class StepRequest(BaseModel):
    action: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# App + global env instance
# ─────────────────────────────────────────────────────────────────────────────

# Single shared environment instance (stateful per session)
_env: MLOpsEnv = MLOpsEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-reset to a clean state on startup
    _env.reset(TaskID.DATA_TRIAGE)
    yield


app = FastAPI(
    title="MLOpsEnv",
    description=(
        "Production ML pipeline operations environment. "
        "Agent manages data quality, deployment decisions, "
        "and incident response under real operational constraints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Health check — always returns 200 if server is up."""
    return {"status": "ok", "env": "MLOpsEnv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResult)
def reset(body: ResetRequest = ResetRequest()) -> ResetResult:
    """
    Reset the environment for a new episode.

    Body (optional):
        task_id: "data_quality_triage" | "deployment_decision" | "incident_cascade"
                 Defaults to "data_quality_triage".

    Returns initial observation.
    """
    try:
        result = _env.reset(body.task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(body: StepRequest) -> StepResult:
    """
    Execute one action in the environment.

    Body:
        action: {
            "action_type": str,          # required
            "target_id":   str | null,   # optional
            "parameters":  dict,         # optional
            "reasoning":   str           # optional (graded in hard task)
        }

    Returns observation, reward (0.0–1.0), done, info.
    """
    try:
        action = Action(**body.action)
        result = _env.step(action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/state", response_model=StateResult)
def state() -> StateResult:
    """
    Return current environment state without advancing the episode.
    Read-only — does not consume a step.
    """
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    """List all available tasks with metadata."""
    return _env.available_tasks()


@app.get("/")
def root() -> dict[str, Any]:
    """Root endpoint — returns env info and available endpoints."""
    return {
        "name":        "MLOpsEnv",
        "version":     "1.0.0",
        "description": (
            "Production ML pipeline operations environment for RL agent training."
        ),
        "tasks": _env.available_tasks(),
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step":  "Execute an action",
            "GET  /state": "Read current state",
            "GET  /tasks": "List available tasks",
            "GET  /health": "Health check",
        },
    }
