"""
server.py
=========
FastAPI server for MLOpsEnv.
Fix E: WebSocket endpoint added
Fix J: max_concurrent_envs via session isolation
Fix I: /mlops-state rich debug endpoint
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import MLOpsEnv
from env.models import (
    Action, MLOpsState, ResetResult,
    StateResult, StepResult, TaskID,
)

# ── Per-session store (Fix J) ──────────────────────────────────────────────
# Each session gets its own MLOpsEnv instance — no shared state
_sessions: dict[str, MLOpsEnv] = {}
MAX_SESSIONS = 16

# ── Single env for HTTP endpoints ──────────────────────────────────────────
_env = MLOpsEnv()
_episode_id = str(uuid.uuid4())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _env.reset(TaskID.DATA_TRIAGE)
    yield


app = FastAPI(
    title="MLOpsEnv",
    description="Production ML pipeline operations environment for RL agent training.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = TaskID.DATA_TRIAGE.value
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict[str, Any]


# ── Standard endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "MLOpsEnv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResult)
def reset(body: ResetRequest = ResetRequest()) -> ResetResult:
    global _episode_id
    try:
        _episode_id = str(uuid.uuid4())
        return _env.reset(body.task_id, seed=body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(body: StepRequest) -> StepResult:
    try:
        action = Action(**body.action)
        return _env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/state", response_model=StateResult)
def state() -> StateResult:
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return _env.available_tasks()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "MLOpsEnv",
        "version": "1.0.0",
        "tasks": _env.available_tasks(),
        "endpoints": {
            "POST /reset": "Start new episode",
            "POST /step": "Execute action",
            "GET /state": "Read current state",
            "GET /tasks": "List tasks",
            "GET /health": "Health check",
            "WS /ws": "WebSocket session",
            "GET /mlops-state": "Rich debug state",
        },
    }


# ── Fix I: Rich state endpoint ─────────────────────────────────────────────

@app.get("/mlops-state", response_model=MLOpsState)
def mlops_state() -> MLOpsState:
    try:
        sim = _env._sim
        if sim is None:
            return MLOpsState(task_id="none", step=0, episode_id=_episode_id)
        rewards = _env._episode_rewards
        return MLOpsState(
            task_id=sim.task_id.value,
            step=sim.step_count,
            episode_id=_episode_id,
            root_cause=getattr(sim, 'root_cause', None),
            alerts_resolved=sum(1 for a in sim.alerts if a.resolved),
            alerts_total=len(sim.alerts),
            last_action_type=sim.context_history[-1] if sim.context_history else None,
            last_reward=rewards[-1] if rewards else None,
            cum_reward=round(sum(rewards), 4),
            deployment_phase=getattr(sim, 'deployment_phase', None),
            seed=sim.seed,
        )
    except Exception:
        return MLOpsState(task_id="error", step=0, episode_id=_episode_id)


# ── Fix E: WebSocket endpoint ──────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket session — each connection gets isolated MLOpsEnv.
    Fix E: low-latency persistent sessions for RL training.
    Fix J: each WS connection = separate env instance (max 16).
    """
    import json

    if len(_sessions) >= MAX_SESSIONS:
        await websocket.close(code=1008)
        return

    session_id = str(uuid.uuid4())
    session_env = MLOpsEnv()
    _sessions[session_id] = session_env

    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
        })

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_id = data.get("task_id", "data_quality_triage")
                seed    = data.get("seed", None)
                result  = session_env.reset(task_id, seed=seed)
                await websocket.send_json({
                    "type": "reset",
                    "observation": result.observation.model_dump(mode="json"),
                })

            elif msg_type == "step":
                action = Action(**data.get("action", {}))
                result = session_env.step(action)
                await websocket.send_json({
                    "type": "step",
                    "observation": result.observation.model_dump(mode="json"),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                })

            elif msg_type == "state":
                result = session_env.state()
                await websocket.send_json({
                    "type": "state",
                    "observation": result.observation.model_dump(mode="json"),
                })

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)[:200],
            })
        except Exception:
            pass
    finally:
        _sessions.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass