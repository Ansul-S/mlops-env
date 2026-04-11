"""
inference.py
============
Baseline inference script for MLOpsEnv.

Uses MLOpsEnvClient over HTTP — not direct class instantiation.
Supports IMAGE_NAME env var for Docker-based evaluation.

MANDATORY env vars:
  API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
  MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
  HF_TOKEN     = os.getenv("HF_TOKEN")

STDOUT FORMAT:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import os
import sys
import json
import textwrap
from typing import Any, List, Optional

# ── Env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
IMAGE_NAME   = os.getenv("IMAGE_NAME")        # Fix R: Docker image support
SPACE_URL    = os.getenv("SPACE_URL", "http://localhost:8000")

FALLBACK_MODE     = not bool(HF_TOKEN)
BENCHMARK         = "mlops-env"
MAX_STEPS         = 30
MAX_TOTAL_REWARD  = MAX_STEPS * 1.0
SUCCESS_THRESHOLD = 0.5

TASKS = ["data_quality_triage", "deployment_decision", "incident_cascade"]

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI     = None
    _OPENAI_OK = False

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import MLOpsEnvClient
    _CLIENT_OK = True
except Exception:
    MLOpsEnvClient = None
    _CLIENT_OK     = False


# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task: str, env_name: str, model: str) -> None:
    sys.stdout.write(f"[START] task={task} env={env_name} model={model}\n")
    sys.stdout.flush()

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    reward = max(0.0051, min(0.9949, float(reward)))
    err    = str(error)[:60] if error else "null"
    done_s = "true" if done else "false"
    sys.stdout.write(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_s} error={err}\n"
    )
    sys.stdout.flush()

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    if not rewards:
        rewards = [0.01]
    rewards = [max(0.0051, min(0.9949, float(r))) for r in rewards]
    score   = max(0.0051, min(0.9949, float(score)))
    rstr    = ",".join(f"{r:.2f}" for r in rewards)
    succ    = "true" if success else "false"
    sys.stdout.write(
        f"[END] success={succ} steps={steps} score={score:.2f} rewards={rstr}\n"
    )
    sys.stdout.flush()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert MLOps engineer. Choose exactly one action per step.
Reply with valid JSON only — no markdown:
{"action_type":"<action>","target_id":"<id or null>","parameters":{},"reasoning":"<reason>"}

data_quality_triage: fix_null, remove_outlier, cast_type, flag_duplicate, accept_record
deployment_decision: deploy_canary, deploy_full, rollback, hold
incident_cascade: investigate, restart_service, reroute_traffic, rollback_model, escalate

For incidents investigate root cause FIRST. For deployment check SLA error_rate constraint.
""").strip()


# ── LLM action ────────────────────────────────────────────────────────────────

def llm_action(client, task_id: str, obs: dict) -> dict:
    try:
        ctx   = obs.get("task_context", "")
        hist  = obs.get("context_history", [])
        avail = obs.get("available_actions", [])
        m     = obs.get("system_metrics", {})
        msg   = (
            f"TASK:{task_id} step:{obs.get('step',0)}/{obs.get('max_steps',30)}\n"
            f"SITUATION:{ctx}\n"
            f"METRICS:lat={m.get('latency_p99_ms')} err={m.get('error_rate_pct')}% "
            f"acc={m.get('model_accuracy')}\n"
            f"RECENT:{hist[-2:]}\nAVAIL:{avail}\nJSON:"
        )
        resp  = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text  = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        data  = json.loads(text)
        return {
            "action_type": data.get("action_type", avail[0] if avail else "hold"),
            "target_id":   data.get("target_id"),
            "parameters":  data.get("parameters", {}),
            "reasoning":   data.get("reasoning", ""),
        }
    except Exception:
        avail = obs.get("available_actions", [])
        return {
            "action_type": avail[0] if avail else "hold",
            "target_id": None,
            "parameters": {},
            "reasoning": "fallback"
        }


# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(
    env,
    llm_client,
    task_id: str,
) -> None:
    rewards:     List[float]   = []
    steps_taken  = 0
    score        = 0.0
    success      = False

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        # Fix O: use async client over HTTP
        result      = await env.reset(task_id)
        obs         = result.get("observation", {})

        for step_num in range(1, MAX_STEPS + 1):
            # Choose action
            if FALLBACK_MODE or llm_client is None:
                avail  = obs.get("available_actions", [])
                action = {
                    "action_type": avail[0] if avail else "hold",
                    "target_id": None,
                    "parameters": {},
                    "reasoning": "baseline fallback"
                }
            else:
                action = llm_action(llm_client, task_id, obs)

            action_str = action.get("action_type", "unknown")
            error_str: Optional[str] = None

            # Step via HTTP
            try:
                step_result = await env.step(action)
                reward      = float(step_result.get("reward", 0.0))
                done        = bool(step_result.get("done", False))
                obs         = step_result.get("observation", obs)
                # Fix Q: read env-level action errors
                error_str   = obs.get("last_action_error") or None
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_str = str(exc)[:60]

            rewards.append(reward)
            steps_taken = step_num
            log_step(step_num, action_str, reward, done, error_str)

            if done:
                break

        # Fix P: normalize against MAX_TOTAL_REWARD
        score   = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
        score   = max(0.0051, min(0.9949, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        err = str(exc)[:60]
        if steps_taken == 0:
            rewards     = [0.01]
            steps_taken = 1
            log_step(1, "none", 0.01, True, err)
        score   = sum(rewards) / MAX_TOTAL_REWARD
        score   = max(0.0051, min(0.9949, score))
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Build LLM client
    llm_client = None
    if not FALLBACK_MODE and _OPENAI_OK and HF_TOKEN:
        try:
            llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            pass

    # Fix O + R: connect to env via HTTP client
    if not _CLIENT_OK:
        # Fallback: use local env directly if client.py unavailable
        try:
            from env import MLOpsEnv
            from env.models import Action

            class _LocalClient:
                def __init__(self):
                    self._env = MLOpsEnv()
                async def reset(self, task_id, seed=None):
                    r = self._env.reset(task_id, seed=seed)
                    return r.model_dump(mode='json')
                async def step(self, action_dict):
                    from env.models import Action
                    a = Action(**action_dict)
                    r = self._env.step(a)
                    return r.model_dump(mode='json')
                async def close(self): pass

            env = _LocalClient()
        except Exception as e:
            # Ultimate fallback - can't run
            for task_id in TASKS:
                log_start(task_id, BENCHMARK, MODEL_NAME)
                log_step(1, "none", 0.01, True, str(e)[:60])
                log_end(False, 1, 0.01, [0.01])
            return
    else:
        # Fix R: use Docker image if IMAGE_NAME is set
        if IMAGE_NAME:
            env = await MLOpsEnvClient.from_docker_image(IMAGE_NAME)
        else:
            env = MLOpsEnvClient(base_url=SPACE_URL)

    try:
        for task_id in TASKS:
            await run_task(env, llm_client, task_id)
    finally:
        try:
            await env.close()
        except Exception:
            pass

    # Save scores
    try:
        with open("baseline_scores.json", "w") as f:
            json.dump({"model": MODEL_NAME, "tasks": TASKS}, f)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())