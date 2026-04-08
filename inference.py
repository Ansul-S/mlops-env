"""
inference.py
============
Baseline inference script for MLOpsEnv.

Calls the deployed HF Space environment via HTTP.
No local env/ package dependency — works in any Python environment.

MANDATORY env vars:
  API_BASE_URL  — LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier (default: meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN      — HuggingFace API key (no default)

STDOUT FORMAT:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.request
import urllib.error
from typing import Any, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config — exact variable names required by validator
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Environment server URL — our deployed HF Space
ENV_URL      = os.getenv("ENV_URL", "https://ansul-s-mlops-env.hf.space")

FALLBACK_MODE     = not bool(HF_TOKEN)
BENCHMARK         = "mlops-env"
MAX_TOKENS        = 512
TEMPERATURE       = 0.0
MAX_STEPS         = 30
SUCCESS_THRESHOLD = 0.5

TASKS = [
    "data_quality_triage",
    "deployment_decision",
    "incident_cascade",
]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client for environment (calls HF Space endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _http_post(path: str, body: dict) -> dict:
    url     = f"{ENV_URL.rstrip('/')}{path}"
    data    = json.dumps(body).encode("utf-8")
    req     = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(path: str) -> dict:
    url = f"{ENV_URL.rstrip('/')}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def env_reset(task_id: str) -> dict:
    return _http_post("/reset", {"task_id": task_id})


def env_step(action: dict) -> dict:
    return _http_post("/step", {"action": action})


# ─────────────────────────────────────────────────────────────────────────────
# Required structured log helpers  (exact validator format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert MLOps engineer. Choose exactly one action per step.
Reply with valid JSON only:
{
  "action_type": "<action>",
  "target_id": "<id or null>",
  "parameters": {},
  "reasoning": "<one sentence>"
}

data_quality_triage actions:
  fix_null, remove_outlier, cast_type, flag_duplicate, accept_record

deployment_decision actions:
  deploy_canary (parameters: canary_pct, rollback_threshold_pct), deploy_full, rollback, hold

incident_cascade actions:
  investigate (parameters: component), restart_service (parameters: component),
  reroute_traffic, rollback_model, escalate, silence_alert

For incidents: investigate root cause FIRST. For deployment: check SLA constraints.
JSON only — no markdown.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback actions (when no HF_TOKEN or LLM fails)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_action(task_id: str, obs: dict) -> dict:
    if task_id == "data_quality_triage":
        records     = obs.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        if not unprocessed:
            return {"action_type": "accept_record", "target_id": None, "parameters": {}, "reasoning": "All done"}
        rec       = unprocessed[0]
        issues    = rec.get("detected_issues", [])
        record_id = rec["record_id"]
        fields    = rec.get("fields", {})
        schema    = rec.get("schema_expected", {})
        if "null_value" in issues:
            null_field = next((f for f, v in fields.items() if v is None), None)
            expected   = schema.get(null_field, "str") if null_field else "str"
            fill       = 0.0 if expected == "float" else (0 if expected == "int" else "unknown")
            return {"action_type": "fix_null", "target_id": record_id, "parameters": {"field": null_field, "fill_value": fill}, "reasoning": "Filling null"}
        elif "type_mismatch" in issues:
            bad_field   = next((f for f, v in fields.items() if isinstance(v, str) and "_bad" in str(v)), None)
            target_type = schema.get(bad_field, "str") if bad_field else "str"
            return {"action_type": "cast_type", "target_id": record_id, "parameters": {"field": bad_field, "target_type": target_type}, "reasoning": "Fixing type mismatch"}
        elif "outlier" in issues:
            outlier_field = next((f for f, v in fields.items() if isinstance(v, (int, float)) and v > 100000), None)
            return {"action_type": "remove_outlier", "target_id": record_id, "parameters": {"field": outlier_field}, "reasoning": "Removing outlier"}
        elif "duplicate" in issues:
            return {"action_type": "flag_duplicate", "target_id": record_id, "parameters": {}, "reasoning": "Flagging duplicate"}
        else:
            return {"action_type": "accept_record", "target_id": record_id, "parameters": {}, "reasoning": "Record is clean"}

    elif task_id == "deployment_decision":
        return {"action_type": "deploy_canary", "target_id": None, "parameters": {"canary_pct": 5, "rollback_threshold_pct": 0.4}, "reasoning": "Canary deploy — challenger error_rate breaches SLA"}

    else:  # incident_cascade
        history = obs.get("context_history", [])
        if not any("feature_store" in h for h in history):
            return {"action_type": "investigate", "target_id": None, "parameters": {"component": "feature_store"}, "reasoning": "Investigating root cause"}
        elif not any("feature_store" in h and "restart" in h for h in history):
            return {"action_type": "restart_service", "target_id": None, "parameters": {"component": "feature_store"}, "reasoning": "Restarting root cause"}
        elif not any("model_serving" in h for h in history):
            return {"action_type": "restart_service", "target_id": None, "parameters": {"component": "model_serving"}, "reasoning": "Fixing downstream"}
        else:
            return {"action_type": "restart_service", "target_id": None, "parameters": {"component": "data_pipeline"}, "reasoning": "Fixing downstream"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM action  
# ─────────────────────────────────────────────────────────────────────────────

def _llm_action(client: OpenAI, task_id: str, obs: dict) -> dict:
    metrics  = obs.get("system_metrics", {})
    context  = obs.get("task_context", "")
    history  = obs.get("context_history", [])
    available = obs.get("available_actions", [])

    lines = [
        f"TASK: {task_id} | Step {obs.get('step',0)}/{obs.get('max_steps',30)}",
        f"SITUATION: {context}",
        f"METRICS: latency={metrics.get('latency_p99_ms')}ms "
        f"error_rate={metrics.get('error_rate_pct')}% "
        f"accuracy={metrics.get('model_accuracy')}",
    ]
    if task_id == "data_quality_triage":
        unprocessed = [r for r in obs.get("data_records", []) if not r["processed"]]
        lines.append(f"UNPROCESSED ({len(unprocessed)}):")
        for r in unprocessed[:4]:
            lines.append(f"  {r['record_id']}: issues={r.get('detected_issues',[])} fields={json.dumps(r['fields'])}")
    elif task_id == "deployment_decision":
        sla = obs.get("sla_requirements", {})
        for c in obs.get("deployment_candidates", []):
            role = "CHAMPION" if c["is_champion"] else "CHALLENGER"
            lines.append(f"[{role}] {c['name']}: acc={c['accuracy']} lat={c['latency_p99_ms']}ms err={c['error_rate_pct']}%")
        lines.append(f"SLA: max_err={sla.get('max_error_rate_pct')}% max_lat={sla.get('max_latency_p99_ms')}ms")
    elif task_id == "incident_cascade":
        for a in obs.get("alerts", []):
            status = "RESOLVED" if a["resolved"] else "OPEN"
            lines.append(f"[{status}][{a['severity'].upper()}] {a['component']}: {a['message'][:80]}")
    if history:
        lines.append(f"RECENT: {' | '.join(history[-3:])}")
    lines.append(f"AVAILABLE: {available}")
    lines.append("JSON only:")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            parts = text.split("\n")
            text  = "\n".join(parts[1:-1]) if len(parts) > 2 else text
        data = json.loads(text)
        return {
            "action_type": data.get("action_type", available[0] if available else "hold"),
            "target_id":   data.get("target_id"),
            "parameters":  data.get("parameters", {}),
            "reasoning":   data.get("reasoning", ""),
        }
    except Exception:
        return _fallback_action(task_id, obs)


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: Optional[OpenAI], task_id: str) -> None:
    rewards:    List[float]    = []
    steps_taken = 0
    score        = 0.0
    success      = False

    # ── [START] printed FIRST — before anything can fail ─────────────────────
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment via HTTP
        reset_result = env_reset(task_id)
        obs          = reset_result.get("observation", {})

        for step_num in range(1, MAX_STEPS + 1):
            # Choose action
            if FALLBACK_MODE or client is None:
                action = _fallback_action(task_id, obs)
            else:
                action = _llm_action(client, task_id, obs)

            action_str = action.get("action_type", "unknown")
            error_str: Optional[str] = None

            # Step environment via HTTP
            try:
                step_result = env_step(action)
                reward      = float(step_result.get("reward", 0.0))
                done        = bool(step_result.get("done", False))
                obs         = step_result.get("observation", obs)
            except Exception as exc:
                reward     = 0.0
                done       = False
                error_str  = str(exc)[:60]

            rewards.append(reward)
            steps_taken = step_num

            # ── [STEP] printed immediately after env.step() ───────────────────
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        # Compute final score
        if rewards:
            score   = round(sum(rewards) / len(rewards), 4)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # Even on total failure — we still print [END]
        error_msg = str(exc)[:80]
        if not rewards:
            log_step(step=1, action="none", reward=0.0, done=True, error=error_msg)
            steps_taken = 1
        score   = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        success = False

    finally:
        # ── [END] ALWAYS printed — even on exception ─────────────────────────
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if not FALLBACK_MODE else None

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()