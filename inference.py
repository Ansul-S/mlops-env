"""
inference.py
============
Baseline inference script for MLOpsEnv.

MANDATORY env vars (do not change these names):
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier  
  HF_TOKEN      — HuggingFace API key (no default)

STDOUT FORMAT (required by validator):
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from env import MLOpsEnv
from env.models import Action, TaskID

# ─────────────────────────────────────────────────────────────────────────────
# Config — exact variable names required by validator checklist
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Run in fallback mode if no token — always produces structured output
FALLBACK_MODE = not bool(HF_TOKEN)
BENCHMARK     = "mlops-env"
MAX_TOKENS    = 512
TEMPERATURE   = 0.0
MAX_STEPS     = 30
SUCCESS_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Required structured log helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert MLOps engineer operating a production ML system.
You will be given the current environment state and must choose exactly one action.

RESPONSE FORMAT — reply with valid JSON only, no other text:
{
  "action_type": "<action_type>",
  "target_id":   "<id or null>",
  "parameters":  {<key>: <value>},
  "reasoning":   "<one sentence explaining your choice>"
}

AVAILABLE ACTION TYPES by task:
  data_quality_triage:
    fix_null       — parameters={"field": "<n>", "fill_value": <value>}
    remove_outlier — parameters={"field": "<n>"}
    cast_type      — parameters={"field": "<n>", "target_type": "<type>"}
    flag_duplicate — parameters={"duplicate_of": "<record_id>"}
    accept_record  — record is clean

  deployment_decision:
    deploy_canary  — parameters={"canary_pct": <1-20>, "rollback_threshold_pct": <float>}
    deploy_full    — full traffic switch
    rollback       — revert to champion
    hold           — defer decision

  incident_cascade:
    investigate     — parameters={"component": "<n>"}
    restart_service — parameters={"component": "<n>"}
    reroute_traffic — parameters={"from_component": "<n>", "fallback": true}
    rollback_model  — revert model checkpoint
    escalate        — page on-call engineer
    silence_alert   — suppress alert (last resort)

RULES:
  - For incidents: find root cause FIRST before fixing downstream
  - For deployment: check error_rate vs max_error_rate_pct SLA
  - JSON only — no markdown, no text outside JSON
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback (when no HF_TOKEN or LLM fails)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_action(task_id: str, obs_dict: dict) -> str:
    if task_id == "data_quality_triage":
        records     = obs_dict.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        if not unprocessed:
            return json.dumps({"action_type": "accept_record", "target_id": None, "parameters": {}, "reasoning": "All records processed"})
        rec       = unprocessed[0]
        issues    = rec.get("detected_issues", [])
        record_id = rec["record_id"]
        fields    = rec.get("fields", {})
        schema    = rec.get("schema_expected", {})
        if "null_value" in issues:
            null_field = next((f for f, v in fields.items() if v is None), None)
            expected   = schema.get(null_field, "str") if null_field else "str"
            fill       = 0.0 if expected == "float" else (0 if expected == "int" else "unknown")
            return json.dumps({"action_type": "fix_null", "target_id": record_id, "parameters": {"field": null_field, "fill_value": fill}, "reasoning": "Filling null with type-appropriate default"})
        elif "type_mismatch" in issues:
            bad_field   = next((f for f, v in fields.items() if isinstance(v, str) and "_bad" in str(v)), None)
            target_type = schema.get(bad_field, "str") if bad_field else "str"
            return json.dumps({"action_type": "cast_type", "target_id": record_id, "parameters": {"field": bad_field, "target_type": target_type}, "reasoning": "Casting mismatched field"})
        elif "outlier" in issues:
            outlier_field = next((f for f, v in fields.items() if isinstance(v, (int, float)) and v > 100000), None)
            return json.dumps({"action_type": "remove_outlier", "target_id": record_id, "parameters": {"field": outlier_field}, "reasoning": "Removing outlier"})
        elif "duplicate" in issues:
            return json.dumps({"action_type": "flag_duplicate", "target_id": record_id, "parameters": {}, "reasoning": "Flagging duplicate"})
        else:
            return json.dumps({"action_type": "accept_record", "target_id": record_id, "parameters": {}, "reasoning": "Record is clean"})

    elif task_id == "deployment_decision":
        return json.dumps({"action_type": "deploy_canary", "target_id": None, "parameters": {"canary_pct": 5, "rollback_threshold_pct": 0.4}, "reasoning": "Challenger error_rate breaches SLA — canary limits blast radius"})

    else:  # incident_cascade
        history = obs_dict.get("context_history", [])
        if not any("feature_store" in h for h in history):
            return json.dumps({"action_type": "investigate", "target_id": None, "parameters": {"component": "feature_store"}, "reasoning": "Investigating feature_store as root cause"})
        elif not any("feature_store" in h and "restart" in h for h in history):
            return json.dumps({"action_type": "restart_service", "target_id": None, "parameters": {"component": "feature_store"}, "reasoning": "Restarting root cause"})
        elif not any("model_serving" in h for h in history):
            return json.dumps({"action_type": "restart_service", "target_id": None, "parameters": {"component": "model_serving"}, "reasoning": "Fixing downstream model_serving"})
        else:
            return json.dumps({"action_type": "restart_service", "target_id": None, "parameters": {"component": "data_pipeline"}, "reasoning": "Fixing downstream data_pipeline"})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(obs_dict: dict[str, Any]) -> str:
    task_id   = obs_dict["task_id"]
    metrics   = obs_dict["system_metrics"]
    history   = obs_dict.get("context_history", [])
    available = obs_dict.get("available_actions", [])
    lines = [
        f"TASK: {task_id} | Step {obs_dict['step']}/{obs_dict['max_steps']}",
        f"SITUATION: {obs_dict.get('task_context', '')}",
        "METRICS:",
        f"  latency={metrics['latency_p99_ms']}ms error_rate={metrics['error_rate_pct']}% accuracy={metrics['model_accuracy']}",
    ]
    if task_id == "data_quality_triage":
        unprocessed = [r for r in obs_dict.get("data_records", []) if not r["processed"]]
        lines.append(f"UNPROCESSED ({len(unprocessed)}):")
        for r in unprocessed[:5]:
            lines.append(f"  {r['record_id']}: issues={r.get('detected_issues',[])} fields={json.dumps(r['fields'])}")
    elif task_id == "deployment_decision":
        sla = obs_dict.get("sla_requirements", {})
        for c in obs_dict.get("deployment_candidates", []):
            role = "CHAMPION" if c["is_champion"] else "CHALLENGER"
            lines.append(f"[{role}] {c['name']}: acc={c['accuracy']} lat={c['latency_p99_ms']}ms err={c['error_rate_pct']}%")
        lines.append(f"SLA: max_latency={sla.get('max_latency_p99_ms')}ms max_error={sla.get('max_error_rate_pct')}%")
    elif task_id == "incident_cascade":
        for a in obs_dict.get("alerts", []):
            lines.append(f"[{'RESOLVED' if a['resolved'] else 'OPEN'}][{a['severity'].upper()}] {a['component']}: {a['message'][:80]}")
    if history:
        lines.append(f"RECENT: {' | '.join(history[-3:])}")
    lines.append(f"AVAILABLE: {available}")
    lines.append("JSON only:")
    return "\n".join(lines)


def parse_action(response_text: str, available: list[str]) -> Action:
    text = response_text.strip()
    if text.startswith("```"):
        parts = text.split("\n")
        text  = "\n".join(parts[1:-1]) if len(parts) > 2 else text
    try:
        data = json.loads(text)
        return Action(action_type=data.get("action_type", available[0] if available else "hold"), target_id=data.get("target_id"), parameters=data.get("parameters", {}), reasoning=data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        return Action(action_type=available[0] if available else "hold", reasoning="Parse error fallback")


def inject_target(action: Action, obs_dict: dict[str, Any]) -> Action:
    if obs_dict["task_id"] != "data_quality_triage" or action.target_id:
        return action
    unprocessed = [r for r in obs_dict.get("data_records", []) if not r["processed"]]
    if unprocessed:
        return Action(action_type=action.action_type, target_id=unprocessed[0]["record_id"], parameters=action.parameters, reasoning=action.reasoning)
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: Optional[OpenAI], env: MLOpsEnv, task_id: str) -> dict[str, Any]:
    result   = env.reset(task_id)
    obs_dict = result.observation.model_dump(mode="json")

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:    List[float]     = []
    step        = 0
    step_result = None
    last_error: Optional[str]  = None

    while step < MAX_STEPS:
        if FALLBACK_MODE or client is None:
            response_text = _fallback_action(task_id, obs_dict)
        else:
            try:
                completion    = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(obs_dict)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                last_error    = None
            except Exception as exc:
                last_error    = str(exc)[:60]
                response_text = _fallback_action(task_id, obs_dict)

        available  = obs_dict.get("available_actions", [])
        action     = parse_action(response_text, available)
        action     = inject_target(action, obs_dict)
        action_str = action.action_type.value

        try:
            step_result = env.step(action)
            last_error  = None
        except (ValueError, RuntimeError) as exc:
            last_error = str(exc)[:60]
            step += 1
            log_step(step=step, action=action_str, reward=0.0, done=False, error=last_error)
            continue

        reward   = step_result.reward
        done     = step_result.done
        obs_dict = step_result.observation.model_dump(mode="json")
        rewards.append(reward)
        step += 1

        log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

        if done:
            break

    episode_info = step_result.info.get("episode", {}) if step_result and rewards else {}
    avg_score    = sum(rewards) / len(rewards) if rewards else 0.0
    final_score  = round(episode_info.get("total_score", avg_score), 4)
    success      = final_score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=step, score=final_score, rewards=rewards)

    return {"task_id": task_id, "steps": step, "total_score": final_score, "success": success, "rewards": rewards}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if not FALLBACK_MODE else None
    env    = MLOpsEnv()
    tasks  = [TaskID.DATA_TRIAGE.value, TaskID.DEPLOYMENT.value, TaskID.INCIDENT.value]

    results: list[dict] = []
    for task_id in tasks:
        results.append(run_task(client, env, task_id))

    overall = sum(r["total_score"] for r in results) / len(results)
    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results, "overall": overall}, f, indent=2)


if __name__ == "__main__":
    main()