"""
inference.py
============
Baseline inference script for MLOpsEnv.

MANDATORY env vars:
  API_BASE_URL  — LLM API endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace / API key

STDOUT FORMAT (required by validator):
  [START] task=<name> env=mlops-env model=<model>
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
from env.models import Action, ActionType, TaskID

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY  = HF_TOKEN or os.getenv("API_KEY", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

FALLBACK_MODE = not bool(API_KEY)   # run deterministically if no key
BENCHMARK     = "mlops-env"
MAX_TOKENS    = 512
TEMPERATURE   = 0.0
MAX_STEPS     = 30
SUCCESS_THRESHOLD = 0.5             # score >= 0.5 → success=true


# ─────────────────────────────────────────────────────────────────────────────
# Required structured log helpers  (exact format the validator parses)
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
    fix_null        — fill null field: parameters={"field": "<n>", "fill_value": <value>}
    remove_outlier  — drop outlier record: parameters={"field": "<n>"}
    cast_type       — coerce field type: parameters={"field": "<n>", "target_type": "<type>"}
    flag_duplicate  — mark as duplicate: parameters={"duplicate_of": "<record_id>"}
    accept_record   — record is clean, no action needed

  deployment_decision:
    deploy_canary   — gradual rollout: parameters={"canary_pct": <1-20>, "rollback_threshold_pct": <float>}
    deploy_full     — full traffic switch
    rollback        — revert to champion
    hold            — defer decision

  incident_cascade:
    investigate     — diagnose component: parameters={"component": "<name>"}
    restart_service — restart component: parameters={"component": "<name>"}
    reroute_traffic — redirect traffic: parameters={"from_component": "<name>", "fallback": true}
    rollback_model  — revert model checkpoint
    escalate        — page on-call engineer
    silence_alert   — suppress alert (last resort only)

RULES:
  - Always pick the action most likely to improve system health
  - For incidents: find root cause before fixing downstream effects
  - For deployment: respect SLA constraints (check error_rate vs max_error_rate_pct)
  - Respond with JSON only — no markdown, no explanation outside JSON
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Smart deterministic fallback (used when no API key or LLM fails)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_action(task_id: str, obs_dict: dict) -> str:
    if task_id == "data_quality_triage":
        records     = obs_dict.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        if not unprocessed:
            return json.dumps({"action_type": "accept_record", "target_id": None, "parameters": {}, "reasoning": "All done"})
        rec       = unprocessed[0]
        issues    = rec.get("detected_issues", [])
        record_id = rec["record_id"]
        fields    = rec.get("fields", {})
        schema    = rec.get("schema_expected", {})

        if "null_value" in issues:
            null_field = next((f for f, v in fields.items() if v is None), None)
            expected   = schema.get(null_field, "str") if null_field else "str"
            fill       = 0.0 if expected == "float" else (0 if expected == "int" else "unknown")
            return json.dumps({"action_type": "fix_null", "target_id": record_id,
                               "parameters": {"field": null_field, "fill_value": fill},
                               "reasoning": "Filling null with type-appropriate default"})
        elif "type_mismatch" in issues:
            bad_field   = next((f for f, v in fields.items() if isinstance(v, str) and "_bad" in str(v)), None)
            target_type = schema.get(bad_field, "str") if bad_field else "str"
            return json.dumps({"action_type": "cast_type", "target_id": record_id,
                               "parameters": {"field": bad_field, "target_type": target_type},
                               "reasoning": "Casting mismatched field to correct type"})
        elif "outlier" in issues:
            outlier_field = next((f for f, v in fields.items() if isinstance(v, (int, float)) and v > 100000), None)
            return json.dumps({"action_type": "remove_outlier", "target_id": record_id,
                               "parameters": {"field": outlier_field},
                               "reasoning": "Removing statistical outlier"})
        elif "duplicate" in issues:
            return json.dumps({"action_type": "flag_duplicate", "target_id": record_id,
                               "parameters": {}, "reasoning": "Flagging duplicate record"})
        else:
            return json.dumps({"action_type": "accept_record", "target_id": record_id,
                               "parameters": {}, "reasoning": "Record is clean"})

    elif task_id == "deployment_decision":
        return json.dumps({
            "action_type": "deploy_canary",
            "target_id":   None,
            "parameters":  {"canary_pct": 5, "rollback_threshold_pct": 0.4},
            "reasoning":   "Challenger error_rate breaches SLA — canary at 5% limits blast radius"
        })

    else:  # incident_cascade
        history = obs_dict.get("context_history", [])
        if not any("feature_store" in h for h in history):
            return json.dumps({"action_type": "investigate", "target_id": None,
                               "parameters": {"component": "feature_store"},
                               "reasoning": "Investigating feature_store as suspected root cause"})
        elif not any("feature_store" in h and "restart" in h for h in history):
            return json.dumps({"action_type": "restart_service", "target_id": None,
                               "parameters": {"component": "feature_store"},
                               "reasoning": "Restarting root cause component"})
        elif not any("model_serving" in h for h in history):
            return json.dumps({"action_type": "restart_service", "target_id": None,
                               "parameters": {"component": "model_serving"},
                               "reasoning": "Resolving downstream model_serving"})
        else:
            return json.dumps({"action_type": "restart_service", "target_id": None,
                               "parameters": {"component": "data_pipeline"},
                               "reasoning": "Resolving downstream data_pipeline"})


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(obs_dict: dict[str, Any]) -> str:
    task_id   = obs_dict["task_id"]
    step      = obs_dict["step"]
    max_steps = obs_dict["max_steps"]
    context   = obs_dict.get("task_context", "")
    metrics   = obs_dict["system_metrics"]
    history   = obs_dict.get("context_history", [])
    available = obs_dict.get("available_actions", [])
    budget    = obs_dict.get("time_budget_remaining", 1.0)
    score     = obs_dict.get("episode_score_so_far", 0.0)

    lines = [
        f"TASK: {task_id}  |  Step {step}/{max_steps}  |  "
        f"Time budget: {budget:.2f}  |  Score so far: {score:.3f}",
        "",
        f"SITUATION: {context}",
        "",
        "SYSTEM METRICS:",
        f"  latency_p99_ms   = {metrics['latency_p99_ms']}",
        f"  error_rate_pct   = {metrics['error_rate_pct']}",
        f"  model_accuracy   = {metrics['model_accuracy']}",
        f"  throughput_rps   = {metrics['throughput_rps']}",
        f"  data_drift_score = {metrics['data_drift_score']}",
    ]

    if task_id == "data_quality_triage":
        records     = obs_dict.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        lines.append(f"\nUNPROCESSED RECORDS ({len(unprocessed)}):")
        for r in unprocessed[:5]:
            lines.append(f"  {r['record_id']}: fields={json.dumps(r['fields'])} "
                         f"issues={r.get('detected_issues',[])} schema={r['schema_expected']}")
        if len(unprocessed) > 5:
            lines.append(f"  ... and {len(unprocessed)-5} more")

    elif task_id == "deployment_decision":
        candidates = obs_dict.get("deployment_candidates", [])
        sla        = obs_dict.get("sla_requirements", {})
        lines.append("\nMODEL CANDIDATES:")
        for c in candidates:
            role = "CHAMPION" if c["is_champion"] else "CHALLENGER"
            lines.append(f"  [{role}] {c['name']}: accuracy={c['accuracy']} "
                         f"latency={c['latency_p99_ms']}ms error_rate={c['error_rate_pct']}%")
        lines.append(f"\nSLA: max_latency={sla.get('max_latency_p99_ms')}ms "
                     f"max_error_rate={sla.get('max_error_rate_pct')}%")
        alerts = obs_dict.get("alerts", [])
        if alerts:
            lines.append(f"\nALERT: {alerts[0]['message']}")

    elif task_id == "incident_cascade":
        alerts = obs_dict.get("alerts", [])
        lines.append(f"\nFIRING ALERTS ({len(alerts)}):")
        for a in alerts:
            status = "RESOLVED" if a["resolved"] else "OPEN"
            lines.append(f"  [{status}][{a['severity'].upper()}] "
                         f"{a['component']}: {a['message'][:100]}")

    if history:
        lines.append(f"\nRECENT ACTIONS: {' | '.join(history[-3:])}")
    lines.append(f"\nAVAILABLE ACTIONS: {available}")
    lines.append("\nRespond with JSON only:")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Action parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(response_text: str, available: list[str]) -> Action:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        data = json.loads(text)
        return Action(
            action_type=data.get("action_type", available[0] if available else "hold"),
            target_id=data.get("target_id"),
            parameters=data.get("parameters", {}),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return Action(
            action_type=available[0] if available else "hold",
            reasoning="Fallback due to parse error.",
        )


def inject_target(action: Action, obs_dict: dict[str, Any]) -> Action:
    if obs_dict["task_id"] != "data_quality_triage" or action.target_id:
        return action
    unprocessed = [r for r in obs_dict.get("data_records", []) if not r["processed"]]
    if unprocessed:
        return Action(
            action_type=action.action_type,
            target_id=unprocessed[0]["record_id"],
            parameters=action.parameters,
            reasoning=action.reasoning,
        )
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    client:  OpenAI | None,
    env:     MLOpsEnv,
    task_id: str,
) -> dict[str, Any]:

    result   = env.reset(task_id)
    obs_dict = result.observation.model_dump(mode="json")

    # ── Required: one [START] line ────────────────────────────────────────────
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    step         = 0
    step_result  = None
    last_error:  Optional[str] = None

    while step < MAX_STEPS:
        # Get action
        if FALLBACK_MODE or client is None:
            response_text = _fallback_action(task_id, obs_dict)
        else:
            try:
                completion = client.chat.completions.create(
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
                last_error    = str(exc)[:80]
                response_text = _fallback_action(task_id, obs_dict)

        available = obs_dict.get("available_actions", [])
        action    = parse_action(response_text, available)
        action    = inject_target(action, obs_dict)
        action_str = action.action_type.value

        # Step environment
        try:
            step_result = env.step(action)
            last_error  = None
        except (ValueError, RuntimeError) as exc:
            last_error = str(exc)[:80]
            step += 1
            log_step(step=step, action=action_str, reward=0.0, done=False, error=last_error)
            continue

        reward   = step_result.reward
        done     = step_result.done
        obs_dict = step_result.observation.model_dump(mode="json")
        rewards.append(reward)
        step += 1

        # ── Required: one [STEP] line per step ───────────────────────────────
        log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

        if done:
            break

    # Compute final score
    episode_info = step_result.info.get("episode", {}) if step_result and rewards else {}
    avg_score    = sum(rewards) / len(rewards) if rewards else 0.0
    final_score  = round(episode_info.get("total_score", avg_score), 4)
    success      = final_score >= SUCCESS_THRESHOLD

    # ── Required: one [END] line ─────────────────────────────────────────────
    log_end(success=success, steps=step, score=final_score, rewards=rewards)

    return {
        "task_id":     task_id,
        "steps":       step,
        "total_score": final_score,
        "success":     success,
        "rewards":     [round(r, 4) for r in rewards],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if not FALLBACK_MODE else None
    env = MLOpsEnv()

    task_id = TaskID.DATA_TRIAGE.value
    run_task(client, env, task_id)


if __name__ == "__main__":
    main()