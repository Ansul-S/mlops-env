"""
inference.py
============
Baseline inference script for MLOpsEnv.

MANDATORY env vars:
  API_BASE_URL  — LLM API endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace / API key

Run:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py

Output:
  Per-step rewards + final score per task.
  Reproduces deterministically (same seed, same model, same scores).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

from openai import OpenAI

from env import MLOpsEnv
from env.models import Action, ActionType, TaskID

# ─────────────────────────────────────────────────────────────────────────────
# Config — all from environment variables (spec requirement)
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

MAX_TOKENS   = 512
TEMPERATURE  = 0.0    # Deterministic — reproducible scores
MAX_STEPS    = 30     # Hard cap per task (well under 20 min limit)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert MLOps engineer operating a production ML system.
You will be given the current environment state and must choose exactly one action.

RESPONSE FORMAT — you must reply with valid JSON only, no other text:
{
  "action_type": "<action_type>",
  "target_id":   "<id or null>",
  "parameters":  {<key>: <value>},
  "reasoning":   "<one sentence explaining your choice>"
}

AVAILABLE ACTION TYPES by task:
  data_quality_triage:
    fix_null        — fill null field: parameters={"field": "<name>", "fill_value": <value>}
    remove_outlier  — drop outlier record: parameters={"field": "<name>"}
    cast_type       — coerce field type: parameters={"field": "<name>", "target_type": "<type>"}
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
    silence_alert   — suppress alert (use only as last resort)

RULES:
  - Always pick the action most likely to improve system health
  - For incidents: find root cause before fixing downstream effects
  - For deployment: respect SLA constraints (check error_rate vs max_error_rate_pct)
  - Respond with JSON only — no markdown, no explanation outside the JSON
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(obs_dict: dict[str, Any]) -> str:
    """Convert observation to a focused LLM prompt."""

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

    # Task-specific context
    if task_id == "data_quality_triage":
        records = obs_dict.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        lines.append(f"\nUNPROCESSED RECORDS ({len(unprocessed)}):")
        for r in unprocessed[:5]:   # Show first 5 to stay within token limit
            issues = r.get("detected_issues", [])
            lines.append(
                f"  {r['record_id']}: fields={json.dumps(r['fields'])} "
                f"issues={issues} schema={r['schema_expected']}"
            )
        if len(unprocessed) > 5:
            lines.append(f"  ... and {len(unprocessed) - 5} more records")

    elif task_id == "deployment_decision":
        candidates = obs_dict.get("deployment_candidates", [])
        sla = obs_dict.get("sla_requirements", {})
        lines.append("\nMODEL CANDIDATES:")
        for c in candidates:
            role = "CHAMPION" if c["is_champion"] else "CHALLENGER"
            lines.append(
                f"  [{role}] {c['name']}: accuracy={c['accuracy']} "
                f"latency={c['latency_p99_ms']}ms "
                f"error_rate={c['error_rate_pct']}%"
            )
        lines.append(
            f"\nSLA CONSTRAINTS: max_latency={sla.get('max_latency_p99_ms')}ms "
            f"max_error_rate={sla.get('max_error_rate_pct')}%"
        )
        alerts = obs_dict.get("alerts", [])
        if alerts:
            lines.append(f"\nALERT: {alerts[0]['message']}")

    elif task_id == "incident_cascade":
        alerts = obs_dict.get("alerts", [])
        lines.append(f"\nFIRING ALERTS ({len(alerts)}):")
        for a in alerts:
            status = "RESOLVED" if a["resolved"] else "OPEN"
            lines.append(
                f"  [{status}][{a['severity'].upper()}] "
                f"{a['component']}: {a['message'][:100]}"
            )

    if history:
        lines.append(f"\nRECENT ACTIONS: {' | '.join(history[-3:])}")

    lines.append(f"\nAVAILABLE ACTIONS: {available}")
    lines.append("\nRespond with JSON only:")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Action parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(response_text: str, available: list[str]) -> Action:
    """
    Parse LLM JSON response → Action.
    Falls back to a safe default if parsing fails.
    """
    # Strip markdown fences if model wraps in ```json
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
        return Action(
            action_type=data.get("action_type", available[0] if available else "hold"),
            target_id=data.get("target_id"),
            parameters=data.get("parameters", {}),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: use first available action with no target
        fallback_type = available[0] if available else "hold"
        return Action(
            action_type=fallback_type,
            reasoning="Fallback action due to parse error.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific target injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_target(action: Action, obs_dict: dict[str, Any]) -> Action:
    """
    For data_quality_triage: auto-inject target_id from first unprocessed
    record when model forgets to set it. Prevents wasted steps.
    """
    if obs_dict["task_id"] != "data_quality_triage":
        return action

    if action.target_id:
        return action

    records = obs_dict.get("data_records", [])
    unprocessed = [r for r in records if not r["processed"]]
    if unprocessed:
        action = Action(
            action_type=action.action_type,
            target_id=unprocessed[0]["record_id"],
            parameters=action.parameters,
            reasoning=action.reasoning,
        )
    return action



# ─────────────────────────────────────────────────────────────────────────────
# Smart per-task fallback (used when LLM call fails)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_action(task_id: str, obs_dict: dict) -> str:
    """
    Return a JSON action string when LLM is unavailable.
    Each task gets a meaningful default — not just available[0].
    """
    if task_id == "data_quality_triage":
        # Pick first unprocessed record and accept it
        records = obs_dict.get("data_records", [])
        unprocessed = [r for r in records if not r["processed"]]
        target = unprocessed[0]["record_id"] if unprocessed else None
        return json.dumps({
            "action_type": "accept_record",
            "target_id": target,
            "parameters": {},
            "reasoning": "Fallback: accepting record as clean"
        })

    elif task_id == "deployment_decision":
        # Safe default: hold — never deploy blindly on fallback
        return json.dumps({
            "action_type": "hold",
            "target_id": None,
            "parameters": {},
            "reasoning": "Fallback: holding deployment pending further analysis"
        })

    else:  # incident_cascade
        # Investigate root cause component first
        return json.dumps({
            "action_type": "investigate",
            "target_id": None,
            "parameters": {"component": "feature_store"},
            "reasoning": "Fallback: investigating feature_store as likely root cause"
        })


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    client:  OpenAI,
    env:     MLOpsEnv,
    task_id: str,
) -> dict[str, Any]:
    """Run one full episode. Returns score summary."""

    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    result      = env.reset(task_id)
    obs         = result.observation
    obs_dict    = obs.model_dump(mode="json")

    # ── Required structured output ────────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    episode_rewards: list[float] = []
    step = 0

    while step < MAX_STEPS:
        # Build prompt
        user_prompt = build_user_prompt(obs_dict)

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [Step {step+1}] LLM call failed: {exc}")
            response_text = _fallback_action(task_id, obs_dict)

        # Parse action
        available = obs_dict.get("available_actions", [])
        action    = parse_action(response_text, available)
        action    = inject_target(action, obs_dict)

        # Step environment
        try:
            step_result = env.step(action)
        except (ValueError, RuntimeError) as exc:
            print(f"  [Step {step+1}] Invalid action ({exc}) — skipping")
            step += 1
            continue

        reward = step_result.reward
        done   = step_result.done
        obs_dict = step_result.observation.model_dump(mode="json")
        episode_rewards.append(reward)
        step += 1

        feedback = step_result.info.get("feedback", "")[:80]
        print(
            f"  Step {step:2d}: {action.action_type.value:20s} "
            f"→ reward={reward:.4f}  done={done}"
        )
        if feedback:
            print(f"          {feedback}")
        # Required structured output
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

        if done:
            break

    avg_score = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0

    # Pull final episode summary from info if available
    episode_info = step_result.info.get("episode", {}) if episode_rewards else {}

    print(f"\n  ✓ Steps: {step} | Avg reward: {avg_score:.4f}")
    if episode_info:
        print(f"  ✓ {episode_info.get('final_state_summary', '')}")

    # Required structured output
    final_score = round(episode_info.get("total_score", avg_score), 4)
    print(f"[END] task={task_id} score={final_score} steps={step}", flush=True)

    return {
        "task_id":      task_id,
        "steps":        step,
        "avg_reward":   round(avg_score, 4),
        "total_score":  final_score,
        "rewards":      [round(r, 4) for r in episode_rewards],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Validate env vars
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    print("MLOpsEnv — Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'set ✓' if API_KEY else 'NOT SET ✗'}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = MLOpsEnv()

    tasks = [
        TaskID.DATA_TRIAGE.value,
        TaskID.DEPLOYMENT.value,
        TaskID.INCIDENT.value,
    ]

    results: list[dict] = []
    for task_id in tasks:
        summary = run_task(client, env, task_id)
        results.append(summary)

    # ── Final score table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    print(f"  {'Task':<30} {'Steps':>6} {'Score':>8}")
    print(f"  {'-'*46}")

    total = 0.0
    for r in results:
        print(
            f"  {r['task_id']:<30} {r['steps']:>6} {r['total_score']:>8.4f}"
        )
        total += r["total_score"]

    overall = total / len(results)
    print(f"  {'-'*46}")
    print(f"  {'OVERALL':.<30} {'':>6} {overall:>8.4f}")
    print(f"{'='*60}\n")

    # Write results to file for reproducibility verification
    with open("baseline_scores.json", "w") as f:
        json.dump(
            {"model": MODEL_NAME, "results": results, "overall": overall},
            f, indent=2
        )
    print("  Scores saved to baseline_scores.json")


if __name__ == "__main__":
    main()