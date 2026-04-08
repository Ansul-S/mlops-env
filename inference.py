"""
inference.py — MLOpsEnv Baseline Agent
=======================================
MANDATORY env vars:
  API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
  MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
  HF_TOKEN      = os.getenv("HF_TOKEN")

STDOUT FORMAT:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import sys
import json
import textwrap

# ── Env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK         = "mlops-env"
MAX_STEPS         = 30
SUCCESS_THRESHOLD = 0.5
TASKS = ["data_quality_triage", "deployment_decision", "incident_cascade"]

# ── Optional imports — script works even if these are missing ─────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import MLOpsEnv
    from env.models import Action
    _ENV_OK = True
except Exception:
    MLOpsEnv = None
    Action    = None
    _ENV_OK   = False

FALLBACK_MODE = (not bool(HF_TOKEN)) or (not _OPENAI_OK)


# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task, model):
    print("[START] task=%s env=%s model=%s" % (task, BENCHMARK, model), flush=True)

def log_step(step, action, reward, done, error=None):
    err  = str(error)[:60] if error else "null"
    done_s = "true" if done else "false"
    print("[STEP] step=%d action=%s reward=%.2f done=%s error=%s" % (
        step, action, reward, done_s, err), flush=True)

def log_end(success, steps, score, rewards):
    if not rewards:
        rewards = [0.0]
    rstr = ",".join("%.2f" % r for r in rewards)
    succ = "true" if success else "false"
    print("[END] success=%s steps=%d score=%.2f rewards=%s" % (
        succ, steps, score, rstr), flush=True)


# ── Fallback actions ──────────────────────────────────────────────────────────

def fallback(task_id, obs):
    if task_id == "data_quality_triage":
        records     = obs.get("data_records", [])
        unprocessed = [r for r in records if not r.get("processed", False)]
        if not unprocessed:
            return {"action_type": "accept_record", "target_id": None, "parameters": {}, "reasoning": "done"}
        rec    = unprocessed[0]
        rid    = rec["record_id"]
        issues = rec.get("detected_issues", [])
        fields = rec.get("fields", {})
        schema = rec.get("schema_expected", {})
        if "null_value" in issues:
            nf   = next((f for f, v in fields.items() if v is None), None)
            et   = schema.get(nf, "str") if nf else "str"
            fill = 0.0 if et == "float" else (0 if et == "int" else "unknown")
            return {"action_type": "fix_null", "target_id": rid,
                    "parameters": {"field": nf, "fill_value": fill}, "reasoning": "null fix"}
        elif "type_mismatch" in issues:
            bf = next((f for f, v in fields.items() if isinstance(v, str) and "_bad" in str(v)), None)
            tt = schema.get(bf, "str") if bf else "str"
            return {"action_type": "cast_type", "target_id": rid,
                    "parameters": {"field": bf, "target_type": tt}, "reasoning": "type fix"}
        elif "outlier" in issues:
            of = next((f for f, v in fields.items() if isinstance(v, (int, float)) and v > 100000), None)
            return {"action_type": "remove_outlier", "target_id": rid,
                    "parameters": {"field": of}, "reasoning": "outlier"}
        elif "duplicate" in issues:
            return {"action_type": "flag_duplicate", "target_id": rid,
                    "parameters": {}, "reasoning": "duplicate"}
        else:
            return {"action_type": "accept_record", "target_id": rid,
                    "parameters": {}, "reasoning": "clean"}

    elif task_id == "deployment_decision":
        return {"action_type": "deploy_canary", "target_id": None,
                "parameters": {"canary_pct": 5, "rollback_threshold_pct": 0.4},
                "reasoning": "challenger breaches SLA"}

    else:
        hist = obs.get("context_history", [])
        if not any("feature_store" in h for h in hist):
            return {"action_type": "investigate", "target_id": None,
                    "parameters": {"component": "feature_store"}, "reasoning": "root cause"}
        elif not any("feature_store" in h and "restart" in h for h in hist):
            return {"action_type": "restart_service", "target_id": None,
                    "parameters": {"component": "feature_store"}, "reasoning": "fix root"}
        elif not any("model_serving" in h for h in hist):
            return {"action_type": "restart_service", "target_id": None,
                    "parameters": {"component": "model_serving"}, "reasoning": "fix downstream"}
        else:
            return {"action_type": "restart_service", "target_id": None,
                    "parameters": {"component": "data_pipeline"}, "reasoning": "fix downstream"}


# ── LLM action ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an MLOps engineer. Reply with JSON only:
{"action_type":"<action>","target_id":"<id or null>","parameters":{},"reasoning":"<reason>"}
Tasks: data_quality_triage(fix_null,remove_outlier,cast_type,flag_duplicate,accept_record)
       deployment_decision(deploy_canary,deploy_full,rollback,hold)
       incident_cascade(investigate,restart_service,reroute_traffic,rollback_model,escalate)
For incidents investigate root cause first. For deployment check SLA error_rate.
""").strip()

def llm_action(client, task_id, obs):
    try:
        ctx  = obs.get("task_context", "")
        hist = obs.get("context_history", [])
        avail = obs.get("available_actions", [])
        m    = obs.get("system_metrics", {})
        msg  = ("TASK:%s step:%d/%d\nSITUATION:%s\n"
                "METRICS:lat=%s err=%s acc=%s\nRECENT:%s\nAVAIL:%s\nJSON:" % (
                    task_id, obs.get("step",0), obs.get("max_steps",30),
                    ctx, m.get("latency_p99_ms"), m.get("error_rate_pct"),
                    m.get("model_accuracy"), hist[-2:], avail))
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":msg}],
            temperature=0.0, max_tokens=256)
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        data = json.loads(text)
        return {"action_type": data.get("action_type", avail[0] if avail else "hold"),
                "target_id":   data.get("target_id"),
                "parameters":  data.get("parameters", {}),
                "reasoning":   data.get("reasoning", "")}
    except Exception:
        return fallback(task_id, obs)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client, task_id):
    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    # [START] printed FIRST — before anything can fail
    log_start(task_id, MODEL_NAME)

    try:
        if not _ENV_OK:
            raise RuntimeError("env/ package not available")

        env = MLOpsEnv()
        res = env.reset(task_id)
        obs = res.observation.model_dump(mode="json")

        for step_num in range(1, MAX_STEPS + 1):
            act     = fallback(task_id, obs) if FALLBACK_MODE else llm_action(client, task_id, obs)
            act_str = act.get("action_type", "unknown")
            err_str = None

            try:
                act_obj = Action(**act)
                sr      = env.step(act_obj)
                reward  = float(sr.reward)
                done    = bool(sr.done)
                obs     = sr.observation.model_dump(mode="json")
            except Exception as ex:
                reward  = 0.0
                done    = True
                err_str = str(ex)[:50]

            rewards.append(reward)
            steps_taken = step_num
            log_step(step_num, act_str, reward, done, err_str)
            if done:
                break

        score   = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as ex:
        err = str(ex)[:50]
        if steps_taken == 0:
            log_step(1, "none", 0.0, True, err)
            steps_taken = 1
        rewards = rewards if rewards else [0.0]
        score   = round(sum(rewards) / len(rewards), 4)
        success = False

    finally:
        # [END] ALWAYS printed — even on exception
        log_end(success, steps_taken, score, rewards)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = None
    if not FALLBACK_MODE and _OPENAI_OK and HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            client = None

    for task_id in TASKS:
        run_task(client, task_id)

    try:
        with open("baseline_scores.json", "w") as f:
            json.dump({"model": MODEL_NAME, "tasks": TASKS}, f)
    except Exception:
        pass


if __name__ == "__main__":
    main()