---
title: MLOps Environment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
.
# MLOpsEnv

**Production ML Pipeline Operations Environment for RL Agent Training**

An OpenEnv-compliant environment where an AI agent manages real MLOps workflows under operational pressure — triaging data quality issues before training runs, making deployment decisions under SLA constraints, and resolving cascading production incidents in the correct causal order.

> Built for the OpenEnv Hackathon. Judged on real-world utility, task quality, environment design, and code quality.

---

## Why This Environment?

Every ML team at every company faces these exact problems daily:

- **Data quality issues** block training runs and silently corrupt models
- **Deployment decisions** require multi-constraint reasoning (accuracy vs. latency vs. error rate)
- **Production incidents** cascade — fixing the wrong component first makes things worse

No existing RL benchmark covers this domain. MLOpsEnv fills that gap with deterministic, reproducible, gradeable tasks that reflect genuine operational complexity.

---

## Environment Overview

```
Entry point : env.environment:MLOpsEnv
Framework   : FastAPI (port 7860)
Tasks       : 3 (easy → medium → hard)
Reward type : Dense (0.0 – 1.0 every step)
Seed        : 42 (fully reproducible)
```

---

## Action Space

Actions are submitted as `Action` Pydantic models:

```python
Action(
    action_type: ActionType,        # required — see table below
    target_id:   str | None,        # record_id, alert_id, or candidate_id
    parameters:  dict[str, Any],    # action-specific params
    reasoning:   str,               # graded in hard task
)
```

### Available Action Types by Task

| Task | Action Type | Parameters |
|---|---|---|
| data_quality_triage | `fix_null` | `{"field": str, "fill_value": any}` |
| data_quality_triage | `remove_outlier` | `{"field": str}` |
| data_quality_triage | `cast_type` | `{"field": str, "target_type": str}` |
| data_quality_triage | `flag_duplicate` | `{"duplicate_of": str}` |
| data_quality_triage | `accept_record` | `{}` |
| deployment_decision | `deploy_canary` | `{"canary_pct": int, "rollback_threshold_pct": float}` |
| deployment_decision | `deploy_full` | `{}` |
| deployment_decision | `rollback` | `{}` |
| deployment_decision | `hold` | `{}` |
| incident_cascade | `investigate` | `{"component": str}` |
| incident_cascade | `restart_service` | `{"component": str}` |
| incident_cascade | `reroute_traffic` | `{"from_component": str, "fallback": bool}` |
| incident_cascade | `rollback_model` | `{}` |
| incident_cascade | `escalate` | `{}` |
| incident_cascade | `silence_alert` | `{}` |

---

## Observation Space

Each step returns an `Observation` containing:

| Field | Type | Description |
|---|---|---|
| `task_id` | `TaskID` | Current task identifier |
| `step` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Episode step limit |
| `system_metrics` | `SystemMetrics` | Live latency, error rate, accuracy, drift |
| `data_records` | `list[DataRecord]` | Records awaiting triage (Task 1) |
| `alerts` | `list[Alert]` | Firing system alerts (Tasks 2 & 3) |
| `deployment_candidates` | `list[ModelCandidate]` | Champion + challenger models (Task 2) |
| `context_history` | `list[str]` | Last 5 action summaries |
| `available_actions` | `list[ActionType]` | Legal actions this step |
| `time_budget_remaining` | `float` | Normalized time left (1.0 → 0.0) |
| `sla_requirements` | `SLARequirements` | Operational constraints |
| `task_context` | `str` | Natural language situation summary |
| `episode_score_so_far` | `float` | Running average score |

---

## Reward Function

Dense reward every step — never sparse:

```
score = correctness  × 0.50   # Was the action right?
      + efficiency   × 0.15   # Did agent act without waste?
      + completeness × 0.25   # Coverage of all open issues?
      + safety       × 0.10   # No metric regression or alert suppression?
```

**Safety gate:** Silencing an alert without fixing it sets `safety = 0.0` regardless of other scores. Deploying a model that violates SLA constraints scores `correctness = 0.0`.

---

## Tasks

### Task 1 — Easy: `data_quality_triage`

**Scenario:** 20 incoming data records before a training run. Each has exactly one issue or is clean.

**Issue types:** null values, type mismatches, statistical outliers (6σ), exact duplicates

**Agent must:** Apply the correct fix action per record within 30 steps.

**Grader:** Deterministic per-record ground truth. Scores correctness, parameter quality, completeness, and data drift safety.

**Baseline score:** ~0.83

---

### Task 2 — Medium: `deployment_decision`

**Scenario:** A/B test complete. Challenger shows +1.4% accuracy improvement but error rate of 0.8% breaches the 0.5% SLA budget.

**Agent must:** Choose deployment strategy + parameters. Reasoning quality is graded.

**Correct answer:** `deploy_canary(canary_pct=5, rollback_threshold_pct=0.4)`

**Why wrong answers fail:**
- `deploy_full` → challenger error rate violates SLA → `correctness = 0.0`
- `rollback` → no regression to roll back from → `correctness = 0.2`
- `hold` → safe but no progress → `correctness = 0.4`

**Baseline score:** ~0.51 (fallback), ~0.90+ (strong model)

---

### Task 3 — Hard: `incident_cascade`

**Scenario:** 3 simultaneous alerts. One is the root cause (feature_store latency spike at 847ms). Two are downstream effects.

**Agent must:**
1. `investigate(component=feature_store)` → confirm root cause
2. `restart_service(component=feature_store)` → fix root cause
3. `restart_service(component=model_serving)` → resolve downstream
4. `restart_service(component=data_pipeline)` → drain backlog

**Optimal resolution: 4 steps.**

**Why it's hard:**
- Fixing downstream components before root cause wastes steps and scores low
- Silencing any alert without fixing → `safety = 0` immediately
- Efficiency score decays non-linearly past step 6

**Baseline score:** ~0.35 (requires causal reasoning frontier models struggle with)

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "..."}` |
| `POST` | `/step` | Execute action. Body: `{"action": {...}}` |
| `GET` | `/state` | Read current state (no step consumed) |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/health` | Health check |

---

## Setup & Usage

### Local Development

```bash
# Clone and install
git clone <your-repo-url>
cd mlops_env
pip install -r requirements.txt

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# Test it
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "data_quality_triage"}'
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

Scores are saved to `baseline_scores.json`.

### Docker

```bash
docker build -t mlops-env .
docker run -p 7860:7860 mlops-env
```

---

## Baseline Scores

Tested with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Inference Providers:

| Task | Difficulty | Steps | Score |
|---|---|---|---|
| data_quality_triage | Easy | 20/30 | **0.8287** |
| deployment_decision | Medium | 1/10 | **0.5100** |
| incident_cascade | Hard | 15/15 | **0.3478** |
| **Overall** | | | **0.5622** |

The difficulty curve is intentional and real — stronger models score significantly higher on Tasks 2 and 3.

---

## Project Structure

```
mlops_env/
├── server.py              # FastAPI server (exposes /reset /step /state)
├── inference.py           # Baseline agent using OpenAI client
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── README.md              # This file
└── env/
    ├── __init__.py
    ├── environment.py     # MLOpsEnv class (step/reset/state)
    ├── models.py          # All Pydantic types
    ├── simulator.py       # Deterministic state machine
    ├── tasks/
    │   ├── base.py
    │   ├── easy_data_triage.py
    │   ├── medium_deployment.py
    │   └── hard_incident.py
    └── graders/
        └── __init__.py
```

---

## Design Decisions

**Why MLOps?** Meta and HuggingFace engineers live in this problem domain. The environment models workflows they encounter daily — making it immediately useful for agent evaluation beyond the hackathon.

**Why deterministic graders?** No LLM calls in grading logic. Every score is computed from pure Python rules against pre-seeded ground truth. Reproducible across any hardware.

**Why dense rewards?** Every step returns a breakdown across 4 dimensions. Training frameworks can log each independently. Agents always receive learning signal — no sparse reward dead zones.

**Why causal consequences?** Wrong actions in Task 1 raise `data_drift_score`. Silencing alerts in Task 3 raises `error_rate_pct`. The environment responds realistically — agents can't act randomly and expect decent scores.
