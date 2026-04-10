---
title: MLOps Environment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MLOpsEnv

**Production ML Pipeline Operations Environment for RL Agent Training**

> An OpenEnv-compliant environment where an AI agent manages real MLOps workflows — triaging data quality issues before training runs, making deployment decisions under SLA constraints, and resolving cascading production incidents in the correct causal order.

---

## Why This Environment?

Every ML team at every company faces these exact problems **every single day**:

- **Data quality issues** silently corrupt models before training even starts
- **Deployment decisions** require balancing accuracy gains against latency and error rate SLAs
- **Production incidents** cascade — fixing the wrong component first makes everything worse

No existing RL benchmark covers this domain. MLOpsEnv fills that gap with **deterministic, reproducible, gradeable tasks** that reflect genuine operational complexity that ML engineers face in production systems.

This environment was built specifically for the kinds of problems Meta and HuggingFace engineers encounter daily — making it immediately useful for agent evaluation beyond the hackathon.

---

## What Makes This Unique

### 1. Causal State Transitions
Unlike toy environments where each step is independent, actions here have **real consequences across future steps**:

```python
# Wrong data fix → data drift increases, affecting training quality
metrics.data_drift_score += 0.03

# Silencing an alert without fixing root cause → error rate worsens
metrics.error_rate_pct += 0.5

# Fixing feature_store → automatically cascades recovery to model_serving
metrics.error_rate_pct = max(0.5, error_rate - 2.0)
```

An agent cannot act randomly and expect decent scores. It must understand cause and effect.

### 2. Dense Multi-Dimensional Reward
Every step returns a 4-dimensional breakdown — never sparse:

```
score = correctness  × 0.50   # Was the action correct?
      + efficiency   × 0.15   # Did agent act without waste?
      + completeness × 0.25   # Coverage of all open issues?
      + safety       × 0.10   # No metric regression or alert suppression?
```

Training frameworks can log each dimension independently, giving rich signal for learning.

### 3. Genuine Difficulty Curve
The difficulty progression is real and provable — tested against Llama-3.3-70B:

| Task | Difficulty | Baseline Score | Why |
|---|---|---|---|
| data_quality_triage | Easy | **0.83** | Clear pattern matching |
| deployment_decision | Medium | **0.96** | Multi-constraint reasoning |
| incident_cascade | Hard | **0.43** | Causal chain reasoning under time pressure |

The hard task genuinely challenges frontier models — even strong LLMs score ~0.43 because they must identify root cause before fixing downstream effects, in the correct order, within 15 steps.

### 4. Safety Gate
Any action that worsens system metrics scores `safety = 0.0` regardless of other dimensions. Silencing an alert without fixing root cause is immediately penalized. Agents cannot game the environment.

---

## Environment Overview

```
Entry point : env.environment:MLOpsEnv
Framework   : FastAPI (port 7860)
Tasks       : 3 (easy → medium → hard)
Reward type : Dense, multi-dimensional (0.0–1.0 per step)
Seed        : 42 (fully deterministic and reproducible)
```

---

## Tasks

### Task 1 — Easy: `data_quality_triage`

**Scenario:** 20 incoming data records before a model training run. Each record contains exactly one issue or is clean.

**Issue types:** null values, type mismatches, statistical outliers (6σ), exact duplicates

**Agent must:** Apply the correct fix action per record within 30 steps.

**Grader:** Deterministic per-record ground truth. Scores correctness, parameter quality, completeness, and data drift safety.

**Baseline score: 0.83**

---

### Task 2 — Medium: `deployment_decision`

**Scenario:** A/B test complete. Challenger shows +1.4% accuracy improvement but error rate of 0.8% breaches the 0.5% SLA budget.

**Agent must:** Choose deployment strategy + parameters. Reasoning quality is graded.

**Correct answer:** `deploy_canary(canary_pct=5, rollback_threshold_pct=0.4)`

**Why wrong answers fail:**
- `deploy_full` → challenger error rate violates SLA → `correctness = 0.0`
- `rollback` → no regression to roll back from → `correctness = 0.2`
- `hold` → safe but no progress → `correctness = 0.4`

**Baseline score: 0.96**

---

### Task 3 — Hard: `incident_cascade`

**Scenario:** 3 simultaneous alerts. One is root cause (feature_store latency spike at 847ms). Two are downstream effects. Agent has 15 steps.

**Optimal sequence (4 steps):**
1. `investigate(feature_store)` → confirm root cause
2. `restart_service(feature_store)` → fix root cause
3. `restart_service(model_serving)` → resolve downstream
4. `restart_service(data_pipeline)` → drain backlog

**Why it's hard:**
- Fixing downstream before root cause wastes steps and scores low
- Silencing any alert → `safety = 0` immediately
- Efficiency score decays non-linearly past step 6
- Requires causal chain reasoning, not pattern matching

**Baseline score: 0.43** — even frontier models struggle with causal ordering

---

## Action Space

```python
Action(
    action_type: ActionType,     # required
    target_id:   str | None,     # record_id, alert_id, candidate_id
    parameters:  dict[str, Any], # action-specific params
    reasoning:   str,            # graded in hard task
)
```

| Task | Actions |
|---|---|
| data_quality_triage | `fix_null`, `remove_outlier`, `cast_type`, `flag_duplicate`, `accept_record` |
| deployment_decision | `deploy_canary`, `deploy_full`, `rollback`, `hold` |
| incident_cascade | `investigate`, `restart_service`, `reroute_traffic`, `rollback_model`, `escalate`, `silence_alert` |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `TaskID` | Current task |
| `step` | `int` | Current step |
| `system_metrics` | `SystemMetrics` | Live latency, error rate, accuracy, drift |
| `data_records` | `list[DataRecord]` | Records awaiting triage (Task 1) |
| `alerts` | `list[Alert]` | Firing alerts (Tasks 2 & 3) |
| `deployment_candidates` | `list[ModelCandidate]` | Champion + challenger (Task 2) |
| `available_actions` | `list[ActionType]` | Legal actions this step |
| `time_budget_remaining` | `float` | Normalized time left (1.0 → 0.0) |
| `task_context` | `str` | Natural language situation summary |
| `episode_score_so_far` | `float` | Running average — partial credit signal |

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
git clone https://github.com/Ansul-S/mlops-env
cd mlops-env
pip install -r requirements.txt

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# Test
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
| data_quality_triage | Easy | 20/30 | **0.83** |
| deployment_decision | Medium | 1/10 | **0.96** |
| incident_cascade | Hard | 4/15 | **0.43** |
| **Overall** | | | **0.74** |

The difficulty curve is intentional — stronger models score significantly higher on Tasks 2 and 3 because they can reason about multi-constraint tradeoffs and causal chains.

---

## Project Structure

```
mlops_env/
├── server.py              # FastAPI server (/reset /step /state)
├── inference.py           # Baseline agent using OpenAI client
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── env/
    ├── __init__.py
    ├── environment.py     # MLOpsEnv — step/reset/state
    ├── models.py          # All Pydantic types
    ├── simulator.py       # Deterministic state machine with causal propagation
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

**Domain choice:** MLOps was chosen because it models workflows Meta and HuggingFace engineers encounter daily — making it immediately useful for agent evaluation in a real organizational context.

**Causal propagation:** Wrong actions have consequences across future steps. This forces agents to think ahead rather than greedily optimizing each step independently.

**Dense rewards over sparse:** Every step returns signal across 4 dimensions. Agents always receive feedback — no dead zones where reward is zero regardless of behavior.

**Deterministic grading:** All graders use pure Python logic against pre-seeded ground truth. No LLM calls in grading. Fully reproducible across any hardware.

**Safety gate:** Actions that worsen system state score `safety = 0.0` regardless of other dimensions. Agents cannot accidentally exploit reward by making destructive actions.
