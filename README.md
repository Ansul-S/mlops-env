---
title: MLOps Environment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
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

# Fixing root cause → automatically cascades recovery to downstream components
metrics.error_rate_pct = max(0.5, error_rate - 2.0)
```

An agent cannot act randomly and expect decent scores. It must understand cause and effect.

### 2. Dense Multi-Dimensional Reward
Every step returns a 4-dimensional breakdown — never sparse:
score = correctness  × 0.50   # Was the action correct?
+ efficiency   × 0.15   # Did agent act without waste?
+ completeness × 0.25   # Coverage of all open issues?
+ safety       × 0.10   # No metric regression or alert suppression?

Training frameworks can log each dimension independently, giving rich signal for learning.

### 3. Genuine Difficulty Curve
The difficulty progression is real and provable — tested against Llama-3.3-70B with a generic baseline agent:

| Task | Difficulty | Baseline Score | Why |
|---|---|---|---|
| data_quality_triage | Easy | **0.28** | Requires correct action per record type |
| deployment_decision | Medium | **0.09** | Multi-step monitoring under SLA constraints |
| incident_cascade | Hard | **0.19** | Causal chain reasoning under time pressure |

A strong LLM agent scores significantly higher than the generic baseline — especially on Tasks 2 and 3 — because they can reason about multi-constraint tradeoffs and causal chains.

### 4. Safety Gate
Any action that worsens system metrics scores `safety = 0.0` regardless of other dimensions. Silencing an alert without fixing root cause is immediately penalized. Agents cannot game the environment.

### 5. Seed-Based Randomization
Every episode is unique. Pass a `seed` in the `/reset` body for reproducible episodes, or omit it for a fresh random episode. Root cause components, data record issue positions, and deployment metrics all vary per seed — agents cannot memorize solutions.

---

## Environment Overview
Entry point : env.environment:MLOpsEnv
Framework   : FastAPI + WebSocket (port 8000)
Tasks       : 3 (easy → medium → hard)
Reward type : Dense, multi-dimensional (0.0–1.0 per step)
Seed        : Configurable per episode (pass seed in /reset body)
Sessions    : Up to 16 concurrent isolated sessions via WebSocket

---

## Tasks

### Task 1 — Easy: `data_quality_triage`

**Scenario:** 20 incoming data records before a model training run. Each record contains exactly one issue or is clean. Issue positions and types vary per episode seed.

**Issue types:** null values, type mismatches, statistical outliers (6σ), exact duplicates

**Agent must:** Apply the correct fix action per record within 30 steps.

**Grader:** Deterministic per-record ground truth. Scores correctness, parameter quality, completeness, and data drift safety.

**Baseline score: 0.28**

---

### Task 2 — Medium: `deployment_decision`

**Scenario:** A/B test complete. Challenger shows accuracy improvement but error rate breaches the 0.5% SLA budget. Metrics vary per episode seed.

**Multi-step flow:**
- **Step 1:** Agent chooses deployment strategy (canary/full/hold/rollback)
- **Steps 2–4:** Metrics evolve per step — agent monitors and decides to promote or rollback
- **Terminal:** Agent promotes canary to full OR triggers rollback OR max steps reached

**Agent must reason about:** accuracy improvement vs. SLA violation tradeoff across multiple steps of metric observation.

**Baseline score: 0.09**

---

### Task 3 — Hard: `incident_cascade`

**Scenario:** 3 simultaneous alerts fire. One is the root cause component (randomized per episode — varies across `feature_store`, `model_serving`, `data_pipeline`). Two are downstream effects caused by the root component failing. Agent has 15 steps.

**Why it's hard:**
- Root cause component changes every episode — agents cannot memorize
- Fixing downstream before root cause wastes steps and scores low
- Silencing any alert → `safety = 0` immediately
- Efficiency score decays non-linearly past step 6
- Requires causal chain reasoning, not pattern matching

**Baseline score: 0.19** — a generic agent struggles because it must identify the root cause before addressing downstream effects, in the correct causal order.

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
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "...", "seed": 42}` |
| `POST` | `/step` | Execute action. Body: `{"action": {...}}` |
| `GET` | `/state` | Read current state (no step consumed) |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/health` | Health check |
| `GET` | `/mlops-state` | Rich debug state (episode_id, root_cause, seed) |
| `WS` | `/ws` | WebSocket session for low-latency RL training |

---

## Setup & Usage

### Local Development

```bash
git clone https://github.com/Ansul-S/mlops-env
cd mlops-env
pip install -r requirements.txt

# Start server
uvicorn server:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "data_quality_triage", "seed": 42}'
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
export SPACE_URL="https://Ansul-S-mlops-env.hf.space"

python inference.py
```

### Docker

```bash
docker build -t mlops-env .
docker run -p 8000:8000 mlops-env
```

### WebSocket Usage

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("wss://Ansul-S-mlops-env.hf.space/ws") as ws:
        await ws.recv()  # connected message
        await ws.send(json.dumps({
            "type": "reset",
            "task_id": "incident_cascade",
            "seed": 42
        }))
        result = json.loads(await ws.recv())
        print(result["observation"]["task_context"])

asyncio.run(main())
```

---

## Baseline Scores

Tested with a generic baseline agent (picks first available action):

| Task | Difficulty | Steps | Score |
|---|---|---|---|
| data_quality_triage | Easy | 20/30 | **0.28** |
| deployment_decision | Medium | 4/10 | **0.09** |
| incident_cascade | Hard | 15/15 | **0.19** |
| **Overall** | | | **0.19** |

A capable LLM agent scores significantly higher — especially on Tasks 2 and 3 — because it can reason about SLA constraints, causal ordering, and multi-step consequences.

---

## Project Structure

```text
mlops_env/
├── server.py              # FastAPI + WebSocket server
├── client.py              # Async HTTP client for MLOpsEnv
├── inference.py           # Baseline agent using OpenAI client
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── tests/
│   └── test_mlops_env.py  # Test suite
└── env/
    ├── __init__.py
    ├── environment.py     # MLOpsEnv — step/reset/state
    ├── models.py          # All Pydantic types
    ├── simulator.py       # Seed-based state machine with causal propagation
    ├── tasks/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── easy_data_triage.py
    │   ├── medium_deployment.py
    │   └── hard_incident.py
    └── graders/
        └── __init__.py
```

## Design Decisions

**Domain choice:** MLOps was chosen because it models workflows Meta and HuggingFace engineers encounter daily — making it immediately useful for agent evaluation in a real organizational context.

**Seed-based randomization:** Root cause components, data record issues, and deployment metrics all vary per seed. This prevents agents from memorizing solutions and ensures genuine evaluation across episodes.

**Causal propagation:** Wrong actions have consequences across future steps. This forces agents to think ahead rather than greedily optimizing each step independently.

**Multi-step deployment:** The deployment task evolves over 3–5 steps with live metric monitoring, transforming it from a classification problem into genuine sequential decision-making.

**Dense rewards over sparse:** Every step returns signal across 4 dimensions. Agents always receive feedback — no dead zones where reward is zero regardless of behavior.

**Deterministic grading:** All graders use pure Python logic against pre-seeded ground truth. No LLM calls in grading. Fully reproducible across any hardware.

**Safety gate:** Actions that worsen system state score `safety = 0.0` regardless of other dimensions. Agents cannot accidentally exploit reward by making destructive actions.

**Concurrent sessions:** Up to 16 isolated WebSocket sessions supported simultaneously — each connection gets its own MLOpsEnv instance with no shared state.