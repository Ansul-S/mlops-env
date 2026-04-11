"""Basic tests for MLOpsEnv — validates grader fairness."""
from env.environment import MLOpsEnv
from env.models import Action

def test_reset_returns_valid_observation():
    env = MLOpsEnv()
    for task_id in ["data_quality_triage", "deployment_decision", "incident_cascade"]:
        result = env.reset(task_id, seed=42)
        assert result is not None
        assert result.observation is not None

def test_different_seeds_give_different_scenarios():
    env = MLOpsEnv()
    r1 = env.reset("incident_cascade", seed=42)
    r2 = env.reset("incident_cascade", seed=137)
    assert r1.observation.task_context != r2.observation.task_context

def test_ground_truth_not_in_observation():
    env = MLOpsEnv()
    result = env.reset("data_quality_triage", seed=42)
    obs_dict = result.observation.model_dump(mode="json")
    for rec in obs_dict["data_records"]:
        assert "ground_truth_action" not in rec
        assert "ground_truth_params" not in rec

def test_optimal_incident_scores_above_threshold():
    env = MLOpsEnv()
    result = env.reset("incident_cascade", seed=42)
    root = env._sim.root_cause
    actions = [
        Action(action_type="investigate", parameters={"component": root}, reasoning="find root cause"),
        Action(action_type="restart_service", parameters={"component": root}, reasoning="fix root"),
    ]
    rewards = []
    for action in actions:
        result = env.step(action)
        rewards.append(result.reward)
        if result.done:
            break
    assert sum(rewards) / len(rewards) >= 0.50