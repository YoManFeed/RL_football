from football_rl.wrappers.gym_env import FootballGymEnv


def test_reset_returns_valid_observation():
    env = FootballGymEnv("scenario_1_single_striker", flatten_observation=True)
    obs, info = env.reset(seed=123)
    assert obs.ndim == 1
    assert obs.shape[0] > 0
    assert info["controlled_agent_id"] == "agent_0"
    env.close()
