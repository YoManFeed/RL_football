from football_rl.wrappers.multi_agent_env import ParallelFootballEnv


def test_multiagent_full_match_reset_and_step():
    env = ParallelFootballEnv("scenario_7_full_match", flatten_observation=True)
    obs, infos = env.reset(seed=7)
    assert len(obs) == 6
    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(rewards, dict)
    assert env.state().ndim == 1
    env.close()
