from football_rl.scenarios.registry import create_scenario, list_scenarios
from football_rl.wrappers.gym_env import FootballGymEnv


def test_all_scenarios_load():
    names = list_scenarios()
    assert len(names) == 7
    for name in names:
        scenario = create_scenario(name)
        assert scenario.name == name
        env = FootballGymEnv(name, flatten_observation=True)
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] > 0
        env.close()
