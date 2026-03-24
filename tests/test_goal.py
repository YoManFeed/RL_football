import numpy as np

from football_rl.wrappers.gym_env import FootballGymEnv


def test_goal_detection_single_striker():
    env = FootballGymEnv("scenario_1_single_striker", flatten_observation=True)
    obs, _ = env.reset(
        seed=1,
        options={
            "attack_direction": 1,
            "player_positions": {"agent_0": (112.0, 40.0)},
            "ball_position": (115.0, 40.0),
        },
    )
    action = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert "goal_valid" in info["events"]
    assert reward > 0.0
    env.close()
