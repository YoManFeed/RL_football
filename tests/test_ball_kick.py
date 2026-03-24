import numpy as np

from football_rl.wrappers.gym_env import FootballGymEnv


def test_ball_kick_changes_ball_velocity():
    env = FootballGymEnv("scenario_1_single_striker", flatten_observation=True)
    env.reset(
        seed=2,
        options={
            "attack_direction": 1,
            "player_positions": {"agent_0": (50.0, 40.0)},
            "ball_position": (53.0, 40.0),
        },
    )
    action = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    env.step(action)
    assert env.simulator.ball.velocity[0] > 0.0
    env.close()
