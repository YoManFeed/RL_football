from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_rl.wrappers.gym_env import FootballGymEnv


def main() -> None:
    env = FootballGymEnv(
        scenario_name="scenario_2_moving_wall",
        render_mode=None,
        flatten_observation=True,
        canonical_observation=True,
    )
    obs, _ = env.reset(seed=11)
    total_reward = 0.0
    for _ in range(32):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print("final observation shape:", obs.shape)
    print("total reward:", total_reward)
    print("events:", info["events"])
    env.close()


if __name__ == "__main__":
    main()
