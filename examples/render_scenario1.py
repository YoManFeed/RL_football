from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_rl.wrappers.gym_env import FootballGymEnv


def main() -> None:
    env = FootballGymEnv(
        scenario_name="scenario_1_single_striker",
        render_mode="human",
        flatten_observation=True,
        canonical_observation=True,
    )
    obs, info = env.reset(seed=7)
    print("obs shape:", obs.shape)
    print("info:", info)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
        if info["step_count"] > 600:
            break
    env.close()


if __name__ == "__main__":
    main()
