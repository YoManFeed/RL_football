from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_rl.wrappers.multi_agent_env import ParallelFootballEnv


def main() -> None:
    env = ParallelFootballEnv(
        scenario_name="scenario_7_full_match",
        render_mode=None,
        flatten_observation=True,
        canonical_observation=True,
    )
    observations, infos = env.reset(seed=3)
    print("agents:", list(observations))
    for _ in range(20):
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if not env.agents:
            break
    print("global state shape:", env.state().shape)
    env.close()


if __name__ == "__main__":
    main()
