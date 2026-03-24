from __future__ import annotations

from football_rl.utils.gym_compat import gym

from football_rl.core.observation import build_flat_observation_space, flatten_observation


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = build_flat_observation_space(env.simulator, env.controlled_agent_id, canonical=env.canonical_observation)

    def observation(self, observation):
        return flatten_observation(observation)
