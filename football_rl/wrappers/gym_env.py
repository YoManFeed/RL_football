from __future__ import annotations

from typing import Any

from football_rl.utils.gym_compat import gym
import numpy as np
from football_rl.utils.gym_compat import spaces

from football_rl.core.actions import ACTION_SIZE
from football_rl.core.observation import build_flat_observation_space, build_observation_space, flatten_observation
from football_rl.core.simulator import SoccerSimulator
from football_rl.policies.scripted import ZeroPolicy
from football_rl.scenarios.registry import create_scenario


class FootballGymEnv(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(
        self,
        scenario_name: str,
        config=None,
        render_mode: str | None = None,
        controlled_agent_id: str | None = None,
        canonical_observation: bool = True,
        flatten_observation: bool = False,
        scenario_kwargs: dict[str, Any] | None = None,
        default_policy_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.scenario_kwargs = scenario_kwargs or {}
        self.scenario = create_scenario(scenario_name, **self.scenario_kwargs)
        self.simulator = SoccerSimulator(self.scenario, config=config, render_mode=render_mode)
        self.canonical_observation = canonical_observation
        self.flatten_observation = flatten_observation
        self.default_policy_overrides = default_policy_overrides or {}
        self.simulator.reset(seed=0)
        self.controlled_agent_id = controlled_agent_id or self.simulator.default_controlled_agents[0]
        if self.flatten_observation:
            self.observation_space = build_flat_observation_space(self.simulator, self.controlled_agent_id, canonical=self.canonical_observation)
        else:
            self.observation_space = build_observation_space(self.simulator, self.controlled_agent_id, canonical=self.canonical_observation)
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _assemble_actions(self, action: np.ndarray) -> dict[str, np.ndarray]:
        actions = {}
        for agent_id, player in self.simulator.players.items():
            if agent_id == self.controlled_agent_id:
                actions[agent_id] = np.asarray(action, dtype=np.float32)
            else:
                policy = self.default_policy_overrides.get(agent_id) or self.simulator.default_scripted_agents.get(agent_id) or ZeroPolicy()
                actions[agent_id] = policy.act(self.simulator, agent_id)
        return actions

    def _obs(self):
        obs = self.simulator.get_observation(self.controlled_agent_id, canonical=self.canonical_observation)
        if self.flatten_observation:
            return flatten_observation(obs)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.simulator.reset(seed=seed, options=options)
        if self.controlled_agent_id not in self.simulator.players:
            self.controlled_agent_id = self.simulator.default_controlled_agents[0]
        return self._obs(), {"controlled_agent_id": self.controlled_agent_id}

    def step(self, action: np.ndarray):
        rewards, terminated, truncated, info = self.simulator.step(self._assemble_actions(action))
        obs = self._obs()
        reward = float(rewards[self.controlled_agent_id])
        info = {**info, "agent_reward_breakdown": rewards, "controlled_agent_id": self.controlled_agent_id}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.simulator.render()

    def close(self) -> None:
        self.simulator.close()
