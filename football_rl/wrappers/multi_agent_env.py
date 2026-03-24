from __future__ import annotations

from typing import Any

import numpy as np
from football_rl.utils.gym_compat import spaces

from football_rl.core.actions import ACTION_SIZE
from football_rl.core.observation import build_flat_observation_space, build_observation_space, flatten_observation
from football_rl.core.simulator import SoccerSimulator
from football_rl.policies.scripted import ZeroPolicy
from football_rl.scenarios.registry import create_scenario


class ParallelFootballEnv:
    metadata = {"name": "ParallelFootballEnv"}

    def __init__(
        self,
        scenario_name: str,
        config=None,
        render_mode: str | None = None,
        canonical_observation: bool = True,
        flatten_observation: bool = False,
        controlled_agents: list[str] | None = None,
        scenario_kwargs: dict[str, Any] | None = None,
        default_policy_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.scenario = create_scenario(scenario_name, **(scenario_kwargs or {}))
        self.simulator = SoccerSimulator(self.scenario, config=config, render_mode=render_mode)
        self.canonical_observation = canonical_observation
        self.flatten_observation = flatten_observation
        self.default_policy_overrides = default_policy_overrides or {}
        self.simulator.reset(seed=0)
        self.possible_agents = controlled_agents or list(self.simulator.default_controlled_agents)
        self.agents = list(self.possible_agents)
        self._action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs_spaces = {}
        for agent_id in self.possible_agents:
            if self.flatten_observation:
                self._obs_spaces[agent_id] = build_flat_observation_space(self.simulator, agent_id, canonical=self.canonical_observation)
            else:
                self._obs_spaces[agent_id] = build_observation_space(self.simulator, agent_id, canonical=self.canonical_observation)

    def observation_space(self, agent_id: str):
        return self._obs_spaces[agent_id]

    def action_space(self, agent_id: str):
        return self._action_space

    def _obs(self, agent_id: str):
        obs = self.simulator.get_observation(agent_id, canonical=self.canonical_observation)
        return flatten_observation(obs) if self.flatten_observation else obs

    def _assemble_actions(self, actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        merged = {}
        for agent_id in self.simulator.players:
            if agent_id in actions:
                merged[agent_id] = np.asarray(actions[agent_id], dtype=np.float32)
            else:
                policy = self.default_policy_overrides.get(agent_id) or self.simulator.default_scripted_agents.get(agent_id) or ZeroPolicy()
                merged[agent_id] = policy.act(self.simulator, agent_id)
        return merged

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.simulator.reset(seed=seed, options=options)
        self.agents = [agent_id for agent_id in self.possible_agents if agent_id in self.simulator.players]
        obs = {agent_id: self._obs(agent_id) for agent_id in self.agents}
        infos = {agent_id: {"team_id": self.simulator.players[agent_id].team_id} for agent_id in self.agents}
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        rewards, terminated, truncated, info = self.simulator.step(self._assemble_actions(actions))
        obs = {agent_id: self._obs(agent_id) for agent_id in self.agents}
        rew = {agent_id: float(rewards.get(agent_id, 0.0)) for agent_id in self.agents}
        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}
        infos = {agent_id: dict(info) for agent_id in self.agents}
        if terminated or truncated:
            self.agents = []
        return obs, rew, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self.simulator.get_global_state()

    def render(self):
        return self.simulator.render()

    def close(self) -> None:
        self.simulator.close()
