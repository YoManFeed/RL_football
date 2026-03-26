"""Potential-based reward shaping: reward the agent for getting closer to the ball.

Without this wrapper the agent has no incentive to move toward the ball in the
first place — ball_progress only fires once the ball is already moving.

The shaping reward is:  APPROACH_SCALE * (prev_dist - curr_dist) / field_width
  * positive when approaching the ball
  * negative when moving away
  * zero-sum over any closed trajectory (potential-based ⇒ optimal policy preserved)
"""

from __future__ import annotations

import numpy as np


class BallApproachWrapper:
    """Wraps a FootballGymEnv to add ball-approach potential shaping."""

    def __init__(self, env, approach_scale: float = 3.0):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.approach_scale = approach_scale
        self._prev_dist: float | None = None

    @staticmethod
    def _dist_to_ball(obs: np.ndarray) -> float:
        """Extract agent-to-ball distance from flattened canonical observation.

        obs layout (flattened, no teammates/opponents in scenario_1):
          [0:11]  self block
          [11:17] ball block — [15:17] = normalize_position(ball - self + [fw/2, fh/2])
            => actual_rel_x = obs[15] * 60,  actual_rel_y = obs[16] * 40
        """
        rel_x = float(obs[15]) * 60.0
        rel_y = float(obs[16]) * 40.0
        return float((rel_x ** 2 + rel_y ** 2) ** 0.5)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = self._dist_to_ball(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_dist = self._dist_to_ball(obs)
        if self._prev_dist is not None:
            shaping = self.approach_scale * (self._prev_dist - curr_dist) / 120.0
            reward += shaping
        self._prev_dist = curr_dist if not (terminated or truncated) else None
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
