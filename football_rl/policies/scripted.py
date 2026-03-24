from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_rl.core.actions import PlayerAction
from football_rl.utils.math_utils import normalize_direction


class BasePolicy:
    def act(self, simulator, agent_id: str) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class ZeroPolicy(BasePolicy):
    def act(self, simulator, agent_id: str) -> np.ndarray:
        return PlayerAction.zero().to_array()


@dataclass(slots=True)
class RandomPolicy(BasePolicy):
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, simulator, agent_id: str) -> np.ndarray:
        action = self.rng.uniform(low=-1.0, high=1.0, size=6).astype(np.float32)
        action[4:] = self.rng.uniform(low=0.0, high=1.0, size=2).astype(np.float32)
        return action


@dataclass(slots=True)
class ChaseBallPolicy(BasePolicy):
    sprint: float = 1.0

    def act(self, simulator, agent_id: str) -> np.ndarray:
        player = simulator.players[agent_id]
        target_dir = normalize_direction(simulator.ball.position - player.position)
        kick_dir = normalize_direction(simulator.opponent_goal_center(player.team_id) - simulator.ball.position)
        can_kick = simulator.can_player_kick(player)
        power = 1.0 if can_kick else 0.0
        return np.asarray([target_dir[0], target_dir[1], kick_dir[0], kick_dir[1], power, self.sprint], dtype=np.float32)


@dataclass(slots=True)
class SimpleAttackerPolicy(BasePolicy):
    pass_preference: float = 0.4

    def act(self, simulator, agent_id: str) -> np.ndarray:
        player = simulator.players[agent_id]
        ball = simulator.ball
        move = normalize_direction(ball.position - player.position)
        if simulator.can_player_kick(player):
            teammates = [p for p in simulator.players.values() if p.team_id == player.team_id and p.entity_id != player.entity_id]
            kick_target = simulator.opponent_goal_center(player.team_id)
            if teammates and simulator.rng.random() < self.pass_preference:
                teammate = min(teammates, key=lambda t: float(np.linalg.norm(t.position - player.position)))
                kick_target = teammate.position
            kick_dir = normalize_direction(kick_target - ball.position)
            return np.asarray([move[0], move[1], kick_dir[0], kick_dir[1], 1.0, 0.7], dtype=np.float32)
        return np.asarray([move[0], move[1], 0.0, 0.0, 0.0, 0.7], dtype=np.float32)


@dataclass(slots=True)
class GoalkeeperPolicy(BasePolicy):
    clear_power: float = 1.0

    def act(self, simulator, agent_id: str) -> np.ndarray:
        player = simulator.players[agent_id]
        ball = simulator.ball
        own_goal = simulator.own_goal_center(player.team_id)
        x_target = own_goal[0] + player.attack_direction * 8.0
        y_target = float(np.clip(ball.position[1], player.goalkeeper_zone[2] + player.radius, player.goalkeeper_zone[3] - player.radius))
        target = np.asarray([x_target, y_target], dtype=np.float32)
        move = normalize_direction(target - player.position)
        if simulator.can_player_kick(player):
            clear_target = simulator.opponent_goal_center(player.team_id)
            kick_dir = normalize_direction(clear_target - ball.position)
            return np.asarray([move[0], move[1], kick_dir[0], kick_dir[1], self.clear_power, 0.8], dtype=np.float32)
        return np.asarray([move[0], move[1], 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
