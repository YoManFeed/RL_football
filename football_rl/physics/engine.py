from __future__ import annotations

import numpy as np

from football_rl.configs.defaults import SimulatorConfig
from football_rl.entities.ball import Ball
from football_rl.entities.obstacle import MovingWall
from football_rl.entities.player import Player
from football_rl.utils.math_utils import clip_vector, l2norm, normalize_direction


class PhysicsEngine:
    def __init__(self, config: SimulatorConfig):
        self.config = config

    def step_player(self, player: Player, action_move: np.ndarray, sprint: float) -> None:
        cfg = self.config
        desired_dir = normalize_direction(action_move)
        sprint_enabled = sprint > 0.5 and player.stamina > cfg.player.stamina_min_for_sprint
        speed_multiplier = cfg.player.sprint_multiplier if sprint_enabled else 1.0
        accel = player.acceleration
        max_speed = player.max_speed * speed_multiplier
        player.velocity = player.velocity * cfg.physics.player_drag + desired_dir * accel * cfg.physics.dt
        player.velocity = clip_vector(player.velocity, max_speed)
        player.position = player.position + player.velocity * cfg.physics.dt
        self._clip_player_to_bounds(player)
        if sprint_enabled:
            player.stamina = max(0.0, player.stamina - cfg.player.stamina_sprint_drain * cfg.physics.dt)
        else:
            player.stamina = min(player.stamina_max, player.stamina + cfg.player.stamina_recovery * cfg.physics.dt)

    def _clip_player_to_bounds(self, player: Player) -> None:
        cfg = self.config.physics
        x_low = player.radius
        x_high = cfg.field_width - player.radius
        y_low = player.radius
        y_high = cfg.field_height - player.radius
        if player.goalkeeper_zone is not None:
            x_low, x_high, y_low, y_high = player.goalkeeper_zone
            x_low += player.radius
            x_high -= player.radius
            y_low += player.radius
            y_high -= player.radius
        player.position[0] = float(np.clip(player.position[0], x_low, x_high))
        player.position[1] = float(np.clip(player.position[1], y_low, y_high))

    def step_ball(self, ball: Ball) -> None:
        ball.position = ball.position + ball.velocity * self.config.physics.dt
        ball.velocity = ball.velocity * self.config.physics.ball_decay

    def apply_player_ball_overlap(self, player: Player, ball: Ball) -> None:
        delta = ball.position - player.position
        dist = l2norm(delta)
        min_dist = player.radius + ball.radius
        if dist < 1e-6:
            delta = np.asarray([1.0, 0.0], dtype=np.float32)
            dist = 1.0
        if dist < min_dist:
            direction = delta / dist
            correction = min_dist - dist + 1e-3
            ball.position = ball.position + direction * correction
            ball.velocity = ball.velocity + player.velocity * self.config.physics.player_ball_restitution

    def step_obstacle(self, obstacle: MovingWall) -> None:
        axis = obstacle.movement_axis
        obstacle.position[axis] += obstacle.direction * obstacle.speed * self.config.physics.dt
        if obstacle.position[axis] < obstacle.min_coord:
            obstacle.position[axis] = obstacle.min_coord
            obstacle.direction *= -1.0
        elif obstacle.position[axis] > obstacle.max_coord:
            obstacle.position[axis] = obstacle.max_coord
            obstacle.direction *= -1.0

    def collide_ball_with_obstacle(self, ball: Ball, obstacle: MovingWall) -> bool:
        delta = ball.position - obstacle.position
        closest = np.clip(delta, -obstacle.half_extents, obstacle.half_extents)
        nearest = obstacle.position + closest
        offset = ball.position - nearest
        dist = l2norm(offset)
        if dist >= ball.radius:
            return False
        if dist < 1e-6:
            if abs(delta[0]) > abs(delta[1]):
                normal = np.asarray([np.sign(delta[0]) or 1.0, 0.0], dtype=np.float32)
            else:
                normal = np.asarray([0.0, np.sign(delta[1]) or 1.0], dtype=np.float32)
        else:
            normal = offset / dist
        penetration = ball.radius - max(dist, 1e-6)
        ball.position = ball.position + normal * (penetration + 1e-3)
        ball.velocity = ball.velocity - 2.0 * np.dot(ball.velocity, normal) * normal
        ball.velocity = ball.velocity * self.config.physics.wall_restitution
        return True
