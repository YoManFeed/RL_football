from __future__ import annotations

import numpy as np

from football_rl.core.types import PlayerRole
from football_rl.entities.ball import Ball
from football_rl.entities.obstacle import MovingWall
from football_rl.entities.player import Player


def team_attack_dirs(rng, randomize: bool, two_teams: bool) -> dict[int, int]:
    base = 1
    if randomize and rng.random() < 0.5:
        base = -1
    if two_teams:
        return {0: base, 1: -base}
    return {0: base}


def team_color_ids(rng, randomize: bool, teams: list[int]) -> dict[int, int]:
    mapping = {team_id: idx for idx, team_id in enumerate(sorted(teams))}
    if randomize and len(teams) == 2 and rng.random() < 0.5:
        mapping = {teams[0]: 1, teams[1]: 0}
    return mapping


def player(
    agent_id: str,
    team_id: int,
    pos: tuple[float, float],
    attack_direction: int,
    color_id: int,
    cfg,
    role: PlayerRole = PlayerRole.STRIKER,
    scripted: bool = False,
    goalkeeper_zone: tuple[float, float, float, float] | None = None,
) -> Player:
    speed = cfg.player.base_max_speed
    accel = cfg.player.acceleration
    if role is PlayerRole.GOALKEEPER:
        speed *= cfg.player.goalkeeper_speed_multiplier
        accel *= cfg.player.goalkeeper_acceleration_multiplier
    return Player(
        entity_id=agent_id,
        team_id=team_id,
        role=role,
        color_id=color_id,
        attack_direction=attack_direction,
        position=np.asarray(pos, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        radius=cfg.physics.player_radius,
        max_speed=speed,
        acceleration=accel,
        stamina=cfg.player.stamina_max,
        stamina_max=cfg.player.stamina_max,
        is_scripted=scripted,
        goalkeeper_zone=goalkeeper_zone,
    )


def ball(pos: tuple[float, float], cfg) -> Ball:
    return Ball(
        entity_id="ball",
        position=np.asarray(pos, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        radius=cfg.physics.ball_radius,
    )


def moving_wall(entity_id: str, pos: tuple[float, float], cfg, rng, speed: float = 10.0) -> MovingWall:
    jitter = cfg.randomization.obstacle_size_jitter
    half_h = cfg.physics.obstacle_half_height + rng.uniform(-jitter, jitter)
    half_w = cfg.physics.obstacle_half_width
    scale_low, scale_high = cfg.randomization.obstacle_speed_scale_range
    speed = speed * rng.uniform(scale_low, scale_high)
    return MovingWall(
        entity_id=entity_id,
        position=np.asarray(pos, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        radius=0.0,
        half_extents=np.asarray([half_w, half_h], dtype=np.float32),
        movement_axis=1,
        min_coord=cfg.physics.goal_width * 0.45,
        max_coord=cfg.physics.field_height - cfg.physics.goal_width * 0.45,
        speed=float(speed),
        direction=1.0,
    )


def apply_spawn_jitter(value: tuple[float, float], rng, magnitude: float) -> tuple[float, float]:
    dx = rng.uniform(-magnitude, magnitude)
    dy = rng.uniform(-magnitude, magnitude)
    return float(value[0] + dx), float(value[1] + dy)
