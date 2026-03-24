from __future__ import annotations

import numpy as np
from football_rl.utils.gym_compat import spaces

from football_rl.utils.math_utils import normalize_position, normalize_velocity


def _transform(position: np.ndarray, velocity: np.ndarray, attack_direction: int, field_width: float) -> tuple[np.ndarray, np.ndarray]:
    pos = position.copy()
    vel = velocity.copy()
    if attack_direction == -1:
        pos[0] = field_width - pos[0]
        vel[0] = -vel[0]
    return pos, vel


def build_agent_observation(simulator, agent_id: str, canonical: bool | None = None) -> dict[str, np.ndarray]:
    canonical = simulator.config.observation.canonical_default if canonical is None else canonical
    player = simulator.players[agent_id]
    fw = simulator.config.physics.field_width
    fh = simulator.config.physics.field_height
    max_player_speed = simulator.config.player.base_max_speed * simulator.config.player.sprint_multiplier
    max_ball_speed = simulator.config.player.kick_power_max + max_player_speed
    attack_direction = player.attack_direction if canonical else 1

    self_pos, self_vel = _transform(player.position, player.velocity, attack_direction, fw)
    ball_pos, ball_vel = _transform(simulator.ball.position, simulator.ball.velocity, attack_direction, fw)
    own_goal = simulator.own_goal_center(player.team_id).copy()
    opp_goal = simulator.opponent_goal_center(player.team_id).copy()
    if canonical and player.attack_direction == -1:
        own_goal[0] = fw - own_goal[0]
        opp_goal[0] = fw - opp_goal[0]

    teammates = [p for p in simulator.players.values() if p.team_id == player.team_id and p.entity_id != player.entity_id]
    opponents = [p for p in simulator.players.values() if p.team_id != player.team_id]
    teammates.sort(key=lambda x: x.entity_id)
    opponents.sort(key=lambda x: x.entity_id)

    max_teammates = simulator.max_teammates
    max_opponents = simulator.max_opponents
    teammate_array = np.zeros((max_teammates, 8), dtype=np.float32)
    opponent_array = np.zeros((max_opponents, 8), dtype=np.float32)

    for idx, mate in enumerate(teammates[:max_teammates]):
        pos, vel = _transform(mate.position, mate.velocity, attack_direction, fw)
        teammate_array[idx] = np.asarray([
            *normalize_position(pos, fw, fh),
            *normalize_velocity(vel, max_player_speed),
            mate.stamina / mate.stamina_max,
            1.0 if simulator.ball.owner_id == mate.entity_id else 0.0,
            float(mate.role == mate.role.GOALKEEPER),
            float(mate.color_id),
        ], dtype=np.float32)

    for idx, opp in enumerate(opponents[:max_opponents]):
        pos, vel = _transform(opp.position, opp.velocity, attack_direction, fw)
        opponent_array[idx] = np.asarray([
            *normalize_position(pos, fw, fh),
            *normalize_velocity(vel, max_player_speed),
            opp.stamina / opp.stamina_max,
            1.0 if simulator.ball.owner_id == opp.entity_id else 0.0,
            float(opp.role == opp.role.GOALKEEPER),
            float(opp.color_id),
        ], dtype=np.float32)

    self_block = np.asarray([
        *normalize_position(self_pos, fw, fh),
        *normalize_velocity(self_vel, max_player_speed),
        player.stamina / player.stamina_max,
        float(player.team_id),
        float(player.color_id),
        float(player.attack_direction),
        1.0 if simulator.ball.owner_id == player.entity_id else 0.0,
        1.0 if simulator.can_player_kick(player) else 0.0,
        float(player.role == player.role.GOALKEEPER),
    ], dtype=np.float32)

    ball_block = np.asarray([
        *normalize_position(ball_pos, fw, fh),
        *normalize_velocity(ball_vel, max_ball_speed),
        *normalize_position(ball_pos - self_pos + np.asarray([fw / 2, fh / 2], dtype=np.float32), fw, fh),
    ], dtype=np.float32)

    goal_block = np.asarray([
        *normalize_position(own_goal, fw, fh),
        *normalize_position(opp_goal, fw, fh),
        float(player.attack_direction),
        simulator.remaining_progress(),
    ], dtype=np.float32)

    score_for = simulator.score_by_team.get(player.team_id, 0)
    score_against = max([score for team_id, score in simulator.score_by_team.items() if team_id != player.team_id] + [0])
    meta = np.asarray([
        simulator.remaining_progress(),
        float(score_for),
        float(score_against),
        float(simulator.ball.last_touch_team_id == player.team_id if simulator.ball.last_touch_team_id is not None else 0.0),
        float(simulator.pass_validation_progress(player.team_id)),
        float(len(simulator.events)),
    ], dtype=np.float32)

    return {
        "self": self_block,
        "ball": ball_block,
        "teammates": teammate_array,
        "opponents": opponent_array,
        "goals": goal_block,
        "meta": meta,
    }


def flatten_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for key in ("self", "ball", "teammates", "opponents", "goals", "meta"):
        value = obs[key]
        chunks.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_observation_space(simulator, agent_id: str, canonical: bool | None = None) -> spaces.Dict:
    obs = build_agent_observation(simulator, agent_id, canonical=canonical)
    return spaces.Dict({
        key: spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)
        for key, value in obs.items()
    })


def build_flat_observation_space(simulator, agent_id: str, canonical: bool | None = None) -> spaces.Box:
    obs = flatten_observation(build_agent_observation(simulator, agent_id, canonical=canonical))
    return spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)


def build_global_state(simulator) -> np.ndarray:
    fw = simulator.config.physics.field_width
    fh = simulator.config.physics.field_height
    max_player_speed = simulator.config.player.base_max_speed * simulator.config.player.sprint_multiplier
    values: list[np.ndarray] = []
    values.append(normalize_position(simulator.ball.position, fw, fh))
    values.append(normalize_velocity(simulator.ball.velocity, simulator.config.player.kick_power_max + max_player_speed))
    for agent_id in sorted(simulator.players):
        player = simulator.players[agent_id]
        values.append(np.asarray([
            *normalize_position(player.position, fw, fh),
            *normalize_velocity(player.velocity, max_player_speed),
            player.stamina / player.stamina_max,
            float(player.team_id),
            float(player.color_id),
            float(player.attack_direction),
            float(player.role == player.role.GOALKEEPER),
        ], dtype=np.float32))
    values.append(np.asarray([simulator.remaining_progress()], dtype=np.float32))
    values.append(np.asarray([float(simulator.score_by_team.get(0, 0)), float(simulator.score_by_team.get(1, 0))], dtype=np.float32))
    return np.concatenate([v.reshape(-1) for v in values]).astype(np.float32)
