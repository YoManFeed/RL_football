from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_rl.configs.defaults import SimulatorConfig, make_default_config
from football_rl.core.actions import PlayerAction
from football_rl.core.events import Event, EventType
from football_rl.core.observation import build_agent_observation, build_global_state
from football_rl.entities.ball import Ball
from football_rl.entities.obstacle import MovingWall
from football_rl.entities.player import Player
from football_rl.physics.engine import PhysicsEngine
from football_rl.render.pygame_renderer import PygameRenderer
from football_rl.rewards.manager import RewardManager, build_default_reward_manager
from football_rl.utils.math_utils import l2norm, normalize_direction
from football_rl.utils.seeding import make_rng


@dataclass(slots=True)
class KickState:
    kicker_id: str
    kicker_team_id: int
    origin: np.ndarray
    step: int


class SoccerSimulator:
    def __init__(
        self,
        scenario,
        config: SimulatorConfig | None = None,
        reward_manager: RewardManager | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or make_default_config()
        self.reward_manager = reward_manager or build_default_reward_manager()
        self.scenario = scenario
        self.render_mode = render_mode
        self.physics = PhysicsEngine(self.config)
        self.renderer = PygameRenderer(self.config) if render_mode is not None else None
        self.rng = make_rng(None)
        self.players: dict[str, Player] = {}
        self.obstacles: list[MovingWall] = []
        self.ball = Ball(entity_id="ball")
        self.events: list[Event] = []
        self.extra_agent_rewards: dict[str, float] = {}
        self.score_by_team: dict[int, int] = {0: 0, 1: 0}
        self.step_count = 0
        self.max_episode_steps = self.config.physics.max_episode_steps
        self.terminated = False
        self.truncated = False
        self.scenario_metadata: dict[str, Any] = {}
        self.default_scripted_agents: dict[str, Any] = {}
        self.default_controlled_agents: list[str] = []
        self.prev_ball_position = np.zeros(2, dtype=np.float32)
        self.prev_owner_id: str | None = None
        self.pending_pass: KickState | None = None
        self.kickoff_snapshot: dict[str, Any] | None = None
        self.max_teammates = 0
        self.max_opponents = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = make_rng(seed)
        self.events = []
        self.extra_agent_rewards = {}
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.score_by_team = {0: 0, 1: 0}
        state = self.scenario.reset(self.config, self.rng, options=options)
        self.players = {player.entity_id: player for player in state.players}
        self.ball = state.ball
        self.obstacles = list(state.obstacles)
        self.scenario_metadata = deepcopy(state.metadata)
        self.default_scripted_agents = dict(state.default_scripted_agents)
        self.default_controlled_agents = list(state.default_controlled_agents)
        self.max_episode_steps = state.max_episode_steps or self.config.physics.max_episode_steps
        self.prev_ball_position = self.ball.position.copy()
        self.prev_owner_id = self.ball.owner_id
        self.pending_pass = None
        self._compute_team_sizes()
        self.capture_kickoff_snapshot()
        self.scenario.on_reset(self)
        return {"events": [], "score": self.score_by_team.copy()}

    def _compute_team_sizes(self) -> None:
        by_team: dict[int, int] = {}
        for player in self.players.values():
            by_team[player.team_id] = by_team.get(player.team_id, 0) + 1
        max_team = max(by_team.values()) if by_team else 1
        self.max_teammates = max_team - 1
        opponent_total = sum(by_team.values()) - min(by_team.values()) if len(by_team) > 1 else 0
        self.max_opponents = max(opponent_total, max(by_team.values()) if len(by_team) > 1 else 0)

    def capture_kickoff_snapshot(self) -> None:
        self.kickoff_snapshot = {
            "players": {agent_id: player.copy() for agent_id, player in self.players.items()},
            "ball": self.ball.copy(),
            "obstacles": [obstacle.copy() for obstacle in self.obstacles],
        }

    def restore_kickoff_snapshot(self) -> None:
        if self.kickoff_snapshot is None:
            return
        self.players = {agent_id: player.copy() for agent_id, player in self.kickoff_snapshot["players"].items()}
        self.ball = self.kickoff_snapshot["ball"].copy()
        self.obstacles = [obstacle.copy() for obstacle in self.kickoff_snapshot["obstacles"]]
        self.pending_pass = None
        self.prev_owner_id = self.ball.owner_id
        self.prev_ball_position = self.ball.position.copy()

    def own_goal_center(self, team_id: int) -> np.ndarray:
        attack_dir = next(player.attack_direction for player in self.players.values() if player.team_id == team_id)
        x = 0.0 if attack_dir == 1 else self.config.physics.field_width
        return np.asarray([x, self.config.physics.field_height / 2], dtype=np.float32)

    def opponent_goal_center(self, team_id: int) -> np.ndarray:
        attack_dir = next(player.attack_direction for player in self.players.values() if player.team_id == team_id)
        x = self.config.physics.field_width if attack_dir == 1 else 0.0
        return np.asarray([x, self.config.physics.field_height / 2], dtype=np.float32)

    def remaining_progress(self) -> float:
        if self.max_episode_steps <= 0:
            return 0.0
        return float(1.0 - (self.step_count / self.max_episode_steps))

    def pass_validation_progress(self, team_id: int) -> float:
        required = int(self.scenario_metadata.get("minimum_required_passes_before_goal_counts", 0))
        if required <= 0:
            return 1.0
        completed = int(self.scenario_metadata.get("completed_passes", 0))
        return min(1.0, completed / max(required, 1))

    def can_player_control_ball(self, player: Player) -> bool:
        dist = l2norm(self.ball.position - player.position)
        threshold = player.radius + self.ball.radius + self.config.physics.control_margin
        return dist <= threshold and l2norm(self.ball.velocity) <= self.config.physics.possession_speed_threshold

    def can_player_kick(self, player: Player) -> bool:
        dist = l2norm(self.ball.position - player.position)
        threshold = player.radius + self.ball.radius + self.config.physics.kick_distance_margin
        return dist <= threshold

    def get_observation(self, agent_id: str, canonical: bool | None = None) -> dict[str, np.ndarray]:
        return build_agent_observation(self, agent_id, canonical=canonical)

    def get_global_state(self) -> np.ndarray:
        return build_global_state(self)

    def active_agents(self) -> list[str]:
        return sorted(self.players.keys())

    def _record(self, event_type: EventType, **data: Any) -> None:
        self.events.append(Event(type=event_type, step=self.step_count, data=data))

    def _clear_step_transients(self) -> None:
        self.events = []
        self.extra_agent_rewards = {agent_id: 0.0 for agent_id in self.players}

    def _kick_ball(self, player: Player, action: PlayerAction) -> bool:
        if action.kick_power <= 0.01 or not self.can_player_kick(player):
            return False
        direction = normalize_direction(action.kick_direction)
        if l2norm(direction) < 1e-6:
            direction = normalize_direction(self.opponent_goal_center(player.team_id) - self.ball.position)
        power = self.config.player.kick_power_min + action.kick_power * (self.config.player.kick_power_max - self.config.player.kick_power_min)
        self.ball.velocity = direction * power + player.velocity * self.config.ball.launch_from_player_velocity_scale
        self.ball.last_touch_player_id = player.entity_id
        self.ball.last_touch_team_id = player.team_id
        self.ball.owner_id = None
        self.pending_pass = KickState(
            kicker_id=player.entity_id,
            kicker_team_id=player.team_id,
            origin=self.ball.position.copy(),
            step=self.step_count,
        )
        self._record(EventType.KICK, player_id=player.entity_id, team_id=player.team_id, power=float(power))
        return True

    def _update_possession(self) -> None:
        prev_owner = self.ball.owner_id
        candidates = [player for player in self.players.values() if self.can_player_control_ball(player)]
        if not candidates:
            self.ball.owner_id = None
            return
        owner = min(candidates, key=lambda p: l2norm(self.ball.position - p.position))
        self.ball.owner_id = owner.entity_id
        self.ball.last_touch_player_id = owner.entity_id
        self.ball.last_touch_team_id = owner.team_id
        if prev_owner != owner.entity_id:
            self._record(EventType.POSSESSION_GAIN, player_id=owner.entity_id, team_id=owner.team_id)
            if prev_owner is not None:
                self._record(EventType.POSSESSION_LOST, player_id=prev_owner)
            if self.pending_pass is not None and self.pending_pass.kicker_id != owner.entity_id:
                origin_dist = l2norm(owner.position - self.pending_pass.origin)
                if owner.team_id == self.pending_pass.kicker_team_id and origin_dist > self.config.physics.player_radius * 2:
                    self._record(
                        EventType.PASS_COMPLETED,
                        from_player_id=self.pending_pass.kicker_id,
                        to_player_id=owner.entity_id,
                        team_id=owner.team_id,
                    )
                    self.scenario_metadata["completed_passes"] = int(self.scenario_metadata.get("completed_passes", 0)) + 1
                elif owner.team_id != self.pending_pass.kicker_team_id:
                    self._record(EventType.INTERCEPTION, player_id=owner.entity_id, team_id=owner.team_id)
                self.pending_pass = None
            elif prev_owner is not None and self.players[prev_owner].team_id != owner.team_id:
                self._record(EventType.STEAL, player_id=owner.entity_id, team_id=owner.team_id)
        if self.ball.owner_id is not None and self.config.ball.allow_dribble:
            player = self.players[self.ball.owner_id]
            offset = self.ball.position - player.position
            if l2norm(offset) < 1e-6:
                offset = np.asarray([player.attack_direction, 0.0], dtype=np.float32)
            direction = normalize_direction(offset)
            self.ball.position = player.position + direction * (player.radius + self.ball.radius + self.config.ball.dribble_follow_distance)
            self.ball.velocity = self.ball.velocity * 0.4 + player.velocity * self.config.ball.dribble_velocity_scale

    def _handle_ball_bounds_and_goal(self) -> int | None:
        fw = self.config.physics.field_width
        fh = self.config.physics.field_height
        gw = self.config.physics.goal_width / 2
        x = self.ball.position[0]
        y = self.ball.position[1]
        crossed_left = x - self.ball.radius <= 0.0
        crossed_right = x + self.ball.radius >= fw
        in_goal_mouth = (fh / 2 - gw) <= y <= (fh / 2 + gw)
        if in_goal_mouth and (crossed_left or crossed_right):
            goal_side = -1 if crossed_left else 1
            scoring_team = None
            for player in self.players.values():
                if player.attack_direction == goal_side:
                    scoring_team = player.team_id
                    break
            if scoring_team is not None:
                return scoring_team
        bounced = False
        if self.ball.position[1] - self.ball.radius < 0.0:
            self.ball.position[1] = self.ball.radius
            self.ball.velocity[1] *= -self.config.physics.restitution
            bounced = True
        elif self.ball.position[1] + self.ball.radius > fh:
            self.ball.position[1] = fh - self.ball.radius
            self.ball.velocity[1] *= -self.config.physics.restitution
            bounced = True
        if self.ball.position[0] - self.ball.radius < 0.0:
            self.ball.position[0] = self.ball.radius
            self.ball.velocity[0] *= -self.config.physics.restitution
            bounced = True
            self._record(EventType.OUT_OF_BOUNDS, side="left")
        elif self.ball.position[0] + self.ball.radius > fw:
            self.ball.position[0] = fw - self.ball.radius
            self.ball.velocity[0] *= -self.config.physics.restitution
            bounced = True
            self._record(EventType.OUT_OF_BOUNDS, side="right")
        if bounced:
            self.pending_pass = None
        return None

    def step(self, action_map: dict[str, np.ndarray | PlayerAction]) -> tuple[dict[str, float], bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a finished episode. Call reset().")
        self._clear_step_transients()
        self.prev_ball_position = self.ball.position.copy()
        self.prev_owner_id = self.ball.owner_id
        self.step_count += 1

        parsed_actions: dict[str, PlayerAction] = {}
        for agent_id, player in self.players.items():
            raw = action_map.get(agent_id, PlayerAction.zero())
            action = raw if isinstance(raw, PlayerAction) else PlayerAction.from_array(raw)
            parsed_actions[agent_id] = action
            player.last_action = action.to_array()

        for agent_id, action in parsed_actions.items():
            self.physics.step_player(self.players[agent_id], action.move, action.sprint)

        kickers = sorted(self.players.values(), key=lambda p: l2norm(self.ball.position - p.position))
        for player in kickers:
            self._kick_ball(player, parsed_actions[player.entity_id])

        for obstacle in self.obstacles:
            self.physics.step_obstacle(obstacle)

        self.physics.step_ball(self.ball)

        for player in self.players.values():
            self.physics.apply_player_ball_overlap(player, self.ball)

        for obstacle in self.obstacles:
            if self.physics.collide_ball_with_obstacle(self.ball, obstacle):
                self._record(EventType.BALL_WALL_BOUNCE, obstacle_id=obstacle.entity_id)
                self.pending_pass = None

        self._update_possession()

        scoring_team = self._handle_ball_bounds_and_goal()
        if scoring_team is not None:
            valid = self.scenario.on_goal_scored(self, scoring_team)
            if valid:
                self.score_by_team[scoring_team] = self.score_by_team.get(scoring_team, 0) + 1
                self._record(EventType.GOAL_VALID, team_id=scoring_team)
                self.capture_kickoff_snapshot()
                if self.scenario.terminate_on_goal:
                    self.terminated = True
            else:
                self._record(EventType.GOAL_INVALID, team_id=scoring_team)
                self.scenario.on_invalid_goal(self, scoring_team)

        self.scenario.on_step_end(self)

        if self.scenario.score_limit is not None:
            if any(score >= self.scenario.score_limit for score in self.score_by_team.values()):
                self.terminated = True

        if self.step_count >= self.max_episode_steps:
            self.truncated = True

        rewards = self.reward_manager.compute(self)
        info = {
            "events": [event.type.value for event in self.events],
            "event_payloads": [event.data for event in self.events],
            "score": self.score_by_team.copy(),
            "ball_owner": self.ball.owner_id,
            "completed_passes": int(self.scenario_metadata.get("completed_passes", 0)),
            "required_passes": int(self.scenario_metadata.get("minimum_required_passes_before_goal_counts", 0)),
            "step_count": self.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        return rewards, self.terminated, self.truncated, info

    def render(self):
        if self.renderer is None:
            return None
        return self.renderer.render(self, mode=self.render_mode or "human")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
