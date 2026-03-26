from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PhysicsConfig:
    field_width: float = 120.0
    field_height: float = 80.0
    goal_width: float = 24.0
    dt: float = 0.12
    player_radius: float = 2.0
    ball_radius: float = 1.0
    obstacle_half_width: float = 1.8
    obstacle_half_height: float = 8.0
    restitution: float = 0.82
    player_ball_restitution: float = 0.35
    wall_restitution: float = 0.92
    control_margin: float = 1.35
    kick_distance_margin: float = 1.5
    possession_speed_threshold: float = 22.0
    player_drag: float = 0.88
    ball_decay: float = 0.965
    max_episode_steps: int = 320


@dataclass(slots=True)
class PlayerConfig:
    base_max_speed: float = 14.0
    sprint_multiplier: float = 1.35
    acceleration: float = 32.0
    kick_power_min: float = 10.0
    kick_power_max: float = 36.0
    stamina_max: float = 100.0
    stamina_sprint_drain: float = 26.0
    stamina_recovery: float = 15.0
    stamina_min_for_sprint: float = 5.0
    goalkeeper_speed_multiplier: float = 0.92
    goalkeeper_acceleration_multiplier: float = 1.05


@dataclass(slots=True)
class BallConfig:
    launch_from_player_velocity_scale: float = 0.2
    dribble_velocity_scale: float = 0.62
    dribble_follow_distance: float = 0.8
    allow_dribble: bool = True  # False = kick-only, ball never sticks to player


@dataclass(slots=True)
class ObservationConfig:
    canonical_default: bool = True
    teammate_features: int = 8
    opponent_features: int = 8
    include_global_positions: bool = True


@dataclass(slots=True)
class RewardConfig:
    goal_reward: float = 50.0
    concede_penalty: float = 5.0
    ball_progress_scale: float = 0.15
    ball_progress_euclidean: bool = False
    pass_reward: float = 0.45
    receive_pass_reward: float = 0.25
    steal_reward: float = 0.6
    interception_reward: float = 0.45
    touch_reward: float = 0.15
    idle_penalty: float = 0.003
    wall_bounce_penalty: float = 0.1
    time_penalty: float = 0.002
    scenario_bonus_scale: float = 1.0


@dataclass(slots=True)
class RandomizationConfig:
    enable: bool = True
    randomize_attack_direction: bool = True
    randomize_team_colors: bool = True
    player_spawn_jitter: float = 5.0
    ball_spawn_jitter: float = 3.0
    obstacle_size_jitter: float = 1.5
    obstacle_speed_scale_range: tuple[float, float] = (0.85, 1.2)
    goalkeeper_spawn_jitter: float = 2.5
    episode_length_jitter: int = 25


@dataclass(slots=True)
class SimulatorConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    render_fps: int = 60
    background_color: tuple[int, int, int] = (16, 90, 32)
    team_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = ((220, 55, 55), (55, 110, 220))


def make_default_config() -> SimulatorConfig:
    return SimulatorConfig()
