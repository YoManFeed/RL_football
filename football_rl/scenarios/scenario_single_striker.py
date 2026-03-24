from __future__ import annotations

from football_rl.core.types import PlayerRole
from football_rl.scenarios.base import BaseScenario, ScenarioState
from football_rl.scenarios.common import apply_spawn_jitter, ball, player, team_attack_dirs, team_color_ids


class SingleStrikerScenario(BaseScenario):
    name = "scenario_1_single_striker"
    description = "Single player near ball, empty goal."
    terminate_on_goal = True

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        options = options or {}
        dirs = team_attack_dirs(rng, cfg.randomization.enable and cfg.randomization.randomize_attack_direction, two_teams=False)
        if "attack_direction" in options:
            dirs = {0: int(options["attack_direction"])}
        colors = team_color_ids(rng, cfg.randomization.enable and cfg.randomization.randomize_team_colors, [0])
        base_player = (cfg.physics.field_width * 0.42, cfg.physics.field_height * 0.5)
        base_ball = (cfg.physics.field_width * 0.5, cfg.physics.field_height * 0.5)
        if cfg.randomization.enable:
            base_player = apply_spawn_jitter(base_player, rng, cfg.randomization.player_spawn_jitter)
            base_ball = apply_spawn_jitter(base_ball, rng, cfg.randomization.ball_spawn_jitter)
        if "player_positions" in options:
            base_player = tuple(options["player_positions"]["agent_0"])
        if "ball_position" in options:
            base_ball = tuple(options["ball_position"])
        p0 = player("agent_0", 0, base_player, dirs[0], colors[0], cfg, role=PlayerRole.STRIKER, scripted=False)
        ball_obj = ball(base_ball, cfg)
        steps = cfg.physics.max_episode_steps
        if cfg.randomization.enable:
            steps += int(rng.integers(-cfg.randomization.episode_length_jitter, cfg.randomization.episode_length_jitter + 1))
        return ScenarioState(
            players=[p0],
            ball=ball_obj,
            metadata={"completed_passes": 0, "minimum_required_passes_before_goal_counts": 0},
            max_episode_steps=max(50, steps),
            default_controlled_agents=["agent_0"],
        )
