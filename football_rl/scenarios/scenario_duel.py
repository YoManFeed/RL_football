from __future__ import annotations

from football_rl.core.types import PlayerRole
from football_rl.scenarios.base import BaseScenario, ScenarioState
from football_rl.scenarios.common import apply_spawn_jitter, ball, player, team_attack_dirs, team_color_ids


class DuelScenario(BaseScenario):
    name = "scenario_6_duel"
    description = "One-vs-one duel with possession, steal, and empty goals."
    terminate_on_goal = True

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        options = options or {}
        dirs = team_attack_dirs(rng, cfg.randomization.enable and cfg.randomization.randomize_attack_direction, two_teams=True)
        if "attack_direction" in options:
            base = int(options["attack_direction"])
            dirs = {0: base, 1: -base}
        colors = team_color_ids(rng, cfg.randomization.enable and cfg.randomization.randomize_team_colors, [0, 1])
        p0_pos = (cfg.physics.field_width * 0.4, cfg.physics.field_height * 0.5)
        p1_pos = (cfg.physics.field_width * 0.6, cfg.physics.field_height * 0.5)
        b_pos = (cfg.physics.field_width * 0.43, cfg.physics.field_height * 0.5)
        if cfg.randomization.enable:
            p0_pos = apply_spawn_jitter(p0_pos, rng, cfg.randomization.player_spawn_jitter)
            p1_pos = apply_spawn_jitter(p1_pos, rng, cfg.randomization.player_spawn_jitter)
            b_pos = apply_spawn_jitter(b_pos, rng, cfg.randomization.ball_spawn_jitter)
        if "player_positions" in options:
            p0_pos = tuple(options["player_positions"]["agent_0"])
            p1_pos = tuple(options["player_positions"]["agent_1"])
        if "ball_position" in options:
            b_pos = tuple(options["ball_position"])
        players = [
            player("agent_0", 0, p0_pos, dirs[0], colors[0], cfg, role=PlayerRole.STRIKER),
            player("agent_1", 1, p1_pos, dirs[1], colors[1], cfg, role=PlayerRole.DEFENDER),
        ]
        ball_obj = ball(b_pos, cfg)
        return ScenarioState(
            players=players,
            ball=ball_obj,
            metadata={"completed_passes": 0, "minimum_required_passes_before_goal_counts": 0},
            default_controlled_agents=["agent_0", "agent_1"],
        )
