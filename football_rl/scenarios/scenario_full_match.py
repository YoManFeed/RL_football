from __future__ import annotations

from football_rl.core.types import PlayerRole
from football_rl.policies.scripted import GoalkeeperPolicy, SimpleAttackerPolicy
from football_rl.scenarios.base import BaseScenario, ScenarioState
from football_rl.scenarios.common import apply_spawn_jitter, ball, player, team_attack_dirs, team_color_ids


class FullMatchScenario(BaseScenario):
    name = "scenario_7_full_match"
    description = "Two teams, each with goalkeeper and two field players."
    terminate_on_goal = False
    score_limit = 5

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        options = options or {}
        dirs = team_attack_dirs(rng, cfg.randomization.enable and cfg.randomization.randomize_attack_direction, two_teams=True)
        if "attack_direction" in options:
            base = int(options["attack_direction"])
            dirs = {0: base, 1: -base}
        colors = team_color_ids(rng, cfg.randomization.enable and cfg.randomization.randomize_team_colors, [0, 1])
        fw = cfg.physics.field_width
        fh = cfg.physics.field_height
        team0 = [
            ("t0_gk", (10 if dirs[0] == 1 else fw - 10, fh / 2), PlayerRole.GOALKEEPER),
            ("t0_a0", (fw * 0.28, fh * 0.35), PlayerRole.STRIKER),
            ("t0_a1", (fw * 0.35, fh * 0.65), PlayerRole.MIDFIELDER),
        ]
        team1 = [
            ("t1_gk", (fw - 10 if dirs[1] == 1 else 10, fh / 2), PlayerRole.GOALKEEPER),
            ("t1_a0", (fw * 0.72, fh * 0.35), PlayerRole.STRIKER),
            ("t1_a1", (fw * 0.65, fh * 0.65), PlayerRole.MIDFIELDER),
        ]
        players = []
        for agent_id, pos, role in team0:
            if cfg.randomization.enable and role is not PlayerRole.GOALKEEPER:
                pos = apply_spawn_jitter(pos, rng, cfg.randomization.player_spawn_jitter)
            zone = None
            scripted = False
            if role is PlayerRole.GOALKEEPER:
                zone = (0, 18, fh / 2 - cfg.physics.goal_width / 2 - 8, fh / 2 + cfg.physics.goal_width / 2 + 8) if dirs[0] == 1 else (fw - 18, fw, fh / 2 - cfg.physics.goal_width / 2 - 8, fh / 2 + cfg.physics.goal_width / 2 + 8)
            players.append(player(agent_id, 0, pos, dirs[0], colors[0], cfg, role=role, scripted=scripted, goalkeeper_zone=zone))
        for agent_id, pos, role in team1:
            if cfg.randomization.enable and role is not PlayerRole.GOALKEEPER:
                pos = apply_spawn_jitter(pos, rng, cfg.randomization.player_spawn_jitter)
            zone = None
            scripted = False
            if role is PlayerRole.GOALKEEPER:
                zone = (0, 18, fh / 2 - cfg.physics.goal_width / 2 - 8, fh / 2 + cfg.physics.goal_width / 2 + 8) if dirs[1] == 1 else (fw - 18, fw, fh / 2 - cfg.physics.goal_width / 2 - 8, fh / 2 + cfg.physics.goal_width / 2 + 8)
            players.append(player(agent_id, 1, pos, dirs[1], colors[1], cfg, role=role, scripted=scripted, goalkeeper_zone=zone))
        ball_pos = (fw / 2, fh / 2)
        if cfg.randomization.enable:
            ball_pos = apply_spawn_jitter(ball_pos, rng, cfg.randomization.ball_spawn_jitter)
        scripted = {"t0_gk": GoalkeeperPolicy(), "t1_gk": GoalkeeperPolicy()}
        if options.get("scripted_field_players", False):
            scripted.update({"t0_a1": SimpleAttackerPolicy(), "t1_a1": SimpleAttackerPolicy()})
        return ScenarioState(
            players=players,
            ball=ball(ball_pos, cfg),
            metadata={"completed_passes": 0, "minimum_required_passes_before_goal_counts": 0},
            default_controlled_agents=[p.entity_id for p in players],
            default_scripted_agents=scripted,
        )
