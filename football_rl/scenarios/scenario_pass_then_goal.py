from __future__ import annotations

from football_rl.core.types import PlayerRole
from football_rl.scenarios.base import BaseScenario, ScenarioState
from football_rl.scenarios.common import apply_spawn_jitter, ball, player, team_attack_dirs, team_color_ids


class PassThenGoalScenario(BaseScenario):
    name = "scenario_3_pass_then_goal"
    description = "Two attackers must complete N passes before the goal counts."
    terminate_on_goal = True

    def __init__(self, minimum_required_passes_before_goal_counts: int = 2):
        self.minimum_required_passes_before_goal_counts = minimum_required_passes_before_goal_counts

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        options = options or {}
        dirs = team_attack_dirs(rng, cfg.randomization.enable and cfg.randomization.randomize_attack_direction, two_teams=False)
        if "attack_direction" in options:
            dirs = {0: int(options["attack_direction"])}
        colors = team_color_ids(rng, cfg.randomization.enable and cfg.randomization.randomize_team_colors, [0])
        p0_pos = (cfg.physics.field_width * 0.35, cfg.physics.field_height * 0.42)
        p1_pos = (cfg.physics.field_width * 0.35, cfg.physics.field_height * 0.58)
        b_pos = (cfg.physics.field_width * 0.44, cfg.physics.field_height * 0.5)
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
            player("agent_1", 0, p1_pos, dirs[0], colors[0], cfg, role=PlayerRole.MIDFIELDER),
        ]
        min_passes = int(options.get("minimum_required_passes_before_goal_counts", self.minimum_required_passes_before_goal_counts))
        return ScenarioState(
            players=players,
            ball=ball(b_pos, cfg),
            metadata={"completed_passes": 0, "minimum_required_passes_before_goal_counts": min_passes},
            default_controlled_agents=["agent_0", "agent_1"],
        )

    def on_goal_scored(self, simulator, scoring_team_id: int) -> bool:
        completed = int(simulator.scenario_metadata.get("completed_passes", 0))
        required = int(simulator.scenario_metadata.get("minimum_required_passes_before_goal_counts", 0))
        return completed >= required
