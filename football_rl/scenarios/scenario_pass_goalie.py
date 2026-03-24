from __future__ import annotations

from football_rl.core.types import PlayerRole
from football_rl.policies.scripted import GoalkeeperPolicy
from football_rl.scenarios.base import ScenarioState
from football_rl.scenarios.common import player
from football_rl.scenarios.scenario_pass_then_goal import PassThenGoalScenario


class PassGoalkeeperScenario(PassThenGoalScenario):
    name = "scenario_5_pass_goalkeeper"
    description = "Two attackers must pass and score against a scripted goalkeeper."

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        state = super().reset(cfg, rng, options)
        attack_direction = state.players[0].attack_direction
        goalie_team = 1
        goalie_attack = -attack_direction
        color_id = 1 if state.players[0].color_id == 0 else 0
        goal_x = cfg.physics.field_width - 12 if attack_direction == 1 else 12
        zone = (
            cfg.physics.field_width - 18 if attack_direction == 1 else 0,
            cfg.physics.field_width if attack_direction == 1 else 18,
            cfg.physics.field_height / 2 - cfg.physics.goal_width / 2 - 8,
            cfg.physics.field_height / 2 + cfg.physics.goal_width / 2 + 8,
        )
        goalie = player(
            "goalkeeper_1",
            goalie_team,
            (goal_x, cfg.physics.field_height / 2 + rng.uniform(-cfg.randomization.goalkeeper_spawn_jitter, cfg.randomization.goalkeeper_spawn_jitter)),
            goalie_attack,
            color_id,
            cfg,
            role=PlayerRole.GOALKEEPER,
            scripted=True,
            goalkeeper_zone=zone,
        )
        state.players.append(goalie)
        state.default_scripted_agents[goalie.entity_id] = GoalkeeperPolicy()
        return state
