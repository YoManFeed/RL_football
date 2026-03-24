from __future__ import annotations

from football_rl.scenarios.base import ScenarioState
from football_rl.scenarios.common import moving_wall
from football_rl.scenarios.scenario_pass_then_goal import PassThenGoalScenario


class PassWallScenario(PassThenGoalScenario):
    name = "scenario_4_pass_wall"
    description = "Pass-then-goal with moving wall."

    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        state = super().reset(cfg, rng, options)
        attack_direction = state.players[0].attack_direction
        goal_x = cfg.physics.field_width - 8 if attack_direction == 1 else 8
        wall_y = cfg.physics.field_height / 2 + rng.uniform(-8.0, 8.0)
        state.obstacles = [moving_wall("wall_0", (goal_x, wall_y), cfg, rng, speed=10.0)]
        return state
