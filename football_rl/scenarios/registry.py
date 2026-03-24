from __future__ import annotations

from football_rl.scenarios.scenario_duel import DuelScenario
from football_rl.scenarios.scenario_full_match import FullMatchScenario
from football_rl.scenarios.scenario_moving_wall import MovingWallScenario
from football_rl.scenarios.scenario_pass_goalie import PassGoalkeeperScenario
from football_rl.scenarios.scenario_pass_then_goal import PassThenGoalScenario
from football_rl.scenarios.scenario_pass_wall import PassWallScenario
from football_rl.scenarios.scenario_single_striker import SingleStrikerScenario


REGISTRY = {
    SingleStrikerScenario.name: SingleStrikerScenario,
    MovingWallScenario.name: MovingWallScenario,
    PassThenGoalScenario.name: PassThenGoalScenario,
    PassWallScenario.name: PassWallScenario,
    PassGoalkeeperScenario.name: PassGoalkeeperScenario,
    DuelScenario.name: DuelScenario,
    FullMatchScenario.name: FullMatchScenario,
}


def create_scenario(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown scenario: {name}")
    return REGISTRY[name](**kwargs)


def list_scenarios() -> list[str]:
    return sorted(REGISTRY)
