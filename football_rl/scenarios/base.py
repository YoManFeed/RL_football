from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from football_rl.entities.ball import Ball
from football_rl.entities.obstacle import MovingWall
from football_rl.entities.player import Player


@dataclass(slots=True)
class ScenarioState:
    players: list[Player]
    ball: Ball
    obstacles: list[MovingWall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_episode_steps: int | None = None
    default_scripted_agents: dict[str, Any] = field(default_factory=dict)
    default_controlled_agents: list[str] = field(default_factory=list)


class BaseScenario(ABC):
    name: str
    description: str = ""
    terminate_on_goal: bool = True
    score_limit: int | None = None

    @abstractmethod
    def reset(self, cfg, rng, options: dict | None = None) -> ScenarioState:
        raise NotImplementedError

    def on_reset(self, simulator) -> None:
        return None

    def on_step_end(self, simulator) -> None:
        return None

    def on_goal_scored(self, simulator, scoring_team_id: int) -> bool:
        return True

    def on_invalid_goal(self, simulator, attempted_team_id: int) -> None:
        simulator.restore_kickoff_snapshot()
