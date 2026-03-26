from __future__ import annotations

from dataclasses import dataclass, field

from football_rl.rewards.base import RewardTerm
from football_rl.rewards.terms import (
    BallProgressRewardTerm,
    GoalRewardTerm,
    IdlePenaltyTerm,
    PassRewardTerm,
    ScenarioHookRewardTerm,
    StealRewardTerm,
    TimePenaltyTerm,
    WallBouncePenaltyTerm,
)


@dataclass(slots=True)
class RewardResult:
    total: dict[str, float]
    breakdown: dict[str, dict[str, float]]


@dataclass(slots=True)
class RewardManager:
    terms: list[RewardTerm] = field(default_factory=list)

    def compute(self, simulator) -> RewardResult:
        total = {agent_id: 0.0 for agent_id in simulator.players}
        breakdown: dict[str, dict[str, float]] = {}
        for term in self.terms:
            partial = term.compute(simulator)
            term_name = type(term).__name__
            breakdown[term_name] = partial
            for agent_id, reward in partial.items():
                total[agent_id] += float(reward)
        return RewardResult(total=total, breakdown=breakdown)


def build_default_reward_manager() -> RewardManager:
    return RewardManager(
        terms=[
            GoalRewardTerm(),
            BallProgressRewardTerm(),
            PassRewardTerm(),
            StealRewardTerm(),
            IdlePenaltyTerm(),
            WallBouncePenaltyTerm(),
            TimePenaltyTerm(),
            ScenarioHookRewardTerm(),
        ]
    )
