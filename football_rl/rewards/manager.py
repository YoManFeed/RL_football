from __future__ import annotations

from dataclasses import dataclass, field

from football_rl.rewards.base import RewardTerm
from football_rl.rewards.terms import (
    BallProgressRewardTerm,
    BallProximityRewardTerm,
    GoalRewardTerm,
    IdlePenaltyTerm,
    KickRewardTerm,
    PassRewardTerm,
    ScenarioHookRewardTerm,
    StealRewardTerm,
    TimePenaltyTerm,
    WallBouncePenaltyTerm,
)


@dataclass(slots=True)
class RewardManager:
    terms: list[RewardTerm] = field(default_factory=list)

    def compute(self, simulator) -> dict[str, float]:
        total = {agent_id: 0.0 for agent_id in simulator.players}
        for term in self.terms:
            partial = term.compute(simulator)
            for agent_id, reward in partial.items():
                total[agent_id] += float(reward)
        return total


def build_default_reward_manager() -> RewardManager:
    return RewardManager(
        terms=[
            GoalRewardTerm(),
            BallProgressRewardTerm(),
            BallProximityRewardTerm(),
            KickRewardTerm(),
            PassRewardTerm(),
            StealRewardTerm(),
            IdlePenaltyTerm(),
            ScenarioHookRewardTerm(),
            TimePenaltyTerm(),
            WallBouncePenaltyTerm(),
        ]
    )
