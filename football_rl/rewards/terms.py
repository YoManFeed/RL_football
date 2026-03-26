from __future__ import annotations

import numpy as np

from football_rl.core.events import EventType
from football_rl.rewards.base import RewardTerm


class GoalRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        for event in simulator.events:
            if event.type is EventType.GOAL_VALID:
                team_id = int(event.data["team_id"])
                for agent_id, player in simulator.players.items():
                    if player.team_id == team_id:
                        rewards[agent_id] += simulator.config.rewards.goal_reward
                    else:
                        rewards[agent_id] -= simulator.config.rewards.concede_penalty
        return rewards


class BallProgressRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        scale = simulator.config.rewards.ball_progress_scale

        if simulator.config.rewards.ball_progress_euclidean:
            fw = simulator.config.physics.field_width
            fh = simulator.config.physics.field_height
            max_dist = float(np.sqrt(fw ** 2 + fh ** 2))
            for agent_id, player in simulator.players.items():
                goal = simulator.opponent_goal_center(player.team_id)
                prev_dist = float(np.linalg.norm(simulator.prev_ball_position - goal))
                curr_dist = float(np.linalg.norm(simulator.ball.position - goal))
                rewards[agent_id] += (prev_dist - curr_dist) * scale / max_dist
        else:
            prev_x = simulator.prev_ball_position[0]
            curr_x = simulator.ball.position[0]
            delta = curr_x - prev_x
            norm_scale = scale / simulator.config.physics.field_width
            for agent_id, player in simulator.players.items():
                rewards[agent_id] += float(delta * player.attack_direction * norm_scale)

        return rewards


class PassRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        for event in simulator.events:
            if event.type is EventType.PASS_COMPLETED:
                passer = event.data["from_player_id"]
                receiver = event.data["to_player_id"]
                rewards[passer] += simulator.config.rewards.pass_reward
                rewards[receiver] += simulator.config.rewards.receive_pass_reward
        return rewards


class StealRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        for event in simulator.events:
            if event.type is EventType.STEAL:
                rewards[event.data["player_id"]] += simulator.config.rewards.steal_reward
            elif event.type is EventType.INTERCEPTION:
                rewards[event.data["player_id"]] += simulator.config.rewards.interception_reward
            elif event.type is EventType.POSSESSION_GAIN:
                rewards[event.data["player_id"]] += simulator.config.rewards.touch_reward
        return rewards


class IdlePenaltyTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        penalty = simulator.config.rewards.idle_penalty
        for agent_id, player in simulator.players.items():
            move_mag = float((player.last_action[0] ** 2 + player.last_action[1] ** 2) ** 0.5)
            kick_mag = float((player.last_action[2] ** 2 + player.last_action[3] ** 2) ** 0.5) * float(player.last_action[4])
            if move_mag < 0.05 and kick_mag < 0.05:
                rewards[agent_id] -= penalty
        return rewards


class WallBouncePenaltyTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        penalty = simulator.config.rewards.wall_bounce_penalty
        for event in simulator.events:
            if event.type is EventType.BALL_WALL_BOUNCE:
                touch_team = simulator.ball.last_touch_team_id
                if touch_team is not None:
                    for agent_id, player in simulator.players.items():
                        if player.team_id == touch_team:
                            rewards[agent_id] -= penalty
        return rewards


class TimePenaltyTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        penalty = simulator.config.rewards.time_penalty
        return {agent_id: -penalty for agent_id in simulator.players}


class ScenarioHookRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        for agent_id, value in simulator.extra_agent_rewards.items():
            rewards[agent_id] += value * simulator.config.rewards.scenario_bonus_scale
        return rewards
