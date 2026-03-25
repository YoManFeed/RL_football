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


# class BallProgressRewardTerm(RewardTerm):
#     def compute(self, simulator) -> dict[str, float]:
#         rewards = {agent_id: 0.0 for agent_id in simulator.players}
#         prev_x = simulator.prev_ball_position[0]
#         curr_x = simulator.ball.position[0]
#         delta = curr_x - prev_x
#         scale = simulator.config.rewards.ball_progress_scale / simulator.config.physics.field_width
#         for agent_id, player in simulator.players.items():
#             rewards[agent_id] += float(delta * player.attack_direction * scale)
#         return rewards

class BallProgressRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}

        prev_pos = np.array(simulator.prev_ball_position)
        curr_pos = np.array(simulator.ball.position)

        # define goal center (you may need to adjust depending on your coordinate system)
        goal_x = simulator.config.physics.field_width / 2.0
        goal_y = 0.0  # assuming center of field in y

        # If attack direction matters:
        for agent_id, player in simulator.players.items():
            if player.attack_direction == -1:
                goal_x = -goal_x  # flip goal for opposite team

            goal_pos = np.array([goal_x, goal_y])

            prev_dist = np.linalg.norm(prev_pos - goal_pos)
            curr_dist = np.linalg.norm(curr_pos - goal_pos)

            delta = prev_dist - curr_dist  # positive if closer to goal

            scale = simulator.config.rewards.ball_progress_scale
            rewards[agent_id] += float(delta * scale)

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


class ScenarioHookRewardTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        for agent_id, value in simulator.extra_agent_rewards.items():
            rewards[agent_id] += value * simulator.config.rewards.scenario_bonus_scale
        return rewards
    
class TimePenaltyTerm(RewardTerm):
    def compute(self, simulator) -> dict[str, float]:
        return {agent_id: -0.001 for agent_id in simulator.players}
