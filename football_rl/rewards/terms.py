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

        for agent_id, player in simulator.players.items():
            goal_pos = simulator.opponent_goal_center(player.team_id)
            prev_dist = float(np.linalg.norm(prev_pos - goal_pos))
            curr_dist = float(np.linalg.norm(curr_pos - goal_pos))
            delta = prev_dist - curr_dist  # positive if ball moved closer to goal
            rewards[agent_id] += delta * simulator.config.rewards.ball_progress_scale

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


class WallBouncePenaltyTerm(RewardTerm):
    """Penalise every ball bounce off a field boundary or moving obstacle.

    Prevents the agent from farming ball-progress reward by repeatedly
    kicking the ball into a corner wall.
    """
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        penalty = simulator.config.rewards.wall_bounce_penalty
        for event in simulator.events:
            if event.type in (EventType.BALL_WALL_BOUNCE, EventType.OUT_OF_BOUNDS):
                for agent_id in simulator.players:
                    rewards[agent_id] -= penalty
        return rewards


class BallProximityRewardTerm(RewardTerm):
    """Reward for staying close to the ball.

    Unlike a potential-based approach (delta distance), this absolute
    proximity term does NOT penalise kicking the ball away — it simply
    gives a continuous incentive to chase the ball after each kick.
    """
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        scale = simulator.config.rewards.ball_proximity_scale
        if scale == 0.0:
            return rewards
        fw = simulator.config.physics.field_width
        fh = simulator.config.physics.field_height
        max_dist = (fw ** 2 + fh ** 2) ** 0.5
        for agent_id, player in simulator.players.items():
            dist = float(np.linalg.norm(simulator.ball.position - player.position))
            proximity = max(0.0, 1.0 - dist / max_dist)
            rewards[agent_id] += proximity * scale
        return rewards


class KickRewardTerm(RewardTerm):
    """Small reward for each kick, encouraging active ball interaction."""
    def compute(self, simulator) -> dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in simulator.players}
        kick_r = simulator.config.rewards.kick_reward
        if kick_r == 0.0:
            return rewards
        for event in simulator.events:
            if event.type is EventType.KICK:
                player_id = event.data.get("player_id")
                if player_id in rewards:
                    rewards[player_id] += kick_r
        return rewards
