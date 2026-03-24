from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from football_rl.training.policies import SharedEncoderRolePolicy
from football_rl.wrappers.gym_env import FootballGymEnv


class RolloutCollector:
    def __init__(self, env: FootballGymEnv):
        self.env = env

    def collect(self, policy, role: str = "striker", horizon: int = 64) -> dict[str, np.ndarray]:
        obs, _ = self.env.reset(seed=13)
        observations = []
        actions = []
        rewards = []
        dones = []
        for _ in range(horizon):
            action = policy.act(obs, role)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(terminated or truncated))
            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset(seed=13)
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }


class PseudoPPOTrainer:
    def __init__(self, policy: SharedEncoderRolePolicy, learning_rate: float = 1e-3):
        self.policy = policy
        self.learning_rate = learning_rate

    def train_one_iteration(self, batch: dict[str, np.ndarray], role: str = "striker") -> float:
        advantage = batch["rewards"] - batch["rewards"].mean()
        obs = batch["observations"]
        acts = batch["actions"]
        hidden = np.tanh(obs @ self.policy.encoder_w + self.policy.encoder_b)
        head_w, head_b = self.policy.role_heads[role]
        pred = np.tanh(hidden @ head_w + head_b)
        target = np.clip(acts * advantage[:, None], -1.0, 1.0)
        error = (pred - target).astype(np.float32)
        grad_w = hidden.T @ error / len(obs)
        grad_b = error.mean(axis=0)
        self.policy.role_heads[role] = (head_w - self.learning_rate * grad_w, head_b - self.learning_rate * grad_b)
        return float((error ** 2).mean())


def main() -> None:
    env = FootballGymEnv(
        scenario_name="scenario_1_single_striker",
        render_mode=None,
        flatten_observation=True,
        canonical_observation=True,
    )
    obs, _ = env.reset(seed=17)
    random_action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(random_action)
    print("Random policy sample:")
    print("obs shape:", obs.shape)
    print("action shape:", random_action.shape)
    print("reward:", reward)
    print("events:", info["events"])

    collector = RolloutCollector(env)
    policy = SharedEncoderRolePolicy(observation_dim=obs.shape[0])
    trainer = PseudoPPOTrainer(policy)
    batch = collector.collect(policy, role="striker", horizon=96)
    loss = trainer.train_one_iteration(batch, role="striker")
    print("Pseudo-PPO one-iteration loss:", loss)
    env.close()


if __name__ == "__main__":
    main()
