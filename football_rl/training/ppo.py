"""PPO with Actor-Critic for Football RL.

Architecture
------------
- Shared MLP encoder  (obs → latent)
- Policy head         (latent → action mean), separate log_std parameter
- Value head          (latent → scalar V)

Action space: 6 continuous dims
  [0:2]  move direction    ∈ [-1, 1]
  [2:4]  kick direction    ∈ [-1, 1]
  [4]    kick power        ∈ [ 0, 1]
  [5]    sprint            ∈ [ 0, 1]

We use a Normal distribution on the raw (pre-clip) space.
The environment clips actions internally via PlayerAction.from_array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticNet(nn.Module):
    """Shared encoder with separate policy and value heads."""

    def __init__(self, obs_dim: int, action_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        # Learnable log std, independent of the observation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor):
        latent = self.encoder(obs)
        mean = self.policy_head(latent)
        value = self.value_head(latent).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(obs)
        return self.value_head(latent).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.obs      = np.zeros((self.capacity, self.obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.log_probs= np.zeros(self.capacity,                    dtype=np.float32)
        self.rewards  = np.zeros(self.capacity,                    dtype=np.float32)
        self.dones    = np.zeros(self.capacity,                    dtype=np.float32)
        self.values   = np.zeros(self.capacity,                    dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: float,
        value: float,
    ) -> None:
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95) -> Batch:
        """Generalised Advantage Estimation + returns."""
        T = self.ptr
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            next_done = 0.0 if t == T - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1.0 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + self.values[:T]

        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return Batch(
            obs        = to_t(self.obs[:T]),
            actions    = to_t(self.actions[:T]),
            log_probs  = to_t(self.log_probs[:T]),
            advantages = to_t(advantages),
            returns    = to_t(returns),
            values     = to_t(self.values[:T]),
        )


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    rollout_steps: int      = 2048
    num_epochs: int         = 4
    minibatch_size: int     = 512
    gamma: float            = 0.99
    gae_lambda: float       = 0.95
    clip_epsilon: float     = 0.2
    value_coef: float       = 0.5
    entropy_coef: float     = 0.01
    max_grad_norm: float    = 0.5
    lr: float               = 3e-4
    normalize_advantages: bool = True


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 128,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = config or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCriticNet(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.buffer = RolloutBuffer(self.cfg.rollout_steps, obs_dim, action_dim, self.device)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Returns (action, log_prob, value) as numpy / python scalars."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env) -> dict:
        """Fill the rollout buffer with `rollout_steps` environment transitions."""
        self.buffer.reset()
        obs, _ = env.reset()
        ep_returns, ep_lengths = [], []
        ep_ret, ep_len = 0.0, 0

        for _ in range(self.cfg.rollout_steps):
            action, log_prob, value = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)

            self.buffer.add(obs, action, log_prob, reward, done, value)

            ep_ret += reward
            ep_len += 1
            obs = next_obs

            if terminated or truncated:
                ep_returns.append(ep_ret)
                ep_lengths.append(ep_len)
                ep_ret, ep_len = 0.0, 0
                obs, _ = env.reset()

        # Bootstrap value for the last step
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.net.get_value(obs_t).item()

        stats = {
            "mean_ep_return": np.mean(ep_returns) if ep_returns else float("nan"),
            "mean_ep_length": np.mean(ep_lengths) if ep_lengths else float("nan"),
            "num_episodes": len(ep_returns),
        }
        return last_value, stats

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float) -> dict:
        batch = self.buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)

        T = batch.obs.shape[0]
        indices = np.arange(T)
        pg_losses, v_losses, ent_bonuses, clip_fracs = [], [], [], []

        for _ in range(self.cfg.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.cfg.minibatch_size):
                mb_idx = torch.as_tensor(
                    indices[start : start + self.cfg.minibatch_size], device=self.device
                )
                mb_obs    = batch.obs[mb_idx]
                mb_act    = batch.actions[mb_idx]
                mb_old_lp = batch.log_probs[mb_idx]
                mb_adv    = batch.advantages[mb_idx]
                mb_ret    = batch.returns[mb_idx]
                mb_old_v  = batch.values[mb_idx]

                if self.cfg.normalize_advantages:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_log_prob, entropy, new_value = self.net.get_action_and_value(mb_obs, mb_act)

                # Policy loss (clipped surrogate)
                ratio = (new_log_prob - mb_old_lp).exp()
                pg_unclipped = -mb_adv * ratio
                pg_clipped   = -mb_adv * ratio.clamp(1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)
                pg_loss = torch.max(pg_unclipped, pg_clipped).mean()

                # Value loss (clipped)
                v_unclipped = (new_value - mb_ret) ** 2
                v_clipped = (
                    mb_old_v + (new_value - mb_old_v).clamp(-self.cfg.clip_epsilon, self.cfg.clip_epsilon) - mb_ret
                ) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_epsilon).float().mean().item()
                    clip_fracs.append(clip_frac)
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_bonuses.append(ent_loss.item())

        return {
            "pg_loss": np.mean(pg_losses),
            "v_loss": np.mean(v_losses),
            "entropy": np.mean(ent_bonuses),
            "clip_frac": np.mean(clip_fracs),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
