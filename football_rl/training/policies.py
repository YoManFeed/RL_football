from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@dataclass(slots=True)
class SharedEncoderRolePolicy:
    observation_dim: int
    action_dim: int = 6
    hidden_dim: int = 64
    roles: tuple[str, ...] = ("striker", "goalkeeper")
    seed: int | None = None
    freeze_encoder_flag: bool = False
    encoder_w: np.ndarray = field(init=False)
    encoder_b: np.ndarray = field(init=False)
    role_heads: dict[str, tuple[np.ndarray, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.encoder_w = rng.normal(scale=0.08, size=(self.observation_dim, self.hidden_dim)).astype(np.float32)
        self.encoder_b = np.zeros(self.hidden_dim, dtype=np.float32)
        self.role_heads = {}
        for role in self.roles:
            w = rng.normal(scale=0.08, size=(self.hidden_dim, self.action_dim)).astype(np.float32)
            b = np.zeros(self.action_dim, dtype=np.float32)
            self.role_heads[role] = (w, b)

    def encode(self, observation: np.ndarray) -> np.ndarray:
        return tanh(observation @ self.encoder_w + self.encoder_b)

    def act(self, observation: np.ndarray, role: str) -> np.ndarray:
        hidden = self.encode(observation.astype(np.float32))
        w, b = self.role_heads[role]
        raw = tanh(hidden @ w + b).astype(np.float32)
        raw[4:] = (raw[4:] + 1.0) * 0.5
        return raw

    def replace_role_head(self, role: str, action_dim: int | None = None, seed: int | None = None) -> None:
        action_dim = action_dim or self.action_dim
        rng = np.random.default_rng(seed)
        self.role_heads[role] = (
            rng.normal(scale=0.08, size=(self.hidden_dim, action_dim)).astype(np.float32),
            np.zeros(action_dim, dtype=np.float32),
        )

    def freeze_encoder(self) -> None:
        self.freeze_encoder_flag = True

    def save(self, path: str | Path) -> None:
        arrays = {
            "encoder_w": self.encoder_w,
            "encoder_b": self.encoder_b,
            "roles": np.asarray(self.roles),
        }
        for role, (w, b) in self.role_heads.items():
            arrays[f"head_w::{role}"] = w
            arrays[f"head_b::{role}"] = b
        np.savez(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SharedEncoderRolePolicy":
        data = np.load(Path(path), allow_pickle=True)
        roles = tuple(str(x) for x in data["roles"])
        policy = cls(
            observation_dim=int(data["encoder_w"].shape[0]),
            hidden_dim=int(data["encoder_w"].shape[1]),
            action_dim=int(data[f"head_w::{roles[0]}"] .shape[1]),
            roles=roles,
        )
        policy.encoder_w = data["encoder_w"]
        policy.encoder_b = data["encoder_b"]
        policy.role_heads = {}
        for role in roles:
            policy.role_heads[role] = (data[f"head_w::{role}"], data[f"head_b::{role}"])
        return policy

    def load_encoder_from(self, path: str | Path) -> None:
        other = self.load(path)
        self.encoder_w = other.encoder_w.copy()
        self.encoder_b = other.encoder_b.copy()
