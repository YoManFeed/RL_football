from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_SIZE = 6


@dataclass(slots=True)
class PlayerAction:
    move: np.ndarray
    kick_direction: np.ndarray
    kick_power: float
    sprint: float

    @classmethod
    def zero(cls) -> "PlayerAction":
        return cls(
            move=np.zeros(2, dtype=np.float32),
            kick_direction=np.zeros(2, dtype=np.float32),
            kick_power=0.0,
            sprint=0.0,
        )

    @classmethod
    def from_array(cls, value: np.ndarray | list[float] | tuple[float, ...]) -> "PlayerAction":
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != ACTION_SIZE:
            raise ValueError(f"Expected action size {ACTION_SIZE}, got {array.size}")
        return cls(
            move=np.clip(array[0:2], -1.0, 1.0).astype(np.float32),
            kick_direction=np.clip(array[2:4], -1.0, 1.0).astype(np.float32),
            kick_power=float(np.clip(array[4], 0.0, 1.0)),
            sprint=float(np.clip(array[5], 0.0, 1.0)),
        )

    def to_array(self) -> np.ndarray:
        return np.asarray(
            [self.move[0], self.move[1], self.kick_direction[0], self.kick_direction[1], self.kick_power, self.sprint],
            dtype=np.float32,
        )
