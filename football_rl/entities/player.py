from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from football_rl.core.types import PlayerRole
from football_rl.entities.base import BaseEntity


@dataclass(slots=True)
class Player(BaseEntity):
    team_id: int = 0
    role: PlayerRole = PlayerRole.STRIKER
    color_id: int = 0
    attack_direction: int = 1
    max_speed: float = 14.0
    acceleration: float = 32.0
    stamina: float = 100.0
    stamina_max: float = 100.0
    is_scripted: bool = False
    goalkeeper_zone: tuple[float, float, float, float] | None = None
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
