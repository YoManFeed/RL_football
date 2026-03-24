from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from football_rl.entities.base import BaseEntity


@dataclass(slots=True)
class MovingWall(BaseEntity):
    half_extents: np.ndarray = field(default_factory=lambda: np.asarray([1.8, 8.0], dtype=np.float32))
    movement_axis: int = 1
    min_coord: float = 0.0
    max_coord: float = 80.0
    speed: float = 12.0
    direction: float = 1.0
