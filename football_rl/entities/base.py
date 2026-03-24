from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class BaseEntity:
    entity_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    radius: float = 1.0

    def copy(self):
        cls = self.__class__
        kwargs = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for key, value in list(kwargs.items()):
            if isinstance(value, np.ndarray):
                kwargs[key] = value.copy()
        return cls(**kwargs)
