from __future__ import annotations

from abc import ABC, abstractmethod


class RewardTerm(ABC):
    @abstractmethod
    def compute(self, simulator) -> dict[str, float]:
        raise NotImplementedError
