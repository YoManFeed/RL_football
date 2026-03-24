from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass(slots=True)
class PolicyRecord:
    name: str
    kind: str
    path: str | None = None
    frozen: bool = True


@dataclass(slots=True)
class OpponentPool:
    records: list[PolicyRecord] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def add(self, name: str, kind: str, path: str | None = None, frozen: bool = True) -> None:
        self.records.append(PolicyRecord(name=name, kind=kind, path=path, frozen=frozen))

    def sample(self) -> PolicyRecord | None:
        if not self.records:
            return None
        idx = int(self.rng.integers(0, len(self.records)))
        return self.records[idx]

    def save(self, path: str | Path) -> None:
        payload = [record.__dict__ for record in self.records]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "OpponentPool":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls(seed=seed)
        for record in payload:
            pool.add(**record)
        return pool
