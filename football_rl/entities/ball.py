from __future__ import annotations

from dataclasses import dataclass

from football_rl.entities.base import BaseEntity


@dataclass(slots=True)
class Ball(BaseEntity):
    last_touch_player_id: str | None = None
    last_touch_team_id: int | None = None
    owner_id: str | None = None
