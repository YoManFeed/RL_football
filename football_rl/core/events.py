from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    GOAL_VALID = "goal_valid"
    GOAL_INVALID = "goal_invalid"
    KICK = "kick"
    PASS_COMPLETED = "pass_completed"
    STEAL = "steal"
    INTERCEPTION = "interception"
    POSSESSION_GAIN = "possession_gain"
    POSSESSION_LOST = "possession_lost"
    OUT_OF_BOUNDS = "out_of_bounds"
    BALL_WALL_BOUNCE = "ball_wall_bounce"


@dataclass(slots=True)
class Event:
    type: EventType
    step: int
    data: dict[str, Any] = field(default_factory=dict)
