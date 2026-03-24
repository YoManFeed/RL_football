from __future__ import annotations

import math
import numpy as np


EPS = 1e-8


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def l2norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def normalize_direction(vector: np.ndarray) -> np.ndarray:
    norm = l2norm(vector)
    if norm < EPS:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = l2norm(vector)
    if norm <= max_norm or norm < EPS:
        return vector.astype(np.float32)
    return (vector / norm * max_norm).astype(np.float32)


def normalize_position(position: np.ndarray, field_width: float, field_height: float) -> np.ndarray:
    x = (position[0] / field_width) * 2.0 - 1.0
    y = (position[1] / field_height) * 2.0 - 1.0
    return np.asarray([x, y], dtype=np.float32)


def normalize_velocity(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    if max_speed <= EPS:
        return np.zeros(2, dtype=np.float32)
    return np.clip(np.asarray(velocity, dtype=np.float32) / max_speed, -1.0, 1.0)


def signed_angle(vector: np.ndarray) -> float:
    return float(math.atan2(float(vector[1]), float(vector[0])))
