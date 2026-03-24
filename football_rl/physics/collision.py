from __future__ import annotations

import numpy as np

from football_rl.utils.math_utils import norm_or_zero


def resolve_circle_circle(
    pos_a: np.ndarray,
    radius_a: float,
    pos_b: np.ndarray,
    radius_b: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    delta = pos_b - pos_a
    distance = float(np.linalg.norm(delta))
    target_distance = radius_a + radius_b
    if distance <= 1e-8:
        normal = np.asarray([1.0, 0.0], dtype=np.float32)
        overlap = target_distance
    else:
        normal = delta / distance
        overlap = target_distance - distance
    if overlap <= 0.0:
        return pos_a, pos_b, 0.0
    correction = normal * (overlap / 2.0)
    return pos_a - correction, pos_b + correction, overlap


def closest_point_on_aabb(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    low = center - half_extents
    high = center + half_extents
    return np.minimum(np.maximum(point, low), high)


def resolve_circle_aabb(
    circle_pos: np.ndarray,
    circle_radius: float,
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    closest = closest_point_on_aabb(circle_pos, box_center, box_half_extents)
    delta = circle_pos - closest
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        normal = np.asarray([1.0, 0.0], dtype=np.float32)
        overlap = circle_radius
    else:
        normal = norm_or_zero(delta)
        overlap = circle_radius - distance
    if overlap <= 0.0:
        return circle_pos, closest, 0.0
    corrected_circle = circle_pos + normal * overlap
    return corrected_circle, closest, overlap
