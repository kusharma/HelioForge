"""Projection utilities for chimney shadows on tilted surfaces."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from shapely.geometry import Polygon


def chimney_top_corners(height: float, base_size: float) -> list[Tuple[float, float, float]]:
    """
    Return the four top corners of a square chimney centered on the origin.

    The list of corners describes the chimney apex in the roof frame as:

        (±half_size, ±half_size, height)

    Parameters
    ----------
    height:
        Chimney height above the roof plane.
    base_size:
        Length of the square base edge in meters.
    """
    half = base_size / 2.0
    return [
        (half, half, height),
        (half, -half, height),
        (-half, -half, height),
        (-half, half, height),
    ]


def project_chimney_shadow_area(
    sun_vector: np.ndarray,
    chimney_corners: Iterable[Tuple[float, float, float]],
    roof_normal: np.ndarray,
) -> float:
    """
    Calculate the roof-projected shadow area of a rectangular chimney.

    Each vertex is projected onto the roof plane along the sun ray using the
    ray-plane intersection:

        t = -z / (v · roof_normal)
        projection = vertex + t * (-v)

    The projected points are consolidated into a polygon whose area is
    computed with Shapely for geometric robustness.

    Parameters
    ----------
    sun_vector:
        Unit-length sun direction vector pointing toward the sun.
    chimney_corners:
        Iterables of (x, y, z) roof-frame coordinates for the chimney top.
    roof_normal:
        Unit normal pointing away from the roof surface.
    """
    if sun_vector[2] <= 0 or np.dot(sun_vector, roof_normal) <= 0:
        return 0.0

    vertices = []
    shadow_direction = -sun_vector

    for vertex in chimney_corners:
        x, y, z = vertex
        if abs(z) < 1e-9:
            vertices.append((x, y))
            continue

        denom = shadow_direction[2]
        if abs(denom) < 1e-9:
            continue

        t = -z / denom
        if t < 0:
            continue

        proj_x = x + shadow_direction[0] * t
        proj_y = y + shadow_direction[1] * t
        vertices.append((proj_x, proj_y))

    unique = list({(round(v[0], 9), round(v[1], 9)) for v in vertices})
    if len(unique) < 3:
        return 0.0

    polygon = Polygon(unique)
    if not polygon.is_valid:
        polygon = polygon.convex_hull

    return max(0.0, polygon.area)

