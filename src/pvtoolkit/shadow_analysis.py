"""Search utilities for shadow minima and descriptive summaries."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional, Tuple

import numpy as np

from .shadow_projection import chimney_top_corners, project_chimney_shadow_area
from .solar_geometry import roof_frame_vectors, solar_direction_vector


def find_minimum_chimney_shadow(
    latitude: float,
    longitude: float,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 135.0,
    height: float = 1.0,
    base_size: float = 1.0,
    year: int = 2022,
    timezone: str = "UTC",
) -> Tuple[Optional[datetime], float]:
    """
    Systematically sample timestamps across the year to locate the minimum shadow area.

    The search uses daily sampling with finer granularity during periods of higher solar
    elevation and filters out invalid sun positions where the sun is below the horizon
    or illuminates the roof from its backside.
    """
    _, _, e3 = roof_frame_vectors(tilt_deg, azimuth_deg)
    corners = chimney_top_corners(height, base_size)

    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    min_area = float("inf")
    min_time: Optional[datetime] = None

    while current <= end:
        if 3 <= current.month <= 8:
            step = 1
            hours = range(8, 16)
        else:
            step = 3
            hours = range(9, 15)

        for hour in hours:
            timestamp = current.replace(hour=hour, minute=0, second=0)
            sun_vec = solar_direction_vector(timestamp.isoformat(), latitude, longitude, timezone)
            area = project_chimney_shadow_area(sun_vec, corners, e3)

            if area > 0 and area < min_area:
                min_area = area
                min_time = timestamp

        current += timedelta(days=step)

    return min_time, min_area


def describe_minimum_shadow(
    observation_time: datetime,
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    timezone: str = "UTC",
) -> dict[str, float]:
    """
    Return descriptive metrics (elevation, azimuth, and roof alignment) for a timestamp.
    """
    sun_vec = solar_direction_vector(observation_time.isoformat(), latitude, longitude, timezone)
    elevation = np.rad2deg(np.arcsin(sun_vec[2]))
    azimuth = np.rad2deg(np.arctan2(sun_vec[0], sun_vec[1]))
    _, _, roof_normal = roof_frame_vectors(tilt_deg, azimuth_deg)
    alignment = np.dot(sun_vec, roof_normal)

    roof_azimuth = azimuth_deg
    azimuth_difference = abs(azimuth - roof_azimuth)
    if azimuth_difference > 180:
        azimuth_difference = 360 - azimuth_difference

    return {
        "sun_elevation_deg": elevation,
        "sun_azimuth_deg": azimuth,
        "sun_roof_alignment": alignment,
        "azimuth_difference_deg": azimuth_difference,
    }

