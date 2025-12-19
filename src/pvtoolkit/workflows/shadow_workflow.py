"""Standalone workflows that mirror the original shadow calculation narrative."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np

from ..shadow_analysis import describe_minimum_shadow, find_minimum_chimney_shadow
from ..shadow_projection import chimney_top_corners, project_chimney_shadow_area
from ..solar_geometry import roof_frame_vectors, solar_direction_vector


def run_shadow_example(
    timestamp: datetime,
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    height: float,
    base_size: float,
    timezone: str = "UTC",
) -> Dict[str, float]:
    """
    Return shadow area and sun geometry for the provided timestamp.
    """
    _, _, e3 = roof_frame_vectors(tilt_deg, azimuth_deg)
    corners = chimney_top_corners(height, base_size)
    sun_vec = solar_direction_vector(timestamp.isoformat(), latitude, longitude, timezone)
    area = project_chimney_shadow_area(sun_vec, corners, e3)
    elevation = np.rad2deg(np.arcsin(max(-1.0, min(1.0, sun_vec[2]))))
    azimuth = np.rad2deg(np.arctan2(sun_vec[0], sun_vec[1]))
    alignment = np.dot(sun_vec, e3)

    return {
        "shadow_area_m2": area,
        "sun_elevation_deg": elevation,
        "sun_azimuth_deg": azimuth,
        "sun_roof_alignment": alignment,
    }


def run_shadow_minimization(
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    height: float,
    base_size: float,
    year: int,
    timezone: str = "UTC",
) -> Dict[str, object]:
    """
    Perform a systematic search to find the minimum shadow and describe its geometry.
    """
    min_time, min_area = find_minimum_chimney_shadow(
        latitude, longitude, tilt_deg, azimuth_deg, height, base_size, year, timezone
    )
    if min_time is None:
        return {"shadow_area_m2": None}

    description = describe_minimum_shadow(min_time, latitude, longitude, tilt_deg, azimuth_deg, timezone)
    return {
        "shadow_area_m2": min_area,
        "observation_time": min_time.isoformat(),
        **description,
    }

