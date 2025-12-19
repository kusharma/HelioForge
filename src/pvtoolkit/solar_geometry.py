"""Solar geometry helpers for computing sun direction vectors and roof coordinate frames."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pvlib


def solar_direction_vector(
    timestamp: str | pd.Timestamp,
    latitude: float,
    longitude: float,
    timezone: str = "UTC",
) -> np.ndarray:
    """
    Return the sun direction unit vector in the global east-north-up frame.

    The sun vector is computed from pvlib's zenith and azimuth angles using
    the spherical-to-Cartesian transform:

        s_east  = sin(zenith) * sin(azimuth)
        s_north = sin(zenith) * cos(azimuth)
        s_up    = cos(zenith)

    Parameters
    ----------
    timestamp:
        ISO-formatted time or pandas.Timestamp describing the observation instant.
    latitude:
        Observer latitude in decimal degrees.
    longitude:
        Observer longitude in decimal degrees.
    timezone:
        Timezone name passed to pvlib for solar position lookup.
    """
    times = pd.DatetimeIndex([pd.to_datetime(timestamp)], tz=timezone)
    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude).iloc[0]
    zenith_rad, azimuth_rad = np.deg2rad([solar_pos.zenith, solar_pos.azimuth])

    sin_zenith = np.sin(zenith_rad)
    return np.array(
        [
            sin_zenith * np.sin(azimuth_rad),  # east component
            sin_zenith * np.cos(azimuth_rad),  # north component
            np.cos(zenith_rad),                # up component
        ]
    )


def roof_frame_vectors(tilt_deg: float, azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a local orthonormal basis for the roof surface.

    The roof normal and slope directions follow these steps:

    * tilt angle defines the magnitude of the roof normal's horizontal projection.
    * azimuth defines the compass direction of the roof normal.
    * The down-slope direction (e1) is the cross product of the normal with the vertical unit vector.
    * The cross-slope direction (e2) completes the right-hand system.

    Parameters
    ----------
    tilt_deg:
        Tilt angle measured from the horizontal plane in degrees.
    azimuth_deg:
        Azimuth of the roof normal measured clockwise from north.
    """
    tilt_rad = np.deg2rad(tilt_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    roof_normal = np.array(
        [
            np.sin(tilt_rad) * np.sin(azimuth_rad),
            np.sin(tilt_rad) * np.cos(azimuth_rad),
            np.cos(tilt_rad),
        ]
    )

    vertical = np.array([0.0, 0.0, 1.0])
    down_slope = np.cross(roof_normal, vertical)
    down_norm = np.linalg.norm(down_slope)
    if down_norm < 1e-10:
        down_slope = np.array([1.0, 0.0, 0.0])
    else:
        down_slope /= down_norm

    cross_slope = np.cross(roof_normal, down_slope)

    return down_slope, cross_slope, roof_normal



