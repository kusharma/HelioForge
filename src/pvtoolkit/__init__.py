"""Top-level pvtoolkit package for solar geometry and PV data utilities."""

from .anomaly_detection import (
    compare_flag_shares,
    fit_cblof_model,
    fit_isolation_forest_model,
    sigma_anomaly_flags,
    train_autoencoder_model,
)
from .shadow_analysis import describe_minimum_shadow, find_minimum_chimney_shadow
from .shadow_projection import chimney_top_corners, project_chimney_shadow_area
from .solar_geometry import roof_frame_vectors, solar_direction_vector
from .time_series import compute_power_metrics, load_power_series
from .workflows.anomaly_workflow import assemble_anomaly_workflow
from .workflows.shadow_workflow import run_shadow_example, run_shadow_minimization

__all__ = [
    "chimney_top_corners",
    "compute_power_metrics",
    "compare_flag_shares",
    "describe_minimum_shadow",
    "fit_cblof_model",
    "fit_isolation_forest_model",
    "load_power_series",
    "project_chimney_shadow_area",
    "roof_frame_vectors",
    "run_shadow_example",
    "run_shadow_minimization",
    "sigma_anomaly_flags",
    "solar_direction_vector",
    "train_autoencoder_model",
    "find_minimum_chimney_shadow",
    "assemble_anomaly_workflow",
]

__version__ = "0.1.0"



