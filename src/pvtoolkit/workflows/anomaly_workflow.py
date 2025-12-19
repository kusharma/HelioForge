"""Workflows capturing the sequence of PV data analysis steps from the original notebook."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..anomaly_detection import (
    compare_flag_shares,
    fit_cblof_model,
    fit_isolation_forest_model,
    sigma_anomaly_flags,
    train_autoencoder_model,
)
from ..time_series import compute_power_metrics, compute_sigma_thresholds


def assemble_anomaly_workflow(
    df: pd.DataFrame,
    expected_column: str = "P_exp",
    actual_column: str = "P",
    contamination: float = 0.01,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the analysis pipeline and return metrics, thresholds, and anomaly flags.
    """
    metrics = compute_power_metrics(df, expected_column, actual_column)
    thresholds = compute_sigma_thresholds(metrics["residual"])
    sigma_flags = sigma_anomaly_flags(metrics["residual"])

    _, iso_flags, iso_scores = fit_isolation_forest_model(
        metrics, ["residual"], contamination=contamination
    )
    _, cblof_flags, cblof_scores = fit_cblof_model(
        metrics, ["residual", expected_column], contamination=contamination
    )
    _, ae_flags, ae_errors = train_autoencoder_model(
        metrics, ["residual", expected_column], contamination=contamination
    )

    enriched = metrics.assign(
        sigma_anomaly=sigma_flags,
        isolation_forest=iso_flags,
        cblof=cblof_flags,
        autoencoder=ae_flags,
        iso_score=iso_scores,
        cblof_score=cblof_scores,
        autoencoder_error=ae_errors,
    )

    flag_summary = compare_flag_shares(
        enriched,
        ["sigma_anomaly", "isolation_forest", "cblof", "autoencoder"],
    )

    return {
        "thresholds": pd.Series(thresholds),
        "enriched_metrics": enriched,
        "flag_summary": flag_summary,
    }



