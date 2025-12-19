"""Power time-series helpers for metrics derivation and statistical thresholds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_power_series(path: str | Path, datetime_column: str = "Datetime") -> pd.DataFrame:
    """
    Load power generation data and index it by datetime.

    Parameters
    ----------
    path:
        File path to the CSV dataset.
    datetime_column:
        Name of the column containing timestamps.
    """
    source = Path(path)
    df = pd.read_csv(source)
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    return df.set_index(datetime_column)


def compute_power_metrics(
    df: pd.DataFrame,
    expected_column: str = "P_exp",
    actual_column: str = "P",
) -> pd.DataFrame:
    """
    Append residual and efficiency metrics to the dataset.

    residual = expected − actual
    efficiency = actual / expected
    """
    result = df.copy()
    result["residual"] = result[expected_column] - result[actual_column]
    result["efficiency"] = (
        result[actual_column]
        .div(result[expected_column])
        .replace([float("inf"), -float("inf")], float("nan"))
    )
    return result


def compute_sigma_thresholds(series: pd.Series, multiplier: float = 3.0) -> dict[str, float]:
    """
    Return mean and μ ± multiplier·σ bounds for a residual series.
    """
    mu = series.mean()
    sigma = series.std()
    return {
        "mean": mu,
        "sigma": sigma,
        "lower_bound": mu - multiplier * sigma,
        "upper_bound": mu + multiplier * sigma,
    }

