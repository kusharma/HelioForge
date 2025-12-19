"""Anomaly detection utilities for photovoltaic residual analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pyod.models.cblof import CBLOF
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models


def sigma_anomaly_flags(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Flag residuals that lie outside μ ± multiplier·σ.
    """
    mu = series.mean()
    sigma = series.std()
    lower = mu - multiplier * sigma
    upper = mu + multiplier * sigma
    return ((series < lower) | (series > upper)).astype(int)


def fit_isolation_forest_model(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[IsolationForest, pd.Series, pd.Series]:
    """
    Train an Isolation Forest on selected features; return model, flags, and scores.
    """
    clean = df[feature_columns].dropna()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(clean)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    flags = pd.Series(model.predict(X) == -1, index=clean.index).astype(int)
    scores = pd.Series(model.decision_function(X), index=clean.index)
    return model, flags, scores


def fit_cblof_model(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[CBLOF, pd.Series, pd.Series]:
    """
    Fit a CBLOF detector and return scores and binary flags for matched indices.
    """
    clean = df[feature_columns].dropna()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(clean)
    model = CBLOF(contamination=contamination, random_state=random_state)
    model.fit(X)
    flags = pd.Series(model.predict(X), index=clean.index).astype(int)
    scores = pd.Series(model.decision_function(X), index=clean.index)
    return model, flags, scores


def train_autoencoder_model(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    hidden_units: int = 4,
    bottleneck_units: int = 1,
    epochs: int = 50,
    contamination: float = 0.01,
) -> tuple[models.Model, pd.Series, pd.Series]:
    """
    Train a feed-forward autoencoder and flag top reconstruction errors.
    """
    clean = df[feature_columns].dropna()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(clean)

    autoencoder = models.Sequential(
        [
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(bottleneck_units, activation="relu", name="bottleneck"),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(X.shape[1], activation="sigmoid"),
        ]
    )
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=epochs, batch_size=128, shuffle=True, verbose=0)

    reconstruction_error = np.mean(np.square(X - autoencoder.predict(X, verbose=0)), axis=1)
    threshold = np.percentile(reconstruction_error, 100 * (1 - contamination))
    flags = (reconstruction_error > threshold).astype(int)

    return (
        autoencoder,
        pd.Series(flags, index=clean.index),
        pd.Series(reconstruction_error, index=clean.index),
    )


def compare_flag_shares(df: pd.DataFrame, flag_columns: Sequence[str]) -> pd.DataFrame:
    """
    Return a DataFrame summarizing intersection counts and shares for flag columns.
    """
    flags = df[flag_columns].fillna(0).astype(bool)
    n_total = len(flags)
    counts = {}
    for column in flag_columns:
        counts[column] = flags[column].sum()
    counts["ALL"] = flags.all(axis=1).sum()
    summary = pd.Series(counts, name="count").to_frame()
    summary["share"] = summary["count"] / max(n_total, 1)
    return summary

