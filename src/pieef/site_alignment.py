"""Causal observation sampling with measured-only evaluation targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def align_scada(frame: pd.DataFrame, *, timezone: str, step_minutes: int = 15):
    data = frame.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(data.pop("Timestamp")))
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timezone)
    else:
        timestamps = timestamps.tz_convert(timezone)
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Source SCADA must be chronological with unique timestamps.")
    data.index = timestamps
    grid = pd.date_range(timestamps[0].ceil(f"{step_minutes}min"),
                         timestamps[-1].floor(f"{step_minutes}min"),
                         freq=f"{step_minutes}min")
    # Sample the latest observation available at each model time. No future value
    # contributes; in particular, a 00:15 input can use 00:10 but never 00:20.
    inputs = data.reindex(data.index.union(grid)).sort_index().ffill().reindex(grid)
    powers = [c for c in data if c.startswith("power") and c[5:].isdigit()]
    if not powers:
        raise ValueError("Expected measured power1..powerN fields.")
    actual = data[powers].reindex(grid)
    sampled_power = inputs[powers].rename(columns={column: f"input_{column}" for column in powers})
    output = pd.concat([inputs, sampled_power], axis=1)
    output[powers] = actual
    observation_time = pd.Series(timestamps, index=timestamps)
    output["SCADA_observation_time"] = observation_time.reindex(timestamps.union(grid)).sort_index().ffill().reindex(grid)
    if (pd.DatetimeIndex(output["SCADA_observation_time"]) > grid).any():
        raise ValueError("SCADA observations cannot follow the model input time.")
    output.insert(0, "Timestamp", grid)
    return output.reset_index(drop=True)


def align_prepared_nwp(scada: pd.DataFrame, weather: pd.DataFrame, *, timezone: str, source_columns=None, feature_columns=None):
    """Join native weather columns at exact valid times without renaming or filling."""
    from .data import NWP_COLS

    if source_columns is not None and feature_columns is not None:
        raise ValueError("Specify source_columns or feature_columns, not both.")
    source_columns = feature_columns if feature_columns is not None else source_columns
    source_columns = list(NWP_COLS if source_columns is None else source_columns)
    if not source_columns or len(set(source_columns)) != len(source_columns) or any(c in {"Timestamp", "Total_Power"} for c in source_columns):
        raise ValueError("Declare distinct original weather fields, excluding timestamps and measured farm power.")
    if "Timestamp" not in scada or "Timestamp" not in weather or not all(c in weather for c in source_columns):
        raise ValueError("SCADA/weather require Timestamp and the declared original weather fields.")
    def times(values):
        index = pd.DatetimeIndex(pd.to_datetime(values))
        index = index.tz_localize(timezone) if index.tz is None else index.tz_convert(timezone)
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("Prepared SCADA and weather timestamps must be unique and chronological.")
        return index
    scada_times, weather_times = times(scada["Timestamp"]), times(weather["Timestamp"])
    aligned = weather[source_columns].copy()
    aligned.index = weather_times
    aligned = aligned.reindex(scada_times)
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Prepared weather must exist at every aligned SCADA time; no forecast interpolation or filling is performed.")
    result = scada.copy()
    for column in source_columns:
        result[column] = aligned[column].to_numpy()
    return result
