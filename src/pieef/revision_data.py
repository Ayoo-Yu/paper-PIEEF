from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def require_verified_forecast_input(protocol: dict | None):
    """Accept either verified vintages or the author's prepared ECMWF inputs."""
    if protocol is None:
        return
    if protocol.get("nwp_timing_rule") == "author_prepared_ecmwf":
        if not protocol.get("nwp_preparation_author_confirmed", False):
            raise ValueError("Prepared ECMWF inputs require explicit author confirmation of their source and availability.")
        return
    if not protocol.get("nwp_alignment_verified", False):
        raise ValueError(
            "Formal revision training requires origin-aligned forecast vintages "
            "or author-confirmed prepared ECMWF inputs. Declare the appropriate source protocol."
        )
    if protocol.get("nwp_timing_rule") == "author_daily_18utc":
        raise ValueError("A production-time assumption cannot substitute for actual forecast vintages.")


class ForecastWindows(Dataset):
    """Create windows on demand; measured targets remain separate from input filling."""

    def __init__(self, power, wind, nwp, targets, indices, seq_len, horizon, future_nwp=None):
        self.power = torch.as_tensor(power, dtype=torch.float32)
        self.wind = torch.as_tensor(wind, dtype=torch.float32)
        self.nwp = torch.as_tensor(nwp, dtype=torch.float32)
        self.future_nwp = self.nwp if future_nwp is None else torch.as_tensor(future_nwp, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len, self.horizon = seq_len, horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        target = int(self.indices[item])
        start = target - self.horizon - self.seq_len + 1
        stop = start + self.seq_len
        n_turb = self.power.shape[1]
        hist_nwp = self.nwp[start:stop].unsqueeze(0).expand(n_turb, -1, -1)
        return (
            self.power[start:stop].T.unsqueeze(-1),
            self.wind[start:stop].T.unsqueeze(-1),
            hist_nwp,
            self.future_nwp[target].unsqueeze(0).expand(n_turb, -1),
            self.targets[target],
        )


def _causal_fill(values, fallback=0.0):
    # No backwards filling or interpolation from later observations.
    return pd.DataFrame(values).ffill().fillna(fallback).to_numpy(dtype=np.float32)


def _power_events(power_mw, wind, train_end, capacity_mw):
    valid = np.isfinite(power_mw) & (power_mw > -0.02)
    low = valid & (wind > 5.0) & (power_mw < 0.01)
    outlier = np.zeros_like(valid)
    for turbine in range(power_mw.shape[1]):
        bins = np.floor(np.where(np.isfinite(wind[:, turbine]), wind[:, turbine], -99)).astype(int)
        for speed_bin in range(25):
            fitting = (
                (bins[:train_end] == speed_bin)
                & valid[:train_end, turbine]
                & ~low[:train_end, turbine]
            )
            y = power_mw[:train_end, turbine][fitting]
            if y.size < 10 or np.std(y) == 0:
                continue
            mean, std = float(np.mean(y)), float(np.std(y))
            outlier[:, turbine] |= (
                (bins == speed_bin) & valid[:, turbine]
                & (np.abs(power_mw[:, turbine] - mean) > 3 * std)
            )
    repeated = np.zeros_like(valid)
    same = np.isfinite(wind[1:]) & (wind[1:] == wind[:-1])
    run = np.zeros(wind.shape[1], dtype=int)
    for t in range(1, len(wind)):
        run = np.where(same[t - 1], run + 1, 0)
        repeated[t] = run >= 4
    events = np.full(valid.shape, "normal", dtype="<U28")
    events[repeated] = "frozen_wind_candidate"
    events[outlier] = "power_outlier"
    events[low] = "low_power_candidate"
    events[~valid] = "invalid_or_missing_target"
    return valid, events


def forecast_production_times(timestamp: pd.DatetimeIndex, rule: str, frame=None):
    if rule == "author_daily_18utc":
        # Author: UTC D 18:00 -> Beijing D+2 00:00, a 22-hour lead to interval start.
        start_utc = timestamp.normalize().tz_convert("UTC")
        return start_utc - pd.Timedelta(hours=22)
    if rule == "column":
        if frame is None or "NWP_produced_time" not in frame:
            raise ValueError("column timing requires NWP_produced_time in UTC.")
        return pd.DatetimeIndex(pd.to_datetime(frame["NWP_produced_time"], utc=True))
    raise ValueError("Explicit NWP timing rule is required for revision experiments.")


def build_revision_bundle(
    dms_path: Path, turbines_path: Path, seq_len: int, horizon: int,
    batch_size: int, train_ratio: float, val_ratio: float,
    capacity_mw: float | np.ndarray, max_turbines: int | None, protocol: dict,
):
    from .data import DataBundle, NWP_COLS, _extract_power_cols

    frame = pd.read_csv(dms_path)
    timestamp = pd.DatetimeIndex(pd.to_datetime(frame["Timestamp"]))
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize(protocol.get("timezone", "Asia/Shanghai"))
    if not timestamp.is_monotonic_increasing or not timestamp.is_unique:
        raise ValueError("Timestamps must be unique and chronological.")
    step = pd.Timedelta(minutes=int(protocol.get("step_minutes", 15)))
    if len(timestamp) > 1 and not (np.diff(timestamp.asi8) == step.value).all():
        raise ValueError("Irregular sampling: align the source data before windowing.")
    power_cols = _extract_power_cols(frame)
    if max_turbines is not None:
        if not 0 < max_turbines <= len(power_cols):
            raise ValueError("Revision experiments do not silently synthesize extra turbines.")
        power_cols = power_cols[:max_turbines]
    n_turb = len(power_cols)
    turbines = pd.read_csv(turbines_path).iloc[:n_turb]
    if len(turbines) != n_turb or not {"x_coord", "y_coord"}.issubset(turbines):
        raise ValueError("A real coordinate row is required for every turbine.")
    capacity_column = protocol.get("capacity_column")
    if capacity_column is not None:
        if capacity_column not in turbines:
            raise ValueError(f"Missing declared per-turbine capacity column: {capacity_column}")
        capacity = pd.to_numeric(turbines[capacity_column], errors="raise").to_numpy(dtype=float)
    else:
        capacity = np.asarray(capacity_mw, dtype=float)
        if capacity.ndim == 0:
            capacity = float(capacity)
    capacity_array = np.asarray(capacity)
    if capacity_array.ndim > 1 or (capacity_array.ndim == 1 and capacity_array.shape != (n_turb,)):
        raise ValueError("Capacity vector must have one entry per selected turbine.")
    if not np.isfinite(capacity_array).all() or not (capacity_array > 0).all():
        raise ValueError("Declared turbine capacities in MW must be positive and finite.")
    power_unit = protocol.get("power_unit")
    if power_unit not in {"kW", "MW"}:
        raise ValueError("Declare power_unit as kW or MW, rather than inferring it.")
    raw_power = frame[power_cols].to_numpy(dtype=np.float32) / (1000 if power_unit == "kW" else 1)
    raw_wind = frame[[f"turbine_ws{i + 1}" for i in range(n_turb)]].to_numpy(dtype=np.float32)
    train_end, val_end = int(len(frame) * train_ratio), int(len(frame) * (train_ratio + val_ratio))
    valid, events = _power_events(raw_power, raw_wind, train_end, capacity)
    input_prefix = protocol.get("input_power_prefix")
    if input_prefix is not None:
        input_columns = [f"{input_prefix}{i + 1}" for i in range(n_turb)]
        if not all(column in frame for column in input_columns):
            raise ValueError("Separate causal SCADA power input fields are missing.")
        observed_power = frame[input_columns].to_numpy(dtype=np.float32) / (1000 if power_unit == "kW" else 1)
        input_valid = np.isfinite(observed_power) & (observed_power > -0.02)
    else:
        observed_power, input_valid = raw_power, valid
    input_power = _causal_fill(np.where(input_valid, observed_power, np.nan)) / capacity
    input_wind = _causal_fill(np.where(raw_wind >= 0, raw_wind, np.nan))
    wind_mean = float(input_wind[:train_end].mean())
    wind_std = float(input_wind[:train_end].std() + 1e-5)
    wind_norm = (input_wind - wind_mean) / wind_std
    targets = np.where(valid, raw_power / capacity, np.nan)

    prepared_mode = protocol.get("nwp_timing_rule") == "author_prepared_ecmwf"
    if prepared_mode:
        require_verified_forecast_input(protocol)
    columns = list(protocol.get("nwp_feature_columns", NWP_COLS))
    if not prepared_mode and "nwp_feature_columns" not in protocol and protocol.get("wind_height_m", 100) == 100:
        columns[4:6] = ["100_metre_U_wind_component_10", "100_metre_V_wind_component_10"]
    if len(columns) < 4 or len(set(columns)) != len(columns):
        raise ValueError("Declare distinct NWP fields including pressure, temperature and wind components.")
    semantic_defaults = {"pressure": 2, "temperature": 3, "u": 4, "v": 5, "target_speed": 7}
    if prepared_mode and columns != NWP_COLS:
        missing = [f"nwp_{name}_column" for name in semantic_defaults if not protocol.get(f"nwp_{name}_column")]
        if missing:
            raise ValueError(f"Native prepared NWP fields require explicit column semantics: {', '.join(missing)}")
    semantic_indices = {}
    for name, default in semantic_defaults.items():
        declared = protocol.get(f"nwp_{name}_column")
        if declared is not None:
            if declared not in columns:
                raise ValueError(f"Declared nwp_{name}_column is absent: {declared}")
            semantic_indices[name] = columns.index(declared)
        else:
            semantic_indices[name] = default if name != "target_speed" or prepared_mode else len(columns)
    pressure_index, temperature_index = semantic_indices["pressure"], semantic_indices["temperature"]
    u_index, v_index = semantic_indices["u"], semantic_indices["v"]
    if max(pressure_index, temperature_index, u_index, v_index) >= len(columns) or u_index == v_index:
        raise ValueError("NWP column semantics must identify actual pressure, temperature and separate U/V fields.")
    cube_mode = protocol.get("nwp_timing_rule") == "forecast_vintages"
    if cube_mode:
        from .forecast_vintages import ForecastVintageStore
        archive_path = Path(protocol["nwp_vintages_path"])
        if not archive_path.is_absolute():
            archive_path = Path(__file__).resolve().parents[2] / archive_path
        store = ForecastVintageStore(archive_path, columns)
        utc = timestamp.tz_convert("UTC").to_numpy(dtype="datetime64[ns]")
        historical = store.lookup(utc, utc)
        origins = utc[np.maximum(np.arange(len(frame)) - horizon, 0)]
        future = store.lookup(origins, utc)
        weather, future_weather = historical.weather, future.weather
        history_available, future_available = historical.valid, future.valid
        produced = pd.DatetimeIndex(future.available_time).tz_localize("UTC")
    else:
        if not all(c in frame for c in columns):
            raise ValueError("The declared NWP feature set is not present in this site file.")
        weather = frame[columns].to_numpy(dtype=np.float32)
        if not np.isfinite(weather).all():
            raise ValueError("NWP missing values require an explicit availability protocol.")
        future_weather = weather
        history_available = np.ones(len(frame), dtype=bool)
        produced = None if prepared_mode else forecast_production_times(timestamp, protocol.get("nwp_timing_rule", ""), frame)
        future_available = np.ones(len(frame), dtype=bool) if prepared_mode else None
        if prepared_mode:
            pressure, temperature = weather[:, pressure_index], weather[:, temperature_index]
            if not ((pressure >= 20000) & (pressure <= 120000)).all():
                raise ValueError("Prepared ECMWF surface pressure must be supplied in Pa (expected 20000..120000 Pa).")
            if not ((temperature >= 150) & (temperature <= 350)).all():
                raise ValueError("Prepared ECMWF 2 m temperature must be supplied in Kelvin (expected 150..350 K).")

    def with_wind_features(values):
        values = values.copy()
        u, v = values[:, u_index], values[:, v_index]
        if not prepared_mode and len(columns) > 7 and columns[7] == "wind_speed100_ms":
            values[:, 7] = np.hypot(u, v)
        if not prepared_mode and len(columns) > 8 and columns[:2] == ["u10_ms", "v10_ms"] and columns[8] == "wind_speed10_ms":
            values[:, 8] = np.hypot(values[:, 0], values[:, 1])
        angle = np.arctan2(u, v)
        return np.column_stack([values, np.hypot(u, v), np.sin(angle), np.cos(angle)])

    weather, future_weather = with_wind_features(weather), with_wind_features(future_weather)
    fitting_weather = weather[:train_end][history_available[:train_end]]
    if not len(fitting_weather):
        raise ValueError("No available training forecasts for fitting NWP normalization.")
    mean, std = fitting_weather.mean(axis=0), fitting_weather.std(axis=0) + 1e-5
    weather_norm = (weather - mean) / std
    future_weather_norm = (future_weather - mean) / std
    # Unavailable rows remain excluded from windows; no forecast imputation.
    missing_history = np.concatenate([[0], np.cumsum(~history_available)])

    ids = turbines["turbine_id"].astype(str).to_numpy() if "turbine_id" in turbines else np.asarray(power_cols)
    coords = turbines[["x_coord", "y_coord"]].to_numpy(dtype=np.float32)
    rel = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(rel, axis=-1)
    terrain_edges = None
    if "elevation_m" in turbines:
        dz = turbines["elevation_m"].to_numpy(dtype=np.float32)
        difference = dz[:, None] - dz[None, :]
        # Between-turbine height gradient, not a DEM-derived local surface slope.
        gradient = difference / np.maximum(dist, 1.0)
        terrain_edges = torch.as_tensor(np.stack([difference / 100.0, gradient], axis=-1))

    indices, metadata, dropped = {}, {}, {}
    for split, lo, hi in [("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, len(frame))]:
        candidate = np.arange(max(lo, seq_len + horizon - 1), hi)
        origin = candidate - horizon
        if cube_mode or prepared_mode:
            available = future_available[candidate].copy()
        else:
            available = produced.asi8[candidate] <= timestamp.tz_convert("UTC").asi8[origin]
        starts, stops = origin - seq_len + 1, origin + 1
        available &= (missing_history[stops] - missing_history[starts]) == 0
        measured = valid[candidate].any(axis=1)
        chosen = candidate[available & measured]
        if not len(chosen):
            raise ValueError(f"No valid, NWP-available {split} samples for horizon {horizon}.")
        indices[split] = chosen
        dropped[split] = int((~available).sum())
        metadata[split] = {
            "timestamp": timestamp[chosen].astype(str).to_numpy(),
            "origin_timestamp": timestamp[chosen - horizon].astype(str).to_numpy(),
            "target_valid": valid[chosen], "events": events[chosen],
        }
        if prepared_mode:
            # This is author-confirmed preprocessing, not an independently
            # reconstructed forecast run or an inferred dissemination time.
            metadata[split]["nwp_provenance"] = np.full(len(chosen), "author_confirmed_prepared_ecmwf")
        else:
            metadata[split]["nwp_produced_time"] = produced[chosen].astype(str).to_numpy()
        if cube_mode:
            metadata[split].update(
                nwp_reference_time=future.reference_time[chosen],
                nwp_available_time=future.available_time[chosen],
                nwp_lead_hours=future.lead_hours[chosen],
            )

    datasets = {name: ForecastWindows(input_power, wind_norm, weather_norm, targets, idx, seq_len, horizon, future_weather_norm) for name, idx in indices.items()}
    # Keep the existing single-process sampler and RNG sequence. Pinned CPU
    # batches only change the transfer path used by cloud CUDA training.
    loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train",
                               num_workers=0, pin_memory=torch.cuda.is_available())
               for name, ds in datasets.items()}
    handler = SimpleNamespace(
        scada_w_mean=wind_mean, scada_w_std=wind_std,
        nwp_mean=mean, nwp_std=std, coords=coords,
        terrain_edge_features=terrain_edges,
        operating_limits={name: turbines[name].to_numpy(dtype=np.float32)
                          for name in ("cut_in_speed", "rated_wind_speed", "cut_out_speed")
                          if name in turbines},
        train_p_norm=input_power[:train_end], val_p_norm=input_power[train_end:val_end], test_p_norm=input_power[val_end:],
        train_w_norm=wind_norm[:train_end], val_w_norm=wind_norm[train_end:val_end], test_w_norm=wind_norm[val_end:],
        train_n_norm=weather_norm[:train_end], val_n_norm=weather_norm[train_end:val_end], test_n_norm=weather_norm[val_end:],
        seq_len=seq_len, horizon=horizon, metadata=metadata, turbine_ids=ids,
        protocol=dict(protocol), feature_columns=columns + ["derived_ws", "derived_sin", "derived_cos"],
        nwp_pressure_index=pressure_index, nwp_temperature_index=temperature_index,
        nwp_u_index=u_index, nwp_v_index=v_index,
        nwp_target_speed_index=semantic_indices["target_speed"],
        nwp_derived_wind_height_m=int(re.search(r"\d+", columns[u_index]).group()) if re.search(r"\d+", columns[u_index]) else None,
        nwp_target_speed_column=columns[semantic_indices["target_speed"]] if semantic_indices["target_speed"] < len(columns) else "derived_ws",
        unavailable_sample_count=dropped,
        capacity_mw=capacity, capacity_kw=capacity * 1000, source_data_path=str(dms_path),
        power_training_raw_mw=raw_power[:train_end].copy(),
        power_training_end_index=train_end,
        power_training_timestamps=timestamp[:train_end].astype(str).to_numpy(),
    )
    bundle = DataBundle(
        loaders["train"], loaders["val"], loaders["test"], handler,
        n_turb, weather.shape[1], capacity,
        torch.as_tensor(dist / max(float(dist.max()), 1.0)), torch.as_tensor(rel),
    )
    return bundle
