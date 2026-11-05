"""Training-only measured per-turbine moments for the isolated F1 candidate."""
from __future__ import annotations

import numpy as np

POLICY = "train_stats_20"
SCHEMA = "raw_training_power_moments_v1"
EPS = 1e-5
PREFIX = "power_training_"
FIELDS = ("schema", "mean_pu", "stdev_pu", "valid_count", "turbine_ids",
          "capacity_mw", "end_index", "start_timestamp", "end_timestamp")


def fit_training_power_statistics(raw_power_mw, capacity_mw, turbine_ids, train_end, timestamps):
    raw = np.asarray(raw_power_mw)
    if raw.ndim != 2 or type(train_end) not in (int, np.int64, np.int32) or not 0 < train_end <= len(raw):
        raise ValueError("Raw measured power and an explicit nonempty training split are required")
    n = raw.shape[1]
    ids = np.asarray(turbine_ids, dtype=str)
    capacity = np.asarray(capacity_mw, dtype=np.float64)
    if capacity.ndim == 0:
        capacity = np.full(n, float(capacity))
    if (ids.shape != (n,) or len(set(ids)) != n or capacity.shape != (n,)
            or not np.isfinite(capacity).all() or not (capacity > 0).all()):
        raise ValueError("Training statistics need ordered unique turbine IDs and positive per-turbine capacities")
    time = np.asarray(timestamps, dtype=str)
    if time.ndim != 1 or len(time) < train_end:
        raise ValueError("Training statistics need the original training timestamps")
    measured_raw = raw[:train_end]
    # Keep the loader's raw dtype when applying its exact target-valid boundary.
    valid = np.isfinite(measured_raw) & (measured_raw > -0.02)
    measured = measured_raw.astype(np.float64)
    count = valid.sum(axis=0, dtype=np.int64)
    if (count == 0).any():
        raise ValueError("No real valid training power for turbines: " + ", ".join(ids[count == 0]))
    power = measured / capacity
    mean = np.where(valid, power, 0.).sum(axis=0) / count
    variance = np.where(valid, (power - mean) ** 2, 0.).sum(axis=0) / count
    result = dict(schema=np.asarray(SCHEMA), mean_pu=mean, stdev_pu=np.sqrt(variance + EPS),
                  valid_count=count, turbine_ids=ids, capacity_mw=capacity,
                  end_index=np.asarray(train_end, dtype=np.int64),
                  start_timestamp=np.asarray(time[0]), end_timestamp=np.asarray(time[train_end - 1]))
    validate_training_power_statistics(result, ids, capacity)
    return result


def validate_training_power_statistics(value, turbine_ids, capacity_mw):
    if not isinstance(value, dict) or set(value) != set(FIELDS) or str(np.asarray(value["schema"]).item()) != SCHEMA:
        raise ValueError("Missing or unsupported F1 training-power statistics schema")
    ids = np.asarray(turbine_ids, dtype=str)
    capacity = np.asarray(capacity_mw, dtype=np.float64)
    if capacity.ndim == 0:
        capacity = np.full(len(ids), float(capacity))
    if not np.array_equal(np.asarray(value["turbine_ids"]).astype(str), ids) or not np.array_equal(value["capacity_mw"], capacity):
        raise ValueError("F1 training statistics turbine order/capacity differs from this bundle")
    for key in ("mean_pu", "stdev_pu", "valid_count"):
        a = np.asarray(value[key])
        if a.shape != ids.shape or not np.isfinite(a).all():
            raise ValueError("F1 training statistics must contain finite values for each turbine: " + key)
    if not (np.asarray(value["stdev_pu"]) > 0).all() or not (np.asarray(value["valid_count"]) > 0).all():
        raise ValueError("F1 training statistics require positive scale and real observation counts")
    if np.asarray(value["valid_count"]).dtype.kind not in "iu" or np.asarray(value["end_index"]).dtype.kind not in "iu":
        raise ValueError("F1 observation counts and training split must be integers")
    end = np.asarray(value["end_index"])
    if end.shape != () or int(end) < 1 or (np.asarray(value["valid_count"]) > int(end)).any():
        raise ValueError("F1 training split/count metadata is inconsistent")
    for key in ("start_timestamp", "end_timestamp"):
        if np.asarray(value[key]).shape != () or not str(np.asarray(value[key]).item()):
            raise ValueError("F1 training statistics lack the actual training time range")


def ensure_training_power_statistics(bundle, saved=None):
    h = bundle.handler
    required = ("power_training_raw_mw", "power_training_end_index", "power_training_timestamps", "turbine_ids")
    if any(not hasattr(h, name) for name in required):
        raise ValueError("F1 requires raw measured training power, actual split and ordered turbine metadata")
    actual = fit_training_power_statistics(h.power_training_raw_mw, bundle.capacity_mw, h.turbine_ids,
                                          h.power_training_end_index, h.power_training_timestamps)
    if saved is not None:
        missing = [PREFIX + key for key in FIELDS if PREFIX + key not in saved]
        if missing:
            raise ValueError("Saved F1 preprocessing lacks training statistics: " + ", ".join(missing))
        restored = {key: np.asarray(saved[PREFIX + key]) for key in FIELDS}
        validate_training_power_statistics(restored, h.turbine_ids, bundle.capacity_mw)
        for key in FIELDS:
            if not np.array_equal(restored[key], actual[key]):
                raise ValueError("Saved F1 preprocessing differs from raw training source: " + key)
        actual = restored
    h.power_training_statistics = actual
    return actual


def preprocessing_fields(bundle):
    stats = ensure_training_power_statistics(bundle)
    return {PREFIX + key: stats[key] for key in FIELDS}


def check_model_training_statistics(model, bundle):
    revin = getattr(model, "revin", None)
    if revin is None or revin.missing_policy == "legacy":
        return
    stats = bundle.handler.power_training_statistics
    for name, field in (("training_mean", "mean_pu"), ("training_stdev", "stdev_pu")):
        expected = np.asarray(stats[field], dtype=np.float32).reshape(1, 1, -1)
        actual = getattr(revin, name).detach().cpu().numpy()
        if not np.array_equal(actual, expected):
            raise ValueError("Checkpoint F1 statistics differ from saved preprocessing: " + name)
