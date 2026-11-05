"""Select weather forecasts available at each prediction origin.

The NPZ schema stores reference_time/available_time UTC strings [R], strictly
increasing lead_hours [L], feature_names unicode [F], and values [R, L, F].
Missing run/lead/feature values remain NaN; they are never filled or extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

_NS_PER_HOUR = 3_600_000_000_000


def _utc_times(values, name: str, *, strings_only: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if strings_only and array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must contain UTC strings")
    if array.dtype.kind == "M" and not strings_only:
        result = array.astype("datetime64[ns]")
    else:
        parsed = []
        for value in array:
            if isinstance(value, np.datetime64) and not strings_only:
                parsed.append(value.astype("datetime64[ns]"))
                continue
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(f"{name} contains an invalid datetime") from exc
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} needs explicit timezone information; convert local times to UTC")
            utc = value.astimezone(timezone.utc).replace(tzinfo=None)
            parsed.append(np.datetime64(utc, "ns"))
        result = np.asarray(parsed, dtype="datetime64[ns]")
    if np.isnat(result).any():
        raise ValueError(f"{name} cannot contain missing datetimes")
    return result


@dataclass(frozen=True)
class ForecastLookupResult:
    weather: np.ndarray
    reference_time: np.ndarray
    available_time: np.ndarray
    lead_hours: np.ndarray
    valid: np.ndarray
    feature_names: tuple[str, ...]


class ForecastVintageStore:
    def __init__(self, path: Path | str, required_features: Sequence[str] | None = None):
        required_keys = {"reference_time", "available_time", "lead_hours", "feature_names", "values"}
        with np.load(Path(path), allow_pickle=False) as archive:
            missing = required_keys.difference(archive.files)
            if missing:
                raise ValueError(f"Forecast archive missing keys: {', '.join(sorted(missing))}")
            self.reference_time = _utc_times(archive["reference_time"], "reference_time", strings_only=True)
            self.available_time = _utc_times(archive["available_time"], "available_time", strings_only=True)
            leads = archive["lead_hours"]
            names = archive["feature_names"]
            values = archive["values"]
            if leads.ndim != 1 or leads.dtype.kind not in {"f", "i", "u"}:
                raise ValueError("lead_hours must be a numeric one-dimensional array")
            self.lead_hours = leads.astype(np.float64)
            if names.ndim != 1 or names.dtype.kind != "U":
                raise ValueError("feature_names must be a one-dimensional unicode array")
            self.feature_names = tuple(str(name) for name in names)
            if values.dtype.kind != "f":
                raise ValueError("values must be a floating-point array; use NaN for missing values")
            self.values = values.astype(np.float64)

        nr, nl, nf = len(self.reference_time), len(self.lead_hours), len(self.feature_names)
        if not nr or not nl or not nf:
            raise ValueError("Forecast archive must contain runs, leads, and features")
        if self.available_time.shape != self.reference_time.shape:
            raise ValueError("available_time and reference_time must have the same length")
        if self.values.shape != (nr, nl, nf):
            raise ValueError(f"values must have shape {(nr, nl, nf)}, got {self.values.shape}")
        self._reference_ns = self.reference_time.astype(np.int64)
        self._available_ns = self.available_time.astype(np.int64)
        if np.any(np.diff(self._reference_ns) <= 0):
            raise ValueError("reference_time must be strictly sorted with no duplicates")
        if np.any(self._available_ns < self._reference_ns):
            raise ValueError("available_time cannot precede initialization")
        if not np.isfinite(self.lead_hours).all() or np.any(self.lead_hours < 0) or np.any(np.diff(self.lead_hours) <= 0):
            raise ValueError("lead_hours must be finite, non-negative, unique, and strictly sorted")
        if not all(name and name == name.strip() for name in self.feature_names) or len(set(self.feature_names)) != nf:
            raise ValueError("feature_names must be non-empty and unique")
        if np.isinf(self.values).any():
            raise ValueError("values may contain NaN for missing fields, but not infinity")
        lead_ns = self.lead_hours * _NS_PER_HOUR
        if np.any(lead_ns >= np.iinfo(np.int64).max):
            raise ValueError("lead_hours exceeds datetime precision range")
        self._lead_ns = np.rint(lead_ns).astype(np.int64)
        if np.any(np.diff(self._lead_ns) <= 0):
            raise ValueError("Distinct lead_hours must remain distinct at nanosecond precision")
        self._required = self._resolve_features(required_features)

    def _resolve_features(self, names: Sequence[str] | None) -> tuple[str, ...]:
        selected = self.feature_names if names is None else tuple(names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("required_features must be non-empty with no duplicates")
        missing = set(selected).difference(self.feature_names)
        if missing:
            raise ValueError(f"Unknown required features: {', '.join(sorted(missing))}")
        return selected

    def lookup(self, origins, valid_times, required_features: Sequence[str] | None = None) -> ForecastLookupResult:
        """Choose the newest complete run already available at each origin.

        Inputs are aware datetimes/strings, or numpy datetime64 arrays already
        converted to UTC. Interpolation always uses two adjacent leads within
        one run. An incomplete newest run may fall back to a complete older run.
        """
        origins = _utc_times(origins, "origins")
        valid_times = _utc_times(valid_times, "valid_times")
        if origins.shape != valid_times.shape:
            raise ValueError("origins and valid_times must have the same length")
        if np.any(origins > valid_times):
            raise ValueError("Prediction origin cannot follow forecast valid time")
        names = self._required if required_features is None else self._resolve_features(required_features)
        columns = [self.feature_names.index(name) for name in names]
        count = len(origins)
        weather = np.full((count, len(names)), np.nan, dtype=np.float64)
        reference = np.full(count, np.datetime64("NaT"), dtype="datetime64[ns]")
        available = np.full(count, np.datetime64("NaT"), dtype="datetime64[ns]")
        lead_hours = np.full(count, np.nan, dtype=np.float64)
        valid = np.zeros(count, dtype=bool)
        origin_ns, valid_ns = origins.astype(np.int64), valid_times.astype(np.int64)

        for row, (origin, target) in enumerate(zip(origin_ns, valid_ns)):
            candidates = np.flatnonzero(self._available_ns <= origin)[::-1]
            for run in candidates:
                query = int(target) - int(self._reference_ns[run])
                right = int(np.searchsorted(self._lead_ns, query, side="left"))
                if right < len(self._lead_ns) and self._lead_ns[right] == query:
                    estimate = self.values[run, right, columns]
                elif right == 0 or right == len(self._lead_ns):
                    continue  # No extrapolation before/after this run's leads.
                else:
                    lower, upper = self.values[run, right - 1, columns], self.values[run, right, columns]
                    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                        continue
                    ratio = (query - int(self._lead_ns[right - 1])) / int(self._lead_ns[right] - self._lead_ns[right - 1])
                    estimate = lower + (upper - lower) * ratio
                if not np.isfinite(estimate).all():
                    continue
                selected_lead = query / _NS_PER_HOUR
                # Returned run/lead must reconstruct the requested UTC valid time.
                if int(self._reference_ns[run]) + int(round(selected_lead * _NS_PER_HOUR)) != int(target):
                    raise ValueError("Selected run and lead do not reproduce forecast valid time")
                weather[row] = estimate
                reference[row] = self.reference_time[run]
                available[row] = self.available_time[run]
                lead_hours[row] = selected_lead
                valid[row] = True
                break
        return ForecastLookupResult(weather, reference, available, lead_hours, valid, names)
