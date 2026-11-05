#!/usr/bin/env python3
"""Build an origin-selectable ECMWF archive from actual forecast GRIB1 files.

Missing run/lead/field values remain NaN. This script does not use valid-time-only
CSV weather, extrapolate forecasts, or mark incomplete data ready for training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pieef.forecast_vintages import ForecastVintageStore
from pieef.grib1 import iter_grib1


FEATURE_NAMES = (
    "u10_ms", "v10_ms", "surface_pressure_pa", "temperature_2m_k",
    "u100_ms", "v100_ms", "dewpoint_2m_k", "wind_speed100_ms", "wind_speed10_ms",
)
PARAMETERS = {
    "165.128": 0, "166.128": 1, "134.128": 2, "167.128": 3,
    "246.228": 4, "247.228": 5, "168.128": 6,
}


def _date(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _files(config):
    """Find only declared raw sources; there is no CSV fallback."""
    result = []
    for value in config["raw_roots"]:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue  # An unavailable supplement leaves explicit missing fields.
        if path.is_file():
            result.append(path)
        else:
            for path in sorted(path.rglob("*.grib")):
                if path.name.startswith("ecmwf_data_"):
                    if not any(path.name.startswith(f"ecmwf_data_{p}_") for p in PARAMETERS):
                        continue
                result.append(path)
    return sorted(set(result))


def build_archive(config, output):
    latitude, longitude = float(config["latitude"]), float(config["longitude"])
    start, end = _date(config["reference_start"]), _date(config["reference_end"])
    hours = tuple(int(h) for h in config.get("cycle_hours_utc", [0]))
    delay = timedelta(minutes=int(config["availability_delay_minutes"]))
    if start > end or delay < timedelta(0) or not hours or len(set(hours)) != len(hours):
        raise ValueError("Invalid reference range, availability delay, or cycle hours")
    if any(h < 0 or h > 23 for h in hours):
        raise ValueError("Cycle hours must be UTC hours in [0, 23]")
    runs = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        runs.extend(day.replace(hour=h) for h in sorted(hours) if start <= day.replace(hour=h) <= end)
        day += timedelta(days=1)
    leads = np.asarray(config["lead_hours"], dtype=float)
    if not runs or not len(leads) or not np.isfinite(leads).all() or np.any(np.diff(leads) <= 0) or np.any(leads < 0):
        raise ValueError("Expected runs and non-negative strictly sorted leads are required")
    values = np.full((len(runs), len(leads), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    run_index = {run: i for i, run in enumerate(runs)}
    lead_index = {float(lead): i for i, lead in enumerate(leads)}
    allowed_types = set(config.get("allowed_mars_types", [9]))
    allowed_streams = set(config.get("allowed_mars_streams", [1025]))
    tolerance = float(config.get("coordinate_tolerance_degrees", 1e-6))
    point_cache = {}
    files = _files(config)
    if not files:
        raise ValueError("No declared raw GRIB1 forecast files exist")
    accepted, duplicates, rejected_nonforecast = 0, 0, 0
    for file_number, path in enumerate(files, 1):
        for message in iter_grib1(path):
            parameter = message.mars_parameter
            if parameter not in PARAMETERS or message.reference_time not in run_index or message.lead_hours not in lead_index:
                continue
            if message.mars_type not in allowed_types or message.mars_stream not in allowed_streams:
                rejected_nonforecast += 1
                continue
            if message.level_type != 1 or message.time_range_indicator not in (0, 1, 10):
                raise ValueError(f"Expected instantaneous surface forecast for {parameter} in {path}")
            if message.grid not in point_cache:
                latitudes, longitudes = message.grid.coordinates()
                match = np.flatnonzero((np.abs(latitudes - latitude) <= tolerance)
                                       & (np.abs(longitudes - longitude) <= tolerance))
                if len(match) != 1:
                    raise ValueError(f"Declared site point is not exactly present in {path}")
                point_cache[message.grid] = int(match[0])
            value = float(message.values[point_cache[message.grid]])
            if not np.isfinite(value):
                continue
            index = (run_index[message.reference_time], lead_index[message.lead_hours], PARAMETERS[parameter])
            old = values[index]
            if np.isfinite(old):
                if not np.isclose(old, value, rtol=1e-7, atol=1e-5):
                    raise ValueError(f"Conflicting duplicate run/lead/parameter {parameter} in {path}")
                duplicates += 1
            else:
                values[index] = value
                accepted += 1
        if file_number % 50 == 0 or file_number == len(files):
            print(f"Decoded {file_number}/{len(files)} raw files", flush=True)
    # Same-run wind diagnostics; NaN naturally propagates from missing U or V.
    values[:, :, 7] = np.hypot(values[:, :, 4], values[:, :, 5])
    values[:, :, 8] = np.hypot(values[:, :, 0], values[:, :, 1])
    reference = np.asarray([run.isoformat().replace("+00:00", "Z") for run in runs])
    available = np.asarray([(run + delay).isoformat().replace("+00:00", "Z") for run in runs])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write through a file handle so the supplied output filename remains exact.
    with output.open("wb") as stream:
        np.savez_compressed(
            stream, reference_time=reference, available_time=available,
            lead_hours=leads, feature_names=np.asarray(FEATURE_NAMES), values=values,
            finite_mask=np.isfinite(values), latitude=np.asarray(latitude),
            longitude=np.asarray(longitude),
            feature_units=np.asarray(["m/s", "m/s", "Pa", "K", "m/s", "m/s", "K", "m/s", "m/s"]),
            source_model=np.asarray("ECMWF operational deterministic forecast"),
            availability_basis=np.asarray(config["availability_basis"]),
        )
    ForecastVintageStore(output, FEATURE_NAMES)  # Validate real output, including UTC/run ordering.
    counts = np.isfinite(values).sum(axis=(0, 1))
    result = {
        "output": str(output), "runs": len(runs), "leads": len(leads),
        "raw_files": len(files), "decoded_values": accepted, "identical_duplicates": duplicates,
        "rejected_nonforecast_messages": rejected_nonforecast,
        "finite_values_by_feature": dict(zip(FEATURE_NAMES, map(int, counts))),
        "complete_run_leads": int(np.isfinite(values).all(axis=2).sum()),
        "formal_training_ready": False,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default=str(ROOT / "configs/ecmwf_wind_farm_A_source.yaml"))
    parser.add_argument("--output", default=str(ROOT / "data/raw/nwp/wind_farm_A_ecmwf_partial.npz"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.source_config).read_text(encoding="utf-8"))
    build_archive(config, args.output)


if __name__ == "__main__":
    main()
