#!/usr/bin/env python3
"""Prepare the six-turbine 2024 SCADA subset without unpacking the wide CSVs.

Measurements stay on the native UTC 10-minute grid. Repeated timestamps with
different finite values become NaN for that field; no target interpolation or
averaging of conflicts is performed. The original archive and status exports
remain available beside the derived CSVs.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIME_COLUMN = "# Date and time"
COLUMNS = {
    "Power (kW)": "power",
    "Wind speed (m/s)": "turbine_ws",
    "Wind speed, Standard deviation (m/s)": "wind_speed_std",
    "Wind direction (°)": "wind_direction",
    "Nacelle position (°)": "nacelle_position",
    "Turbine Power setpoint (kW)": "power_setpoint",
    "Nacelle ambient temperature (°C)": "nacelle_ambient_temperature",
}
STEP_NS = 600_000_000_000


def turbine_number(name: str, prefix: str) -> int:
    match = re.fullmatch(prefix + r"_Kelmarsh_(\d+)_.*\.csv", Path(name).name)
    if match is None:
        raise ValueError(f"Unexpected turbine export filename: {name}")
    return int(match.group(1))


def local_xy(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular metres about the six-turbine coordinate centroid.

    x = R cos(lat0) (lon-lon0); y = R (lat-lat0), with angles in radians and
    spherical R = 6,371,000 m. This small site spans less than one kilometre.
    """
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    lat0, lon0 = lat.mean(), lon.mean()
    return 6_371_000.0 * np.cos(lat0) * (lon - lon0), 6_371_000.0 * (lat - lat0)


def prepare_turbines(directory: Path) -> pd.DataFrame:
    source = pd.read_csv(directory / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
    source["turbine_number"] = source["Title"].str.extract(r"Kelmarsh (\d+)$", expand=False).astype(int)
    source = source.sort_values("turbine_number")
    if source["turbine_number"].tolist() != list(range(1, 7)):
        raise ValueError("Static metadata must contain Kelmarsh turbines 1 through 6 exactly once")
    latitude = source["Latitude"].to_numpy(dtype=float)
    longitude = source["Longitude"].to_numpy(dtype=float)
    x, y = local_xy(latitude, longitude)
    frame = pd.DataFrame({
        "turbine_id": source["turbine_number"].to_numpy(),
        "turbine_name": source["Title"].to_numpy(),
        "source_identity": source["Identity"].to_numpy(),
        "manufacturer": source["Manufacturer"].to_numpy(),
        "model": source["Model"].to_numpy(),
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": source["Elevation (m)"].to_numpy(dtype=float),
        "hub_height": source["Hub Height (m)"].to_numpy(dtype=float),
        "rotor_diameter": source["Rotor Diameter (m)"].to_numpy(dtype=float),
        "rated_power": source["Rated power (kW)"].to_numpy(dtype=float),
        "capacity_mw": source["Rated power (kW)"].to_numpy(dtype=float) / 1000.0,
        # Senvion MM92/2050 manufacturer's technical page, 2017-03-01.
        # Nominal design thresholds; site controller settings are not supplied.
        "cut_in_speed": 3.0,
        "rated_wind_speed": 12.5,
        "cut_out_speed": 24.0,
        "x_coord": x,
        "y_coord": y,
    })
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Static turbine metadata contains missing or non-finite values")
    frame.to_csv(directory / "turbines.csv", index=False)
    return frame


def prepare_scada_member(archive: ZipFile, member, grid: pd.DatetimeIndex, chunk_rows: int) -> dict[str, np.ndarray]:
    turbine = turbine_number(member.filename, "Turbine_Data")
    with archive.open(member) as raw:
        preamble = [raw.readline().decode("utf-8-sig").strip() for _ in range(9)]
        header = next(csv.reader([raw.readline().decode("utf-8")]))
    if "# Time zone: UTC" not in preamble:
        raise ValueError(f"Turbine {turbine}: export does not explicitly declare UTC")
    missing = set([TIME_COLUMN, *COLUMNS]).difference(header)
    if missing:
        raise ValueError(f"Turbine {turbine}: missing required SCADA columns {sorted(missing)}")

    n_rows, n_fields = len(grid), len(COLUMNS)
    minimum = np.full((n_rows, n_fields), np.inf)
    maximum = np.full((n_rows, n_fields), -np.inf)
    timestamp_count = np.zeros(n_rows, dtype=np.int64)
    start_ns, stop_ns = grid[0].value, grid[-1].value + STEP_NS
    rows_read, outside_rows, infinite_values = 0, 0, 0
    earliest, latest = None, None
    begun, last_print = time.perf_counter(), time.perf_counter()
    print(f"[start] turbine={turbine} compressed_mb={member.compress_size / 1e6:.1f} "
          f"streamed_csv_mb={member.file_size / 1e6:.1f}", flush=True)
    dtypes = {TIME_COLUMN: "string", **{column: "float64" for column in COLUMNS}}
    with archive.open(member) as raw:
        reader = pd.read_csv(raw, skiprows=9, usecols=[TIME_COLUMN, *COLUMNS],
                             dtype=dtypes, encoding="utf-8-sig", chunksize=chunk_rows)
        for chunk in reader:
            timestamps = pd.to_datetime(chunk[TIME_COLUMN], format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise")
            if timestamps.isna().any():
                raise ValueError(f"Turbine {turbine}: missing timestamp")
            now_min, now_max = timestamps.min(), timestamps.max()
            earliest = now_min if earliest is None else min(earliest, now_min)
            latest = now_max if latest is None else max(latest, now_max)
            ns = timestamps.astype("int64").to_numpy()
            if np.any((ns - start_ns) % STEP_NS):
                raise ValueError(f"Turbine {turbine}: timestamp is not on the native 10-minute grid")
            inside = (ns >= start_ns) & (ns < stop_ns)
            outside_rows += int((~inside).sum())
            indices = ((ns[inside] - start_ns) // STEP_NS).astype(np.int64)
            values = chunk[list(COLUMNS)].to_numpy(dtype=float)[inside]
            infinite_values += int(np.isinf(values).sum())
            np.add.at(timestamp_count, indices, 1)
            for column in range(n_fields):
                finite = np.isfinite(values[:, column])
                np.minimum.at(minimum[:, column], indices[finite], values[finite, column])
                np.maximum.at(maximum[:, column], indices[finite], values[finite, column])
            rows_read += len(chunk)
            if time.perf_counter() - last_print >= 15:
                elapsed = time.perf_counter() - begun
                print(f"[progress] turbine={turbine} rows={rows_read:,} "
                      f"unique={int((timestamp_count > 0).sum()):,} elapsed_s={elapsed:.1f}", flush=True)
                last_print = time.perf_counter()

    have_value = np.isfinite(minimum)
    conflict = have_value & (minimum != maximum)
    values = np.where(have_value & ~conflict, minimum, np.nan)
    unique = int((timestamp_count > 0).sum())
    duplicates = int(np.maximum(timestamp_count - 1, 0).sum())
    conflicts = {COLUMNS[column]: int(conflict[:, index].sum()) for index, column in enumerate(COLUMNS)}
    print(f"[done] turbine={turbine} rows={rows_read:,} unique={unique:,} "
          f"range_utc={earliest}..{latest} repeated_rows={duplicates:,} "
          f"missing_grid_times={n_rows - unique:,} conflict_times={int(conflict.any(axis=1).sum()):,} "
          f"conflict_fields={conflicts} outside_year_rows={outside_rows} infinite_values={infinite_values} "
          f"elapsed_s={time.perf_counter() - begun:.1f}", flush=True)
    return {f"{output}{turbine}": values[:, index] for index, output in enumerate(COLUMNS.values())}


def prepare(directory: Path, archive_path: Path, chunk_rows: int = 100_000) -> Path:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    begun = time.perf_counter()
    grid = pd.date_range("2024-01-01", "2025-01-01", freq="10min", inclusive="left", tz="UTC")
    output = {"Timestamp": grid.strftime("%Y-%m-%dT%H:%M:%SZ")}
    with ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if Path(member.filename).name.startswith("Turbine_Data_Kelmarsh_")]
        members.sort(key=lambda member: turbine_number(member.filename, "Turbine_Data"))
        if [turbine_number(member.filename, "Turbine_Data") for member in members] != list(range(1, 7)):
            raise ValueError("Archive must contain exactly six 2024 SCADA turbine CSVs")
        for member in members:
            output.update(prepare_scada_member(archive, member, grid, chunk_rows))
        status_members = [member for member in archive.infolist() if Path(member.filename).name.startswith("Status_Kelmarsh_")]
        if sorted(turbine_number(member.filename, "Status") for member in status_members) != list(range(1, 7)):
            raise ValueError("Archive must contain exactly six original status exports")
        for member in status_members:
            destination = directory / Path(member.filename).name
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
    frame = pd.DataFrame(output)
    # Power and historical wind inputs are placed first for existing data loaders.
    first = ["Timestamp", *[f"power{i}" for i in range(1, 7)], *[f"turbine_ws{i}" for i in range(1, 7)]]
    frame = frame[first + [column for column in frame if column not in first]]
    target_path = directory / "Kelmarsh_SCADA_2024_selected.csv"
    frame.to_csv(target_path, index=False)
    turbines = prepare_turbines(directory)
    print(f"[saved] scada={target_path} rows={len(frame):,} columns={len(frame.columns)} "
          f"missing_power_cells={int(frame[[f'power{i}' for i in range(1, 7)]].isna().sum().sum()):,} "
          f"turbines={len(turbines)} capacities_mw={turbines.capacity_mw.tolist()} "
          f"elapsed_s={time.perf_counter() - begun:.1f}", flush=True)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/kelmarsh")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    archive = args.archive or args.directory / "Kelmarsh_SCADA_2024_5962.zip"
    prepare(args.directory, archive, args.chunk_rows)


if __name__ == "__main__":
    main()
