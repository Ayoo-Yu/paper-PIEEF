#!/usr/bin/env python3
"""Prepare measured-only targets and causal 15-minute inputs for the two new sites."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pieef.site_alignment import align_scada, align_prepared_nwp

wind_farm_B_SOURCE = Path("results/paper")


def prepare_wind_farm_B(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source / "趋势分析_十分钟数据(0)_1.csv", encoding="gb18030")
    fields = {"Timestamp": pd.to_datetime(raw["统计时间"])}
    features = {"平均网侧有功功率": "power", "平均风速": "turbine_ws", "平均风向角": "wind_direction"}
    ids = list(range(17, 89))
    for number, source_id in enumerate(ids, 1):
        for source_name, output_name in features.items():
            matching = [c for c in raw if re.search(rf"F{source_id}_", c) and source_name in c]
            if len(matching) != 1:
                raise ValueError(f"F{source_id}: expected one {source_name} column, got {matching}")
            fields[f"{output_name}{number}"] = pd.to_numeric(raw[matching[0]], errors="raise")
    native = pd.DataFrame(fields).sort_values("Timestamp").reset_index(drop=True)
    aligned = align_scada(native, timezone="Asia/Shanghai")
    weather = pd.read_csv(source / "wind_farm_B_已有处理数据.csv")
    weather_columns = [column for column in weather if column not in {"Timestamp", "Total_Power"}]
    if len(weather_columns) != 22:
        raise ValueError("Expected the 22 actual weather fields in the original scada_ecmwf_15min.csv.")
    aligned = align_prepared_nwp(
        aligned, weather, timezone="Asia/Shanghai", source_columns=weather_columns,
    )
    aligned.to_csv(destination / "SCADA_15min.csv", index=False)

    workbook = pd.read_excel(source / "Wind Farm B风电场风机坐标及高程.xlsx", header=None)
    sites = workbook[workbook[2].astype(str).isin([f"MLX{i}" for i in ids])].copy()
    sites["source_id"] = sites[2].str.extract(r"MLX(\d+)$", expand=False).astype(int)
    sites = sites.sort_values("source_id")
    if sites["source_id"].tolist() != ids:
        raise ValueError("Coordinate source must map each adjusted T01..T72 once.")
    def numeric(column):
        return pd.to_numeric(sites[column], errors="raise").to_numpy(dtype=float)
    capacity, longitude, latitude = numeric(7), numeric(8), numeric(9)
    x, y = numeric(3), numeric(4)
    if np.corrcoef(x, longitude)[0, 1] < .9 or np.corrcoef(y, latitude)[0, 1] < .9:
        raise ValueError("Coordinate axes do not match east/north orientation.")
    if np.sum(capacity == 6.7) != 55 or np.sum(capacity == 5.0) != 17:
        raise ValueError("Unexpected mixed machine capacity mapping.")
    static = pd.DataFrame({
        "turbine_id": [f"F{i}" for i in ids], "source_adjusted_id": sites[2].to_numpy(),
        "model": sites[6].to_numpy(), "capacity_mw": capacity,
        "rated_power": capacity * 1000, "latitude": latitude, "longitude": longitude,
        "elevation_m": numeric(5), "x_coord": x - x.mean(), "y_coord": y - y.mean(),
        "hub_height": 110.0, "rotor_diameter": 191.0,
        "cut_in_speed": 2.5, "rated_wind_speed": np.where(capacity == 6.7, 10.0, 8.9),
        "cut_out_speed": np.where(capacity == 6.7, 24.0, 25.0),
    })
    static.to_csv(destination / "turbines.csv", index=False)
    for name, wind_column, power_column, rated in [("GWH191_5MW", 12, 13, 5000), ("GWH191_6_7MW", 14, 15, 6700)]:
        points = workbook[[wind_column, power_column]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(wind_column)
        pd.DataFrame({"wind_speed_ms": points[wind_column], "power_pu": points[power_column] / rated}).to_csv(destination / f"manufacturer_{name}.csv", index=False)
    print(f"wind_farm_B: native_rows={len(native)}, aligned_rows={len(aligned)}, machines={len(static)}, "
          f"capacity_mw={capacity.sum():.1f}, original_ecmwf_fields={len(weather_columns)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind_farm_B-source", type=Path, default=wind_farm_B_SOURCE)
    parser.add_argument("--site", choices=["wind_farm_B", "kelmarsh", "both"], default="both")
    args = parser.parse_args()
    if args.site in {"wind_farm_B", "both"}:
        prepare_wind_farm_B(args.wind_farm_B_source, ROOT / "data/wind_farm_B")
    if args.site in {"kelmarsh", "both"}:
        location = ROOT / "data/kelmarsh"
        original = pd.read_csv(location / "Kelmarsh_SCADA_2024_selected.csv")
        aligned = align_scada(original, timezone="UTC")
        aligned.to_csv(location / "SCADA_15min.csv", index=False)
        print(f"Kelmarsh: native_rows={len(original)}, aligned_rows={len(aligned)}", flush=True)


if __name__ == "__main__":
    main()
