#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

NWP_COLS = [
    "nwp_direct_solar_radiation",
    "nwp_total_sky_direct_shortwave",
    "nwp_surface_pressure_pa",
    "nwp_temperature_2m_k",
    "nwp_u10_ms",
    "nwp_v10_ms",
    "nwp_wind_speed200_ms",
    "nwp_wind_speed100_ms",
    "nwp_wind_speed10_ms",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified feature tensors (power/scada_ws/nwp12/coords).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--config-id", default="build_features_v1")
    p.add_argument("--artifact-path", default=str(ROOT / "artifacts/data/build_features"))
    p.add_argument("--scada-path", default=str(ROOT / "data/wind_farm_A/scada_ecmwf_15min.csv"))
    p.add_argument("--turbines-path", default=str(ROOT / "data/raw/turbines.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scada_path)
    tdf = pd.read_csv(args.turbines_path)

    power_cols = sorted([c for c in df.columns if c.startswith("power") and c[5:].isdigit()], key=lambda x: int(x[5:]))
    n = len(power_cols)
    wind_cols = [f"turbine_ws{i+1}" for i in range(n)]

    power = df[power_cols].values.astype(float)
    ws = df[wind_cols].values.astype(float) if all(c in df.columns for c in wind_cols) else np.zeros_like(power)

    nwp9 = df[[c for c in NWP_COLS if c in df.columns]].values.astype(float)
    if nwp9.shape[1] < 6:
        raise ValueError("NWP columns incomplete")
    u = nwp9[:, 4]
    v = nwp9[:, 5]
    nwp_ws = np.sqrt(u**2 + v**2)
    wd = np.arctan2(u, v)
    nwp12 = np.concatenate([nwp9, nwp_ws[:, None], np.sin(wd)[:, None], np.cos(wd)[:, None]], axis=1)
    nwp12 = np.tile(nwp12[:, None, :], (1, n, 1))

    coords = tdf[["x_coord", "y_coord"]].values.astype(float)[:n]

    out_dir = Path(args.artifact_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "features.npz", power=power, scada_ws=ws, nwp12=nwp12, coords=coords)

    meta = {
        "seed": args.seed,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "feature_signature": "power+scada_ws+nwp12+coords",
        "rule_version": "build_features_v1",
        "artifact_path": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
