#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curtailment bins (0-10%/10-30%/>30%).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--config-id", default="build_curtailment_bins_v1")
    p.add_argument("--artifact-path", default=str(ROOT / "artifacts/data/build_curtailment_bins"))
    p.add_argument("--scada-path", default=str(ROOT / "data/wind_farm_A/scada_ecmwf_15min.csv"))
    p.add_argument("--rule-version", default="v1_utilization_proxy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scada_path)
    power_cols = sorted([c for c in df.columns if c.startswith("power") and c[5:].isdigit()], key=lambda x: int(x[5:]))
    p = df[power_cols].values.astype(float)

    cap = np.nanmax(p)
    util = np.clip(1.0 - p / (cap + 1e-6), 0.0, 1.0)
    flat = util.reshape(-1)

    out = pd.DataFrame({"curtailment_ratio": flat})
    out["curtailment_bin"] = pd.cut(
        out["curtailment_ratio"],
        bins=[-1e-9, 0.10, 0.30, 1.0 + 1e-9],
        labels=["0-10%", "10-30%", ">30%"],
    )

    out_dir = Path(args.artifact_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "curtailment_bins.csv"
    out.to_csv(out_csv, index=False)

    summary = out.groupby("curtailment_bin", as_index=False).size().rename(columns={"size": "samples"})
    summary.to_csv(out_dir / "curtailment_bins_summary.csv", index=False)

    meta = {
        "seed": args.seed,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "rule_version": args.rule_version,
        "artifact_path": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
