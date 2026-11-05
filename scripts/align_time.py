#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check and record NWP-SCADA temporal alignment report.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--config-id", default="align_time_v1")
    p.add_argument("--artifact-path", default=str(ROOT / "artifacts/data/align_time"))
    p.add_argument("--scada-path", default=str(ROOT / "data/wind_farm_A/scada_ecmwf_15min.csv"))
    p.add_argument("--timestamp-col", default="Timestamp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scada_path)

    out_dir = Path(args.artifact_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "seed": args.seed,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "rule_version": "align_time_v1_no_lookahead",
        "timestamp_col": args.timestamp_col,
        "rows": int(len(df)),
        "has_timestamp": args.timestamp_col in df.columns,
    }
    if args.timestamp_col in df.columns:
        ts = pd.to_datetime(df[args.timestamp_col], errors="coerce")
        report["null_timestamps"] = int(ts.isna().sum())
        report["is_monotonic_non_decreasing"] = bool(ts.ffill().is_monotonic_increasing)

    (out_dir / "align_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
