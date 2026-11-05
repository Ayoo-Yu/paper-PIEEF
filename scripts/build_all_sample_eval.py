#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pieef.data import build_data_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build all-sample evaluation mask/index on test set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--config-id", default="build_all_sample_eval_v1")
    p.add_argument("--artifact-path", default=str(ROOT / "artifacts/data/build_all_sample_eval"))
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--scada-path", default=str(ROOT / "data/wind_farm_A/scada_ecmwf_15min.csv"))
    p.add_argument("--turbines-path", default=str(ROOT / "data/raw/turbines.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_data_bundle(
        scada_path=Path(args.scada_path),
        turbines_path=Path(args.turbines_path),
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.1,
        capacity_mw=2.0,
    )

    n_samples = len(bundle.handler.test_set[0])
    n_turb = bundle.num_turbines
    all_sample_mask = np.ones((n_samples, n_turb), dtype=bool)

    out_dir = Path(args.artifact_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "all_sample_eval.npz", mask=all_sample_mask)

    meta = {
        "seed": args.seed,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "eval_scope": "all-sample",
        "n_samples": int(n_samples),
        "n_turbines": int(n_turb),
        "rule_version": "all_sample_eval_v1",
        "artifact_path": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
