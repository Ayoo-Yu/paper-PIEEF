#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run significance testing (DM + BH).")
    p.add_argument("--seed", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--config-id", "--config_id", default="dm_bh_v1")
    p.add_argument("--artifact-path", "--artifact_path", default="NA")
    p.add_argument("--ledger-dir", default=str(ROOT / "ledger"))
    p.add_argument("--ref-model", default="PI-EEF")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/dm_bh.py"),
        "--config-id",
        args.config_id,
        "--artifact-path",
        args.artifact_path,
        "--ledger-dir",
        args.ledger_dir,
        "--ref-model",
        args.ref_model,
        "--alpha",
        str(args.alpha),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.horizon is not None:
        cmd += ["--horizon", str(args.horizon)]
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
