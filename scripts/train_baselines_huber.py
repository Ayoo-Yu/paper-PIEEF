#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
def validate_huber_config(path):
    """No global function replacement and no implicit identity override."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    train = cfg.get("train", {})
    if train.get("loss_name") != "huber" or train.get("huber_delta") != 0.1:
        raise ValueError("Explicit train.loss_name=huber and huber_delta=0.1 are required")
    if not cfg.get("tracking", {}).get("config_id") or not cfg["tracking"].get("artifact_root"):
        raise ValueError("Frozen config_id and artifact_root must be supplied by the YAML")
    return cfg


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    validate_huber_config(args.config)
    runpy.run_path(str(ROOT / "scripts/train_baselines.py"), run_name="__main__")


if __name__ == "__main__":
    main()
