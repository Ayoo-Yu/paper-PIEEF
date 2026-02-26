#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def safe_import(name: str):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"MISSING: {exc}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "paths": {
            "repo": str(root),
            "demo_dms_exists": (root / "data/public_demo/DMS.csv").exists(),
            "demo_turbines_exists": (root / "data/public_demo/turbines.csv").exists(),
            "train_script_exists": (root / "scripts/train.py").exists(),
            "public_demo_config_exists": (root / "configs/main_h16_public_demo.yaml").exists(),
            "quick_config_exists": (root / "configs/main_public_demo_quick.yaml").exists(),
        },
        "deps": {
            "numpy": safe_import("numpy"),
            "pandas": safe_import("pandas"),
            "yaml": safe_import("yaml"),
            "sklearn": safe_import("sklearn"),
            "scipy": safe_import("scipy"),
            "torch": safe_import("torch"),
            "lightgbm": safe_import("lightgbm"),
            "xgboost": safe_import("xgboost"),
            "psutil": safe_import("psutil"),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
