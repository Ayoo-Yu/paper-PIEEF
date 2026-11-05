#!/usr/bin/env python3
# backend/utils/query_speed.py

import os
import sys
import json
import argparse

try:
    import numpy as np
    from visualization_contract import load_speed_data, grid_axes, StrictGridInterpolator, json_safe
except ImportError as e:
    print(json.dumps({"success": False, "error": f"Missing Python dependency: {e}. Use the configured WindSim Python environment."}))
    sys.exit(1)

def get_speed_at_point(case_id, x_query, y_query, z_query, *, uploads_dir=None):
    try:
        base_dir = os.path.join(uploads_dir or os.path.join(os.path.dirname(__file__), "..", "uploads"), case_id)
        meta_path = os.path.join(base_dir, "output.json")
        bin_path = os.path.join(base_dir, "speed.bin")

        if not os.path.exists(meta_path) or not os.path.exists(bin_path):
            return {"success": False, "error": "Data files (output.json/speed.bin) not found."}

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        with open(os.path.join(base_dir, "info.json"), "r", encoding="utf8") as handle:
            info = json.load(handle)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in (x_query, y_query, z_query)):
            raise ValueError("Query coordinates must be finite numbers in meters")
        data, nx, ny, nz = load_speed_data(bin_path, meta)
        axes = grid_axes(meta, info, (nx, ny, nz))
        interpolator = StrictGridInterpolator(axes, data)

        point = (z_query, y_query, x_query)
        
        # --- FIX: Robustly handle interpolator output ---
        interpolated_value = interpolator(point)

        # np.isnan() works correctly on both 0-dim and 1-dim arrays
        if np.isnan(interpolated_value):
            return {"success": True, "speed": None, "message": "Point is outside the domain or has missing contributing samples."}
        else:
            # Use .item() to safely extract the scalar value
            speed = float(interpolated_value.item())
            return {"success": True, "speed": speed}

    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """主执行函数，包含参数解析和错误捕获。"""
    parser = argparse.ArgumentParser(description="Query wind speed at a specific point.")
    parser.add_argument("--caseId", required=True, help="Case ID")
    parser.add_argument("--x", required=True, type=float, help="X coordinate")
    parser.add_argument("--y", required=True, type=float, help="Y coordinate")
    parser.add_argument("--z", required=True, type=float, help="Z coordinate (height)")
    args = parser.parse_args()

    result = get_speed_at_point(args.caseId, args.x, args.y, args.z)
    
    print(json.dumps(json_safe(result), allow_nan=False))
    return 0 if result.get("success") else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(json.dumps({"success": False, "error": f"A critical error occurred in the script execution: {e}"}))
        sys.exit(1)
