# @Author: joe contact withheld
# @Date: 2025-06-27 16:23:28
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-07-03 22:30:24
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\batch_1\windsim_run_20250626_020320\base\solver\post.py
# @Description:
#
# Copyright (c) 2025 by joe, All Rights Reserved.

import os
import math
import numpy as np
from scipy.spatial import Delaunay, QhullError
import json
import sys


def interpolate_speed_magnitude(data, scale, x_grid, y_grid):
    """Piecewise linear speed on the sample hull; unsupported points stay NaN.

    Linear barycentric weights preserve the range of the measured magnitudes.
    Cubic splines can create negative speeds and nearest extrapolation conceals
    missing coverage. Coordinates alone are scaled back to physical metres.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 6 or not np.isfinite(data[:, :3]).all():
        raise ValueError("Height samples must contain finite XYZ coordinates")
    if np.isinf(data[:, 3:6]).any():
        raise ValueError("Infinite velocity in height samples")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mesh scale must be finite and positive")
    xy = data[:, :2] / scale
    speed = np.linalg.norm(data[:, 3:6], axis=1)
    # Duplicate XY samples must not make the triangulation depend on row order.
    points, inverse = np.unique(xy, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    # A NaN sample remains a triangulation vertex. Its incident triangles stay
    # missing, rather than interpolating across an unsupported interior hole.
    values = np.bincount(inverse, weights=speed) / counts
    if len(points) < 3:
        raise ValueError("At least three non-collinear height samples are required")
    try:
        triangulation = Delaunay(points)
    except QhullError as exc:
        raise ValueError("Height samples do not span a two-dimensional surface") from exc
    x_grid, y_grid = np.broadcast_arrays(x_grid, y_grid)
    query = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    simplex = triangulation.find_simplex(query)
    out = np.full(len(query), np.nan)
    inside = simplex >= 0
    indices = simplex[inside]
    transform = triangulation.transform[indices]
    delta = query[inside] - transform[:, 2, :]
    first = np.einsum("nij,nj->ni", transform[:, :2, :], delta)
    weights = np.column_stack([first, 1 - first.sum(axis=1)])
    weights = np.clip(weights, 0., 1.)
    weights /= weights.sum(axis=1, keepdims=True)
    contributors = values[triangulation.simplices[indices]]
    finite = np.isfinite(contributors)
    # A missing vertex with zero weight must not erase an exact valid node or
    # edge sample (NaN*0 in generic interpolators would do so).
    supported = (finite | (weights <= 1e-12)).all(axis=1)
    interpolated = (np.where(finite, contributors, 0.) * weights).sum(axis=1)
    interpolated[~supported] = np.nan
    out[inside] = interpolated
    return out.reshape(x_grid.shape)


def find_height_file(directory, height):
    """Match the recorded physical height, never truncate to another layer."""
    expected = format(height, '.15g')
    canonical = os.path.join(directory, expected)
    if os.path.isfile(canonical):
        return canonical
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            try:
                matches = math.isclose(float(name), height, rel_tol=1e-12, abs_tol=1e-9)
            except ValueError:
                continue
            candidate = os.path.join(directory, name)
            if matches and os.path.isfile(candidate):
                return candidate
    return canonical


def process_data():
    """主处理函数，读取numh和dh来生成高度，并包含详细的错误处理。"""
    from validateInput import resolve_post_heights

    try:
        print("[INFO] Loading configuration from ../info.json...")
        with open("../info.json", 'r') as f:
            info = json.load(f)

        # 从配置中提取参数
        post_info = info['post']
        mesh_info = info['mesh']
        domain_info = info['domain']
        wind_info = info['wind']

        # --- CRITICAL FIX: Read numh and dh, then generate height_levels ---
        height_levels = resolve_post_heights(post_info)
        numh = len(height_levels)
        dh = post_info['dh'] if 'heights' not in post_info else None

        lt = domain_info['lt']
        width = post_info.get('width', 1000)
        height = post_info.get('height', 1000)

        print(f"[INFO] Parameters: numh={numh}, dh={dh}, grid_size=[{width}, {height}]")
        print(f"[INFO] Generated height levels: {height_levels}")

        # 初始化输出数组
        size = [width, height, numh]
        out = np.full([numh, height, width], np.nan, dtype=np.float32)

        # 定义网格化的目标坐标
        x_range = np.linspace(-lt/2, lt/2, size[0])  # 这里保持不变
        y_range = np.linspace(-lt/2, lt/2, size[1])

        # ---- ① 打印网格整体范围 ----
        print(f"[DEBUG] Grid extent  : x [{x_range[0]:.2f}, {x_range[-1]:.2f}], "
              f"y [{y_range[0]:.2f}, {y_range[-1]:.2f}]  (total {width}×{height})")
        X, Y = np.meshgrid(x_range, y_range)

        # --- 主循环 ---
        for i, height_val in enumerate(height_levels):
            filepath = find_height_file('Output/plt', height_val)
            print(f"\n[INFO] Processing height {height_val}m (index {i}) from file: {filepath}")

            if not os.path.exists(filepath):
                print(f"[WARNING] File not found: {filepath}. Skipping this height.")
                out[i,:,:] = np.nan
                continue

            if os.path.getsize(filepath) == 0:
                print(f"[WARNING] File is empty: {filepath}. Skipping this height.")
                out[i,:,:] = np.nan
                continue

            try:
                data = np.loadtxt(filepath)
                if data.ndim == 1: data = data.reshape(1, -1)
                if data.shape[1] < 6:
                    print(f"[WARNING] Not enough columns in {filepath}. Skipping.")
                    out[i,:,:] = np.nan
                    continue
            except ValueError as e:
                print(f"[ERROR] Failed to load {filepath}. Error: {e}. Skipping.")
                out[i,:,:] = np.nan
                continue

            print(f"[INFO] Loaded {data.shape[0]} data points.")
            scale = info['mesh']['scale']
            x = data[:, 0] /scale
            y = data[:, 1]  /scale

            # ---- ② 打印当前层散点范围 ----
            print(f"[DEBUG] Points range: x [{x.min():.2f}, {x.max():.2f}], "
                  f"y [{y.min():.2f}, {y.max():.2f}] "
                  f"({data.shape[0]} pts)")
            U_interp = interpolate_speed_magnitude(data, scale, X, Y)

            # 之前我们猜测这里可能需要转置，这取决于X,Y的顺序和numpy数组索引的对应关系
            # 通常 griddata 的输出形状与 X, Y 相同，而 numpy 数组索引是 (row, col) 即 (y, x)
            # 所以 out[i, :, :] 对应 (height, width)，而 U_interp 是 (height, width) 形状，不需要转置
            out[i,:,:] = U_interp

            # ---- ③ 统计 NaN 比例 ----
            nan_ratio = np.isnan(U_interp).sum() / U_interp.size
            print(f"[DEBUG] Unsupported grid fraction (retained as NaN): {nan_ratio:.3%}")

        if any(not np.isfinite(layer).any() for layer in out):
            raise ValueError('Every recorded height must contain valid wind-speed samples.')

        # 保存结果
        print("\n[INFO] Saving binary data to ../speed.bin")
        out.tofile("../speed.bin")

        print("[INFO] Saving metadata to ../output.json")
        # Compute a robust visualization range (for colorbar) based on the data itself.
        # Using wind_speed * 1.5 is often too small in complex terrain (speed-up can exceed 1.5x),
        # which causes the plots to saturate at vmax and look "all red".
        #
        # To keep runtime/memory reasonable, estimate percentiles using a strided sample.
        wind_speed = float(wind_info.get('speed', 1) or 1)
        stride = max(1, int(min(width, height) / 200))  # ~200x200 samples per layer
        sample = out[:, ::stride, ::stride].reshape(-1)
        sample = sample[np.isfinite(sample)]

        vmin = 0.0
        vmax = wind_speed * 1.5
        if sample.size:
            # 99.5th percentile avoids single-point outliers while preventing saturation.
            vmax = max(vmax, float(np.percentile(sample, 99.5)))

        output_meta = {
            "file": "speed.bin",
            "size": size,
            "range": [float(vmin), float(vmax)],
            "dh": dh,
            "heights": height_levels,
            "interpolation": "linear_inside_convex_hull",
            "extrapolation": "none",
            "valid_fraction_by_height": [float(np.isfinite(layer).mean()) for layer in out]
        }
        with open("../output.json", 'w') as f:
            json.dump(output_meta, f, indent=4)

        print("\n[SUCCESS] FINISHED")

    except FileNotFoundError as e:
        print(f"[FATAL] A required file was not found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"[FATAL] A required key was not found in the JSON configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    process_data()
