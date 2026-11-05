# @Author: joe contact withheld
# @Date: 2025-07-03 22:35:51
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-07-04 00:20:48
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\utils\precompute_visualization.py
# @Description: 
# 
# Copyright (c) 2025 by joe, All Rights Reserved.

#!/usr/bin/env python3
# utils/precompute_visualization.py
# ---------------------------------------------------------------
# 预计算风场可视化所需的所有中间数据（风廓线 / 轴向尾流 /
# 各高度切片 PNG + 像素坐标），并缓存到
#   uploads/<caseId>/visualization_cache/
# ---------------------------------------------------------------

import os, sys, json, math, time, argparse, traceback
import shutil
import fcntl
import tempfile
import uuid
from pathlib import Path
from visualization_contract import load_speed_data, grid_axes, StrictGridInterpolator, json_safe, source_fingerprint
import numpy as np
import matplotlib
matplotlib.use("Agg")           # 服务器无显示环境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

def rotate_point_to_solver_frame(x_value, y_value, wind_angle_deg):
    angle_rad = -(float(wind_angle_deg) + 90.0) * math.pi / 180.0
    cos_value = math.cos(angle_rad)
    sin_value = math.sin(angle_rad)
    solver_x = float(x_value) * cos_value + float(y_value) * sin_value
    solver_y = -float(x_value) * sin_value + float(y_value) * cos_value
    return solver_x, solver_y

# ------------------------------------------------------------------
# 读取 speed.bin 并 reshape 成 (layers, Ny, Nx)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def precompute_all_data(case_id: str, *, uploads_dir=None) -> bool:
    t0 = time.time()
    print(f"Starting precomputation for case: {case_id}")

    # --------- 路径 -------------------------------------------------
    base_dir         = str(Path(uploads_dir or Path(__file__).parent.parent / "uploads") / case_id)
    meta_path        = os.path.join(base_dir, "output.json")
    info_path        = os.path.join(base_dir, "info.json")
    published_cache  = Path(base_dir) / "visualization_cache"
    cache_dir        = tempfile.mkdtemp(prefix=".visualization-build-", dir=base_dir)
    profiles_dir     = os.path.join(cache_dir, "profiles")
    wakes_dir        = os.path.join(cache_dir, "wakes")
    slices_img_dir   = os.path.join(cache_dir, "slices_img")
    slices_info_dir  = os.path.join(cache_dir, "slices_info")

    for p in (profiles_dir, wakes_dir, slices_img_dir, slices_info_dir):
        os.makedirs(p, exist_ok=True)
    print("Cache directories ready:", cache_dir)

    try:
        # --------- 1. 读取 metadata & info -------------------------
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        binfile = meta.get("file", "speed.bin")
        if not os.path.isabs(binfile):
            binfile = os.path.join(base_dir, binfile)
        if not os.path.exists(binfile):
            raise FileNotFoundError(binfile)

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        turbines = info.get("turbines", [])
        scale_factor = info.get("mesh", {}).get("scale")
        wind_angle = info.get("wind", {}).get("angle")
        if not isinstance(wind_angle, (int, float)) or not np.isfinite(wind_angle):
            raise ValueError("info.wind.angle must be finite")
        data, Nx, Ny, Nz = load_speed_data(binfile, meta)
        H_levels, y_coords, x_coords = grid_axes(meta, info, (Nx, Ny, Nz))
        if not np.isfinite(data).any():
            raise ValueError("speed.bin contains no valid speed samples")
        source_files = [meta_path, info_path, binfile]
        fingerprint = source_fingerprint(source_files)
        dx = x_coords[1] - x_coords[0] if Nx > 1 else 0
        dy = y_coords[1] - y_coords[0] if Ny > 1 else 0
        extent_m = [float(x_coords[0] - dx / 2), float(x_coords[-1] + dx / 2),
                    float(y_coords[0] - dy / 2), float(y_coords[-1] + dy / 2)]

        # --------- 3. 构造插值器 -----------------------------------
        f_interp = StrictGridInterpolator((H_levels, y_coords, x_coords), data)
        print("Interpolator ready. extent (m):", extent_m)

        # --------- 4. 处理风机坐标 ---------------------------------
        # `speed.bin` / OpenFOAM 求解域使用的是 makeGmsh.py 旋转后的 solver 坐标。
        # 这里必须把 info.json 中的原始风机平面坐标转到同一坐标系，
        # 否则切片、廓线、尾流和前端点位都会整体错位。
        solver_streamwise_vec = [1.0, 0.0]

        frontend_turbines = []
        if turbines:
            for i, t in enumerate(turbines):
                rx, ry = t.get("x"), t.get("y")
                if rx is None or ry is None:
                    print(f"Warning: turbine #{i} missing coords, skipped.", file=sys.stderr)
                    continue

                original_x = float(rx)
                original_y = float(ry)
                if not np.isfinite([original_x, original_y]).all():
                    raise ValueError(f"Turbine {i} coordinates are not finite")
                if not all(isinstance(t.get(key), (int, float)) and np.isfinite(t[key]) and t[key] > 0 for key in ("hub", "d")):
                    raise ValueError(f"Turbine {i} hub/diameter must be finite and positive")
                if any(token in str(t.get("id", "")) for token in ("/", "\\")):
                    raise ValueError("Turbine id cannot contain path separators")
                solver_x, solver_y = rotate_point_to_solver_frame(original_x, original_y, wind_angle)

                frontend_turbines.append({
                    "id":  t.get("id") or f"WT{i+1}",
                    "originalX": original_x,
                    "originalY": original_y,
                    "solverX": solver_x,
                    "solverY": solver_y,
                    "x": solver_x,
                    "y": solver_y,
                    "hubHeight":     float(t.get("hub", 90)),
                    "rotorDiameter": float(t.get("d",   120)),
                    "name": t.get("name", f"WT{i+1}")
                })
        else:
            print("Warning: No turbines defined.", file=sys.stderr)

        if frontend_turbines:
            print(
                "First turbine solver xy (m):",
                frontend_turbines[0]["x"],
                frontend_turbines[0]["y"],
                "from original:",
                frontend_turbines[0]["originalX"],
                frontend_turbines[0]["originalY"],
            )

        # --------- 5. 风廓线 ---------------------------------------
        print(f"Computing wind profiles for {len(frontend_turbines)} turbines …")
        h_eval = np.linspace(H_levels[0], H_levels[-1], 50)
        for t in frontend_turbines:
            # With the corrected interpolator and turbine coordinates, this calculation is now correct.
            pts = np.column_stack((h_eval,
                                   np.full_like(h_eval, t["y"]),
                                   np.full_like(h_eval, t["x"])))
            speeds = f_interp(pts)
            out = {"id": t["id"], "heights": h_eval.tolist(), "speeds": json_safe(speeds)}
            with open(os.path.join(profiles_dir, f"turbine_{t['id']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(json_safe(out), f, allow_nan=False, indent=2)
        print("Profiles done.")

        # --------- 6. 尾流 -----------------------------------------
        print("Computing wakes …")
        xmin, xmax = extent_m[0], extent_m[1]
        ymin, ymax = extent_m[2], extent_m[3]

        for t in frontend_turbines:
            R = t["rotorDiameter"]
            s_vals = np.linspace(-2*R, 10*R, 100)           # upstream & downstream (m)
            
            # solver frame 中下游方向固定为 +X，尾流沿 streamwise 方向取样。
            xp = t["x"] + s_vals * solver_streamwise_vec[0]
            yp = t["y"] + s_vals * solver_streamwise_vec[1]

            # ---- ① 仅保留在网格内的点 ----
            inside = (xp >= xmin) & (xp <= xmax) & (yp >= ymin) & (yp <= ymax)
            if not inside.any():
                print(f"  - turbine {t['id']} wake outside domain; retaining null samples.")

            xp_in = xp[inside]
            yp_in = yp[inside]
            s_in  = s_vals[inside]

            hub_z = t["hubHeight"]
            pts   = np.column_stack((np.full_like(s_in, hub_z), yp_in, xp_in))
            speeds_in = f_interp(pts)

            # ---- ② 把“域外”点填 null，保证前端长度一致 ----
            speeds_full = [None]*len(s_vals)
            j = 0
            for i, ok in enumerate(inside):
                if ok:
                    val = float(speeds_in[j])
                    speeds_full[i] = None if (np.isnan(val) or np.isinf(val)) else val
                    j += 1

            out = {
                "id": t["id"],
                "hubHeightUsed": float(hub_z),
                "distances": s_vals.tolist(),
                "speeds": speeds_full
            }
            with open(os.path.join(wakes_dir, f"turbine_{t['id']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(json_safe(out), f, allow_nan=False, indent=2)
        print("Wakes done.")

        # --------- 7. 高度切片 PNG + 像素坐标 -----------------------
        # Color range: prefer output.json's range, but auto-expand using data percentiles
        # to avoid saturation in complex terrain (otherwise the plot looks "all red").
        vmin, vmax = map(float, meta.get("range", [0, 15]))
        stride = max(1, int(min(Nx, Ny) / 200))  # ~200x200 samples per layer
        sample = data[:, ::stride, ::stride].reshape(-1)
        sample = sample[np.isfinite(sample)]
        if sample.size:
            vmax = max(vmax, float(np.percentile(sample, 99.5)))

        cmap_name = os.getenv("WINDSIM_COLORMAP", "viridis")
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Warning: unknown colormap '{cmap_name}', fallback to 'viridis'.", file=sys.stderr)
            cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        img_w, dpi = 800, 100
        domain_w = extent_m[1]-extent_m[0]
        domain_h = extent_m[3]-extent_m[2]
        aspect   = domain_h / domain_w if domain_w > 0 else 1
        img_h = max(1, int(img_w * aspect))
        figsize = (img_w/dpi, img_h/dpi)

        heights_info = []

        print("Generating slice images …")
        for k, z in enumerate(H_levels):
            slice_np = data[k].astype(float)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.90)

            im = ax.imshow(slice_np, cmap=cmap, norm=norm, origin="lower",
                           extent=extent_m, aspect="auto")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Wind speed @ {z:g} m")
            cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85, extend="max")
            cbar.set_label("Wind speed (m/s)")
            cbar.set_ticks(np.linspace(vmin, vmax, 6))
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            cbar.ax.tick_params(labelsize=8)

            # 计算风机像素位置
            fig.canvas.draw()
            fw, fh = fig.get_size_inches()*dpi
            turbines_px = []
            for t in frontend_turbines:
                disp = ax.transData.transform((t["x"], t["y"]))
                px, py = disp[0], fh - disp[1]
                turbines_px.append({"id": t["id"], "x": round(float(px),2), "y": round(float(py),2)})

            # 保存单 slice json
            h_str = f"{z:.1f}" if float(z).is_integer() else format(float(z), ".15g")
            info_obj = {"actualHeight": float(z),
                        "imageDimensions": {"width": int(fw), "height": int(fh), "dpi": dpi},
                        "turbinesPixels": turbines_px}
            with open(os.path.join(slices_info_dir, f"slice_info_{h_str}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(json_safe(info_obj), f, allow_nan=False, indent=2)

            # 保存 PNG
            png_name = f"slice_height_{h_str}.png"
            fig.savefig(os.path.join(slices_img_dir, png_name), dpi=dpi)
            plt.close(fig)

            heights_info.append({"height": float(z),
                                 "imageFile": png_name,
                                 "infoFile": f"slice_info_{h_str}.json"})

        print("Slices done.")

        # --------- 8. 汇总 metadata ---------------------------------
        meta_out = {
            "cacheVersion": 2,
            "sourceFingerprint": fingerprint,
            "heightLevelsInfo": heights_info,
            "heightLevels": [h["height"] for h in heights_info],
            "turbines": frontend_turbines,
            "vmin": vmin, "vmax": vmax,
            "windAngle": float(wind_angle),
            "windDirectionVector": solver_streamwise_vec,
            "turbineCoordinateFrame": "solver",
            "scaleFactor": scale_factor,
            "xCoords_m": x_coords.tolist(),
            "yCoords_m": y_coords.tolist(),
            "extent_m": extent_m
        }
        with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(meta_out), f, allow_nan=False, indent=2)

        with (Path(base_dir) / ".visualization-publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if source_fingerprint(source_files) != fingerprint:
                raise RuntimeError("Source files changed during visualization precomputation")
            backup = published_cache.with_name(f".visualization-previous-{uuid.uuid4().hex}")
            had_cache = published_cache.exists()
            if had_cache:
                os.replace(published_cache, backup)
            try:
                os.replace(cache_dir, published_cache)
            except Exception:
                if had_cache:
                    os.replace(backup, published_cache)
                raise
            if had_cache:
                shutil.rmtree(backup)
        print(f"All done in {time.time()-t0:.2f} s.")
        return True

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        traceback.print_exc()
        return False
    finally:
        plt.close("all")
        shutil.rmtree(cache_dir, ignore_errors=True)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Precompute visualization data.")
    p.add_argument("--caseId", required=True, help="case id under uploads/")
    args = p.parse_args()

    ok = precompute_all_data(args.caseId)
    sys.exit(0 if ok else 1)
