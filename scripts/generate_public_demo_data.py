#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


NWP_COLS = [
    "Direct_solar_radiation_10",
    "Total_sky_direct_short-wave_(solar)_radiation_at_surface_10",
    "Surface_pressure_10",
    "2_metre_temperature_10",
    "10_metre_U_wind_component_10",
    "10_metre V_wind_component_10",
    "wind_speed200_10",
    "wind_speed100_10",
    "wind_speed10_10",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate public synthetic demo data for PI-EEF pipeline.")
    p.add_argument("--output-dir", default="data/public_demo")
    p.add_argument("--turbines", type=int, default=29)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--capacity-kw", type=float, default=2000.0)
    return p.parse_args()


def _power_curve(ws: float, rated_kw: float) -> float:
    cut_in = 3.0
    rated = 12.0
    if ws <= cut_in:
        return 0.0
    if ws >= rated:
        return rated_kw
    x = (ws - cut_in) / (rated - cut_in)
    return rated_kw * (x ** 3)


def build_turbines(path: Path, n_turbines: int, rated_kw: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turbine_id", "x_coord", "y_coord", "rated_power", "hub_height", "rotor_diameter"])
        grid = int(math.ceil(math.sqrt(n_turbines)))
        spacing = 450.0
        for i in range(n_turbines):
            r = i // grid
            c = i % grid
            x = 1000.0 + c * spacing + random.uniform(-35.0, 35.0)
            y = 1000.0 + r * spacing + random.uniform(-35.0, 35.0)
            w.writerow([f"T{i+1:02d}", f"{x:.6f}", f"{y:.6f}", int(rated_kw), 85, 120])


def build_dms(path: Path, n_turbines: int, days: int, rated_kw: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = days * 24 * 4
    start = datetime(2023, 1, 1, 0, 0, 0)

    ws_bias = [random.uniform(-0.8, 0.8) for _ in range(n_turbines)]
    ws_scale = [random.uniform(0.92, 1.08) for _ in range(n_turbines)]

    header = ["Timestamp"] + NWP_COLS + ["Total_Power"]
    header += [f"turbine_ws{i+1}" for i in range(n_turbines)]
    header += [f"power{i+1}" for i in range(n_turbines)]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(steps):
            ts = start + timedelta(minutes=15 * t)
            hour = ts.hour + ts.minute / 60.0
            season = 0.75 + 0.25 * math.sin(2.0 * math.pi * (ts.timetuple().tm_yday / 365.0))
            diurnal = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))

            base_ws10 = max(0.2, random.gauss(7.2 * season, 1.8))
            base_ws100 = max(0.3, base_ws10 + random.gauss(0.9, 0.4))
            base_ws200 = max(0.4, base_ws100 + random.gauss(0.5, 0.35))

            theta = random.uniform(-math.pi, math.pi)
            u10 = base_ws10 * math.cos(theta)
            v10 = base_ws10 * math.sin(theta)

            solar = 850.0 * diurnal * max(0.0, random.uniform(0.65, 1.0))
            surf_p = random.gauss(100800.0, 650.0)
            t2m = random.gauss(15.0 + 10.0 * math.sin(2.0 * math.pi * (ts.timetuple().tm_yday / 365.0)), 3.2)

            row = [
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                f"{solar:.6f}",
                f"{(solar * random.uniform(0.85, 1.05)):.6f}",
                f"{surf_p:.6f}",
                f"{t2m:.6f}",
                f"{u10:.6f}",
                f"{v10:.6f}",
                f"{base_ws200:.6f}",
                f"{base_ws100:.6f}",
                f"{base_ws10:.6f}",
            ]

            p_list = []
            w_list = []
            for i in range(n_turbines):
                ws_i = max(0.0, base_ws100 * ws_scale[i] + ws_bias[i] + random.gauss(0.0, 0.55))
                p_i = _power_curve(ws_i, rated_kw) * random.uniform(0.92, 1.03)

                if random.random() < 0.015:
                    p_i *= random.uniform(0.05, 0.30)

                if random.random() < 0.003:
                    p_i = -random.uniform(0.5, 12.0)

                p_i = max(-20.0, min(1.03 * rated_kw, p_i))
                w_list.append(ws_i)
                p_list.append(p_i)

            row.append(f"{sum(max(0.0, x) for x in p_list):.6f}")
            row += [f"{x:.6f}" for x in w_list]
            row += [f"{x:.6f}" for x in p_list]
            w.writerow(row)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    dms_path = out_dir / "DMS.csv"
    turbines_path = out_dir / "turbines.csv"

    build_turbines(turbines_path, args.turbines, args.capacity_kw)
    build_dms(dms_path, args.turbines, args.days, args.capacity_kw)

    print(f"wrote={dms_path}")
    print(f"wrote={turbines_path}")


if __name__ == "__main__":
    main()
