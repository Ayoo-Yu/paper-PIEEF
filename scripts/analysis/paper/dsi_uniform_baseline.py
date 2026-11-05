"""Compare attention DSI with the value expected under uniform attention.

DSI = A_up / (A_up + A_down). Under uniform attention it equals
n_up / (n_up + n_down), which depends on turbine geometry and flow direction.
Outputs dsi_uniform_baseline.csv next to this script.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"results/paper")
SCADA = Path(r"data/kelmarsh/SCADA_15min.csv")
TI_LOW, TI_HIGH, WS_MIN = 0.124, 0.166, 4.0  # frozen thresholds (plan section 2.1)
OUT = Path(__file__).with_name("dsi_uniform_baseline.csv")


def ti_strata(scada, timestamps):
    s = scada.reindex(pd.to_datetime(timestamps, utc=True))
    ws = np.stack([s[f"turbine_ws{i}"].values for i in range(1, 7)], 1)
    sd = np.stack([s[f"wind_speed_std{i}"].values for i in range(1, 7)], 1)
    ti = np.where(ws >= WS_MIN, sd / ws, np.nan)
    lab = np.full(ti.shape, "undefined", dtype=object)
    lab[ti < TI_LOW] = "low"
    lab[(ti >= TI_LOW) & (ti < TI_HIGH)] = "medium"
    lab[ti >= TI_HIGH] = "high"
    return lab


def summarise(diag, mask):
    up, dn = diag["up_neighbor_count"][mask], diag["down_neighbor_count"][mask]
    tot = up + dn
    uniform = np.where(tot > 0, up / np.maximum(tot, 1), np.nan)
    dsi = diag["dsi"][mask]
    return dict(n_queries=int(mask.sum()), dsi_observed=np.nanmean(dsi), dsi_uniform=np.nanmean(uniform),
                dsi_excess=np.nanmean(dsi) - np.nanmean(uniform), n_up_mean=up.mean(), n_down_mean=dn.mean())


def main():
    scada = pd.read_csv(SCADA, parse_dates=["Timestamp"]).set_index("Timestamp")
    rows = []
    for h in (1, 4, 16):
        for seed in (13, 42, 3407):
            f = ROOT / f"补充验证_20261023/M_controls/VF_Kelmarsh_h{h}_seed{seed}/test_attention_diagnostics.npz"
            diag = np.load(f)
            lab = ti_strata(scada, diag["timestamp"])
            for stratum in ("low", "medium", "high", "undefined"):
                rows.append(dict(site="Kelmarsh", variant="VF", horizon_steps=h, seed=seed, stratum=stratum,
                                 source=str(f), **summarise(diag, lab == stratum)))
    for pattern, variant in [("补充验证_20261023/M_controls/VF_wind_farm_A_h16_seed*", "VF"),
                             ("补充验证_20261023/M_controls/direction_off_wind_farm_A_h16_seed*", "direction-off"),
                             ("补充验证_20261023/M0/runs/energy_r4_M0_wind_farm_A_h16_seed*", "M0 100-m direction"),
                             ("补充验证_20261023/M1/runs/energy_r4_M1_wind_farm_A_h16_seed*", "M1 CFD direction"),
                             ("补充验证_20261023/M2/runs/energy_r4_M2_wind_farm_A_h16_seed*", "M2 CFD direction+speed")]:
        for d in sorted(glob.glob(str(ROOT / pattern))):
            f = Path(d) / "test_attention_diagnostics.npz"
            diag = np.load(f)
            seed = int(d.rsplit("seed", 1)[1])
            rows.append(dict(site="wind_farm_A", variant=variant, horizon_steps=16, seed=seed, stratum="all",
                             source=str(f), **summarise(diag, np.ones(diag["dsi"].shape, bool))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    cols = ["dsi_observed", "dsi_uniform", "dsi_excess", "n_up_mean", "n_down_mean"]
    print(df.groupby(["site", "variant", "horizon_steps", "stratum"])[cols].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
