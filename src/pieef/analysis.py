from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas.errors import ParserError


@dataclass
class DMResult:
    dm_stat: float
    p_value: float
    n: int


def load_main_runs(main_csv: Path) -> pd.DataFrame:
    if not main_csv.exists():
        raise FileNotFoundError(f"Missing main ledger: {main_csv}")

    try:
        df = pd.read_csv(main_csv)
    except ParserError:
        # Be robust to occasional malformed ledger rows (e.g., unescaped commas in fail_reason).
        # We keep valid rows and continue significance/statistical summaries.
        df = pd.read_csv(main_csv, on_bad_lines="skip", engine="python")
        print(f"[analysis] warning: skipped malformed rows while reading {main_csv}")
    if df.empty:
        return df

    df = df[df["status"] == "success"].copy()
    for c in ["seed", "horizon", "nmae", "nrmse", "r2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "horizon",
                "n_runs",
                "nmae_mean",
                "nmae_std",
                "nrmse_mean",
                "nrmse_std",
                "r2_mean",
                "r2_std",
            ]
        )

    g = (
        df.groupby(["model", "horizon"], as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            nmae_mean=("nmae", "mean"),
            nmae_std=("nmae", "std"),
            nrmse_mean=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )
    return g


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def dm_test(loss_ref: np.ndarray, loss_base: np.ndarray) -> DMResult:
    x = np.asarray(loss_ref).reshape(-1)
    y = np.asarray(loss_base).reshape(-1)
    n = min(len(x), len(y))
    if n < 3:
        return DMResult(dm_stat=float("nan"), p_value=float("nan"), n=n)

    # Positive d means ref has lower loss than baseline.
    d = y[:n] - x[:n]
    d_mean = float(np.mean(d))
    d_var = float(np.var(d, ddof=1))
    if d_var <= 1e-12:
        return DMResult(dm_stat=float("nan"), p_value=float("nan"), n=n)

    dm_stat = d_mean / math.sqrt(d_var / n)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))
    return DMResult(dm_stat=dm_stat, p_value=p_value, n=n)


def bh_correction(p_values: Iterable[float]) -> list[float]:
    p = np.array(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return []

    order = np.argsort(np.nan_to_num(p, nan=1.0))
    ranked = p[order]

    adjusted = np.full(m, np.nan, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        raw = ranked[i] * m / rank
        val = min(prev, raw)
        prev = val
        adjusted[i] = min(max(val, 0.0), 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def load_loss_series_from_npz(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {npz_path}")

    d = np.load(npz_path)
    y_true = d["y_true_mw"].reshape(-1)
    y_pred = d["y_pred_mw"].reshape(-1)
    return (y_true - y_pred) ** 2
