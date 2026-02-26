from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_metrics(y_true_mw: np.ndarray, y_pred_mw: np.ndarray, capacity_mw: float) -> tuple[float, float, float]:
    cap = capacity_mw if capacity_mw > 0 else 1.0

    mae = float(np.mean(np.abs(y_true_mw - y_pred_mw)))
    rmse = float(np.sqrt(np.mean((y_true_mw - y_pred_mw) ** 2)))

    nmae = mae / cap
    nrmse = rmse / cap

    ss_res = float(np.sum((y_true_mw - y_pred_mw) ** 2))
    ss_tot = float(np.sum((y_true_mw - np.mean(y_true_mw)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-7)
    return nmae, nrmse, r2
