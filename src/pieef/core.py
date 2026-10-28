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


def get_device(*, require_cuda: bool = False) -> torch.device:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError("CUDA is required for revision neural-network training; CPU fallback is disabled.")
    return torch.device("cuda" if available else "cpu")


def calc_metrics(
    y_true_mw: np.ndarray, y_pred_mw: np.ndarray,
    capacity_mw: float | np.ndarray,
) -> tuple[float, float, float]:
    """Per-turbine capacity-normalized errors; R² pools valid original MW values."""
    if y_true_mw.shape != y_pred_mw.shape:
        raise ValueError("Prediction and target shapes must match.")
    valid = np.isfinite(y_true_mw)
    if not valid.any():
        raise ValueError("No valid measured targets to evaluate.")
    if not np.isfinite(y_pred_mw[valid]).all():
        raise ValueError("Non-finite prediction for a valid measured target.")
    capacity = np.asarray(capacity_mw, dtype=float)
    if capacity.ndim > 1 or not np.isfinite(capacity).all() or not (capacity > 0).all():
        raise ValueError("Nominal capacity must be positive and finite, scalar or per-turbine vector.")
    if capacity.ndim == 1 and (y_true_mw.ndim < 1 or capacity.shape != (y_true_mw.shape[-1],)):
        raise ValueError("Capacity vector must match the final turbine axis of predictions.")
    normalized_error = ((y_true_mw - y_pred_mw) / capacity)[valid]
    nmae = float(np.mean(np.abs(normalized_error)))
    nrmse = float(np.sqrt(np.mean(normalized_error ** 2)))

    y_true_mw, y_pred_mw = y_true_mw[valid], y_pred_mw[valid]

    ss_res = float(np.sum((y_true_mw - y_pred_mw) ** 2))
    ss_tot = float(np.sum((y_true_mw - np.mean(y_true_mw)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-7)
    return nmae, nrmse, r2
