"""Strictly paired forecast inference on one loss value per timestamp.

DM/HLN: doi:10.1080/07350015.1995.10524599; doi:10.1016/S0169-2070(96)00719-4.
Bartlett HAC and HLN correction follow the forecast package's DM formulation,
with an explicit bandwidth that can exceed h-1 for remaining serial dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Prediction:
    timestamp: np.ndarray
    turbine_ids: np.ndarray
    seed: int
    horizon: int
    y_true_mw: np.ndarray
    y_pred_mw: np.ndarray
    target_valid: np.ndarray


@dataclass(frozen=True)
class PairedPredictions:
    timestamp: np.ndarray
    turbine_ids: np.ndarray
    seeds: tuple[int, ...]
    horizon: int
    loss_reference: np.ndarray
    loss_baseline: np.ndarray
    mae_reference: np.ndarray
    mae_baseline: np.ndarray
    valid_turbines_per_time: np.ndarray
    rmse_reference_per_seed: np.ndarray
    rmse_baseline_per_seed: np.ndarray
    mae_reference_per_seed: np.ndarray
    mae_baseline_per_seed: np.ndarray
    loss_unit: str


def load_prediction(path: str | Path) -> Prediction:
    required = {"timestamp", "turbine_ids", "seed", "horizon", "y_true_mw", "y_pred_mw", "target_valid"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"prediction file lacks required fields: {sorted(missing)}")
        arrays = {key: np.array(data[key], copy=True) for key in required}
    truth, prediction = arrays["y_true_mw"], arrays["y_pred_mw"]
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("predictions and truth must have identical [timestamps, turbines] shapes")
    timestamp, turbine_ids = arrays["timestamp"], arrays["turbine_ids"]
    valid = arrays["target_valid"]
    if timestamp.ndim != 1 or len(timestamp) != len(truth) or turbine_ids.ndim != 1 or len(turbine_ids) != truth.shape[1]:
        raise ValueError("timestamp and turbine identifiers must exactly index prediction axes")
    if len(np.unique(timestamp)) != len(timestamp) or len(np.unique(turbine_ids)) != len(turbine_ids):
        raise ValueError("duplicate timestamps or turbine identifiers are not allowed")
    try:
        parsed_time = timestamp.astype("datetime64[ns]")
    except (ValueError, TypeError) as exc:
        raise ValueError("timestamps must be parseable forecast target times") from exc
    if np.isnat(parsed_time).any() or np.any(np.diff(parsed_time.astype(np.int64)) <= 0):
        raise ValueError("timestamps must be finite and strictly chronological")
    if valid.shape != truth.shape or valid.dtype.kind != "b":
        raise ValueError("target_valid must be a boolean matrix matching truth")
    if not np.isfinite(truth[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise ValueError("valid targets and their predictions must be finite")
    if arrays["seed"].size != 1 or arrays["horizon"].size != 1:
        raise ValueError("seed and horizon must be scalar metadata")
    seed, horizon = int(arrays["seed"].item()), int(arrays["horizon"].item())
    if horizon < 1:
        raise ValueError("prediction horizon must be positive")
    return Prediction(timestamp, turbine_ids, seed, horizon, truth, prediction, valid)


def _seed_files(paths: Sequence[str | Path]) -> dict[int, Prediction]:
    if not paths:
        raise ValueError("prediction file group cannot be empty")
    result = {}
    for path in paths:
        record = load_prediction(path)
        if record.seed in result:
            raise ValueError("duplicate seed in prediction file group")
        result[record.seed] = record
    return result


def _same_grid(expected: Prediction, actual: Prediction) -> None:
    if expected.horizon != actual.horizon:
        raise ValueError("cannot pair different forecast horizons")
    for name in ("timestamp", "turbine_ids", "target_valid"):
        if not np.array_equal(getattr(expected, name), getattr(actual, name)):
            raise ValueError(f"prediction pairing mismatch: {name}")
    if expected.y_true_mw.shape != actual.y_true_mw.shape or not np.array_equal(
        expected.y_true_mw, actual.y_true_mw, equal_nan=True
    ):
        raise ValueError("prediction pairing mismatch: ground-truth values")


def pair_predictions(
    reference_paths: Sequence[str | Path], baseline_paths: Sequence[str | Path],
    capacity_mw: float | np.ndarray | None = None,
) -> PairedPredictions:
    """Pair identical grids; optional capacities normalize each turbine before aggregation."""
    reference, baseline = _seed_files(reference_paths), _seed_files(baseline_paths)
    if reference.keys() != baseline.keys():
        raise ValueError("reference and baseline must have exactly the same seed set")
    seeds = tuple(sorted(reference))
    expected = reference[seeds[0]]
    capacity = np.asarray(1.0 if capacity_mw is None else capacity_mw, dtype=float)
    if capacity.ndim > 1 or not np.isfinite(capacity).all() or not (capacity > 0).all():
        raise ValueError("Capacity must be finite and positive, scalar or per-turbine vector")
    if capacity.ndim == 1 and capacity.shape != (len(expected.turbine_ids),):
        raise ValueError("Capacity vector must match turbine_ids in the saved order")
    loss_unit = "MW_squared" if capacity_mw is None else "normalized_squared_error"
    count = expected.target_valid.sum(axis=1)
    if np.any(count == 0):
        raise ValueError("all-invalid target timestamps require an explicit common evaluation scope; no silent time-grid shrinkage")
    squared_ref, squared_base, absolute_ref, absolute_base = [], [], [], []
    for seed in seeds:
        ref, base = reference[seed], baseline[seed]
        _same_grid(expected, ref)
        _same_grid(expected, base)
        for record, squared, absolute in ((ref, squared_ref, absolute_ref), (base, squared_base, absolute_base)):
            error = np.where(expected.target_valid, (record.y_pred_mw - record.y_true_mw) / capacity, 0.0)
            squared.append((error ** 2).sum(axis=1) / count)
            absolute.append(np.abs(error).sum(axis=1) / count)
    return PairedPredictions(
        expected.timestamp, expected.turbine_ids, seeds, expected.horizon,
        np.mean(squared_ref, axis=0), np.mean(squared_base, axis=0),
        np.mean(absolute_ref, axis=0), np.mean(absolute_base, axis=0), count,
        np.sqrt(np.average(squared_ref, weights=count, axis=1)),
        np.sqrt(np.average(squared_base, weights=count, axis=1)),
        np.average(absolute_ref, weights=count, axis=1),
        np.average(absolute_base, weights=count, axis=1), loss_unit,
    )


def _loss_difference(loss_reference, loss_baseline) -> np.ndarray:
    reference, baseline = np.asarray(loss_reference, dtype=float), np.asarray(loss_baseline, dtype=float)
    if reference.ndim != 1 or baseline.shape != reference.shape or len(reference) < 3:
        raise ValueError("losses must be equal-length 1-D time series with at least three timestamps")
    if not np.isfinite(reference).all() or not np.isfinite(baseline).all():
        raise ValueError("loss time series must be finite; missing times are never truncated")
    return reference - baseline


def dm_hac_hln(loss_reference, loss_baseline, horizon: int, hac_lag: int | None = None) -> dict:
    """Two-sided DM test; negative statistic means the reference has lower MSE."""
    difference = _loss_difference(loss_reference, loss_baseline)
    n = len(difference)
    h = int(horizon)
    if h < 1 or h >= n:
        raise ValueError("HLN correction requires 1 <= horizon < number of timestamps")
    lag = max(h - 1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))) if hac_lag is None else int(hac_lag)
    if lag < h - 1 or lag >= n:
        raise ValueError("HAC bandwidth must cover at least horizon-1 lags and be less than sample count")
    average = float(np.mean(difference))
    centered = difference - average
    variance = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        variance += 2 * (1 - k / (lag + 1)) * covariance
    factor = float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    if variance <= np.finfo(float).eps * max(np.finfo(float).tiny, float(np.mean(difference ** 2))):
        statistic, pvalue, status = float("nan"), float("nan"), "degenerate_loss_difference"
    else:
        statistic = average / np.sqrt(variance / n) * factor
        pvalue = float(2 * student_t.sf(abs(statistic), df=n - 1))
        status = "ok"
    return dict(dm_stat=float(statistic), p_value=pvalue, mean_loss_difference=average,
                long_run_variance=variance, hln_factor=factor, hac_lag=lag,
                horizon=h, n_time=n, status=status)


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """One BH family for all provided comparisons; undefined tests stay undefined."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("pvalues must be a vector of probabilities or NaN")
    if np.isinf(p).any():
        raise ValueError("pvalues cannot be infinite")
    adjusted = np.full(p.shape, np.nan)
    if not len(p):
        return adjusted
    # Undefined tests occupy the declared family as p=1, conservatively.
    finite = np.isfinite(p)
    safe = np.where(finite, p, 1.0)
    order = np.argsort(safe)
    ranked = safe[order] * len(p) / np.arange(1, len(p) + 1)
    values = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    adjusted[order] = values
    adjusted[~finite] = np.nan
    return adjusted


def paired_block_bootstrap_ci(
    loss_reference, loss_baseline, *, horizon: int = 1,
    block_length: int | None = None, n_bootstrap: int = 2000,
    confidence: float = 0.95, seed: int = 13,
) -> dict:
    """Circular moving-block bootstrap of paired time-level mean loss differences."""
    difference = _loss_difference(loss_reference, loss_baseline)
    n = len(difference)
    if int(horizon) < 1 or int(horizon) >= n:
        raise ValueError("bootstrap horizon must be positive and less than sample count")
    length = max(int(horizon), int(np.ceil(n ** (1 / 3)))) if block_length is None else int(block_length)
    if not int(horizon) <= length < n:
        raise ValueError("block length must cover the forecast horizon and be less than sample count")
    if n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("bootstrap requires at least 100 replicates and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / length))
    offsets = np.arange(length)
    estimates = np.empty(n_bootstrap)
    for repeat in range(n_bootstrap):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        estimates[repeat] = np.mean(difference[indices])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return dict(mean_loss_difference=float(np.mean(difference)), ci_low=float(low), ci_high=float(high),
                confidence=float(confidence), block_length=length, n_bootstrap=int(n_bootstrap))


def adjust_comparison_family(rows: Sequence[dict], alpha: float = 0.05) -> list[dict]:
    """Apply BH once to a complete declared family, including all retained horizons."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    values = bh_adjust([row["p_value"] for row in rows])
    result = []
    for row, value in zip(rows, values):
        adjusted = dict(row)
        adjusted.update(bh_p_value=float(value), bh_family_size=len(rows),
                        significant=bool(np.isfinite(value) and value <= alpha))
        result.append(adjusted)
    return result


def compare_prediction_groups(
    reference_paths: Sequence[str | Path], baseline_groups: Mapping[str, Sequence[str | Path]],
    *, capacity_mw: float | np.ndarray, n_bootstrap: int = 2000, bootstrap_seed: int = 13,
    hac_lag: int | None = None, block_length: int | None = None,
) -> list[dict]:
    """Normalized loss comparisons; apply BH again over the full multi-horizon family."""
    if not baseline_groups:
        raise ValueError("comparisons need baseline file groups")
    rows = []
    for name, files in baseline_groups.items():
        paired = pair_predictions(reference_paths, files, capacity_mw=capacity_mw)
        result = dm_hac_hln(paired.loss_reference, paired.loss_baseline, paired.horizon, hac_lag)
        result.update(paired_block_bootstrap_ci(
            paired.loss_reference, paired.loss_baseline, horizon=paired.horizon,
            block_length=block_length, n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        ))
        rmse_ref, rmse_base = np.mean(paired.rmse_reference_per_seed), np.mean(paired.rmse_baseline_per_seed)
        def sample_std(values):
            return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        result.update(
            baseline=name, seeds=list(paired.seeds), n_turbines=len(paired.turbine_ids),
            loss_unit=paired.loss_unit,
            capacity_mw=np.asarray(capacity_mw).tolist(),
            reference_nrmse=float(rmse_ref), baseline_nrmse=float(rmse_base),
            reference_nrmse_std=sample_std(paired.rmse_reference_per_seed),
            baseline_nrmse_std=sample_std(paired.rmse_baseline_per_seed),
            reference_nmae=float(np.mean(paired.mae_reference_per_seed)),
            baseline_nmae=float(np.mean(paired.mae_baseline_per_seed)),
            reference_nmae_std=sample_std(paired.mae_reference_per_seed),
            baseline_nmae_std=sample_std(paired.mae_baseline_per_seed),
            reference_pooled_nrmse=float(np.sqrt(np.average(paired.loss_reference, weights=paired.valid_turbines_per_time))),
            baseline_pooled_nrmse=float(np.sqrt(np.average(paired.loss_baseline, weights=paired.valid_turbines_per_time))),
            relative_nrmse_gain_pct=float((rmse_base - rmse_ref) / rmse_base * 100) if rmse_base > 0 else float("nan"),
        )
        rows.append(result)
    strongest = min(rows, key=lambda row: row["baseline_nrmse"])
    for row in rows:
        row.update(strongest_baseline=strongest["baseline"],
                   gain_vs_strongest_nrmse_pct=strongest["relative_nrmse_gain_pct"],
                   is_strongest_baseline=row is strongest)
    return adjust_comparison_family(rows)
