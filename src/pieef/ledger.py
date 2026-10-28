from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping

DEFAULT_SEEDS = [13, 21, 42, 87, 3407]

RUN_REGISTRY_FIELDS = [
    "run_id",
    "exp_type",
    "model",
    "variant",
    "seed",
    "horizon",
    "eval_scope",
    "data_version",
    "config_id",
    "code_commit",
    "artifact_path",
    "start_time",
    "end_time",
    "status",
    "fail_reason",
]

MAIN_BENCHMARK_FIELDS = RUN_REGISTRY_FIELDS + [
    "input_signature",
    "arch_family",
    "backbone",
    "spatial_module",
    "physics_module",
    "key_hparams",
    "nmae",
    "nrmse",
    "r2",
]
SIGNIFICANCE_FIELDS = RUN_REGISTRY_FIELDS + [
    "ref_model",
    "baseline_model",
    "dm_stat",
    "dm_pvalue",
    "bh_pvalue",
    "significant",
    "error_series_id",
]
ABLATION_FIELDS = RUN_REGISTRY_FIELDS + [
    "pi_mae",
    "td_dcn",
    "sa_dwgn",
    "nmae",
    "nrmse",
    "r2",
    "delta_nmae_pct",
    "delta_nrmse_pct",
    "delta_r2_pct",
]
SENS_MASK_RATIO_FIELDS = RUN_REGISTRY_FIELDS + [
    "scan_id",
    "mask_ratio",
    "other_hparams_frozen",
    "best_value_by_nrmse",
    "nmae",
    "nrmse",
    "r2",
]
SENS_CONTROL_POINTS_FIELDS = RUN_REGISTRY_FIELDS + [
    "scan_id",
    "control_points",
    "other_hparams_frozen",
    "best_value_by_nrmse",
    "nmae",
    "nrmse",
    "r2",
]
ROBUST_WS_FIELDS = RUN_REGISTRY_FIELDS + [
    "noise_type",
    "noise_dist",
    "noise_level",
    "injection_target",
    "injection_stage",
    "injection_rule_version",
    "nrmse",
]
ROBUST_WD_FIELDS = RUN_REGISTRY_FIELDS + [
    "noise_type",
    "noise_dist",
    "noise_level",
    "injection_target",
    "injection_stage",
    "injection_rule_version",
    "nrmse",
]
ROBUST_CURTAIL_FIELDS = RUN_REGISTRY_FIELDS + [
    "curtailment_bin",
    "samples",
    "bin_rule_version",
    "nrmse",
]
EFFICIENCY_FIELDS = RUN_REGISTRY_FIELDS + [
    "train_time_s",
    "inference_latency_ms",
    "params_k",
    "memory_gb",
    "hardware_env",
    "latency_warmup",
    "latency_repeat",
    "latency_stat",
    "train_time_sec",
    "inference_time_ms_per_batch",
    "peak_memory_mb",
    "num_params",
    "flops",
]
SCALABILITY_FIELDS = RUN_REGISTRY_FIELDS + [
    "turbine_count",
    "split_or_sampling_rule",
    "inference_latency_ms",
    "nmae",
    "nrmse",
    "r2",
    "train_time_s",
    "train_time_sec",
    "memory_gb",
    "peak_memory_mb",
]


@dataclass(frozen=True)
class LedgerSpec:
    file_name: str
    fields: Iterable[str]


LEDGER_SPECS: Dict[str, LedgerSpec] = {
    "run_registry": LedgerSpec("run_registry.csv", RUN_REGISTRY_FIELDS),
    "main": LedgerSpec("main_benchmark_runs.csv", MAIN_BENCHMARK_FIELDS),
    "significance": LedgerSpec("significance_dm_bh.csv", SIGNIFICANCE_FIELDS),
    "ablation": LedgerSpec("ablation_runs.csv", ABLATION_FIELDS),
    "sensitivity_mask_ratio": LedgerSpec("sensitivity_mask_ratio.csv", SENS_MASK_RATIO_FIELDS),
    "sensitivity_control_points": LedgerSpec("sensitivity_control_points.csv", SENS_CONTROL_POINTS_FIELDS),
    "robustness_wind_speed_noise": LedgerSpec("robustness_wind_speed_noise.csv", ROBUST_WS_FIELDS),
    "robustness_wind_direction_noise": LedgerSpec("robustness_wind_direction_noise.csv", ROBUST_WD_FIELDS),
    "robustness_curtailment_bins": LedgerSpec("robustness_curtailment_bins.csv", ROBUST_CURTAIL_FIELDS),
    "efficiency": LedgerSpec("efficiency_cost.csv", EFFICIENCY_FIELDS),
    "scalability": LedgerSpec("scalability_turbine_count.csv", SCALABILITY_FIELDS),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "NA"
    except Exception:
        return "NA"


def build_run_id(exp_type: str, model_or_variant: str, horizon: int, seed: int, ts: datetime | None = None) -> str:
    safe_model = model_or_variant.replace(" ", "_")
    t = (ts or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{exp_type}_{safe_model}_h{horizon}_s{seed}_{t}"


def ensure_csv(path: Path, headers: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()


def normalize_csv_schema(path: Path, headers: Iterable[str]) -> tuple[int, int]:
    """Normalize an existing CSV file to the target headers.

    Returns (kept_rows, skipped_rows).
    """
    fields = list(headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) or path.stat().st_size == 0:
        ensure_csv(path, fields)
        return (0, 0)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            old_header = next(reader)
        except StopIteration:
            ensure_csv(path, fields)
            return (0, 0)
        rows = list(reader)

    if old_header == fields and all(len(r) == len(fields) for r in rows):
        return (len(rows), 0)

    normalized: list[dict[str, object]] = []
    skipped = 0
    for r in rows:
        if len(r) == len(fields):
            rec = {k: v for k, v in zip(fields, r)}
            normalized.append(rec)
            continue
        if len(r) == len(old_header):
            rec_old = {k: v for k, v in zip(old_header, r)}
            rec = {k: rec_old.get(k, "NA") for k in fields}
            normalized.append(rec)
            continue
        skipped += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in normalized:
            writer.writerow({k: rec.get(k, "NA") for k in fields})
    return (len(normalized), skipped)


def append_row(path: Path, headers: Iterable[str], row: Mapping[str, object]) -> None:
    fields = list(headers)
    normalize_csv_schema(path, fields)
    normalized = {k: row.get(k, "NA") for k in fields}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(normalized)


def ensure_ledger_templates(ledger_dir: Path) -> None:
    for spec in LEDGER_SPECS.values():
        ensure_csv(ledger_dir / spec.file_name, spec.fields)


def base_row(
    *,
    run_id: str,
    exp_type: str,
    model: str,
    variant: str,
    seed: int,
    horizon: int,
    data_version: str,
    config_id: str,
    artifact_path: str,
    status: str,
    start_time: str,
    end_time: str,
    fail_reason: str,
    eval_scope: str = "all-sample",
    code_commit: str | None = None,
) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "exp_type": exp_type,
        "model": model,
        "variant": variant,
        "seed": seed,
        "horizon": horizon,
        "eval_scope": eval_scope,
        "data_version": data_version,
        "config_id": config_id,
        "code_commit": code_commit or get_git_commit(),
        "artifact_path": artifact_path,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "fail_reason": fail_reason,
    }
