#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from pieef.baselines import (
    ACFormerModel,
    DLTrainConfig,
    DLinearModel,
    GRUModel,
    ITransformerModel,
    JensenNetModel,
    PatchTSTModel,
    STANModel,
    TCNModel,
    TimeMixerModel,
    TimesNetModel,
    WakeGNNModel,
    eval_dl_model,
    eval_persistence,
    eval_tabular_model,
    train_dl_model,
    train_eval_hgbr,
    train_eval_lightgbm,
    train_eval_mlp,
    train_eval_xgboost,
)
from pieef.core import get_device, set_seed
from pieef.data import build_data_bundle
from pieef.revision_data import require_verified_forecast_input
from pieef.artifacts import save_preprocessing, save_run_predictions, write_json_atomic
from pieef.ledger import LEDGER_SPECS, append_row, base_row, build_run_id, ensure_ledger_templates
from pieef.model_registry import get_arch_info
from pieef.training_timing import save_training_timing


DL_MODELS = {"GRU", "DLinear", "TCN", "iTransformer", "PatchTST", "ACFormer", "TimesNet", "TimeMixer", "STAN", "Wake-GNN", "Jensen-Net"}
ML_MODELS = {"HistGBR", "XGBoost", "LightGBM", "MLP"}
OTHER_MODELS = {"Persistence"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline models and log to main benchmark ledger.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--config-id", "--config_id")
    p.add_argument("--artifact-path", "--artifact_path")
    p.add_argument("--max-seeds", type=int)
    p.add_argument("--max-batches-per-epoch", type=int)
    p.add_argument("--max-test-batches", type=int)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_path(value: str | None, default_rel: str) -> Path:
    if value:
        p = Path(value)
        return p if p.is_absolute() else (ROOT / p)
    return ROOT / default_rel


def _load_turbine_static(turbines_path: Path, num_turbines: int) -> torch.Tensor:
    rows = []
    with turbines_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if len(rows) < num_turbines:
        raise ValueError(f"turbines.csv has {len(rows)} rows, expected >= {num_turbines}")

    xs = np.array([float(rows[i]["x_coord"]) for i in range(num_turbines)], dtype=np.float32)
    ys = np.array([float(rows[i]["y_coord"]) for i in range(num_turbines)], dtype=np.float32)
    rated = np.array([float(rows[i].get("rated_power", 2000.0)) for i in range(num_turbines)], dtype=np.float32)

    x_n = (xs - xs.mean()) / (xs.std() + 1e-6)
    y_n = (ys - ys.mean()) / (ys.std() + 1e-6)
    p_n = rated / (rated.mean() + 1e-6)

    coords = np.stack([xs, ys], axis=-1)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    density = 1.0 / (dist.mean(axis=1) + 1e-6)
    density_n = density / (density.mean() + 1e-6)

    static = np.stack([x_n, y_n, p_n, density_n], axis=-1).astype(np.float32)
    return torch.from_numpy(static)


def _load_turbine_meta(turbines_path: Path, num_turbines: int) -> dict[str, torch.Tensor]:
    rows = []
    with turbines_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if len(rows) < num_turbines:
        raise ValueError(f"turbines.csv has {len(rows)} rows, expected >= {num_turbines}")

    xs = np.array([float(rows[i]["x_coord"]) for i in range(num_turbines)], dtype=np.float32)
    ys = np.array([float(rows[i]["y_coord"]) for i in range(num_turbines)], dtype=np.float32)
    rotor = np.array([float(rows[i].get("rotor_diameter", 120.0)) for i in range(num_turbines)], dtype=np.float32)
    return {
        "coords_xy": torch.from_numpy(np.stack([xs, ys], axis=-1).astype(np.float32)),
        "rotor_d": torch.from_numpy(rotor.astype(np.float32)),
    }


def build_model(name: str, seq_len: int, bundle=None, turbine_meta: dict | None = None, turbines_path: Path | None = None):
    nwp_dim = int(bundle.nwp_dim) if bundle is not None else 12
    if name == "GRU":
        return GRUModel(hidden_dim=64, nwp_dim=nwp_dim)
    if name == "DLinear":
        return DLinearModel(seq_len=seq_len, nwp_dim=nwp_dim)
    if name == "TCN":
        return TCNModel(channels=32, nwp_dim=nwp_dim)
    if name == "iTransformer":
        return ITransformerModel(seq_len=seq_len, d_model=128, n_heads=4, nwp_dim=nwp_dim)
    if name == "PatchTST":
        return PatchTSTModel(seq_len=seq_len, patch_len=16, stride=8, d_model=64, nwp_dim=nwp_dim)
    if name == "ACFormer":
        return ACFormerModel(
            seq_len=seq_len,
            d_model=96,
            patch_kernel=10,
            stride=2,
            n_heads=8,
            n_layers=1,
            nwp_dim=nwp_dim,
        )
    if name == "TimesNet":
        return TimesNetModel(seq_len=seq_len, d_model=64, nwp_dim=nwp_dim)
    if name == "TimeMixer":
        return TimeMixerModel(seq_len=seq_len, hidden=64, nwp_dim=nwp_dim)
    if name == "STAN":
        node_static = None
        dist_matrix = None
        if bundle is not None:
            dist_matrix = bundle.dist_tensor
        if turbines_path is not None and bundle is not None:
            node_static = _load_turbine_static(turbines_path, bundle.num_turbines)
        return STANModel(
            seq_len=seq_len,
            hidden=64,
            gru_hidden=200,
            gat_hidden=64,
            dist_matrix=dist_matrix,
            node_static=node_static,
            gravity_alpha=0.35,
            gravity_beta=0.37,
            gravity_r=30.0,
            nwp_dim=nwp_dim,
        )
    if name == "Wake-GNN":
        kwargs = {"nwp_dim": nwp_dim}
        if turbine_meta is not None:
            kwargs["coords_xy"] = turbine_meta["coords_xy"]
            kwargs["rotor_d"] = turbine_meta["rotor_d"]
        if bundle is not None:
            kwargs["nwp_mean"] = torch.from_numpy(bundle.handler.nwp_mean.astype(np.float32))
            kwargs["nwp_std"] = torch.from_numpy(bundle.handler.nwp_std.astype(np.float32))
            kwargs["nwp_target_speed_index"] = getattr(bundle.handler, "nwp_target_speed_index", -3)
        return WakeGNNModel(seq_len=seq_len, hidden=64, kw=0.05, **kwargs)
    if name == "Jensen-Net":
        kwargs = {"nwp_dim": nwp_dim}
        if turbine_meta is not None:
            kwargs["coords_xy"] = turbine_meta["coords_xy"]
            kwargs["rotor_d"] = turbine_meta["rotor_d"]
        if bundle is not None:
            kwargs["nwp_mean"] = torch.from_numpy(bundle.handler.nwp_mean.astype(np.float32))
            kwargs["nwp_std"] = torch.from_numpy(bundle.handler.nwp_std.astype(np.float32))
            kwargs["nwp_target_speed_index"] = getattr(bundle.handler, "nwp_target_speed_index", -3)
        return JensenNetModel(seq_len=seq_len, hidden=64, kw=0.05, **kwargs)
    raise ValueError(f"Unsupported DL model: {name}")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    train_cfg_raw = cfg["train"]
    track_cfg = cfg["tracking"]
    baseline_cfg = cfg["baselines"]
    if not args.dry_run:
        require_verified_forecast_input(data_cfg.get("revision_protocol"))

    models = baseline_cfg["models"]
    seeds = list(exp_cfg["seeds"])
    if args.seed is not None:
        seeds = [int(args.seed)]
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    if args.horizon is not None:
        exp_cfg["horizon"] = int(args.horizon)
    if args.config_id:
        track_cfg["config_id"] = args.config_id
    if args.artifact_path:
        track_cfg["artifact_root"] = args.artifact_path

    device = get_device(
        require_cuda=not args.dry_run
        and bool(data_cfg.get("revision_protocol"))
        and bool(set(models) & DL_MODELS)
    )

    scada_path = _resolve_data_path(data_cfg.get("scada_path"), "data/wind_farm_A/scada_ecmwf_15min.csv")
    turbines_path = _resolve_data_path(data_cfg.get("turbines_path"), "data/raw/turbines.csv")

    bundle = build_data_bundle(
        scada_path=scada_path,
        turbines_path=turbines_path,
        seq_len=int(data_cfg["seq_len"]),
        horizon=int(exp_cfg["horizon"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        train_ratio=float(data_cfg["split"][0]),
        val_ratio=float(data_cfg["split"][1]),
        capacity_mw=float(data_cfg.get("capacity_mw", 2.0)),
        revision_protocol=data_cfg.get("revision_protocol"),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "models": models,
                    "num_turbines": bundle.num_turbines,
                    "train_batches": len(bundle.train_loader),
                    "test_batches": len(bundle.test_loader),
                },
                ensure_ascii=False,
            )
        )
        return

    ledger_dir = ROOT / track_cfg.get("ledger_dir", "ledger")
    ensure_ledger_templates(ledger_dir)

    dl_cfg = DLTrainConfig(
        epochs=int(train_cfg_raw.get("epochs", 5)),
        lr=float(train_cfg_raw.get("lr", 1e-3)),
        max_batches_per_epoch=train_cfg_raw.get("max_batches_per_epoch"),
        max_test_batches=train_cfg_raw.get("max_test_batches"),
        patience=train_cfg_raw.get("patience"),
        min_delta=float(train_cfg_raw.get("min_delta", 0.0)),
        loss_name=train_cfg_raw.get("loss_name"),
        huber_delta=float(train_cfg_raw.get("huber_delta", 0.1)),
    )
    if args.max_batches_per_epoch is not None:
        dl_cfg.max_batches_per_epoch = args.max_batches_per_epoch
    if args.max_test_batches is not None:
        dl_cfg.max_test_batches = args.max_test_batches
    turbine_meta = _load_turbine_meta(turbines_path, bundle.num_turbines)

    for seed in seeds:
        set_seed(int(seed))
        sample_batch = next(iter(bundle.test_loader))
        sample_x_p = sample_batch[0].cpu().numpy()
        bsz, n_turb, seq_len, _ = sample_x_p.shape
        sample_hist = torch.cat(sample_batch[:3], dim=-1).reshape(bsz * n_turb, -1)
        sample_x_flat = torch.cat([sample_hist, sample_batch[3].reshape(bsz * n_turb, -1)], dim=-1).cpu().numpy()

        for model_name in models:
            run_id = build_run_id(
                exp_type=exp_cfg["exp_type"],
                model_or_variant=model_name,
                horizon=int(exp_cfg["horizon"]),
                seed=int(seed),
            )
            print(
                f"[start] seed={seed} horizon={exp_cfg['horizon']} model={model_name} run_id={run_id}",
                flush=True,
            )
            run_artifact_dir = ROOT / track_cfg.get("artifact_root", "artifacts/baselines") / run_id
            run_artifact_dir.mkdir(parents=True, exist_ok=True)

            start_ts = datetime.now().isoformat(timespec="seconds")
            t0_model = time.perf_counter()

            try:
                save_preprocessing(run_artifact_dir, bundle)
                if model_name in OTHER_MODELS:
                    metrics, y_true_mw, y_pred_mw = eval_persistence(
                        bundle.test_loader,
                        capacity_mw=bundle.capacity_mw,
                        max_test_batches=dl_cfg.max_test_batches,
                    )
                    model_obj = None
                elif model_name in DL_MODELS:
                    model = build_model(
                        model_name,
                        seq_len=int(data_cfg["seq_len"]),
                        bundle=bundle,
                        turbine_meta=turbine_meta,
                        turbines_path=turbines_path,
                    )
                    model = train_dl_model(
                        model,
                        bundle.train_loader,
                        device=device,
                        cfg=dl_cfg,
                        progress_label=f"{model_name}|s{seed}|h{exp_cfg['horizon']}",
                        val_loader=bundle.val_loader,
                        checkpoint_dir=run_artifact_dir,
                    )
                    # The timing writer expects experiment.model, which baseline configs express
                    # through baselines.models instead; stamp the concrete model name here.
                    cfg.setdefault("experiment", {})["model"] = model_name
                    save_training_timing(run_artifact_dir, model, cfg, seed, exp_cfg["horizon"], device)
                    metrics, y_true_mw, y_pred_mw = eval_dl_model(
                        model,
                        bundle.test_loader,
                        device=device,
                        capacity_mw=bundle.capacity_mw,
                        max_test_batches=dl_cfg.max_test_batches,
                    )
                    model_obj = model
                elif model_name == "HistGBR":
                    model_obj, metrics, y_true_mw, y_pred_mw = train_eval_hgbr(
                        bundle.train_loader,
                        bundle.test_loader,
                        capacity_mw=bundle.capacity_mw,
                        seed=int(seed),
                        max_batches=dl_cfg.max_test_batches,
                        cache_dir=run_artifact_dir / "_tabular_cache",
                    )
                elif model_name == "LightGBM":
                    model_obj, metrics, y_true_mw, y_pred_mw = train_eval_lightgbm(
                        bundle.train_loader,
                        bundle.test_loader,
                        capacity_mw=bundle.capacity_mw,
                        seed=int(seed),
                        max_batches=dl_cfg.max_test_batches,
                        cache_dir=run_artifact_dir / "_tabular_cache",
                        loss_name=dl_cfg.loss_name,
                        huber_delta=dl_cfg.huber_delta,
                    )
                elif model_name == "XGBoost":
                    model_obj, metrics, y_true_mw, y_pred_mw = train_eval_xgboost(
                        bundle.train_loader,
                        bundle.test_loader,
                        capacity_mw=bundle.capacity_mw,
                        seed=int(seed),
                        max_batches=dl_cfg.max_test_batches,
                        cache_dir=run_artifact_dir / "_tabular_cache",
                    )
                elif model_name == "MLP":
                    model_obj, metrics, y_true_mw, y_pred_mw = train_eval_mlp(
                        bundle.train_loader,
                        bundle.test_loader,
                        capacity_mw=bundle.capacity_mw,
                        seed=int(seed),
                        max_batches=dl_cfg.max_test_batches,
                        cache_dir=run_artifact_dir / "_tabular_cache",
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                if model_name in OTHER_MODELS:
                    validation_metrics, val_true, val_pred = eval_persistence(
                        bundle.val_loader, bundle.capacity_mw, max_test_batches=dl_cfg.max_test_batches)
                elif model_name in DL_MODELS:
                    validation_metrics, val_true, val_pred = eval_dl_model(
                        model_obj, bundle.val_loader, device, bundle.capacity_mw,
                        max_test_batches=dl_cfg.max_test_batches)
                else:
                    validation_metrics, val_true, val_pred = eval_tabular_model(
                        model_obj, bundle.val_loader, bundle.capacity_mw,
                        max_batches=dl_cfg.max_test_batches)
                save_run_predictions(run_artifact_dir, bundle, val_true, val_pred,
                                     seed=int(seed), horizon=int(exp_cfg["horizon"]), split="val")
                save_run_predictions(
                    run_artifact_dir, bundle, y_true_mw, y_pred_mw,
                    seed=int(seed), horizon=int(exp_cfg["horizon"]),
                )
                np.savez_compressed(
                    run_artifact_dir / "latency_sample.npz",
                    x_p=sample_x_p,
                    x_w=sample_batch[1].cpu().numpy(),
                    x_n_hist=sample_batch[2].cpu().numpy(),
                    x_nwp_target=sample_batch[3].cpu().numpy(),
                    x_flat=sample_x_flat,
                )
                if model_obj is not None and isinstance(model_obj, torch.nn.Module):
                    if data_cfg.get("revision_protocol") is None:
                        torch.save(model_obj.state_dict(), run_artifact_dir / "model.pt")
                elif model_obj is not None:
                    import joblib

                    joblib.dump(model_obj, run_artifact_dir / "model.pkl")

                write_json_atomic(
                    run_artifact_dir / "metrics.json",
                    {
                        "run_id": run_id,
                        "status": "completed",
                        "model": model_name,
                        "seed": int(seed),
                        "horizon": int(exp_cfg["horizon"]),
                        "metrics": metrics,
                        "validation_metrics": validation_metrics,
                        "evaluation_split": "test",
                        "config": cfg,
                        "device": str(device) if model_name in DL_MODELS else "cpu",
                        "training_summary": getattr(model_obj, "training_summary", None),
                    },
                )

                end_ts = datetime.now().isoformat(timespec="seconds")
                common = base_row(
                    run_id=run_id,
                    exp_type=exp_cfg["exp_type"],
                    model=model_name,
                    variant="NA",
                    seed=int(seed),
                    horizon=int(exp_cfg["horizon"]),
                    data_version=data_cfg["data_version"],
                    config_id=track_cfg["config_id"],
                    artifact_path=str(run_artifact_dir),
                    status="success",
                    start_time=start_ts,
                    end_time=end_ts,
                    fail_reason="NA",
                    eval_scope=exp_cfg.get("eval_scope", "all-sample"),
                )
                append_row(ledger_dir / LEDGER_SPECS["run_registry"].file_name, LEDGER_SPECS["run_registry"].fields, common)
                main_row = dict(common)
                main_row.update(get_arch_info(model_name))
                main_row.update(metrics)
                append_row(ledger_dir / LEDGER_SPECS["main"].file_name, LEDGER_SPECS["main"].fields, main_row)
                print(json.dumps({"run_id": run_id, "model": model_name, "metrics": metrics}, ensure_ascii=False))
                print(
                    f"[done] seed={seed} horizon={exp_cfg['horizon']} model={model_name} "
                    f"elapsed={time.perf_counter() - t0_model:.1f}s",
                    flush=True,
                )

            except Exception as exc:
                end_ts = datetime.now().isoformat(timespec="seconds")
                fail_reason = str(exc)[:300]
                (run_artifact_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                fail_row = base_row(
                    run_id=run_id,
                    exp_type=exp_cfg["exp_type"],
                    model=model_name,
                    variant="NA",
                    seed=int(seed),
                    horizon=int(exp_cfg["horizon"]),
                    data_version=data_cfg["data_version"],
                    config_id=track_cfg["config_id"],
                    artifact_path=str(run_artifact_dir),
                    status="failed",
                    start_time=start_ts,
                    end_time=end_ts,
                    fail_reason=fail_reason,
                    eval_scope=exp_cfg.get("eval_scope", "all-sample"),
                )
                append_row(ledger_dir / LEDGER_SPECS["run_registry"].file_name, LEDGER_SPECS["run_registry"].fields, fail_row)
                print(json.dumps({"run_id": run_id, "model": model_name, "status": "failed", "reason": fail_reason}, ensure_ascii=False))
                print(
                    f"[failed] seed={seed} horizon={exp_cfg['horizon']} model={model_name} "
                    f"elapsed={time.perf_counter() - t0_model:.1f}s reason={fail_reason}",
                    flush=True,
                )
                if args.fail_fast:
                    raise


if __name__ == "__main__":
    main()
