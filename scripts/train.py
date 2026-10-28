#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pieef.core import get_device, set_seed
from pieef.data import build_data_bundle
from pieef.forecast_corruption import ForecastCorruption
from pieef.ledger import LEDGER_SPECS, append_row, base_row, build_run_id, ensure_ledger_templates
from pieef.model_main import E2EPhysNet
from pieef.model_factory import build_main_model
from pieef.artifacts import save_preprocessing, save_run_predictions, write_json_atomic
from pieef.revision_data import require_verified_forecast_input
from pieef.model_registry import get_arch_info
from pieef.train_main import TrainConfig, evaluate_multistep, train_multistep
from pieef.training_timing import save_training_timing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PI-EEF main model from YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--config-id", "--config_id")
    p.add_argument("--artifact-path", "--artifact_path")
    p.add_argument("--max-seeds", type=int)
    p.add_argument("--max-turbines", type=int)
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


def _forecast_corruption(tcfg, bundle, seed: int, corruption_seed: int, seq_len: int):
    """Build the F2 sampler from the real data handler, or None when it is off."""
    if tcfg.stage3_forecast_corruption == "off":
        return None
    handler = getattr(bundle, "handler", None)
    coords = getattr(handler, "coords", None)
    if coords is None:
        raise ValueError("F2 corruption requires the real data handler coordinates")
    return ForecastCorruption(
        policy=tcfg.stage3_forecast_corruption,
        coords=coords,
        scada_wind_std=float(getattr(handler, "scada_w_std")),
        seed=int(corruption_seed),
        num_turbines=int(bundle.num_turbines),
        seq_len=int(seq_len),
    )


def main() -> None:
    # Ensure progress logs are flushed line-by-line on all platforms/shells.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    cfg = load_config(Path(args.config))

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    train_cfg_raw = cfg["train"]
    track_cfg = cfg["tracking"]
    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("revin_missing_policy", "legacy")
    # Round-3 switches: the recorded defaults are exactly the V1 behaviour.
    model_cfg.setdefault("graph_score_policy", "v1_dot_gate")
    model_cfg.setdefault("forecast_skip_policy", "none")
    if not args.dry_run:
        require_verified_forecast_input(data_cfg.get("revision_protocol"))

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

    device = get_device(require_cuda=not args.dry_run and bool(data_cfg.get("revision_protocol")))

    dms_path = _resolve_data_path(data_cfg.get("dms_path"), "data/raw/DMS.csv")
    turbines_path = _resolve_data_path(data_cfg.get("turbines_path"), "data/raw/turbines.csv")

    bundle = build_data_bundle(
        dms_path=dms_path,
        turbines_path=turbines_path,
        seq_len=int(data_cfg["seq_len"]),
        horizon=int(exp_cfg["horizon"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        train_ratio=float(data_cfg["split"][0]),
        val_ratio=float(data_cfg["split"][1]),
        capacity_mw=float(data_cfg.get("capacity_mw", 2.0)),
        max_turbines=args.max_turbines,
        revision_protocol=data_cfg.get("revision_protocol"),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "num_turbines": bundle.num_turbines,
                    "nwp_dim": bundle.nwp_dim,
                    "train_batches": len(bundle.train_loader),
                    "val_batches": len(bundle.val_loader),
                    "test_batches": len(bundle.test_loader),
                },
                ensure_ascii=False,
            )
        )
        return

    ledger_dir = ROOT / track_cfg.get("ledger_dir", "ledger")
    ensure_ledger_templates(ledger_dir)

    # Persist the effective legacy default in metrics.config as well as loss_history.config.
    train_cfg_raw.setdefault("stage3_decoder_policy", "frozen")
    train_cfg_raw.setdefault("stage3_forecast_corruption", "off")
    tcfg = TrainConfig(
        epochs_stage1=int(train_cfg_raw["epochs_stage1"]),
        epochs_stage2=int(train_cfg_raw["epochs_stage2"]),
        epochs_stage3=int(train_cfg_raw["epochs_stage3"]),
        lr_stage1=float(train_cfg_raw.get("lr_stage1", 1e-3)),
        lr_stage2=float(train_cfg_raw.get("lr_stage2", 1e-3)),
        lr_stage3=float(train_cfg_raw.get("lr_stage3", 3e-4)),
        stage3_mode=str(train_cfg_raw.get("stage3_mode", "staged")),
        stage3_decoder_policy=str(train_cfg_raw["stage3_decoder_policy"]),
        loss_name=str(train_cfg_raw.get("loss_name", "huber")),
        huber_delta=float(train_cfg_raw.get("huber_delta", 0.1)),
        quantile=float(train_cfg_raw.get("quantile", 0.5)),
        patience=int(train_cfg_raw.get("patience", 0)),
        reconstruction_weight=float(train_cfg_raw.get("reconstruction_weight", 0.0)),
        calibration_weight=float(train_cfg_raw.get("calibration_weight", 0.0)),
        auxiliary_decay=float(train_cfg_raw.get("auxiliary_decay", 0.1)),
        record_gradient_conflicts=bool(train_cfg_raw.get("record_gradient_conflicts", False)),
        checkpoint_metric=str(train_cfg_raw.get("checkpoint_metric", "loss")),
        stage3_forecast_corruption=str(train_cfg_raw["stage3_forecast_corruption"]),
        max_batches_per_epoch=train_cfg_raw.get("max_batches_per_epoch"),
        max_test_batches=train_cfg_raw.get("max_test_batches"),
    )
    if args.max_batches_per_epoch is not None:
        tcfg.max_batches_per_epoch = args.max_batches_per_epoch
    if args.max_test_batches is not None:
        tcfg.max_test_batches = args.max_test_batches

    for seed in seeds:
        set_seed(int(seed))

        run_id = build_run_id(
            exp_type=exp_cfg["exp_type"],
            model_or_variant=exp_cfg.get("variant", exp_cfg["model"]),
            horizon=int(exp_cfg["horizon"]),
            seed=int(seed),
        )

        run_artifact_dir = ROOT / track_cfg.get("artifact_root", "artifacts") / run_id
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        start_ts = datetime.now().isoformat(timespec="seconds")
        print(
            f"[train] start run seed={seed} horizon={exp_cfg['horizon']} "
            f"stage_epochs=({tcfg.epochs_stage1},{tcfg.epochs_stage2},{tcfg.epochs_stage3}) "
            f"stage3_mode={tcfg.stage3_mode} "
            f"max_batches={tcfg.max_batches_per_epoch} max_test_batches={tcfg.max_test_batches}",
            flush=True,
        )

        try:
            # A fresh F2 stream per run keeps the corruption reproducible per seed.
            tcfg.stage3_corruption = _forecast_corruption(
                tcfg, bundle, seed, int(train_cfg_raw.get("corruption_seed", seed)),
                int(data_cfg["seq_len"]))
            model = build_main_model(bundle, data_cfg["seq_len"], model_cfg, device)
            tcfg.checkpoint_dir = str(run_artifact_dir)
            save_preprocessing(run_artifact_dir, bundle, revin_missing_policy=model_cfg["revin_missing_policy"])

            model, loss_history = train_multistep(model, bundle.train_loader, bundle.val_loader, device, tcfg)
            save_training_timing(run_artifact_dir, model, cfg, seed, exp_cfg["horizon"], device)
            validation_metrics, _, _ = evaluate_multistep(
                model, bundle.val_loader, device, bundle.capacity_mw,
                max_test_batches=tcfg.max_test_batches,
            )
            evaluation_split = "val" if exp_cfg.get("selection_scope") == "validation" else "test"
            evaluation_loader = bundle.val_loader if evaluation_split == "val" else bundle.test_loader
            metrics, y_true_mw, y_pred_mw = evaluate_multistep(
                model,
                evaluation_loader,
                device,
                bundle.capacity_mw,
                max_test_batches=tcfg.max_test_batches,
            )

            # local import to keep top imports minimal
            import numpy as np
            import torch

            save_run_predictions(run_artifact_dir, bundle, y_true_mw, y_pred_mw, seed, exp_cfg["horizon"], split=evaluation_split)
            if data_cfg.get("revision_protocol") is None:
                torch.save(model.state_dict(), run_artifact_dir / "model.pt")
            (run_artifact_dir / "loss_history.json").write_text(
                json.dumps(loss_history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Keep one test batch for reproducible latency profiling.
            test_batch = next(iter(bundle.test_loader))
            x_h_p, x_h_w, x_h_n, x_nwp_t, _ = test_batch
            np.savez_compressed(
                run_artifact_dir / "latency_sample.npz",
                x_hist_p=x_h_p.cpu().numpy(),
                x_hist_w=x_h_w.cpu().numpy(),
                x_hist_nwp=x_h_n.cpu().numpy(),
                x_nwp_target=x_nwp_t.cpu().numpy(),
            )

            metrics_payload = {
                "run_id": run_id,
                "status": "completed",
                "seed": int(seed),
                "horizon": int(exp_cfg["horizon"]),
                "metrics": metrics,
                "validation_metrics": validation_metrics,
                "evaluation_split": evaluation_split,
                "config": cfg,
                "device": str(device),
            }
            write_json_atomic(run_artifact_dir / "metrics.json", metrics_payload)

            end_ts = datetime.now().isoformat(timespec="seconds")

            common = base_row(
                run_id=run_id,
                exp_type=exp_cfg["exp_type"],
                model=exp_cfg["model"],
                variant=exp_cfg.get("variant", "NA"),
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

            append_row(
                ledger_dir / LEDGER_SPECS["run_registry"].file_name,
                LEDGER_SPECS["run_registry"].fields,
                common,
            )

            main_row = dict(common)
            arch_info = get_arch_info(exp_cfg["model"])
            # Keep key hyper-parameters visible in ledger for exact experiment replay.
            arch_info["key_hparams"] = (
                f"d_model={model_cfg.get('d_model', 128)},"
                f"mask_ratio={model_cfg.get('mask_ratio', 0.4)},"
                f"num_control_points={model_cfg.get('num_control_points', 15)},"
                f"use_graph_phys_bias={model_cfg.get('use_graph_phys_bias', True)},"
                f"stage3_mode={tcfg.stage3_mode}"
            )
            main_row.update(arch_info)
            main_row.update(metrics)
            append_row(
                ledger_dir / LEDGER_SPECS["main"].file_name,
                LEDGER_SPECS["main"].fields,
                main_row,
            )

            print(json.dumps({"run_id": run_id, "metrics": metrics}, ensure_ascii=False))
        except Exception as exc:
            end_ts = datetime.now().isoformat(timespec="seconds")
            fail_reason = str(exc)[:300]
            tb = traceback.format_exc()
            (run_artifact_dir / "error.log").write_text(tb, encoding="utf-8")

            fail_row = base_row(
                run_id=run_id,
                exp_type=exp_cfg["exp_type"],
                model=exp_cfg["model"],
                variant=exp_cfg.get("variant", "NA"),
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
            append_row(
                ledger_dir / LEDGER_SPECS["run_registry"].file_name,
                LEDGER_SPECS["run_registry"].fields,
                fail_row,
            )
            print(json.dumps({"run_id": run_id, "status": "failed", "reason": fail_reason}, ensure_ascii=False))
            if args.fail_fast:
                raise


if __name__ == "__main__":
    main()
