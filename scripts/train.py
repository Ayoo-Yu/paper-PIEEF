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
from pieef.ledger import LEDGER_SPECS, append_row, base_row, build_run_id, ensure_ledger_templates
from pieef.model_main import E2EPhysNet
from pieef.model_registry import get_arch_info
from pieef.train_main import TrainConfig, evaluate_multistep, train_multistep


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
    model_cfg = cfg.get("model", {})

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

    device = get_device()

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

    tcfg = TrainConfig(
        epochs_stage1=int(train_cfg_raw["epochs_stage1"]),
        epochs_stage2=int(train_cfg_raw["epochs_stage2"]),
        epochs_stage3=int(train_cfg_raw["epochs_stage3"]),
        lr_stage1=float(train_cfg_raw.get("lr_stage1", 1e-3)),
        lr_stage2=float(train_cfg_raw.get("lr_stage2", 1e-3)),
        lr_stage3=float(train_cfg_raw.get("lr_stage3", 3e-4)),
        max_batches_per_epoch=train_cfg_raw.get("max_batches_per_epoch"),
        max_test_batches=train_cfg_raw.get("max_test_batches"),
        log_every_batches=int(train_cfg_raw.get("log_every_batches", 10)),
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
            f"max_batches={tcfg.max_batches_per_epoch} max_test_batches={tcfg.max_test_batches}",
            flush=True,
        )

        try:
            model = E2EPhysNet(
                num_turbines=bundle.num_turbines,
                nwp_dim=bundle.nwp_dim,
                seq_len=int(data_cfg["seq_len"]),
                dist_matrix=bundle.dist_tensor.to(device),
                rel_coords=bundle.rel_coords_tensor.to(device),
                scada_wind_mean=float(bundle.handler.scada_w_mean),
                scada_wind_std=float(bundle.handler.scada_w_std),
                d_model=int(model_cfg.get("d_model", 128)),
                mask_ratio=float(model_cfg.get("mask_ratio", 0.4)),
                num_control_points=int(model_cfg.get("num_control_points", 15)),
                use_graph_phys_bias=bool(model_cfg.get("use_graph_phys_bias", True)),
            ).to(device)

            model, loss_history = train_multistep(model, bundle.train_loader, bundle.val_loader, device, tcfg)
            metrics, y_true_mw, y_pred_mw = evaluate_multistep(
                model,
                bundle.test_loader,
                device,
                bundle.capacity_mw,
                max_test_batches=tcfg.max_test_batches,
            )

            npz_path = run_artifact_dir / "predictions.npz"
            # local import to keep top imports minimal
            import numpy as np
            import torch

            np.savez_compressed(npz_path, y_true_mw=y_true_mw, y_pred_mw=y_pred_mw)
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
                "seed": int(seed),
                "horizon": int(exp_cfg["horizon"]),
                "metrics": metrics,
                "config": cfg,
                "device": str(device),
            }
            (run_artifact_dir / "metrics.json").write_text(
                json.dumps(metrics_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

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
                f"use_graph_phys_bias={model_cfg.get('use_graph_phys_bias', True)}"
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
