from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .core import calc_metrics


@dataclass
class TrainConfig:
    epochs_stage1: int = 5
    epochs_stage2: int = 5
    epochs_stage3: int = 20
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-3
    lr_stage3: float = 3e-4
    max_batches_per_epoch: int | None = None
    max_test_batches: int | None = None
    log_every_batches: int = 10


def _toggle_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _forecast_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "forecast_head"):
        return model.forecast_head
    if hasattr(model, "head"):
        return model.head
    raise AttributeError("Model has neither forecast_head nor head.")


def train_multistep(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[torch.nn.Module, dict]:
    train_batches_total = len(train_loader)
    val_batches_total = len(val_loader)

    def _effective_train_batches() -> int:
        if cfg.max_batches_per_epoch is None:
            return train_batches_total
        return min(train_batches_total, int(cfg.max_batches_per_epoch))

    def _effective_val_batches() -> int:
        if cfg.max_batches_per_epoch is None:
            return val_batches_total
        return min(val_batches_total, int(cfg.max_batches_per_epoch))

    def _log_batch(stage: str, epoch: int, max_epoch: int, batch_idx: int, batch_total: int, loss_sum: float, t0: float) -> None:
        if batch_idx <= 0:
            return
        if cfg.log_every_batches <= 0:
            return
        if batch_idx != 1 and batch_idx % cfg.log_every_batches != 0 and batch_idx != batch_total:
            return
        elapsed = max(time.perf_counter() - t0, 1e-6)
        speed = batch_idx / elapsed
        eta = (batch_total - batch_idx) / max(speed, 1e-6)
        avg_loss = loss_sum / batch_idx
        print(
            f"[{stage}] epoch={epoch}/{max_epoch} batch={batch_idx}/{batch_total} "
            f"loss={avg_loss:.6f} speed={speed:.2f}it/s eta={eta:.1f}s",
            flush=True,
        )

    print(
        f"[train-main] begin stage1={cfg.epochs_stage1}, stage2={cfg.epochs_stage2}, stage3={cfg.epochs_stage3}, "
        f"train_batches={_effective_train_batches()}, val_batches={_effective_val_batches()}",
        flush=True,
    )
    loss_history: dict[str, object] = {
        "config": {
            "epochs_stage1": int(cfg.epochs_stage1),
            "epochs_stage2": int(cfg.epochs_stage2),
            "epochs_stage3": int(cfg.epochs_stage3),
            "lr_stage1": float(cfg.lr_stage1),
            "lr_stage2": float(cfg.lr_stage2),
            "lr_stage3": float(cfg.lr_stage3),
            "max_batches_per_epoch": None if cfg.max_batches_per_epoch is None else int(cfg.max_batches_per_epoch),
            "train_batches_effective": int(_effective_train_batches()),
            "val_batches_effective": int(_effective_val_batches()),
        },
        "stage1": [],
        "stage2": [],
        "stage3": [],
    }

    if cfg.epochs_stage1 > 0:
        print("[stage1] PI-MAE reconstruction start", flush=True)
        _toggle_grad(model, True)
        if hasattr(model, "revin"):
            model.revin.affine_weight.requires_grad = False
            model.revin.affine_bias.requires_grad = False
        _toggle_grad(model.dcn, False)
        _toggle_grad(model.graph, False)
        _toggle_grad(_forecast_module(model), False)

        opt_s1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1)

        for epoch in range(cfg.epochs_stage1):
            model.train()
            batch_count = 0
            train_loss_sum = 0.0
            t_epoch = time.perf_counter()
            for x_h_p, x_h_w, x_h_n, _, _ in train_loader:
                x_h_p, x_h_w, x_h_n = x_h_p.to(device), x_h_w.to(device), x_h_n.to(device)
                opt_s1.zero_grad()
                loss, _, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
                loss.backward()
                opt_s1.step()
                batch_count += 1
                train_loss_sum += float(loss.item())
                _log_batch("stage1", epoch + 1, cfg.epochs_stage1, batch_count, _effective_train_batches(), train_loss_sum, t_epoch)
                if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                    break
            epoch_elapsed = time.perf_counter() - t_epoch
            if batch_count > 0:
                print(
                    f"[stage1] epoch={epoch + 1}/{cfg.epochs_stage1} done "
                    f"train_loss={train_loss_sum / batch_count:.6f} elapsed={epoch_elapsed:.1f}s",
                    flush=True,
                )
                loss_history["stage1"].append(
                    {
                        "epoch": int(epoch + 1),
                        "train_loss": float(train_loss_sum / batch_count),
                        "elapsed_s": float(epoch_elapsed),
                        "num_batches": int(batch_count),
                    }
                )

    if cfg.epochs_stage2 > 0:
        print("[stage2] TD-DCN calibration start", flush=True)
        _toggle_grad(model, False)
        _toggle_grad(model.dcn, True)
        opt_s2 = optim.Adam(model.dcn.parameters(), lr=cfg.lr_stage2)

        for epoch in range(cfg.epochs_stage2):
            model.train()
            batch_count = 0
            train_loss_sum = 0.0
            t_epoch = time.perf_counter()
            for _, _, _, x_n_target, y in train_loader:
                x_n_target, y = x_n_target.to(device), y.to(device)
                v_cal, p_phys, delta_v = model.forward_stage2_training(x_n_target)
                _ = v_cal

                valid_mask = y > 0.01
                if valid_mask.sum() > 0:
                    loss_fit = F.mse_loss(p_phys.squeeze(-1)[valid_mask], y[valid_mask])
                else:
                    loss_fit = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss_fit + 0.05 * torch.mean(delta_v**2)
                opt_s2.zero_grad()
                loss.backward()
                opt_s2.step()

                batch_count += 1
                train_loss_sum += float(loss.item())
                _log_batch("stage2", epoch + 1, cfg.epochs_stage2, batch_count, _effective_train_batches(), train_loss_sum, t_epoch)
                if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                    break
            epoch_elapsed = time.perf_counter() - t_epoch
            if batch_count > 0:
                print(
                    f"[stage2] epoch={epoch + 1}/{cfg.epochs_stage2} done "
                    f"train_loss={train_loss_sum / batch_count:.6f} elapsed={epoch_elapsed:.1f}s",
                    flush=True,
                )
                loss_history["stage2"].append(
                    {
                        "epoch": int(epoch + 1),
                        "train_loss": float(train_loss_sum / batch_count),
                        "elapsed_s": float(epoch_elapsed),
                        "num_batches": int(batch_count),
                    }
                )

    print("[stage3] graph forecasting start", flush=True)
    _toggle_grad(model, False)
    _toggle_grad(model.graph, True)
    _toggle_grad(_forecast_module(model), True)
    if hasattr(model, "revin"):
        model.revin.affine_weight.requires_grad = True
        model.revin.affine_bias.requires_grad = True

    opt_s3 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage3)

    for epoch in range(cfg.epochs_stage3):
        model.train()
        batch_count = 0
        train_loss_sum = 0.0
        t_epoch = time.perf_counter()
        for x_h_p, x_h_w, x_h_n, x_n_target, y in train_loader:
            x_h_p, x_h_w, x_h_n = x_h_p.to(device), x_h_w.to(device), x_h_n.to(device)
            x_n_target, y = x_n_target.to(device), y.to(device)

            opt_s3.zero_grad()
            with torch.no_grad():
                model.eval()
                _, feat, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
                model.train()

            v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
            pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target)
            loss = F.huber_loss(pred, y, delta=0.1)
            loss.backward()
            opt_s3.step()

            batch_count += 1
            train_loss_sum += float(loss.item())
            _log_batch("stage3", epoch + 1, cfg.epochs_stage3, batch_count, _effective_train_batches(), train_loss_sum, t_epoch)
            if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                break

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, (x_h_p, x_h_w, x_h_n, x_n_target, y) in enumerate(val_loader):
                if cfg.max_batches_per_epoch and i >= cfg.max_batches_per_epoch:
                    break
                x_h_p, x_h_w, x_h_n = x_h_p.to(device), x_h_w.to(device), x_h_n.to(device)
                x_n_target, y = x_n_target.to(device), y.to(device)
                _, feat, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
                v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
                pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target)
                val_loss = F.huber_loss(pred, y, delta=0.1)
                val_loss_sum += float(val_loss.item())
                val_batches += 1
        epoch_elapsed = time.perf_counter() - t_epoch
        train_avg = train_loss_sum / max(batch_count, 1)
        val_avg = val_loss_sum / max(val_batches, 1)
        print(
            f"[stage3] epoch={epoch + 1}/{cfg.epochs_stage3} done "
            f"train_loss={train_avg:.6f} val_loss={val_avg:.6f} elapsed={epoch_elapsed:.1f}s",
            flush=True,
        )
        loss_history["stage3"].append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_avg),
                "val_loss": float(val_avg),
                "elapsed_s": float(epoch_elapsed),
                "num_train_batches": int(batch_count),
                "num_val_batches": int(val_batches),
            }
        )

    return model, loss_history


def evaluate_multistep(
    model,
    loader,
    device: torch.device,
    capacity_mw: float,
    max_test_batches: int | None = None,
):
    model.eval()
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for i, (x_h_p, x_h_w, x_h_n, x_n_target, y) in enumerate(loader):
            if max_test_batches and i >= max_test_batches:
                break
            x_h_p, x_h_w, x_h_n = x_h_p.to(device), x_h_w.to(device), x_h_n.to(device)
            x_n_target = x_n_target.to(device)

            _, feat, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
            v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
            pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target)

            preds_list.append(pred.cpu().numpy())
            trues_list.append(y.numpy())

    preds_mw = np.concatenate(preds_list, axis=0) * capacity_mw
    trues_mw = np.concatenate(trues_list, axis=0) * capacity_mw
    preds_mw = np.maximum(preds_mw, 0)

    nmae, nrmse, r2 = calc_metrics(trues_mw, preds_mw, capacity_mw)
    return {
        "nmae": float(nmae),
        "nrmse": float(nrmse),
        "r2": float(r2),
        "num_samples": int(trues_mw.shape[0]),
        "num_turbines": int(trues_mw.shape[1]),
    }, trues_mw, preds_mw
