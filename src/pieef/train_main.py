from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .core import calc_metrics
from .training_timing import TrainingTimer


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
    log_every_batches: int = 50
    stage3_mode: str = "staged"
    stage3_decoder_policy: str = "frozen"
    loss_name: str = "huber"
    huber_delta: float = 0.1
    quantile: float = 0.5
    patience: int = 0
    min_delta: float = 0.0
    checkpoint_dir: str | None = None
    reconstruction_weight: float = 0.0
    calibration_weight: float = 0.0
    auxiliary_decay: float = 0.1
    record_gradient_conflicts: bool = False
    checkpoint_metric: str = "loss"
    # F2: corrupt the historical inputs on the Stage-3 forecast-loss path only.
    stage3_forecast_corruption: str = "off"
    # Injected by the training entry point; holds a pieef.forecast_corruption sampler.
    stage3_corruption: object | None = None

    def __post_init__(self) -> None:
        if self.stage3_mode not in {"staged", "joint"}:
            raise ValueError("stage3_mode must be 'staged' or 'joint'.")
        if self.stage3_decoder_policy not in ("frozen", "auxiliary"):
            raise ValueError("stage3_decoder_policy must be 'frozen' or 'auxiliary'.")
        if self.loss_name not in {"huber", "mse", "quantile"}:
            raise ValueError("loss_name must be huber, mse or quantile.")
        if not 0 < self.quantile < 1 or self.huber_delta <= 0:
            raise ValueError("Invalid loss settings.")
        if self.checkpoint_metric not in {"loss", "nrmse"}:
            raise ValueError("checkpoint_metric must be loss or nrmse.")
        if self.stage3_forecast_corruption not in {"off", "f2_block"}:
            raise ValueError("stage3_forecast_corruption must be 'off' or 'f2_block'.")


def _forecast_loss(pred, target, cfg):
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("No finite measured targets in this batch.")
    if not bool(torch.isfinite(pred[valid]).all()):
        raise ValueError("Non-finite model prediction.")
    pred, target = pred[valid], target[valid]
    if cfg.loss_name == "mse":
        return F.mse_loss(pred, target)
    if cfg.loss_name == "quantile":
        error = target - pred
        return torch.maximum(cfg.quantile * error, (cfg.quantile - 1) * error).mean()
    return F.huber_loss(pred, target, delta=cfg.huber_delta)


def _save_weights(model, directory, filename):
    if directory is not None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / filename)


class _ScalarBuffer:
    """Batch observation copies without changing scalar dtype or Python sum order."""
    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def append(self, *values):
        self.rows.append(tuple(value.detach().reshape(()) for value in values))

    def drain(self):
        if not self.rows:
            return []
        # Separate columns retain integer counts and original floating-point bits.
        columns = [torch.stack(column).cpu().tolist() for column in zip(*self.rows)]
        self.rows.clear()
        return list(zip(*columns))


def _sum_observed_loss(buffer, total):
    for (value,) in buffer.drain():
        total += float(value)
    return total


def _sum_observed_validation(buffer, loss_sum, observations, squared_error):
    for value, count, squared in buffer.drain():
        count = int(count)
        loss_sum += float(value) * count
        observations += count
        squared_error += float(squared)
    return loss_sum, observations, squared_error


def _gradient_cosine(left, right, parameters):
    if not left.requires_grad or not right.requires_grad:
        return None
    a = torch.autograd.grad(left, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(right, parameters, retain_graph=True, allow_unused=True)
    pairs = [(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    aa, bb = torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs])
    denominator = aa.norm() * bb.norm()
    return None if float(denominator) == 0 else float((aa @ bb / denominator).detach())


def _toggle_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _forecast_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "forecast_head"):
        return model.forecast_head
    if hasattr(model, "head"):
        return model.head
    raise AttributeError("Model has neither forecast_head nor head.")


def _configure_stage3(
    model: torch.nn.Module,
    mode: str,
    decoder_policy: str = "frozen",
    reconstruction_weight: float = 0.0,
) -> None:
    """Select the historical staged protocol or differentiable joint fine-tuning."""
    if mode not in {"staged", "joint"}:
        raise ValueError("stage3_mode must be 'staged' or 'joint'.")
    if decoder_policy not in ("frozen", "auxiliary"):
        raise ValueError("stage3_decoder_policy must be 'frozen' or 'auxiliary'.")
    # Gradients left by earlier stages are not evidence of Stage-3 connectivity.
    model.zero_grad(set_to_none=True)
    _toggle_grad(model, mode == "joint")
    if mode == "joint" and hasattr(model, "mae_head"):
        # Preserve the historical frozen decoder unless reconstruction training is explicit.
        _toggle_grad(model.mae_head, decoder_policy == "auxiliary" and reconstruction_weight > 0)
    _toggle_grad(model.graph, True)
    _toggle_grad(_forecast_module(model), True)
    # F6 lives on the model (not in the head), so it needs the same explicit switch.
    if getattr(model, "forecast_skip", None) is not None:
        _toggle_grad(model.forecast_skip, True)
    if hasattr(model, "revin"):
        model.revin.affine_weight.requires_grad = True
        model.revin.affine_bias.requires_grad = True


def _stage3_features(model, x_h_p, x_h_w, x_h_n, mode: str,
                     force_power_mask=None, force_wind_mask=None):
    """Encode unmasked history, retaining autograd only for joint fine-tuning.

    F2 corruption reaches the encoder through the same explicit mask channels that
    `faults.apply_fault` uses at evaluation time.
    """
    was_training = model.training
    grad_enabled = torch.is_grad_enabled() and mode == "joint"
    try:
        # eval() suppresses the pretraining mask/dropout without disabling autograd.
        model.eval()
        with torch.set_grad_enabled(grad_enabled):
            _, feat, _, _ = model.forward_stage1_reconstruct(
                x_h_p, x_h_w, x_h_n,
                force_mask=force_power_mask, force_wind_mask=force_wind_mask)
    finally:
        model.train(was_training)
    return feat


def train_multistep(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
    corruption=None,
) -> tuple[torch.nn.Module, dict]:
    # The sampler may also be attached to the config by the training entry point.
    corruption = cfg.stage3_corruption if corruption is None else corruption
    timing = TrainingTimer(device)
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

    def _log_batch(stage: str, epoch: int, max_epoch: int, batch_idx: int, batch_total: int, loss_sum: float, t0: float, values) -> float:
        due = batch_idx > 0 and cfg.log_every_batches > 0 and (
            batch_idx % cfg.log_every_batches == 0 or batch_idx == batch_total)
        if due or len(values) >= 50:
            loss_sum = _sum_observed_loss(values, loss_sum)
        if not due:
            return loss_sum
        elapsed = max(time.perf_counter() - t0, 1e-6)
        speed = batch_idx / elapsed
        eta = (batch_total - batch_idx) / max(speed, 1e-6)
        avg_loss = loss_sum / batch_idx
        print(
            f"[{stage}] epoch={epoch}/{max_epoch} batch={batch_idx}/{batch_total} "
            f"loss={avg_loss:.6f} speed={speed:.2f}it/s eta={eta:.1f}s",
            flush=True,
        )
        return loss_sum

    print(
        f"[train-main] begin stage1={cfg.epochs_stage1}, stage2={cfg.epochs_stage2}, stage3={cfg.epochs_stage3}, "
        f"stage3_mode={cfg.stage3_mode}, stage3_decoder_policy={cfg.stage3_decoder_policy}, "
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
            "stage3_mode": cfg.stage3_mode,
            "stage3_decoder_policy": cfg.stage3_decoder_policy,
            "loss_name": cfg.loss_name,
            "reconstruction_weight": cfg.reconstruction_weight,
            "calibration_weight": cfg.calibration_weight,
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
        if getattr(model, "forecast_skip", None) is not None:
            _toggle_grad(model.forecast_skip, False)

        opt_s1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1)

        for epoch in range(cfg.epochs_stage1):
            model.train()
            batch_count = 0
            train_loss_sum = 0.0
            loss_values = _ScalarBuffer()
            t_epoch = time.perf_counter()
            timing.start("train", "stage1", epoch + 1)
            for x_h_p, x_h_w, x_h_n, _, _ in train_loader:
                x_h_p, x_h_w, x_h_n = (value.to(device, non_blocking=True) for value in (x_h_p, x_h_w, x_h_n))
                opt_s1.zero_grad()
                loss, _, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
                loss.backward()
                opt_s1.step()
                batch_count += 1
                loss_values.append(loss)
                train_loss_sum = _log_batch("stage1", epoch + 1, cfg.epochs_stage1, batch_count, _effective_train_batches(), train_loss_sum, t_epoch, loss_values)
                if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                    break
            train_loss_sum = _sum_observed_loss(loss_values, train_loss_sum)
            timing.stop(batch_count)
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

    _save_weights(model, cfg.checkpoint_dir, "weights_stage1.pt")
    if cfg.epochs_stage2 > 0 and any(True for _ in model.dcn.parameters()):
        print("[stage2] TD-DCN calibration start", flush=True)
        _toggle_grad(model, False)
        _toggle_grad(model.dcn, True)
        opt_s2 = optim.Adam(model.dcn.parameters(), lr=cfg.lr_stage2)

        for epoch in range(cfg.epochs_stage2):
            model.train()
            batch_count = 0
            train_loss_sum = 0.0
            loss_values = _ScalarBuffer()
            t_epoch = time.perf_counter()
            timing.start("train", "stage2", epoch + 1)
            for _, _, _, x_n_target, y in train_loader:
                x_n_target, y = (value.to(device, non_blocking=True) for value in (x_n_target, y))
                v_cal, p_phys, delta_v = model.forward_stage2_training(x_n_target)
                _ = v_cal

                valid_mask = torch.isfinite(y) & (y > 0.01)
                if valid_mask.sum() > 0:
                    loss_fit = F.mse_loss(p_phys.squeeze(-1)[valid_mask], y[valid_mask])
                else:
                    loss_fit = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss_fit + 0.05 * torch.mean(delta_v**2)
                opt_s2.zero_grad()
                loss.backward()
                opt_s2.step()

                batch_count += 1
                loss_values.append(loss)
                train_loss_sum = _log_batch("stage2", epoch + 1, cfg.epochs_stage2, batch_count, _effective_train_batches(), train_loss_sum, t_epoch, loss_values)
                if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                    break
            train_loss_sum = _sum_observed_loss(loss_values, train_loss_sum)
            timing.stop(batch_count)
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

    _save_weights(model, cfg.checkpoint_dir, "weights_stage2.pt")
    print(f"[stage3] graph forecasting start mode={cfg.stage3_mode}", flush=True)
    _configure_stage3(model, cfg.stage3_mode, cfg.stage3_decoder_policy, cfg.reconstruction_weight)

    opt_s3 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage3)
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(cfg.epochs_stage3):
        model.train()
        batch_count = 0
        train_loss_sum = 0.0
        loss_values = _ScalarBuffer()
        t_epoch = time.perf_counter()
        timing.start("train", "stage3", epoch + 1)
        for x_h_p, x_h_w, x_h_n, x_n_target, y in train_loader:
            # F2: the forecast head sees a corrupted history; the auxiliary reconstruction
            # path keeps the clean history, exactly as in V1.
            power_mask = wind_mask = None
            clean_history = None
            if cfg.stage3_forecast_corruption == "f2_block":
                if corruption is None:
                    raise ValueError(
                        "stage3_forecast_corruption=f2_block requires an explicit F2 sampler")
                clean_history = (x_h_p, x_h_w, x_h_n)
                (x_h_p, x_h_w, x_h_n, _, _), power_mask, wind_mask = corruption.corrupt(
                    (x_h_p, x_h_w, x_h_n, x_n_target, y))
            x_h_p, x_h_w, x_h_n = (value.to(device, non_blocking=True) for value in (x_h_p, x_h_w, x_h_n))
            x_n_target, y = (value.to(device, non_blocking=True) for value in (x_n_target, y))
            if power_mask is not None:
                power_mask, wind_mask = (mask.to(device, non_blocking=True) for mask in (power_mask, wind_mask))
            # The clean-history branch keeps the uncorrupted tensors for the auxiliary reconstruction
            # term. F2 corrupts only a fraction of batches, so power_mask is None on the rest; the
            # transfer must therefore happen whenever clean_history exists, not only when a mask does.
            if clean_history is not None and cfg.stage3_mode == "joint" and (cfg.reconstruction_weight or cfg.calibration_weight):
                clean_history = tuple(value.to(device, non_blocking=True) for value in clean_history)

            opt_s3.zero_grad()
            feat = _stage3_features(model, x_h_p, x_h_w, x_h_n, cfg.stage3_mode,
                                    force_power_mask=power_mask, force_wind_mask=wind_mask)

            v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
            pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target,
                                                 hist_power=x_h_p, hist_power_mask=power_mask)
            forecast_loss = _forecast_loss(pred, y, cfg)
            loss = forecast_loss
            if cfg.stage3_mode == "joint" and (cfg.reconstruction_weight or cfg.calibration_weight):
                history = (x_h_p, x_h_w, x_h_n) if clean_history is None else clean_history
                reconstruction, _, _, _ = model.forward_stage1_reconstruct(*history)
                _, auxiliary_power, delta_v = model.forward_stage2_training(x_n_target)
                valid_calibration = torch.isfinite(y) & (y > 0.01)
                calibration = delta_v.square().mean() * 0.05
                if valid_calibration.any():
                    calibration = calibration + F.mse_loss(auxiliary_power.squeeze(-1)[valid_calibration], y[valid_calibration])
                weight = math.exp(-cfg.auxiliary_decay * epoch)
                loss = loss + weight * (cfg.reconstruction_weight * reconstruction + cfg.calibration_weight * calibration)
                if cfg.record_gradient_conflicts and batch_count == 0:
                    parameters = [p for p in model.parameters() if p.requires_grad]
                    loss_history.setdefault("gradient_conflicts", []).append({
                        "epoch": epoch + 1,
                        "forecast_reconstruction": _gradient_cosine(forecast_loss, reconstruction, parameters),
                        "forecast_calibration": _gradient_cosine(forecast_loss, calibration, parameters),
                        "reconstruction_calibration": _gradient_cosine(reconstruction, calibration, parameters),
                    })
            loss.backward()
            opt_s3.step()

            batch_count += 1
            loss_values.append(loss)
            train_loss_sum = _log_batch("stage3", epoch + 1, cfg.epochs_stage3, batch_count, _effective_train_batches(), train_loss_sum, t_epoch, loss_values)
            if cfg.max_batches_per_epoch and batch_count >= cfg.max_batches_per_epoch:
                break

        train_loss_sum = _sum_observed_loss(loss_values, train_loss_sum)
        timing.stop(batch_count)
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_observations = 0
        val_squared_error = 0.0
        validation_values = _ScalarBuffer()
        timing.start("validation", "stage3", epoch + 1)
        with torch.no_grad():
            for i, (x_h_p, x_h_w, x_h_n, x_n_target, y) in enumerate(val_loader):
                if cfg.max_batches_per_epoch and i >= cfg.max_batches_per_epoch:
                    break
                x_h_p, x_h_w, x_h_n = (value.to(device, non_blocking=True) for value in (x_h_p, x_h_w, x_h_n))
                x_n_target, y = (value.to(device, non_blocking=True) for value in (x_n_target, y))
                _, feat, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
                v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
                pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target,
                                                     hist_power=x_h_p)
                val_loss = _forecast_loss(pred, y, cfg)
                observations = torch.isfinite(y).sum()
                valid = torch.isfinite(y)
                squared_error = ((pred.clamp_min(0)[valid] - y[valid]) ** 2).sum()
                validation_values.append(val_loss, observations, squared_error)
                if len(validation_values) >= 50:
                    val_loss_sum, val_observations, val_squared_error = _sum_observed_validation(
                        validation_values, val_loss_sum, val_observations, val_squared_error)
                val_batches += 1
        val_loss_sum, val_observations, val_squared_error = _sum_observed_validation(
            validation_values, val_loss_sum, val_observations, val_squared_error)
        timing.stop(val_batches)
        epoch_elapsed = time.perf_counter() - t_epoch
        train_avg = train_loss_sum / max(batch_count, 1)
        if not val_observations:
            raise ValueError("Validation set contains no measured targets.")
        val_avg = val_loss_sum / val_observations
        val_nrmse = math.sqrt(val_squared_error / val_observations)
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
                "val_nrmse": val_nrmse,
                "elapsed_s": float(epoch_elapsed),
                "num_train_batches": int(batch_count),
                "num_val_batches": int(val_batches),
            }
        )
        checkpoint_value = val_nrmse if cfg.checkpoint_metric == "nrmse" else val_avg
        if checkpoint_value < best_loss - cfg.min_delta:
            best_loss, stale = checkpoint_value, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_weights(model, cfg.checkpoint_dir, "weights_best.pt")
            loss_history["best_epoch"] = epoch + 1
        else:
            stale += 1
        if cfg.patience and stale >= cfg.patience:
            break

    _save_weights(model, cfg.checkpoint_dir, "weights_last.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_timing = timing.result()
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
            x_h_p, x_h_w, x_h_n = (value.to(device, non_blocking=True) for value in (x_h_p, x_h_w, x_h_n))
            x_n_target = x_n_target.to(device, non_blocking=True)

            _, feat, _, _ = model.forward_stage1_reconstruct(x_h_p, x_h_w, x_h_n)
            v_cal, p_phys = model.forward_stage2_calibration(x_h_w, x_n_target)
            pred = model.forward_stage3_forecast(feat, v_cal, p_phys, x_n_target,
                                                 hist_power=x_h_p)

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
