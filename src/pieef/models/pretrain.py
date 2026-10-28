from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockMaskGenerator:
    def __init__(self, mask_ratio: float = 0.5, min_block_len: int = 4, max_block_len: int = 24) -> None:
        self.mask_ratio = mask_ratio
        self.min_len = min_block_len
        self.max_len = max_block_len

    def generate_mask(self, x_power: torch.Tensor) -> torch.Tensor:
        bsz, n_turbines, seq_len, _ = x_power.shape
        if self.mask_ratio <= 0:
            return torch.zeros((bsz, n_turbines, seq_len, 1), device=x_power.device, dtype=torch.float32)
        if self.mask_ratio >= 1:
            return torch.ones((bsz, n_turbines, seq_len, 1), device=x_power.device, dtype=torch.float32)

        min_len = max(1, min(self.min_len, seq_len))
        max_len = max(min_len, min(self.max_len, seq_len))

        target_masked = max(1, int(round(seq_len * float(self.mask_ratio))))
        avg_len = 0.5 * (min_len + max_len)
        num_blocks = max(1, int(math.ceil(target_masked / max(avg_len, 1.0))))

        starts = torch.randint(0, max(1, seq_len - min_len + 1), (bsz, n_turbines, num_blocks), device=x_power.device)
        lengths = torch.randint(min_len, max_len + 1, (bsz, n_turbines, num_blocks), device=x_power.device)
        ends = torch.clamp(starts + lengths, max=seq_len)

        arange = torch.arange(seq_len, device=x_power.device).view(1, 1, 1, seq_len)
        mask_blocks = (arange >= starts.unsqueeze(-1)) & (arange < ends.unsqueeze(-1))
        mask_s = mask_blocks.any(dim=2)

        # Calibrate each sample to the target masked count so different mask_ratio
        # values are reliably distinguishable in sensitivity experiments.
        # Read all row counts together, rather than synchronizing CUDA once per
        # sample/turbine. Fixed-size index banks also avoid CUDA nonzero's
        # dynamic-output synchronization. Valid indices remain in the exact
        # ascending order returned by the former per-row torch.where calls.
        counts = mask_s.sum(dim=-1).cpu().tolist()
        positions = torch.arange(seq_len, device=x_power.device).view(1, 1, seq_len)
        masked_indices = torch.where(mask_s, positions, seq_len).sort(dim=-1).values
        unmasked_indices = torch.where(~mask_s, positions, seq_len).sort(dim=-1).values
        selected_permutations, bank_ids, row_ids, replacement_values = [], [], [], []
        for b in range(bsz):
            for t in range(n_turbines):
                cur = counts[b][t]
                if cur > target_masked:
                    size, change, bank, value = cur, cur - target_masked, 0, False
                elif cur < target_masked:
                    size, change, bank, value = seq_len - cur, target_masked - cur, 1, True
                else:
                    continue
                # Preserve every original RNG call, including its order, shape
                # and device. Only deterministic indexing/scattering is batched.
                perm = torch.randperm(size, device=x_power.device)
                selected_permutations.append(perm[:change])
                bank_ids.extend([bank] * change)
                row_ids.extend([b * n_turbines + t] * change)
                replacement_values.extend([value] * change)
        if selected_permutations:
            banks = torch.stack((masked_indices.reshape(-1, seq_len),
                                 unmasked_indices.reshape(-1, seq_len)))
            routing = torch.tensor([bank_ids, row_ids], device=x_power.device, dtype=torch.long)
            selected = banks[routing[0], routing[1], torch.cat(selected_permutations)]
            indices = routing[1] * seq_len + selected
            mask_s.reshape(-1)[indices] = torch.tensor(
                replacement_values, device=x_power.device, dtype=torch.bool)
        return mask_s.unsqueeze(-1).float()


class PhysicsConsistencyLoss(nn.Module):
    def __init__(
        self, lambda_bound: float = 0.1, lambda_deriv: float = 0.5,
        use_bound: bool = True, use_derivative: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_deriv = lambda_deriv
        self.use_bound = bool(use_bound)
        self.use_derivative = bool(use_derivative)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        p_phys_prior: torch.Tensor,
        physical_pred: torch.Tensor | None = None,
        prior_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mse = (pred - target) ** 2
        loss_recon = (loss_mse * mask).sum() / (mask.sum() + 1e-6)

        p_per_unit = pred if physical_pred is None else physical_pred
        loss_bound = pred.new_zeros(())
        if self.use_bound:
            loss_bound = (F.relu(-p_per_unit) + F.relu(p_per_unit - 1.05)).mean()

        loss_deriv = pred.new_zeros(())
        if self.use_derivative:
            pred_diff = p_per_unit[:, :, 1:] - p_per_unit[:, :, :-1]
            phys_diff = p_phys_prior[:, :, 1:] - p_phys_prior[:, :, :-1]
            loss_deriv_raw = (pred_diff - phys_diff) ** 2
            mask_slice = mask[:, :, 1:]
            if prior_valid_mask is not None:
                mask_slice = mask_slice * prior_valid_mask[:, :, 1:] * prior_valid_mask[:, :, :-1]
            loss_deriv = (loss_deriv_raw * mask_slice).sum() / (mask_slice.sum() + 1e-6)

        total_loss = loss_recon + self.lambda_bound * loss_bound + self.lambda_deriv * loss_deriv
        return total_loss, loss_recon, loss_deriv


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True,
                 missing_policy: str = "legacy", training_mean=None, training_stdev=None) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if missing_policy not in {"legacy", "train_stats_20"}:
            raise ValueError("revin_missing_policy must be legacy or train_stats_20")
        self.missing_policy = missing_policy
        mean, scale = None, None
        if missing_policy == "train_stats_20":
            if training_mean is None or training_stdev is None:
                raise ValueError("F1 RevIN requires actual per-turbine training mean and stdev")
            mean = torch.as_tensor(training_mean, dtype=torch.float32)
            scale = torch.as_tensor(training_stdev, dtype=torch.float32)
            if (mean.shape != (num_features,) or scale.shape != (num_features,)
                    or not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()) or not bool((scale > 0).all())):
                raise ValueError("F1 RevIN needs finite ordered training moments with positive stdev")
            mean, scale = mean.reshape(1, 1, -1), scale.reshape(1, 1, -1)
        elif training_mean is not None or training_stdev is not None:
            raise ValueError("Training moments must not be silently supplied to legacy RevIN")
        self.register_buffer("training_mean", mean)
        self.register_buffer("training_stdev", scale)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if mask is None:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        else:
            if mask.shape != x.shape:
                raise ValueError("RevIN missing mask must have the same shape as the input")
            weights = (1.0 - mask.to(x.dtype)).clamp(0, 1)
            count = weights.sum(dim=dim2reduce, keepdim=True)
            safe_x = torch.where(weights > 0, x, torch.zeros_like(x))
            mean = safe_x.sum(dim=dim2reduce, keepdim=True) / count.clamp_min(1)
            variance = ((safe_x - mean).square() * weights).sum(dim=dim2reduce, keepdim=True) / count.clamp_min(1)
            self.mean = torch.where(count > 0, mean, torch.zeros_like(mean)).detach()
            # An entirely missing channel uses a deterministic scale; no held-out values.
            variance = torch.where(count > 0, variance, torch.ones_like(variance))
        self.stdev = torch.sqrt(variance + self.eps).detach()
        if self.missing_policy == "train_stats_20" and mask is not None:
            if x.ndim != 3 or x.shape[-1] != self.num_features:
                raise ValueError("F1 RevIN expects [batch, time, ordered turbines]")
            # Device robustness: the F1 moments are buffers, but a corrupted-input path can
            # reach here with them still on the host. Align before combining.
            if self.training_mean is not None and self.training_mean.device != self.mean.device:
                self.training_mean = self.training_mean.to(self.mean.device)
                self.training_stdev = self.training_stdev.to(self.mean.device)
            use_training = (count / x.shape[1] <= 0.2).to(self.mean.device)
            self.mean = torch.where(use_training, self.training_mean, self.mean).detach()
            self.stdev = torch.where(use_training, self.training_stdev, self.stdev).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x

    def apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with the statistics of the most recent `norm` call.

        F6 needs the power history in the same space as the forecast-head output.
        This reuses the recorded per-window statistics instead of refitting them,
        so the forecast target space stays exactly the one used for denormalization.
        """
        if getattr(self, "mean", None) is None or getattr(self, "stdev", None) is None:
            raise ValueError("RevIN statistics are unavailable; run a normalization pass first")
        return self._normalize(x)
