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
        for b in range(bsz):
            for t in range(n_turbines):
                cur = int(mask_s[b, t].sum().item())
                if cur > target_masked:
                    idx = torch.where(mask_s[b, t])[0]
                    perm = torch.randperm(idx.numel(), device=x_power.device)
                    drop = idx[perm[: cur - target_masked]]
                    mask_s[b, t, drop] = False
                elif cur < target_masked:
                    idx = torch.where(~mask_s[b, t])[0]
                    perm = torch.randperm(idx.numel(), device=x_power.device)
                    add = idx[perm[: target_masked - cur]]
                    mask_s[b, t, add] = True
        return mask_s.unsqueeze(-1).float()


class PhysicsConsistencyLoss(nn.Module):
    def __init__(self, lambda_bound: float = 0.1, lambda_deriv: float = 0.5) -> None:
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_deriv = lambda_deriv

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        p_phys_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mse = (pred - target) ** 2
        loss_recon = (loss_mse * mask).sum() / (mask.sum() + 1e-6)

        loss_lower = F.relu(-pred)
        loss_upper = F.relu(pred - 1.05)
        loss_bound = (loss_lower + loss_upper).mean()

        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        phys_diff = p_phys_prior[:, :, 1:] - p_phys_prior[:, :, :-1]
        loss_deriv_raw = (pred_diff - phys_diff) ** 2
        mask_slice = mask[:, :, 1:]
        loss_deriv = (loss_deriv_raw * mask_slice).sum() / (mask_slice.sum() + 1e-6)

        total_loss = loss_recon + self.lambda_bound * loss_bound + self.lambda_deriv * loss_deriv
        return total_loss, loss_recon, loss_deriv


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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
