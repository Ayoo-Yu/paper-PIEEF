"""F6: nearest-observation anchoring plus a shared full-window linear skip.

Both terms are added in the forecast-head output space Z (the affine
RevIN-normalized space) and are therefore applied before the inverse transform.
"""
from __future__ import annotations

import torch
import torch.nn as nn

POLICIES = ("none", "last_obs_linear")


class ForecastSkipBlock(nn.Module):
    """F6a `g * last observed power` and F6b `Linear(seq_len -> 1)`, shared by turbines."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("forecast skip requires a positive history length")
        self.seq_len = int(seq_len)
        # F6a: g = sigmoid(gamma) with gamma initialised to 0, hence g = 0.5.
        self.gate = nn.Parameter(torch.zeros(()))
        # F6b: zero weight and bias, so this path contributes exactly 0 at initialisation.
        self.linear = nn.Linear(self.seq_len, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Latest terms for inspection; interpretation exports read them on the CPU.
        self.last_terms = None

    @property
    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def forward(
        self,
        history_z: torch.Tensor,
        valid_mask: torch.Tensor,
        fallback_z: torch.Tensor,
    ) -> torch.Tensor:
        """Return the additive term for [batch, turbine] predictions.

        `history_z` is the power history already expressed in Z; masked steps are
        forced to exactly 0 here (F6b must never read a filled value as real power).
        `fallback_z` is the Z image of the training mean, used when the whole
        window is masked.
        """
        if history_z.ndim != 3 or history_z.shape[-1] != self.seq_len:
            raise ValueError("forecast skip history must be [batch, turbines, seq_len]")
        if valid_mask.shape != history_z.shape or valid_mask.dtype != torch.bool:
            raise ValueError("forecast skip validity mask must be a boolean tensor of the history shape")
        if fallback_z.shape != history_z.shape[:2]:
            raise ValueError("forecast skip fallback must match [batch, turbines]")
        if not bool(torch.isfinite(history_z[valid_mask]).all()):
            raise ValueError("forecast skip history must be finite where it is observed")
        z = torch.where(valid_mask, history_z, torch.zeros_like(history_z))
        positions = torch.arange(self.seq_len, device=history_z.device)
        last_index = torch.where(
            valid_mask, positions, torch.full_like(positions, -1)).max(dim=-1).values
        observed = last_index >= 0
        gathered = z.gather(-1, last_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        last_obs = torch.where(observed, gathered, fallback_z)
        linear_term = self.linear(z).squeeze(-1)
        self.last_terms = {
            "last_obs": last_obs.detach(),
            "linear": linear_term.detach(),
            "gate": self.gate_value.detach(),
            "observed": observed,
        }
        return self.gate_value * last_obs + linear_term
