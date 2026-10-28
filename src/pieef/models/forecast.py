from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    def __init__(
        self, d_model: int, policy: str = "legacy", nwp_dim: int = 12,
    ) -> None:
        super().__init__()
        if policy not in ("legacy", "target_nwp_physics_mlp"):
            raise ValueError("forecast_head_policy must be legacy or target_nwp_physics_mlp")
        self.policy = policy
        if policy == "legacy":
            # Keep the original module name and sole RNG-consuming initialization.
            self.head = nn.Linear(d_model + 1, 1)
        else:
            # v2: the target-NWP head is parameterised by the actual (d_model, nwp_dim) so a
            # cross-site site with a different NWP feature count can use the same architecture.
            # For DMS (d_model=128, nwp_dim=12) this builds identical modules in the identical
            # RNG order as v1, so DMS behaviour is unchanged.
            self.d_model = int(d_model)
            self.nwp_dim = int(nwp_dim)
            self.nwp_embedding = nn.Sequential(nn.Linear(self.nwp_dim, 16), nn.GELU())
            self.head = nn.Sequential(nn.Linear(self.d_model + 1 + 16 + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self, feat_graph: torch.Tensor, v_cal: torch.Tensor,
        *, x_nwp_target: torch.Tensor | None = None, p_phys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.policy == "legacy":
            feat_final = torch.cat([feat_graph, v_cal], dim=-1)
            return self.head(feat_final).squeeze(-1)
        if feat_graph.ndim != 3 or feat_graph.shape[-1] != self.d_model:
            raise ValueError("F4 graph features must have shape [batch, turbine, d_model]")
        prefix = feat_graph.shape[:2]
        if v_cal.shape != (*prefix, 1):
            raise ValueError("F4 v_cal must have shape [batch, turbine, 1]")
        if x_nwp_target is None or x_nwp_target.shape != (*prefix, self.nwp_dim):
            raise ValueError("F4 target NWP must have shape [batch, turbine, nwp_dim]")
        if p_phys is None or p_phys.shape != (*prefix, 1):
            raise ValueError("F4 P_phys must have shape [batch, turbine, 1]")
        # Use exactly the train-standardized target NWP already supplied to DCN.
        # P_phys comes from its calibrated weather/physical curve, never a target label.
        embedded_nwp = self.nwp_embedding(x_nwp_target)
        feat_final = torch.cat([feat_graph, v_cal, embedded_nwp, p_phys], dim=-1)
        return self.head(feat_final).squeeze(-1)
