from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsPrior:
    @staticmethod
    def get_theoretical_power(wind_speed_norm: torch.Tensor, scada_mean: torch.Tensor, scada_std: torch.Tensor) -> torch.Tensor:
        ws_ms = wind_speed_norm * scada_std + scada_mean
        cut_in = 3.0
        rated_ws = 12.0
        v_effective = F.relu(ws_ms - cut_in)
        p_theory = (v_effective / (rated_ws - cut_in)) ** 3
        return torch.clamp(p_theory, 0.0, 1.0)


class LearnableBSplineCurve(nn.Module):
    def __init__(self, num_turbines: int, num_control_points: int = 15, max_wind_speed: float = 25.0) -> None:
        super().__init__()
        self.num_turbines = num_turbines
        self.num_cp = num_control_points
        self.max_ws = max_wind_speed
        self.grid_size = max_wind_speed / (num_control_points - 3)

        init_curve = np.linspace(0, 1.0, num_control_points)
        init_deltas = init_curve[1:] - init_curve[:-1]
        self.cp_delta_logs = nn.Parameter(
            torch.tensor(init_deltas, dtype=torch.float32).log().unsqueeze(0).repeat(num_turbines, 1)
        )
        self.cp_start = nn.Parameter(torch.zeros(num_turbines, 1))
        self.rated_power = nn.Parameter(torch.ones(num_turbines, 1) * 1.0)

    def get_control_points(self) -> torch.Tensor:
        deltas = F.softplus(self.cp_delta_logs)
        cumulative = torch.cumsum(deltas, dim=1)
        zeros = torch.zeros_like(self.cp_start)
        cps_raw = torch.cat([zeros, cumulative], dim=1)
        cap = torch.clamp(self.rated_power, 0.8, 1.0)
        return torch.minimum(cps_raw, cap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        t = torch.clamp(x, 0.0, self.max_ws - 0.01) / self.grid_size
        indices = torch.floor(t).long()

        w = t - indices.float()
        w2 = w * w
        w3 = w2 * w

        b0 = (1 - w) ** 3 / 6.0
        b1 = (3 * w3 - 6 * w2 + 4) / 6.0
        b2 = (-3 * w3 + 3 * w2 + 3 * w + 1) / 6.0
        b3 = w3 / 6.0

        cps = self.get_control_points()

        bsz, _, _ = indices.shape
        cps_expanded = cps.unsqueeze(0).expand(bsz, -1, -1)

        p0 = torch.gather(cps_expanded, 2, indices)
        p1 = torch.gather(cps_expanded, 2, indices + 1)
        p2 = torch.gather(cps_expanded, 2, indices + 2)
        p3 = torch.gather(cps_expanded, 2, indices + 3)

        out = b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3
        return out.squeeze(-1)


class DifferentiablePhysicsLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_turbines: int,
        hidden_dim: int = 32,
        num_control_points: int = 15,
    ) -> None:
        super().__init__()
        self.num_turbines = num_turbines
        self.calib_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.power_curve = LearnableBSplineCurve(
            num_turbines,
            num_control_points=num_control_points,
            max_wind_speed=25.0,
        )

    def forward(self, x_feat: torch.Tensor, raw_wind_speed_ms: torch.Tensor):
        delta_v = 3.0 * torch.tanh(self.calib_net(x_feat))
        v_cal = raw_wind_speed_ms.unsqueeze(-1) + delta_v
        v_cal = F.relu(v_cal)

        p_phys = self.power_curve(v_cal.squeeze(-1)).unsqueeze(-1)

        cut_in_speed = 2.5
        low_wind_mask = torch.sigmoid(5.0 * (v_cal - cut_in_speed))
        p_phys = p_phys * low_wind_mask
        return v_cal, p_phys, delta_v
