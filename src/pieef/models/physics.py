from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _limit_vector(values, num_turbines: int, name: str):
    if values is None:
        return None
    vector = torch.as_tensor(values, dtype=torch.float32).detach().clone()
    if vector.shape != (num_turbines,) or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must be a finite vector with one value per turbine")
    return vector


def _broadcast_limit(vector, scalar, wind_speed: torch.Tensor):
    if vector is None:
        return scalar
    if wind_speed.ndim < 2 or vector.shape != (wind_speed.shape[1],):
        raise ValueError("Operating limit vector must match the turbine axis")
    return vector.to(wind_speed).reshape(1, -1, *([1] * (wind_speed.ndim - 2)))


class PhysicsPrior:
    @staticmethod
    def get_theoretical_power(
        wind_speed_norm: torch.Tensor,
        scada_mean: torch.Tensor,
        scada_std: torch.Tensor,
        cut_in_speed: float = 3.0,
        rated_wind_speed: float = 12.0,
        cut_out_speed: float | None = None,
        density_ratio: torch.Tensor | None = None,
        cut_in_speeds: torch.Tensor | None = None,
        rated_wind_speeds: torch.Tensor | None = None,
        cut_out_speeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ws_ms = wind_speed_norm * scada_std + scada_mean
        cut_in = _broadcast_limit(cut_in_speeds, cut_in_speed, ws_ms)
        rated = _broadcast_limit(rated_wind_speeds, rated_wind_speed, ws_ms)
        cut_out = _broadcast_limit(cut_out_speeds, cut_out_speed, ws_ms)
        if not bool(torch.as_tensor(rated - cut_in).gt(0).all()):
            raise ValueError("rated_wind_speed must exceed cut_in_speed")
        equivalent_ws = ws_ms
        if density_ratio is not None:
            equivalent_ws = ws_ms * density_ratio.clamp_min(1e-6).pow(1.0 / 3.0)
        p_theory = (F.relu(equivalent_ws - cut_in) / (rated - cut_in)) ** 3
        out = torch.clamp(p_theory, 0.0, 1.0)
        out = torch.where(ws_ms < cut_in, torch.zeros_like(out), out)
        if cut_out is not None:
            out = torch.where(ws_ms >= cut_out, torch.zeros_like(out), out)
        return out


class ManufacturerPowerCurve(nn.Module):
    """Linear interpolation of a supplied manufacturer curve in per-unit power.

    No curve is synthesized: callers must provide genuine tabulated points.
    """

    def __init__(self, wind_speeds, power) -> None:
        super().__init__()
        ws = torch.as_tensor(wind_speeds, dtype=torch.float32)
        p = torch.as_tensor(power, dtype=torch.float32)
        if ws.ndim != 1 or p.ndim != 1 or ws.numel() < 2 or ws.shape != p.shape:
            raise ValueError("manufacturer curve requires equally sized 1-D arrays with at least two points")
        if not bool(torch.isfinite(ws).all() and torch.isfinite(p).all()):
            raise ValueError("manufacturer curve points must be finite")
        if not bool((ws[1:] > ws[:-1]).all()) or not bool(((p >= 0) & (p <= 1)).all()):
            raise ValueError("manufacturer wind speeds must increase; power must be per-unit in [0, 1]")
        self.register_buffer("wind_speeds", ws)
        self.register_buffer("power", p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        value = x.clamp(self.wind_speeds[0], self.wind_speeds[-1])
        idx = torch.searchsorted(self.wind_speeds, value.contiguous(), right=True) - 1
        idx = idx.clamp(0, self.wind_speeds.numel() - 2)
        low, high = self.wind_speeds[idx], self.wind_speeds[idx + 1]
        fraction = (value - low) / (high - low)
        return self.power[idx] + fraction * (self.power[idx + 1] - self.power[idx])


class LearnableBSplineCurve(nn.Module):
    def __init__(self, num_turbines: int, num_control_points: int = 15, max_wind_speed: float = 25.0) -> None:
        super().__init__()
        if num_control_points < 4 or max_wind_speed <= 0:
            raise ValueError("B-spline requires at least four control points and positive maximum wind speed")
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
        cut_in_speed: float = 2.5,
        rated_wind_speed: float = 12.0,
        cut_out_speed: float | None = None,
        use_calibration: bool = True,
        use_physics_curve: bool = True,
        curve_mode: str = "bspline",
        manufacturer_wind_speeds=None,
        manufacturer_power=None,
        cut_in_speeds=None,
        rated_wind_speeds=None,
        cut_out_speeds=None,
    ) -> None:
        super().__init__()
        self.num_turbines = num_turbines
        self.cut_in_speed = float(cut_in_speed)
        self.rated_wind_speed = float(rated_wind_speed)
        self.cut_out_speed = None if cut_out_speed is None else float(cut_out_speed)
        self.register_buffer("cut_in_speeds", _limit_vector(cut_in_speeds, num_turbines, "cut_in_speeds"))
        self.register_buffer("rated_wind_speeds", _limit_vector(rated_wind_speeds, num_turbines, "rated_wind_speeds"))
        self.register_buffer("cut_out_speeds", _limit_vector(cut_out_speeds, num_turbines, "cut_out_speeds"))
        cut_in = self.cut_in_speed if self.cut_in_speeds is None else self.cut_in_speeds
        rated = self.rated_wind_speed if self.rated_wind_speeds is None else self.rated_wind_speeds
        cut_out = self.cut_out_speed if self.cut_out_speeds is None else self.cut_out_speeds
        if not bool(torch.as_tensor(cut_in).isfinite().all()) or not bool(torch.as_tensor(cut_in).ge(0).all()):
            raise ValueError("cut_in speeds must be finite and nonnegative")
        if not bool(torch.as_tensor(rated).isfinite().all()) or not bool(torch.as_tensor(rated - cut_in).gt(0).all()):
            raise ValueError("rated_wind_speed must exceed cut_in_speed")
        if cut_out is not None and (not bool(torch.as_tensor(cut_out).isfinite().all()) or not bool(torch.as_tensor(cut_out - rated).gt(0).all())):
            raise ValueError("cut_out_speed must exceed rated_wind_speed")
        self.use_calibration = bool(use_calibration)
        self.use_physics_curve = bool(use_physics_curve)
        self.curve_mode = curve_mode
        self.calib_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ) if self.use_calibration else None
        if not self.use_physics_curve:
            self.power_curve = None
        elif curve_mode == "bspline":
            self.power_curve = LearnableBSplineCurve(
                num_turbines,
                num_control_points=num_control_points,
                max_wind_speed=max(25.0, float(torch.as_tensor(cut_out).max()) if cut_out is not None else 25.0),
            )
        elif curve_mode == "manufacturer":
            if manufacturer_wind_speeds is None or manufacturer_power is None:
                raise ValueError("manufacturer curve mode requires actual wind speed and per-unit power points")
            self.power_curve = ManufacturerPowerCurve(manufacturer_wind_speeds, manufacturer_power)
        else:
            raise ValueError("curve_mode must be 'bspline' or 'manufacturer'")

    def power_at(self, calibrated_wind_speed: torch.Tensor, density_ratio: torch.Tensor | None = None):
        ws = calibrated_wind_speed
        if not self.use_physics_curve:
            return torch.zeros_like(ws).unsqueeze(-1)
        equivalent_ws = ws if density_ratio is None else ws * density_ratio.clamp_min(1e-6).pow(1.0 / 3.0)
        power = self.power_curve(equivalent_ws).unsqueeze(-1)
        # Physical operating boundaries apply to actual wind, not density-equivalent wind.
        cut_in = _broadcast_limit(self.cut_in_speeds, self.cut_in_speed, ws)
        cut_out = _broadcast_limit(self.cut_out_speeds, self.cut_out_speed, ws)
        operating = ws >= cut_in
        if cut_out is not None:
            operating = operating & (ws < cut_out)
        power = torch.where(operating.unsqueeze(-1), power, torch.zeros_like(power))
        return power

    def power_sensitivity(self, calibrated_wind_speed: torch.Tensor, density_ratio: torch.Tensor | None = None):
        # Differentiable finite-difference d(P/P_r)/d(v), excluding cut-out discontinuity.
        eps = 0.05
        upper = self.power_at(calibrated_wind_speed + eps, density_ratio)
        lower = self.power_at((calibrated_wind_speed - eps).clamp_min(0), density_ratio)
        result = ((upper - lower) / (2.0 * eps)).clamp_min(0.0)
        cut_out = _broadcast_limit(self.cut_out_speeds, self.cut_out_speed, calibrated_wind_speed)
        if cut_out is not None:
            result = torch.where(
                (calibrated_wind_speed >= cut_out - eps).unsqueeze(-1),
                torch.zeros_like(result), result,
            )
        return result

    def forward(self, x_feat: torch.Tensor, raw_wind_speed_ms: torch.Tensor, density_ratio: torch.Tensor | None = None):
        if self.use_calibration:
            delta_v = 3.0 * torch.tanh(self.calib_net(x_feat))
        else:
            delta_v = torch.zeros_like(raw_wind_speed_ms).unsqueeze(-1)
        v_cal = raw_wind_speed_ms.unsqueeze(-1) + delta_v
        v_cal = F.relu(v_cal)
        p_phys = self.power_at(v_cal.squeeze(-1), density_ratio)
        return v_cal, p_phys, delta_v
