from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCORE_POLICIES = ("v1_dot_gate", "cosine_temperature")
TEMPERATURE_INIT = 10.0
TEMPERATURE_MAX = 50.0
NORMALIZE_EPS = 1e-6


class DynamicDirectionalGraphLayer(nn.Module):
    def __init__(
        self, d_model: int, enable_physics_bias: bool = True,
        use_direction_bias: bool | None = None,
        use_sensitivity_gate: bool | None = None,
        sensitivity_mode: str = "symmetric",
        terrain_edge_features: torch.Tensor | None = None,
        use_terrain_bias: bool = False,
        score_policy: str = "v1_dot_gate",
    ) -> None:
        super().__init__()
        if score_policy not in SCORE_POLICIES:
            raise ValueError("graph_score_policy must be v1_dot_gate or cosine_temperature")
        self.score_policy = score_policy
        self.d_model = d_model
        self.enable_physics_bias = enable_physics_bias
        self.use_direction_bias = enable_physics_bias if use_direction_bias is None else bool(use_direction_bias)
        self.use_sensitivity_gate = enable_physics_bias if use_sensitivity_gate is None else bool(use_sensitivity_gate)
        if sensitivity_mode not in {"symmetric", "derivative"}:
            raise ValueError("sensitivity_mode must be 'symmetric' or 'derivative'")
        self.sensitivity_mode = sensitivity_mode
        self.use_terrain_bias = bool(use_terrain_bias)
        if self.use_terrain_bias and terrain_edge_features is None:
            raise ValueError("terrain bias requires externally supplied terrain edge features")
        if terrain_edge_features is not None:
            terrain = torch.as_tensor(terrain_edge_features, dtype=torch.float32)
            if terrain.ndim != 3 or terrain.shape[0] != terrain.shape[1] or terrain.shape[-1] < 1:
                raise ValueError("terrain edge features must have shape [turbines, turbines, features]")
            if not bool(torch.isfinite(terrain).all()):
                raise ValueError("terrain edge features must be finite")
        else:
            terrain = None
        self.register_buffer("terrain_edge_features", terrain)
        self.terrain_encoder = None
        if self.use_terrain_bias:
            self.terrain_encoder = nn.Sequential(nn.Linear(terrain.shape[-1], 16), nn.ReLU(), nn.Linear(16, 1))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        if self.score_policy == "v1_dot_gate":
            self.sens_gate = nn.Linear(1, 1)
        else:
            # F3: s = exp(theta) with theta = ln 10 (s = 10) and a hard cap at 50.
            # Constants carry no RNG draw; the multiplicative (1 + gate) module is removed.
            self.log_temperature = nn.Parameter(torch.tensor(math.log(TEMPERATURE_INIT)))
            if self.use_sensitivity_gate:
                self.sensitivity_weight = nn.Parameter(torch.tensor(1.0))
        self.dir_encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.last_attn_weights = None

    def forward(
        self,
        x: torch.Tensor,
        v_cal: torch.Tensor,
        p_phys: torch.Tensor,
        dist_matrix: torch.Tensor,
        rel_coords: torch.Tensor,
        wind_vec: torch.Tensor,
        curve_sensitivity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = v_cal
        _ = dist_matrix
        if self.score_policy == "v1_dot_gate":
            q = self.query(x)
            k = self.key(x)
        else:
            # F3: unit-normalized queries/keys, so the score is s * cos(q, k).
            q = F.normalize(self.query(x), p=2, dim=-1, eps=NORMALIZE_EPS)
            k = F.normalize(self.key(x), p=2, dim=-1, eps=NORMALIZE_EPS)
        v = self.value(x)
        if self.score_policy == "v1_dot_gate":
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        else:
            scores = self.score_scale() * torch.matmul(q, k.transpose(-2, -1))

        if self.use_direction_bias:
            w_vec = wind_vec.view(-1, 1, 1, 2)
            r_vec = rel_coords.unsqueeze(0)
            w_norm = F.normalize(w_vec, dim=-1)
            r_norm = F.normalize(r_vec, dim=-1)
            alignment = torch.sum(w_norm * r_norm, dim=-1, keepdim=True)
            dir_bias = self.dir_encoder(alignment).squeeze(-1)
            scores = scores + dir_bias
        if self.use_terrain_bias:
            if self.terrain_edge_features.shape[:2] != scores.shape[-2:]:
                raise ValueError("terrain feature layout does not match graph nodes")
            scores = scores + self.terrain_encoder(self.terrain_edge_features).squeeze(-1).unsqueeze(0)
        if self.use_sensitivity_gate:
            if self.sensitivity_mode == "derivative":
                if curve_sensitivity is None:
                    raise ValueError("derivative sensitivity gate requires curve_sensitivity")
                sensitivity = curve_sensitivity
            else:
                sensitivity = p_phys * (1 - p_phys)
            if self.score_policy == "v1_dot_gate":
                phys_bias = torch.sigmoid(self.sens_gate(sensitivity))
                scores = scores * (1 + phys_bias.transpose(-2, -1))
            else:
                # F3: additive w * S_j replaces the multiplicative (1 + gate) entirely.
                scores = scores + self.sensitivity_weight * sensitivity.transpose(-2, -1)
        attn = F.softmax(scores, dim=-1)
        # Keep observation on-device; interpretation exports perform the CPU copy.
        self.last_attn_weights = attn.detach()

        out = torch.matmul(attn, v)
        return out + x

    def score_scale(self) -> torch.Tensor:
        """Learned temperature s = exp(theta), truncated at TEMPERATURE_MAX (F3)."""
        if self.score_policy != "cosine_temperature":
            raise AttributeError("score_scale is only defined for the cosine_temperature policy")
        return torch.exp(self.log_temperature).clamp(max=TEMPERATURE_MAX)


class DirectionalGraphStack(nn.Module):
    """Independent attention layers; graph depth changes the actual computation."""

    def __init__(self, d_model: int, depth: int, **layer_kwargs) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("graph depth must be positive")
        self.layers = nn.ModuleList([
            DynamicDirectionalGraphLayer(d_model, **layer_kwargs) for _ in range(depth)
        ])

    @property
    def last_attn_weights(self):
        return self.layers[-1].last_attn_weights

    def forward(self, x, v_cal, p_phys, dist_matrix, rel_coords, wind_vec, curve_sensitivity=None):
        for layer in self.layers:
            x = layer(x, v_cal, p_phys, dist_matrix, rel_coords, wind_vec, curve_sensitivity)
        return x
