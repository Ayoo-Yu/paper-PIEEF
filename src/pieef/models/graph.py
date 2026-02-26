from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDirectionalGraphLayer(nn.Module):
    def __init__(self, d_model: int, enable_physics_bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.enable_physics_bias = enable_physics_bias
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.sens_gate = nn.Linear(1, 1)
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
    ) -> torch.Tensor:
        _ = v_cal
        _ = dist_matrix
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if self.enable_physics_bias:
            sensitivity = p_phys * (1 - p_phys)
            phys_bias = torch.sigmoid(self.sens_gate(sensitivity))

            w_vec = wind_vec.view(-1, 1, 1, 2)
            r_vec = rel_coords.unsqueeze(0)
            w_norm = F.normalize(w_vec, dim=-1)
            r_norm = F.normalize(r_vec, dim=-1)
            alignment = torch.sum(w_norm * r_norm, dim=-1, keepdim=True)
            dir_bias = self.dir_encoder(alignment).squeeze(-1)
            scores = (scores + dir_bias) * (1 + phys_bias.transpose(-2, -1))
        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach().cpu().numpy()

        out = torch.matmul(attn, v)
        return out + x
