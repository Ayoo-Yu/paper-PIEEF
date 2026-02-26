from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.head = nn.Linear(d_model + 1, 1)

    def forward(self, feat_graph: torch.Tensor, v_cal: torch.Tensor) -> torch.Tensor:
        feat_final = torch.cat([feat_graph, v_cal], dim=-1)
        return self.head(feat_final).squeeze(-1)
