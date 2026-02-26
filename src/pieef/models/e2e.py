from __future__ import annotations

import torch
import torch.nn as nn

from .forecast import ForecastHead
from .graph import DynamicDirectionalGraphLayer
from .physics import DifferentiablePhysicsLayer, PhysicsPrior
from .pretrain import BlockMaskGenerator, PhysicsConsistencyLoss, RevIN


class E2EPhysNet(nn.Module):
    def __init__(
        self,
        num_turbines: int,
        nwp_dim: int,
        seq_len: int,
        dist_matrix: torch.Tensor,
        rel_coords: torch.Tensor,
        scada_wind_mean: float,
        scada_wind_std: float,
        d_model: int = 128,
        mask_ratio: float = 0.4,
        num_control_points: int = 15,
        use_graph_phys_bias: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_turbines = num_turbines

        self.register_buffer("scada_wind_mean", torch.tensor(scada_wind_mean, dtype=torch.float32))
        self.register_buffer("scada_wind_std", torch.tensor(scada_wind_std, dtype=torch.float32))

        self.pi_input_dim = 1 + 1 + 1 + nwp_dim
        self.pi_embedding = nn.Linear(self.pi_input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True,
        )
        self.mae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.mae_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.mask_generator = BlockMaskGenerator(mask_ratio=mask_ratio, min_block_len=6, max_block_len=24)
        self.loss_fn_pi = PhysicsConsistencyLoss(lambda_bound=0.1, lambda_deriv=0.05)

        self.revin = RevIN(num_turbines)
        self.register_buffer("dist_matrix", dist_matrix)
        self.register_buffer("rel_coords", rel_coords)
        self.dcn = DifferentiablePhysicsLayer(
            nwp_dim,
            num_turbines=num_turbines,
            num_control_points=num_control_points,
        )
        self.graph = DynamicDirectionalGraphLayer(d_model, enable_physics_bias=use_graph_phys_bias)
        self.forecast_head = ForecastHead(d_model)

    def forward_stage1_reconstruct(
        self,
        x_hist_p: torch.Tensor,
        x_hist_w: torch.Tensor,
        x_hist_nwp: torch.Tensor,
        force_mask: torch.Tensor | None = None,
    ):
        bsz, n_turb, seq_len, _ = x_hist_p.shape

        x_p_for_revin = x_hist_p.squeeze(-1).permute(0, 2, 1)
        x_p_norm_seq = self.revin(x_p_for_revin, "norm")
        x_p_norm = x_p_norm_seq.permute(0, 2, 1).unsqueeze(-1)

        p_phys_prior = PhysicsPrior.get_theoretical_power(x_hist_w, self.scada_wind_mean, self.scada_wind_std)

        if force_mask is not None:
            mask = force_mask
        elif self.training:
            mask = self.mask_generator.generate_mask(x_hist_p)
        else:
            mask = torch.zeros_like(x_hist_p)

        p_input = x_p_norm * (1 - mask)
        x_combined = torch.cat([p_input, p_phys_prior, x_hist_w, x_hist_nwp], dim=-1)

        x_in_flat = x_combined.reshape(bsz * n_turb, seq_len, -1)
        x_emb = self.pi_embedding(x_in_flat) + self.position_embedding[:, :seq_len, :]
        feat = self.mae_encoder(x_emb)
        rec_power = self.mae_head(feat).reshape(bsz, n_turb, seq_len, 1)

        loss_total = torch.tensor(0.0, device=x_hist_p.device)
        if self.training or force_mask is not None:
            loss_total, _, _ = self.loss_fn_pi(
                rec_power.squeeze(-1),
                x_p_norm.squeeze(-1),
                mask.squeeze(-1),
                p_phys_prior.squeeze(-1),
            )
        return loss_total, feat, rec_power, mask

    def forward_stage2_calibration(self, x_hist_w: torch.Tensor, x_nwp_target: torch.Tensor):
        _ = x_hist_w
        nwp_ws_norm = x_nwp_target[:, :, -3]
        raw_w_approx = nwp_ws_norm * self.scada_wind_std + self.scada_wind_mean
        v_cal, p_phys, _ = self.dcn(x_nwp_target, raw_w_approx)
        return v_cal, p_phys

    def forward_stage2_training(self, x_nwp_target: torch.Tensor):
        nwp_ws_norm = x_nwp_target[:, :, -3]
        raw_w_approx = nwp_ws_norm * self.scada_wind_std + self.scada_wind_mean
        v_cal, p_phys, delta_v = self.dcn(x_nwp_target, raw_w_approx)
        return v_cal, p_phys, delta_v

    def forward_stage3_forecast(
        self,
        feat_seq: torch.Tensor,
        v_cal: torch.Tensor,
        p_phys: torch.Tensor,
        x_nwp_target: torch.Tensor,
    ):
        bsz = v_cal.shape[0]
        feat_last = feat_seq[:, -1, :]
        feat_nodes = feat_last.view(bsz, self.num_turbines, self.d_model)

        wd_sin = x_nwp_target[:, :, -2]
        wd_cos = x_nwp_target[:, :, -1]
        wind_vec = torch.stack([wd_sin, wd_cos], dim=-1)
        global_wind_vec = wind_vec.mean(dim=1)

        feat_graph = self.graph(feat_nodes, v_cal, p_phys, self.dist_matrix, self.rel_coords, global_wind_vec)
        pred_norm = self.forecast_head(feat_graph, v_cal)
        pred_denorm = self.revin(pred_norm.unsqueeze(1), "denorm").squeeze(1)
        return pred_denorm
