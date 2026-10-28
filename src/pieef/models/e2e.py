from __future__ import annotations

import torch
import torch.nn as nn

from .forecast import ForecastHead
from .forecast_skip import POLICIES as FORECAST_SKIP_POLICIES, ForecastSkipBlock
from .graph import SCORE_POLICIES, DirectionalGraphStack, DynamicDirectionalGraphLayer
from .physics import DifferentiablePhysicsLayer, PhysicsPrior, _limit_vector
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
        nwp_mean: torch.Tensor | None = None,
        nwp_std: torch.Tensor | None = None,
        encoder_depth: int = 3,
        graph_depth: int = 1,
        num_heads: int = 4,
        use_prior_input: bool = True,
        use_bound_loss: bool = True,
        use_derivative_loss: bool = True,
        lambda_bound: float = 0.1,
        lambda_deriv: float = 0.05,
        use_density: bool = False,
        reference_density: float = 1.225,
        nwp_pressure_index: int = 2,
        nwp_temperature_index: int = 3,
        nwp_target_speed_index: int = -3,
        cut_in_speed: float = 3.0,
        rated_wind_speed: float = 12.0,
        cut_out_speed: float | None = None,
        cut_in_speeds: torch.Tensor | None = None,
        rated_wind_speeds: torch.Tensor | None = None,
        cut_out_speeds: torch.Tensor | None = None,
        use_calibration: bool = True,
        use_physics_curve: bool = True,
        curve_mode: str = "bspline",
        manufacturer_wind_speeds=None,
        manufacturer_power=None,
        use_direction_bias: bool | None = None,
        use_sensitivity_gate: bool | None = None,
        sensitivity_mode: str = "symmetric",
        terrain_edge_features: torch.Tensor | None = None,
        use_terrain_bias: bool = False,
        joint_masking: bool = False,
        use_mask_indicators: bool = False,
        revin_missing_policy: str = "legacy",
        revin_training_mean=None,
        revin_training_stdev=None,
        forecast_head_policy: str = "legacy",
        graph_score_policy: str = "v1_dot_gate",
        forecast_skip_policy: str = "none",
    ) -> None:
        super().__init__()
        if forecast_head_policy not in ("legacy", "target_nwp_physics_mlp"):
            raise ValueError("forecast_head_policy must be legacy or target_nwp_physics_mlp")
        if graph_score_policy not in SCORE_POLICIES:
            raise ValueError("graph_score_policy must be v1_dot_gate or cosine_temperature")
        if forecast_skip_policy not in FORECAST_SKIP_POLICIES:
            raise ValueError("forecast_skip_policy must be none or last_obs_linear")
        # v2: widths are derived from (d_model, nwp_dim); no fixed-shape guard.
        self.forecast_head_policy = forecast_head_policy
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_turbines = num_turbines
        self.graph_score_policy = graph_score_policy
        self.forecast_skip_policy = forecast_skip_policy
        if encoder_depth < 1 or graph_depth < 1 or d_model % num_heads:
            raise ValueError("depths must be positive and d_model must be divisible by num_heads")
        self.encoder_depth = int(encoder_depth)
        self.graph_depth = int(graph_depth)
        self.use_prior_input = bool(use_prior_input)
        self.joint_masking = bool(joint_masking)
        self.use_mask_indicators = bool(use_mask_indicators)
        self.cut_in_speed = float(cut_in_speed)
        self.rated_wind_speed = float(rated_wind_speed)
        self.cut_out_speed = cut_out_speed
        # None buffers are omitted from state_dict, preserving scalar checkpoints.
        self.register_buffer("cut_in_speeds", _limit_vector(cut_in_speeds, num_turbines, "cut_in_speeds"))
        self.register_buffer("rated_wind_speeds", _limit_vector(rated_wind_speeds, num_turbines, "rated_wind_speeds"))
        self.register_buffer("cut_out_speeds", _limit_vector(cut_out_speeds, num_turbines, "cut_out_speeds"))
        self.use_density = bool(use_density)
        self.reference_density = float(reference_density)
        self.nwp_pressure_index = int(nwp_pressure_index)
        self.nwp_temperature_index = int(nwp_temperature_index)
        self.nwp_target_speed_index = int(nwp_target_speed_index)
        self.nwp_stats_supplied = nwp_mean is not None and nwp_std is not None
        if (nwp_mean is None) != (nwp_std is None):
            raise ValueError("nwp_mean and nwp_std must be supplied together")
        if self.use_density and (not self.nwp_stats_supplied or self.reference_density <= 0):
            raise ValueError("density correction requires explicit NWP statistics and positive reference_density")
        mean = torch.zeros(nwp_dim) if nwp_mean is None else torch.as_tensor(nwp_mean, dtype=torch.float32)
        std = torch.ones(nwp_dim) if nwp_std is None else torch.as_tensor(nwp_std, dtype=torch.float32)
        if mean.shape != (nwp_dim,) or std.shape != (nwp_dim,) or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise ValueError("NWP statistics must be finite vectors of nwp_dim with positive standard deviations")
        self.register_buffer("nwp_mean", mean)
        self.register_buffer("nwp_std", std)
        if not 0 <= self.nwp_pressure_index < nwp_dim or not 0 <= self.nwp_temperature_index < nwp_dim:
            raise ValueError("NWP pressure/temperature feature indices are out of range")
        if not -nwp_dim <= self.nwp_target_speed_index < nwp_dim:
            raise ValueError("NWP target speed feature index is out of range")
        if not self.nwp_stats_supplied and self.nwp_target_speed_index != -3:
            raise ValueError("An explicit target speed field requires NWP normalization statistics")
        self.sensitivity_mode = sensitivity_mode

        self.register_buffer("scada_wind_mean", torch.tensor(scada_wind_mean, dtype=torch.float32))
        self.register_buffer("scada_wind_std", torch.tensor(scada_wind_std, dtype=torch.float32))

        self.pi_input_dim = 1 + 1 + 1 + nwp_dim + (2 if self.use_mask_indicators else 0)
        self.pi_embedding = nn.Linear(self.pi_input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True,
        )
        self.mae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.mae_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.mask_generator = BlockMaskGenerator(mask_ratio=mask_ratio, min_block_len=6, max_block_len=24)
        self.loss_fn_pi = PhysicsConsistencyLoss(
            lambda_bound=lambda_bound, lambda_deriv=lambda_deriv,
            use_bound=use_bound_loss, use_derivative=use_derivative_loss,
        )

        self.revin = RevIN(num_turbines, missing_policy=revin_missing_policy,
                           training_mean=revin_training_mean, training_stdev=revin_training_stdev)
        self.register_buffer("dist_matrix", dist_matrix)
        self.register_buffer("rel_coords", rel_coords)
        self.dcn = DifferentiablePhysicsLayer(
            nwp_dim,
            num_turbines=num_turbines,
            num_control_points=num_control_points,
            cut_in_speed=cut_in_speed,
            rated_wind_speed=rated_wind_speed,
            cut_out_speed=cut_out_speed,
            cut_in_speeds=self.cut_in_speeds,
            rated_wind_speeds=self.rated_wind_speeds,
            cut_out_speeds=self.cut_out_speeds,
            use_calibration=use_calibration,
            use_physics_curve=use_physics_curve,
            curve_mode=curve_mode,
            manufacturer_wind_speeds=manufacturer_wind_speeds,
            manufacturer_power=manufacturer_power,
        )
        graph_kwargs = dict(
            enable_physics_bias=use_graph_phys_bias,
            use_direction_bias=use_direction_bias,
            use_sensitivity_gate=use_sensitivity_gate,
            sensitivity_mode=sensitivity_mode,
            terrain_edge_features=terrain_edge_features,
            use_terrain_bias=use_terrain_bias,
            score_policy=graph_score_policy,
        )
        self.graph = (
            DynamicDirectionalGraphLayer(d_model, **graph_kwargs) if graph_depth == 1
            else DirectionalGraphStack(d_model, graph_depth, **graph_kwargs)
        )
        self.forecast_head = (
            ForecastHead(d_model) if forecast_head_policy == "legacy" else
            ForecastHead(d_model, policy=forecast_head_policy, nwp_dim=nwp_dim)
        )
        # F6 is constructed last so that, when it is off (or merely when it is on),
        # no earlier layer's initialization depends on it.
        if forecast_skip_policy == "last_obs_linear":
            self.forecast_skip = ForecastSkipBlock(seq_len)

    def raw_nwp(self, x_nwp: torch.Tensor) -> torch.Tensor:
        """Invert feature standardization using the statistics fitted on training data."""
        return x_nwp * self.nwp_std + self.nwp_mean

    def density_ratio(self, x_nwp: torch.Tensor) -> torch.Tensor:
        raw = self.raw_nwp(x_nwp)
        pressure = raw[..., self.nwp_pressure_index]
        temperature = raw[..., self.nwp_temperature_index]
        # Dry-air approximation using Pa and Kelvin; no humidity or terrain extrapolation.
        if not bool(torch.isfinite(pressure).all() and torch.isfinite(temperature).all()):
            raise ValueError("NWP density inputs must be finite")
        if not bool((pressure > 0).all() and (temperature > 0).all()):
            raise ValueError("NWP density inputs must use positive pressure in Pa and temperature in K")
        return pressure / (287.05 * temperature * self.reference_density)

    def _raw_target_wind(self, x_nwp_target: torch.Tensor) -> torch.Tensor:
        if self.nwp_stats_supplied:
            return self.raw_nwp(x_nwp_target)[..., self.nwp_target_speed_index]
        # Legacy constructors did not supply NWP statistics. Revised runs must do so.
        return x_nwp_target[..., -3] * self.scada_wind_std + self.scada_wind_mean

    def forward_stage1_reconstruct(
        self,
        x_hist_p: torch.Tensor,
        x_hist_w: torch.Tensor,
        x_hist_nwp: torch.Tensor,
        force_mask: torch.Tensor | None = None,
        force_wind_mask: torch.Tensor | None = None,
    ):
        bsz, n_turb, seq_len, _ = x_hist_p.shape

        if force_mask is not None:
            mask = force_mask.to(x_hist_p)
        elif self.training:
            mask = self.mask_generator.generate_mask(x_hist_p)
        else:
            mask = torch.zeros_like(x_hist_p)
        if mask.shape != x_hist_p.shape or not bool(((mask >= 0) & (mask <= 1)).all()):
            raise ValueError("power missing mask must match power input shape and lie in [0, 1]")
        wind_mask = mask if self.joint_masking else torch.zeros_like(x_hist_w)
        if force_wind_mask is not None:
            supplied = force_wind_mask.to(x_hist_w)
            if supplied.shape != x_hist_w.shape or not bool(((supplied >= 0) & (supplied <= 1)).all()):
                raise ValueError("wind missing mask must match wind input shape and lie in [0, 1]")
            wind_mask = torch.maximum(wind_mask, supplied)

        x_p_for_revin = x_hist_p.squeeze(-1).permute(0, 2, 1)
        mask_for_revin = mask.squeeze(-1).permute(0, 2, 1)
        x_p_norm_seq = self.revin(x_p_for_revin, "norm", mask=mask_for_revin)
        x_p_norm = x_p_norm_seq.permute(0, 2, 1).unsqueeze(-1)
        p_input = torch.where(mask > 0, torch.zeros_like(x_p_norm), x_p_norm)
        wind_input = torch.where(wind_mask > 0, torch.zeros_like(x_hist_w), x_hist_w)
        # Recompute the prior from the masked wind and suppress it where wind is unavailable.
        rho = self.density_ratio(x_hist_nwp).unsqueeze(-1) if self.use_density else None
        p_phys_prior = PhysicsPrior.get_theoretical_power(
            wind_input, self.scada_wind_mean, self.scada_wind_std,
            cut_in_speed=self.cut_in_speed, rated_wind_speed=self.rated_wind_speed,
            cut_out_speed=self.cut_out_speed, density_ratio=rho,
            cut_in_speeds=self.cut_in_speeds, rated_wind_speeds=self.rated_wind_speeds,
            cut_out_speeds=self.cut_out_speeds,
        )
        p_phys_prior = torch.where(wind_mask > 0, torch.zeros_like(p_phys_prior), p_phys_prior)
        prior_input = p_phys_prior if self.use_prior_input else torch.zeros_like(p_phys_prior)
        inputs = [p_input, prior_input, wind_input, x_hist_nwp]
        if self.use_mask_indicators:
            inputs.extend([mask, wind_mask])
        x_combined = torch.cat(inputs, dim=-1)

        x_in_flat = x_combined.reshape(bsz * n_turb, seq_len, -1)
        x_emb = self.pi_embedding(x_in_flat) + self.position_embedding[:, :seq_len, :]
        feat = self.mae_encoder(x_emb)
        rec_power = self.mae_head(feat).reshape(bsz, n_turb, seq_len, 1)

        loss_total = torch.tensor(0.0, device=x_hist_p.device)
        if self.training or force_mask is not None:
            rec_per_unit = self.revin(rec_power.squeeze(-1).permute(0, 2, 1), "denorm").permute(0, 2, 1)
            loss_total, _, _ = self.loss_fn_pi(
                rec_power.squeeze(-1),
                x_p_norm.squeeze(-1),
                mask.squeeze(-1),
                p_phys_prior.squeeze(-1),
                physical_pred=rec_per_unit,
                prior_valid_mask=(1 - wind_mask).squeeze(-1),
            )
        return loss_total, feat, rec_power, mask

    def forward_stage2_calibration(self, x_hist_w: torch.Tensor, x_nwp_target: torch.Tensor):
        _ = x_hist_w
        raw_w = self._raw_target_wind(x_nwp_target)
        rho = self.density_ratio(x_nwp_target) if self.use_density else None
        v_cal, p_phys, _ = self.dcn(x_nwp_target, raw_w, rho)
        return v_cal, p_phys

    def forward_stage2_training(self, x_nwp_target: torch.Tensor):
        raw_w = self._raw_target_wind(x_nwp_target)
        rho = self.density_ratio(x_nwp_target) if self.use_density else None
        v_cal, p_phys, delta_v = self.dcn(x_nwp_target, raw_w, rho)
        return v_cal, p_phys, delta_v

    def forecast_space_history(self, hist_power, hist_power_mask=None):
        """Map the power history into Z, the forecast-head output space (F6).

        Z is the affine RevIN-normalized space produced by the most recent
        `forward_stage1_reconstruct` call, i.e. exactly the space the head predicts in
        and from which `revin(..., "denorm")` returns physical units. Masked steps are
        reported as invalid and the caller zeroes them in Z.
        """
        if hist_power is None:
            raise ValueError("forecast_skip_policy=last_obs_linear requires the historical power window")
        if hist_power.ndim != 4 or hist_power.shape[-1] != 1:
            raise ValueError("historical power must be [batch, turbines, seq_len, 1]")
        if hist_power.shape[1] != self.num_turbines or hist_power.shape[2] != self.seq_len:
            raise ValueError("historical power window does not match the configured turbines/seq_len")
        mask = (torch.zeros_like(hist_power) if hist_power_mask is None
                else hist_power_mask.to(hist_power))
        if mask.shape != hist_power.shape:
            raise ValueError("historical power mask must match the power window")
        series = hist_power.squeeze(-1).permute(0, 2, 1)
        masked = mask.squeeze(-1).permute(0, 2, 1)
        history_z = self.revin.apply_normalization(series)
        valid = (masked <= 0) & torch.isfinite(series)
        fallback = self.forecast_space_training_mean(series.shape[0])
        return history_z.permute(0, 2, 1), valid.permute(0, 2, 1), fallback

    def forecast_space_training_mean(self, batch: int) -> torch.Tensor:
        """The F1 training mean expressed in Z, broadcast to [batch, turbines]."""
        mean, stdev = getattr(self.revin, "mean", None), getattr(self.revin, "stdev", None)
        if mean is None or stdev is None:
            raise ValueError("RevIN statistics are unavailable; run forward_stage1_reconstruct first")
        # Without F1 moments the per-window mean is the only available reference.
        base = (torch.zeros_like(mean) if self.revin.training_mean is None
                else self.revin.training_mean.to(mean))
        z = (base - mean) / stdev
        if self.revin.affine:
            z = z * self.revin.affine_weight + self.revin.affine_bias
        # RevIN records per-sample statistics, so the fallback is per sample as well.
        return z.reshape(batch, self.num_turbines)

    def forward_stage3_forecast(
        self,
        feat_seq: torch.Tensor,
        v_cal: torch.Tensor,
        p_phys: torch.Tensor,
        x_nwp_target: torch.Tensor,
        hist_power: torch.Tensor | None = None,
        hist_power_mask: torch.Tensor | None = None,
    ):
        bsz = v_cal.shape[0]
        feat_last = feat_seq[:, -1, :]
        feat_nodes = feat_last.view(bsz, self.num_turbines, self.d_model)

        direction_nwp = self.raw_nwp(x_nwp_target) if self.nwp_stats_supplied else x_nwp_target
        wd_sin = direction_nwp[:, :, -2]
        wd_cos = direction_nwp[:, :, -1]
        wind_vec = torch.stack([wd_sin, wd_cos], dim=-1)
        global_wind_vec = wind_vec.mean(dim=1)

        curve_sensitivity = None
        if self.sensitivity_mode == "derivative":
            rho = self.density_ratio(x_nwp_target) if self.use_density else None
            curve_sensitivity = self.dcn.power_sensitivity(v_cal.squeeze(-1), rho)
        feat_graph = self.graph(
            feat_nodes, v_cal, p_phys, self.dist_matrix, self.rel_coords,
            global_wind_vec, curve_sensitivity=curve_sensitivity,
        )
        if self.forecast_head_policy == "legacy":
            pred_norm = self.forecast_head(feat_graph, v_cal)
        else:
            pred_norm = self.forecast_head(
                feat_graph, v_cal, x_nwp_target=x_nwp_target, p_phys=p_phys,
            )
        if self.forecast_skip_policy == "last_obs_linear":
            history_z, history_valid, fallback = self.forecast_space_history(hist_power, hist_power_mask)
            pred_norm = pred_norm + self.forecast_skip(history_z, history_valid, fallback)
        pred_denorm = self.revin(pred_norm.unsqueeze(1), "denorm").squeeze(1)
        return pred_denorm
