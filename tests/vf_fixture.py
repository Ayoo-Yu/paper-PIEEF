"""Small, wholly synthetic CPU fixture; it is not a paper experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"


def install_source(source=None):
    source = Path(source) if source else Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source))


def model_and_inputs(**overrides):
    import torch
    from pieef.models.e2e import E2EPhysNet

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    coords = torch.tensor([[0., 0.], [400., 100.], [850., -80.]])
    rel = coords[:, None, :] - coords[None, :, :]
    nwp_mean = torch.zeros(12)
    nwp_mean[2:4] = torch.tensor([101325., 288.15])
    nwp_mean[7] = nwp_mean[9] = 8.
    nwp_mean[-1] = 1.
    nwp_std = torch.ones(12)
    nwp_std[2] = 100.
    terrain = torch.stack([rel[..., 0] / 1000., rel[..., 1] / 1000.,
                           torch.cdist(coords, coords) / 1000.], dim=-1)
    kwargs = dict(num_turbines=3, nwp_dim=12, seq_len=10,
                  dist_matrix=torch.cdist(coords, coords) / 1000., rel_coords=rel,
                  scada_wind_mean=8., scada_wind_std=2., d_model=16,
                  encoder_depth=1, graph_depth=1, num_heads=4,
                  nwp_mean=nwp_mean, nwp_std=nwp_std,
                  use_density=True, nwp_target_speed_index=7,
                  rated_wind_speed=10., cut_out_speed=22.,
                  terrain_edge_features=terrain, use_terrain_bias=True,
                  use_mask_indicators=True, revin_missing_policy="train_stats_20",
                  revin_training_mean=torch.tensor([.3, .4, .5]),
                  revin_training_stdev=torch.tensor([.1, .2, .15]),
                  forecast_head_policy="target_nwp_physics_mlp",
                  graph_score_policy="cosine_temperature",
                  forecast_skip_policy="last_obs_linear")
    kwargs.update(overrides)
    model = E2EPhysNet(**kwargs).cpu().eval()
    inputs = dict(power=.1 + .7 * torch.rand(2, 3, 10, 1),
                  wind=.2 * torch.randn(2, 3, 10, 1),
                  nwp=.1 * torch.randn(2, 3, 10, 12),
                  target_nwp=.1 * torch.randn(2, 3, 12),
                  target=.2 + .4 * torch.rand(2, 3),
                  mask=torch.zeros(2, 3, 10, 1))
    inputs["mask"][0, 0, -3:] = 1
    return model, inputs


def forward(model, inputs):
    loss, feat, reconstruction, mask = model.forward_stage1_reconstruct(
        inputs["power"], inputs["wind"], inputs["nwp"], force_mask=inputs["mask"])
    wind, prior = model.forward_stage2_calibration(inputs["wind"], inputs["target_nwp"])
    prediction = model.forward_stage3_forecast(
        feat, wind, prior, inputs["target_nwp"], hist_power=inputs["power"],
        hist_power_mask=inputs["mask"])
    return dict(reconstruction_loss=loss, features=feat, reconstruction=reconstruction,
                mask=mask, calibrated_wind=wind, physical_power=prior, prediction=prediction,
                attention=model.graph.last_attn_weights,
                revin_mean=model.revin.mean, revin_stdev=model.revin.stdev)


def snapshot():
    import torch
    from pieef.train_main import TrainConfig, _configure_stage3, _forecast_loss

    model, inputs = model_and_inputs()
    _configure_stage3(model, "joint", "frozen", reconstruction_weight=.05)
    result = forward(model, inputs)
    loss = _forecast_loss(result["prediction"], inputs["target"], TrainConfig(huber_delta=.1))
    loss.backward()
    return dict(state={k: v.detach().clone() for k, v in model.state_dict().items()},
                output={k: v.detach().clone() for k, v in result.items()},
                forecast_loss=loss.detach(),
                gradient={k: None if p.grad is None else p.grad.detach().clone()
                          for k, p in model.named_parameters()},
                cuda_initialized=torch.cuda.is_initialized())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    install_source(args.source)
    import torch
    torch.save(snapshot(), args.snapshot)
