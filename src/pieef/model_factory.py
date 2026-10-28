from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import torch

from .models.e2e import E2EPhysNet


def build_main_model(bundle, seq_len, options, device):
    options = dict(options)
    policy = options.get("revin_missing_policy", "legacy")
    if policy not in {"legacy", "train_stats_20"}:
        raise ValueError("revin_missing_policy must be legacy or train_stats_20")
    if any(key in options for key in ("revin_training_mean", "revin_training_stdev")):
        raise ValueError("F1 moments come only from raw training data or verified preprocessing, not model config arrays")
    if policy == "train_stats_20":
        from .power_training_stats import ensure_training_power_statistics
        stats = ensure_training_power_statistics(bundle)
        options["revin_training_mean"] = stats["mean_pu"]
        options["revin_training_stdev"] = stats["stdev_pu"]
    options.setdefault("nwp_target_speed_index", getattr(bundle.handler, "nwp_target_speed_index", -3))
    options.setdefault("nwp_pressure_index", getattr(bundle.handler, "nwp_pressure_index", 2))
    options.setdefault("nwp_temperature_index", getattr(bundle.handler, "nwp_temperature_index", 3))
    if options.pop("use_turbine_operating_limits", False):
        limits = getattr(bundle.handler, "operating_limits", None)
        mapping = {"cut_in_speed": "cut_in_speeds", "rated_wind_speed": "rated_wind_speeds",
                   "cut_out_speed": "cut_out_speeds"}
        if not isinstance(limits, dict) or any(key not in limits or limits[key] is None for key in mapping):
            raise ValueError("Per-turbine operating limits require actual cut-in, rated, and cut-out metadata for every turbine")
        for source, destination in mapping.items():
            options[destination] = torch.as_tensor(limits[source], dtype=torch.float32)
    manufacturer_path = options.pop("manufacturer_curve_path", None)
    if manufacturer_path:
        path = Path(manufacturer_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        curve = pd.read_csv(path)
        options["manufacturer_wind_speeds"] = curve["wind_speed_ms"].to_numpy()
        options["manufacturer_power"] = curve["power_pu"].to_numpy()
    allowed = set(inspect.signature(E2EPhysNet).parameters)
    if set(options) - allowed:
        raise ValueError(f"Unknown model options: {sorted(set(options) - allowed)}")
    options.update(
        nwp_mean=torch.as_tensor(bundle.handler.nwp_mean),
        nwp_std=torch.as_tensor(bundle.handler.nwp_std),
        terrain_edge_features=getattr(bundle.handler, "terrain_edge_features", None),
    )
    return E2EPhysNet(
        num_turbines=bundle.num_turbines, nwp_dim=bundle.nwp_dim,
        seq_len=int(seq_len), dist_matrix=bundle.dist_tensor.to(device),
        rel_coords=bundle.rel_coords_tensor.to(device),
        scada_wind_mean=float(bundle.handler.scada_w_mean),
        scada_wind_std=float(bundle.handler.scada_w_std), **options,
    ).to(device)
