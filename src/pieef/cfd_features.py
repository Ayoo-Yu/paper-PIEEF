"""Frozen round-4, target-only local wind features shared by PI-EEF and LightGBM.

The original weather layout, historical inputs, targets, splits and fitted NWP
normalization are retained. CFD directions are meteorological FROM; the model's
last two channels are flow-TO east/north components. Measured-direction mode
uses only the observation at the prediction origin (target index - horizon).
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SOURCE_SHA = "870a605f3c3ecb609a0bf31a9d2abe38f46fb0a10791595e33b01cd12827fb98"
SETTINGS_SHA = "63824c248c4e0edd29f9ea8b29cde877ae1c89885090134bb12e81b0e368341e"
PROFILES = {"native": 28, "sampling_recovery": 1, "existing_post_only_recovery": 1,
            "mpi_finalize_post_only_recovery": 4, "loaded43_count_correction": 4}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wrap_degrees(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


class FrozenCFDLookup:
    """Exact turbine-ordered default Ct=0 LUT with circular linear interpolation."""
    def __init__(self, rows, turbine_ids, site):
        self.turbine_ids = tuple(map(str, turbine_ids))
        _need(len(set(self.turbine_ids)) == len(self.turbine_ids), "Duplicate CFD turbine identity")
        n = 29 if site == "wind_farm_A" else 6 if site == "kelmarsh" else None
        _need(n is not None and len(self.turbine_ids) == n, "Declared CFD site/turbine count differs")
        angles = np.arange(0.0, 360.0, 22.5 if site == "wind_farm_A" else 30.0)
        expected = {(tid, float(a)) for tid in self.turbine_ids for a in angles}
        mapped = {}
        for row in rows:
            key = (str(row["turbine_id"]), float(row["sector_deg"]))
            _need(key not in mapped, "Duplicate CFD turbine/sector cell")
            _need(row["site"] == site and row["case_kind"] == "sector" and float(row["ct"]) == 0,
                  "Only default terrain-only CFD sectors can feed prediction")
            _need(float(row["z0_m"]) == (0.3 if site == "wind_farm_A" else 0.05), "Default CFD roughness differs")
            d, speedup = float(row["delta_dir_deg"]), float(row["speedup"])
            _need(np.isfinite(d) and -180 <= d < 180 and np.isfinite(speedup) and speedup > 0,
                  "CFD direction/speedup must be finite and physical")
            if row["quality_pass"] != "True":
                rid = row["run_id"]
                permitted = {"energy-r4-wind_farm_A-sector-a" + a + "-z0p3-ct0-v1"
                             for a in ("022p50", "225p00", "292p50", "337p50")}
                _need(rid in permitted and row.get("recovery_accepted") == "True"
                      and row.get("source_kind") == "mpi_finalize_post_only_recovery",
                      "Raw-False CFD cell lacks exact accepted MPI recovery")
            mapped[key] = (d, speedup)
        _need(set(mapped) == expected, "CFD table must contain the exact full turbine/sector grid")
        self.angles = angles
        self.delta = np.asarray([[mapped[(tid, float(a))][0] for a in angles] for tid in self.turbine_ids])
        self.speedup = np.asarray([[mapped[(tid, float(a))][1] for a in angles] for tid in self.turbine_ids])

    def evaluate(self, direction_from_degrees):
        theta = np.asarray(direction_from_degrees, dtype=np.float64)
        _need(theta.ndim >= 1 and theta.shape[-1] == len(self.turbine_ids) and np.isfinite(theta).all(),
              "CFD queries must be finite and ordered [..., turbines]")
        theta = theta % 360.0
        step = float(self.angles[1] - self.angles[0])
        lower = np.floor(theta / step).astype(np.int64)
        upper = (lower + 1) % len(self.angles)
        fraction = (theta - lower * step) / step
        turbine = np.arange(len(self.turbine_ids))
        a, b = self.delta[turbine, lower], self.delta[turbine, upper]
        delta = wrap_degrees(a + fraction * wrap_degrees(b - a))
        scale = self.speedup[turbine, lower] + fraction * (self.speedup[turbine, upper] - self.speedup[turbine, lower])
        _need(np.isfinite(scale).all() and (scale > 0).all(), "Interpolated CFD speedups must be positive")
        return (theta + delta) % 360.0, scale


def _frozen_lookup(options, turbine_ids):
    table, summary = _path(options["lut_path"]), _path(options["summary_path"])
    raw, evidence = table.read_bytes(), summary.read_bytes()
    _need(_digest(raw) == options["lut_sha256"] and _digest(evidence) == options["summary_sha256"],
          "Pinned CFD table/strict-summary SHA differs")
    proof = json.loads(evidence)
    _need(proof.get("complete_cases") == 38 and proof.get("required_cases") == 38
          and proof.get("rejected_cases") == [] and proof.get("source_profile_counts") == PROFILES
          and proof.get("settings_sha256") == SETTINGS_SHA
          and proof.get("summary_script_sha256") == SUMMARY_SOURCE_SHA, "Final strict CFD38 identity is required")
    sources = proof.get("all_case_sources", {})
    _need(len(sources) == 38 and all(s.get("strict_full_file_stream_verified") is True for s in sources.values()),
          "Every frozen CFD source must have full D/E stream verification")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    for row in rows:
        _need(row["run_id"] in sources, "CFD row is absent from the final accepted sources")
    return FrozenCFDLookup(rows, turbine_ids, options["site"])


def _source_options(bundle, data_cfg, model_cfg):
    source = model_cfg.get("direction_source", "farm_mean")
    correction = model_cfg.get("nwp_speed_correction", "none")
    _need(source in {"farm_mean", "per_turbine"} and correction in {"none", "cfd_speedup"}, "Unknown local-wind switches")
    if source == "farm_mean" and correction == "none":
        return None
    _need(source == "per_turbine", "Speed correction requires local directions")
    options = dict(data_cfg.get("local_wind") or {})
    _need(options.get("mode") in {"cfd", "measured_origin"}, "Enabled local wind requires a frozen data.local_wind source")
    _need(options.get("site") in {"wind_farm_A", "kelmarsh"}, "Local wind site is explicit")
    ids = tuple(map(str, bundle.handler.turbine_ids))
    declared = tuple(map(str, options.get("turbine_id_map", ids)))
    _need(declared == ids, "CFD turbine order must exactly match native physical turbine IDs")
    if options["mode"] == "measured_origin":
        _need(options["site"] == "kelmarsh" and correction == "none", "Measured-origin mode is Kelmarsh direction only")
    return options


def compute_local_target_features(bundle, data_cfg, model_cfg):
    """Return target-aligned raw CFD information; also used unchanged by F1.

    Dict keys train/val/test each contain [samples,turbines] flow_sin,
    flow_cos, cfd_speedup, raw_corrected_speed, direction_from_degrees plus
    indices/origin_indices. No normalization or original input is modified.
    """
    options = _source_options(bundle, data_cfg, model_cfg)
    _need(options is not None, "Local features requested with all switches disabled")
    h = bundle.handler
    mean = np.asarray(h.nwp_mean, dtype=np.float32)
    std = np.asarray(h.nwp_std, dtype=np.float32)
    speed_index = int(h.nwp_target_speed_index)
    source_path = _path(data_cfg.get("scada_path", h.source_data_path))
    frame = None
    if options["site"] == "wind_farm_A" or options["mode"] == "measured_origin":
        frame = pd.read_csv(source_path)
    lookup = _frozen_lookup(options, h.turbine_ids) if options["mode"] == "cfd" else None
    result = {}
    for split in ("train", "val", "test"):
        base = getattr(bundle, split + "_loader").dataset
        target = np.asarray(base.indices, dtype=np.int64)
        origin = target - int(base.horizon)
        _need((origin >= 0).all(), "Every local query needs a valid historical prediction origin")
        normalized = base.future_nwp[target].detach().cpu().numpy().astype(np.float32)
        _need(normalized.ndim == 2 and normalized.shape[1] == bundle.nwp_dim, "Base target NWP layout changed")
        raw = normalized * std + mean
        if options["site"] == "wind_farm_A":
            u_name, v_name = options["direction_u_column"], options["direction_v_column"]
            _need(u_name in frame and v_name in frame, "Declared target 100m NWP components are absent")
            u = frame[u_name].to_numpy(dtype=np.float64)[target]
            v = frame[v_name].to_numpy(dtype=np.float64)[target]
        else:
            u, v = raw[:, int(h.nwp_u_index)], raw[:, int(h.nwp_v_index)]
        _need(np.isfinite(u).all() and np.isfinite(v).all() and (np.hypot(u, v) > 0).all(), "Query wind direction is undefined")
        theta = np.broadcast_to((np.degrees(np.arctan2(-u, -v)) % 360)[:, None], (len(target), bundle.num_turbines))
        if lookup is not None:
            local, scale = lookup.evaluate(theta)
        else:
            names = options.get("measured_direction_columns", [f"wind_direction{i + 1}" for i in range(bundle.num_turbines)])
            _need(len(names) == bundle.num_turbines and all(c in frame for c in names), "Measured-origin direction columns missing")
            observed = frame[names].to_numpy(dtype=np.float64)[origin]
            _need(np.isfinite(observed).all(), "Measured origin directions must be actually available; no future fill")
            radians = np.deg2rad(observed)
            sx, cy = np.sin(radians).mean(axis=1), np.cos(radians).mean(axis=1)
            _need((np.hypot(sx, cy) > 1e-6).all(), "Measured origin farm direction has ambiguous circular mean")
            farm = np.degrees(np.arctan2(sx, cy))
            anomaly = wrap_degrees(observed - farm[:, None])
            local, scale = (theta + anomaly) % 360, np.ones_like(theta)
        radians = np.deg2rad(local)
        flow_sin, flow_cos = -np.sin(radians), -np.cos(radians)
        if options["site"] == "wind_farm_A":
            # The prepared source stores this physical field directly. Avoid a
            # second normalization round trip in the shared PI/F1 information.
            speed_column = h.nwp_target_speed_column
            _need(speed_column in frame, "Selected prepared target-speed column is absent")
            raw_speed = frame[speed_column].to_numpy(dtype=np.float64)[target]
        else:
            # GFS must remain the genuine origin-aligned future vintage. A
            # same-valid-time SCADA row must never substitute weather for it.
            raw_speed = raw[:, speed_index]
        corrected = raw_speed[:, None] * scale
        _need(np.isfinite(corrected).all() and (corrected >= 0).all(), "Corrected selected target speed must be finite/nonnegative")
        result[split] = {"indices": target, "origin_indices": origin,
                         "direction_from_degrees": local, "cfd_speedup": scale,
                         "flow_sin": flow_sin, "flow_cos": flow_cos,
                         "raw_corrected_speed": corrected}
    result["provenance"] = {"mode": options["mode"], "site": options["site"], "source_options": options,
                            "direction_source": model_cfg.get("direction_source", "farm_mean"),
                            "nwp_speed_correction": model_cfg.get("nwp_speed_correction", "none"),
                            "target_only": True, "history_targets_and_splits_unchanged": True,
                            "normalization": "original VF train-fitted NWP statistics unchanged",
                            "direction_channels": "flow-TO east/north = -sin/-cos(meteorological FROM)",
                            "target_speed_index": speed_index, "turbine_ids": list(map(str, h.turbine_ids)),
                            "raw_speed_source": (h.nwp_target_speed_column if options["site"] == "wind_farm_A"
                                                 else "origin-aligned future_nwp inverse normalization"),
                            "measured_origin_index": "target_index-horizon; only used by measured_origin",
                            "raw_source_path": str(source_path), "raw_source_sha256": _digest(source_path.read_bytes())}
    return result


class LocalWindForecastWindows(Dataset):
    def __init__(self, base, transformed_target):
        self.base = base
        self.transformed_target = torch.as_tensor(transformed_target, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, item):
        power, wind, historical_weather, _, target = self.base[item]
        return power, wind, historical_weather, self.transformed_target[item], target


def attach_local_forecast_inputs(bundle, data_cfg, model_cfg):
    """Default-off is a strict no-op; enabled modes alter only target NWP."""
    options = _source_options(bundle, data_cfg, model_cfg)
    if options is None:
        return bundle
    features = compute_local_target_features(bundle, data_cfg, model_cfg)
    mean = np.asarray(bundle.handler.nwp_mean, dtype=np.float32)
    std = np.asarray(bundle.handler.nwp_std, dtype=np.float32)
    speed_index = int(bundle.handler.nwp_target_speed_index)
    for split in ("train", "val", "test"):
        loader = getattr(bundle, split + "_loader")
        _need(loader.num_workers == 0 and loader.drop_last is False, "Frozen single-worker loader protocol required")
        base = loader.dataset
        values = base.future_nwp[base.indices].detach().cpu().numpy()
        local = np.broadcast_to(values[:, None, :], (len(base), bundle.num_turbines, bundle.nwp_dim)).copy()
        local[:, :, -2] = (features[split]["flow_sin"] - mean[-2]) / std[-2]
        local[:, :, -1] = (features[split]["flow_cos"] - mean[-1]) / std[-1]
        if model_cfg.get("nwp_speed_correction", "none") == "cfd_speedup":
            local[:, :, speed_index] = (features[split]["raw_corrected_speed"] - mean[speed_index]) / std[speed_index]
        _need(np.isfinite(local).all(), "Transformed target NWP must be finite")
        wrapped = LocalWindForecastWindows(base, local)
        # Reuse the original sampler (it only consults len(base)); it preserves the
        # exact training random permutation and DataLoader base-seed draw sequence.
        replacement = DataLoader(wrapped, batch_size=loader.batch_size, sampler=loader.sampler,
                                 num_workers=0, pin_memory=loader.pin_memory,
                                 collate_fn=loader.collate_fn, generator=loader.generator)
        setattr(bundle, split + "_loader", replacement)
    bundle.handler.local_wind_provenance = features["provenance"]
    return bundle
