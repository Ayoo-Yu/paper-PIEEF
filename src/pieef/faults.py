"""Input faults in SCADA units or raw NWP units; never alter targets or predictions."""
from __future__ import annotations

import hashlib
import re
from typing import Mapping

import numpy as np
import torch


def _wind_field_layout(names):
    """Find original wind components/speeds by height without renaming fields."""
    components, speeds = {}, []
    for index, name in enumerate(names):
        component = re.match(r"^(\d+)_metre[_ ]([uv])_wind_component", name, flags=re.I)
        if component:
            height, axis = component.group(1), component.group(2).lower()
        else:
            component = re.match(r"^(\d+)([uv])_", name, flags=re.I)
            if component:
                height, axis = component.group(1), component.group(2).lower()
            else:
                component = re.match(r"^([uv])(\d+)_", name, flags=re.I)
                height, axis = (component.group(2), component.group(1).lower()) if component else (None, None)
        if component:
            components.setdefault(height, {})[axis] = index
        elif re.match(r"^(wind_speed|ws)\d+(?:_|$)", name, flags=re.I):
            speeds.append(index)
    vectors = [(value["u"], value["v"]) for value in components.values() if "u" in value and "v" in value]
    return vectors, speeds


def _sample_rng(seed: int, identity) -> np.random.Generator:
    payload = f"{int(seed)}|{identity}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(value)


def _nearest_turbines(coords, count: int, anchor: int, num_turbines: int) -> list[int]:
    if not 1 <= count <= num_turbines or not 0 <= anchor < num_turbines:
        raise ValueError("fault turbine_count/anchor_turbine is out of range")
    if count == 1:
        return [anchor]
    if coords is None:
        raise ValueError("multi-turbine faults require real coordinates")
    xy = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    if xy.ndim != 2 or xy.shape[0] != num_turbines or xy.shape[1] < 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be finite [turbines, coordinate dimensions]")
    distance = np.linalg.norm(xy[:, :2] - xy[anchor, :2], axis=1)
    others = np.array([i for i in range(num_turbines) if i != anchor])
    ordered = others[np.lexsort((others, distance[others]))]
    return [anchor, *ordered[:count - 1].tolist()]


def _causal_last(x: torch.Tensor, start: int) -> torch.Tensor:
    if start == 0:
        return x.new_zeros(())
    prefix = x[:start]
    available = torch.nonzero(torch.isfinite(prefix), as_tuple=False).flatten()
    return prefix[available[-1]] if available.numel() else x.new_zeros(())


def _fault_steps(scenario: Mapping, seq_len: int) -> tuple[int, int]:
    if "duration_steps" in scenario:
        steps = int(scenario["duration_steps"])
    else:
        minutes = float(scenario.get("step_minutes", 15.0))
        hours = float(scenario.get("duration_hours", 1.0))
        if minutes <= 0 or hours <= 0:
            raise ValueError("fault duration and sampling interval must be positive")
        raw_steps = hours * 60.0 / minutes
        if not np.isclose(raw_steps, round(raw_steps)):
            raise ValueError("fault duration must be an integral number of samples")
        steps = int(round(raw_steps))
    if not 1 <= steps <= seq_len:
        raise ValueError("fault duration must fit the historical input window; it is never shortened silently")
    start = int(scenario.get("start_index", seq_len - steps))
    if start < 0 or start + steps > seq_len:
        raise ValueError("fault start/duration lies outside the historical input window")
    return start, start + steps


def apply_fault(batch, scenario: Mapping, scada_wind_std, coords, seed: int):
    """Return (five corrupted inputs/target, power missing mask, wind missing mask).

    ``sample_ids`` should contain global forecast-origin timestamps, making faults
    invariant to model names and batch boundaries. Without them, batch-local
    ordinals are used. Missing masks stay zero for frozen/drift/NWP faults.
    """
    if len(batch) != 5 or not all(isinstance(x, torch.Tensor) for x in batch):
        raise ValueError("batch must be the five tensor tuple: power, wind, historical NWP, target NWP, target")
    x_p, x_w, x_hist_nwp, x_target_nwp, target = batch
    if x_p.ndim != 4 or x_p.shape[-1] != 1 or x_w.shape != x_p.shape:
        raise ValueError("SCADA inputs must have equal [batch, turbines, time, 1] shapes")
    bsz, num_turbines, seq_len, _ = x_p.shape
    if x_target_nwp.ndim != 3 or x_target_nwp.shape[:2] != (bsz, num_turbines):
        raise ValueError("target NWP must have shape [batch, turbines, features]")
    identities = scenario.get("sample_ids", list(range(bsz)))
    if len(identities) != bsz:
        raise ValueError("sample_ids must match batch size")
    kind = scenario.get("kind", "missing")
    power, wind = x_p.clone(), x_w.clone()
    target_nwp = x_target_nwp.clone()
    p_mask, w_mask = torch.zeros_like(x_p), torch.zeros_like(x_w)

    if kind in {"missing", "frozen", "drift"}:
        channels = scenario.get("channels", "both")
        if channels not in {"power", "wind", "both"}:
            raise ValueError("SCADA channels must be 'power', 'wind', or 'both'")
        if kind == "drift" and channels != "wind":
            raise ValueError("drift is a wind-speed fault measured in m/s; channels must be 'wind'")
        start, end = _fault_steps(scenario, seq_len)
        count = int(scenario.get("turbine_count", 1))
        std = torch.as_tensor(scada_wind_std, dtype=x_w.dtype, device=x_w.device)
        if std.numel() not in {1, num_turbines} or not bool(torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("scada_wind_std must be positive, scalar or one value per turbine")
        std = std.reshape(-1)
        for b, identity in enumerate(identities):
            rng = _sample_rng(seed, identity)
            anchor = int(scenario.get("anchor_turbine", rng.integers(num_turbines)))
            selected = _nearest_turbines(coords, count, anchor, num_turbines)
            for turbine in selected:
                if kind in {"missing", "frozen"}:
                    if channels in {"power", "both"}:
                        power[b, turbine, start:end, 0] = _causal_last(x_p[b, turbine, :, 0], start)
                        if kind == "missing":
                            p_mask[b, turbine, start:end, 0] = 1
                    if channels in {"wind", "both"}:
                        wind[b, turbine, start:end, 0] = _causal_last(x_w[b, turbine, :, 0], start)
                        if kind == "missing":
                            w_mask[b, turbine, start:end, 0] = 1
                else:
                    amount = float(scenario.get("drift_ms", 1.0))
                    if not np.isfinite(amount):
                        raise ValueError("drift_ms must be finite")
                    scale = std[0] if std.numel() == 1 else std[turbine]
                    ramp = torch.linspace(1.0 / (end - start), 1.0, end - start,
                                          dtype=wind.dtype, device=wind.device)
                    wind[b, turbine, start:end, 0] += amount * ramp / scale
        return (power, wind, x_hist_nwp, x_target_nwp, target), p_mask, w_mask

    if kind not in {"nwp_speed", "nwp_direction"}:
        raise ValueError(f"unknown input fault kind: {kind}")
    if "nwp_mean" not in scenario or "nwp_std" not in scenario:
        raise ValueError("NWP faults require training-set normalization statistics")
    mean = torch.as_tensor(scenario["nwp_mean"], dtype=target_nwp.dtype, device=target_nwp.device)
    std = torch.as_tensor(scenario["nwp_std"], dtype=target_nwp.dtype, device=target_nwp.device)
    n_features = target_nwp.shape[-1]
    if mean.shape != (n_features,) or std.shape != mean.shape or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("NWP statistics must be finite vectors with positive standard deviations")
    declared = scenario.get("nwp_feature_columns")
    if declared is not None:
        if len(declared) not in (n_features, n_features - 3):
            raise ValueError("Declared NWP features must describe the base fields or all base-plus-derived fields.")
        vectors, named_speeds = _wind_field_layout(list(declared)[:n_features - 3])
    else:
        vectors, named_speeds = [], []
    default_pair = next((pair for pair in vectors if re.search(r"100", str(declared[pair[0]]))), vectors[0] if vectors else (4, 5))
    ui, vi = int(scenario.get("u_index", default_pair[0])), int(scenario.get("v_index", default_pair[1]))
    if n_features < 6 or not 0 <= ui < n_features - 3 or not 0 <= vi < n_features - 3 or ui == vi:
        raise ValueError("NWP U/V feature indices must identify separate base features")
    sigma = float(scenario.get("sigma_ms" if kind == "nwp_speed" else "sigma_deg", 0.0))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("NWP perturbation standard deviation must be nonnegative and finite")
    if sigma == 0:
        return (power, wind, x_hist_nwp, target_nwp, target), p_mask, w_mask
    raw = target_nwp * std + mean
    speed_indices = list(scenario.get("speed_feature_indices", []))
    if any(not 0 <= int(idx) < n_features - 3 for idx in speed_indices):
        raise ValueError("speed_feature_indices must refer to base NWP features")
    named_features = declared is not None
    if scenario.get("nwp_timing_rule") == "author_prepared_ecmwf" and not named_features:
        raise ValueError("Prepared ECMWF perturbations require the original feature names and wind semantics.")
    if (ui, vi) not in vectors:
        vectors.append((ui, vi))
    if named_features:
        # Each height keeps its own forecast magnitude. Never copy one height's
        # vector magnitude into a differently named speed field.
        speed_indices = named_speeds if kind == "nwp_speed" else []
    for b, identity in enumerate(identities):
        rng = _sample_rng(seed, identity)
        size = 1 if scenario.get("spatially_shared", True) else num_turbines
        eps = torch.as_tensor(rng.normal(0.0, sigma, size=size), dtype=raw.dtype, device=raw.device)
        for u_field, v_field in vectors:
            u, v = raw[b, :, u_field], raw[b, :, v_field]
            speed = torch.sqrt(u.square() + v.square())
            angle = torch.atan2(u, v)
            if kind == "nwp_speed":
                speed = (speed + eps).clamp_min(0)
            else:
                angle = angle + eps * (np.pi / 180.0)
            raw[b, :, u_field], raw[b, :, v_field] = speed * torch.sin(angle), speed * torch.cos(angle)
        speed = torch.hypot(raw[b, :, ui], raw[b, :, vi])
        angle = torch.atan2(raw[b, :, ui], raw[b, :, vi])
        sine, cosine = torch.sin(angle), torch.cos(angle)
        raw[b, :, -3], raw[b, :, -2], raw[b, :, -1] = speed, sine, cosine
        for idx in speed_indices:
            if named_features:
                raw[b, :, int(idx)] = (raw[b, :, int(idx)] + eps).clamp_min(0)
            else:
                raw[b, :, int(idx)] = speed
    updated_indices = sorted({*(index for pair in vectors for index in pair), n_features - 3, n_features - 2, n_features - 1,
                              *(int(idx) for idx in speed_indices)})
    normalized = (raw - mean) / std
    target_nwp[..., updated_indices] = normalized[..., updated_indices]
    return (power, wind, x_hist_nwp, target_nwp, target), p_mask, w_mask
