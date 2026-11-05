#!/usr/bin/env python3
"""Export real stage curves and three fixed-rule attention cases from main h16 seed42."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pieef.model_factory import build_main_model
from pieef.revision_dispatch import completed_run
from evaluate_revision import _bundle, _run_record

CHECKPOINTS = (("after_stage1", "weights_stage1.pt"), ("after_stage2", "weights_stage2.pt"),
               ("validation_best", "weights_best.pt"))
RULES = {
    "normal": "Earliest test target where every turbine has the existing normal event label and a valid measured target.",
    "low_power_candidate": "Most valid low_power_candidate turbines at a test target; earliest tie. Existing label: wind >5 m/s and measured power <0.01 MW, with power >-0.02 MW. Candidate only; controller curtailment is not established.",
    "high_wind_variability_proxy": "Among all-normal valid test targets with finite raw wind in the preceding 96-point history ending at the forecast origin, maximize the median across turbines of population wind-speed standard deviation (m/s); earliest tie. A 24-hour variability proxy, not measured turbulence intensity.",
}


def select_samples(dataset, metadata, raw_wind, seq_len=96, horizon=16):
    """Use input exposure/event rules only; predictions cannot affect selection."""
    indices = np.asarray(dataset.indices, dtype=int)
    events = np.asarray(metadata["events"]).astype(str)
    valid = np.asarray(metadata["target_valid"], dtype=bool)
    raw_wind = np.asarray(raw_wind, dtype=float)
    if events.shape != valid.shape or len(events) != len(indices) or raw_wind.ndim != 2 or raw_wind.shape[1] != events.shape[1]:
        raise ValueError("Event, measured target, wind and sample indices must match")
    if not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("Test indices must be nonempty and chronological")
    normal = (events == "normal").all(axis=1) & valid.all(axis=1)
    low_counts = ((events == "low_power_candidate") & valid).sum(axis=1)
    low_counts[~valid.all(axis=1)] = -1
    if not normal.any() or low_counts.max() <= 0:
        raise ValueError("The actual test set does not contain all three requested regimes")
    scores = np.full(len(indices), np.nan)
    for item in np.flatnonzero(normal):
        target = indices[item]
        start, stop = target - horizon - seq_len + 1, target - horizon + 1
        if start < 0 or stop > len(raw_wind):
            raise ValueError("Selected history extends beyond the actual input data")
        window = raw_wind[start:stop]
        if np.isfinite(window).all():
            scores[item] = np.median(np.std(window, axis=0, ddof=0))
    if not np.isfinite(scores).any():
        raise ValueError("No normal target has a finite raw-wind variability history")
    chosen = (int(np.flatnonzero(normal)[0]), int(np.argmax(low_counts)), int(np.nanargmax(scores)))
    records = []
    for regime, item in zip(RULES, chosen):
        target = int(indices[item])
        start, end = target - horizon - seq_len + 1, target - horizon
        window = raw_wind[start:end + 1]
        std = float(np.median(np.std(window, axis=0, ddof=0))) if np.isfinite(window).all() else None
        records.append(dict(regime=regime, dataset_item=item, target_row=target,
                            history_start_row=start, history_end_row=end,
                            timestamp=str(metadata["timestamp"][item]),
                            origin_timestamp=str(metadata["origin_timestamp"][item]),
                            n_low_power_candidates=int(((events[item] == "low_power_candidate") & valid[item]).sum()),
                            median_history_wind_std_ms=std))
    return records


def stage_curve(model, device):
    """Nominal per-unit curve versus calibrated wind, at reference air density."""
    if getattr(model.dcn, "power_curve", None) is None:
        raise ValueError("The actual model has no physical power-curve module")
    grid = torch.arange(301, dtype=torch.float32, device=device) / 10
    with torch.enable_grad():
        wind = grid[:, None].repeat(1, model.num_turbines).requires_grad_(True)
        power = model.dcn.power_at(wind, torch.ones_like(wind)).squeeze(-1)
        derivative = torch.autograd.grad(power.sum(), wind)[0]
    if power.shape != wind.shape or not bool(torch.isfinite(power).all() and torch.isfinite(derivative).all()):
        raise ValueError("The real power curve or its local derivative is invalid")
    return grid.cpu().numpy(), power.detach().cpu().numpy(), derivative.detach().cpu().numpy()


def attention_case(model, dataset, sample, device):
    batch = [value.unsqueeze(0).to(device) for value in dataset[sample["dataset_item"]]]
    p, w, history, target, truth = batch
    with torch.no_grad():
        _, features, _, _ = model.forward_stage1_reconstruct(p, w, history)
        calibrated, physical = model.forward_stage2_calibration(w, target)
        predicted = model.forward_stage3_forecast(features, calibrated, physical, target,
                                                  hist_power=p)
    attention = getattr(model.graph, "last_attn_weights", None)
    if attention is None:
        raise ValueError("The real graph forward did not expose attention weights")
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    attention = np.asarray(attention)
    count = model.num_turbines
    if attention.shape != (1, count, count) or not np.isfinite(attention).all() or (attention < 0).any() or not np.allclose(attention.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Actual attention must be finite [1,turbines,turbines] probabilities")
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("Actual prediction contains non-finite values")
    return {"attention": attention[0].copy(), "prediction_pu": predicted[0].cpu().numpy(),
            "target_pu": truth[0].cpu().numpy(), "calibrated_wind_ms": calibrated[0].cpu().numpy(),
            "physical_power_pu": physical[0].cpu().numpy(),
            "x_hist_p_pu": p[0].cpu().numpy(), "x_hist_w_standardized": w[0].cpu().numpy(),
            "x_hist_nwp_standardized": history[0].cpu().numpy(),
            "x_nwp_target_standardized": target[0].cpu().numpy()}


def validate_run(directory):
    record = _run_record(directory)
    cfg = record["config"]
    exp = cfg["experiment"]
    if record["model"] != "PI-EEF" or record["group"] != "main" or record["horizon"] != 16 or record["seed"] != 42:
        raise ValueError("Supply the actual formal main PI-EEF h16 seed42 run")
    if exp.get("finalized") is not True or int(cfg["data"]["seq_len"]) != 96 or int(cfg["data"]["revision_protocol"]["step_minutes"]) != 15:
        raise ValueError("Interpretation requires the frozen model and its declared 24-hour history")
    actual = completed_run(cfg)
    if actual is None or actual.resolve() != (record["run_dir"] / "metrics.json").resolve():
        raise ValueError("The requested run is not the strict matching completed formal artifact")
    if record["source_dir"].resolve() != record["run_dir"].resolve():
        raise ValueError("Use the true main run rather than a control alias")
    for _, name in CHECKPOINTS:
        if not (record["source_dir"] / name).is_file():
            raise ValueError(f"Missing actual checkpoint: {name}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no CPU substitution")
    record = validate_run(args.run_dir)
    bundle = _bundle(record)  # Verifies real dataset and saved train-only preprocessing.
    cfg, ids = record["config"], np.asarray(bundle.handler.turbine_ids).astype(str)
    frame = pd.read_csv(bundle.handler.source_data_path)
    wind = frame[[f"turbine_ws{i + 1}" for i in range(bundle.num_turbines)]].to_numpy(dtype=float)
    dataset, metadata = bundle.test_loader.dataset, bundle.handler.metadata["test"]
    samples = select_samples(dataset, metadata, wind)
    source_times = pd.DatetimeIndex(pd.to_datetime(frame["Timestamp"]))
    if source_times.tz is None:
        source_times = source_times.tz_localize(cfg["data"]["revision_protocol"]["timezone"])
    for sample in samples:
        sample["history_start_timestamp"] = str(source_times[sample["history_start_row"]])
        sample["history_end_timestamp"] = str(source_times[sample["history_end_row"]])
    model = build_main_model(bundle, 96, cfg.get("model", {}), device)
    curves = []
    for label, filename in CHECKPOINTS:
        model.load_state_dict(torch.load(record["source_dir"] / filename, map_location=device, weights_only=True), strict=True)
        model.eval()
        speeds, power, gradient = stage_curve(model, device)
        capacity = np.broadcast_to(np.asarray(bundle.capacity_mw), (bundle.num_turbines,))
        cut_in = np.broadcast_to(model.dcn.cut_in_speed if model.dcn.cut_in_speeds is None else model.dcn.cut_in_speeds.cpu().numpy(), capacity.shape)
        cut_out = np.broadcast_to(np.inf if model.dcn.cut_out_speed is None else model.dcn.cut_out_speed, capacity.shape) if model.dcn.cut_out_speeds is None else model.dcn.cut_out_speeds.cpu().numpy()
        for turbine, turbine_id in enumerate(ids):
            for point, speed in enumerate(speeds):
                region = "below_cut_in" if speed < cut_in[turbine] else "cut_out" if speed >= cut_out[turbine] else "operating"
                curves.append(dict(checkpoint=label, turbine_id=turbine_id, calibrated_wind_ms=float(speed),
                                   power_pu=float(power[point, turbine]), power_mw=float(power[point, turbine] * capacity[turbine]),
                                   local_derivative_pu_per_ms=float(gradient[point, turbine]), region=region))
    # The final iteration loaded validation-best weights. All three cases use it.
    cases = [attention_case(model, dataset, sample, device) for sample in samples]
    arrays = {name: np.stack([case[name] for case in cases]) for name in cases[0]}
    capacity = np.broadcast_to(np.asarray(bundle.capacity_mw), (bundle.num_turbines,))
    arrays["prediction_mw"] = arrays["prediction_pu"] * capacity
    arrays["target_mw"] = arrays["target_pu"] * capacity
    arrays.update(turbine_ids=ids, coords_m=np.asarray(bundle.handler.coords),
                  rel_coords_m=bundle.rel_coords_tensor.cpu().numpy(),
                  normalized_dist_matrix=bundle.dist_tensor.cpu().numpy(),
                  regime=np.asarray([sample["regime"] for sample in samples]),
                  timestamp=np.asarray([sample["timestamp"] for sample in samples]),
                  origin_timestamp=np.asarray([sample["origin_timestamp"] for sample in samples]),
                  target_valid=np.stack([metadata["target_valid"][sample["dataset_item"]] for sample in samples]),
                  event_labels=np.stack([metadata["events"][sample["dataset_item"]] for sample in samples]),
                  capacity_mw=np.broadcast_to(np.asarray(bundle.capacity_mw), (bundle.num_turbines,)),
                  nwp_mean=bundle.handler.nwp_mean, nwp_std=bundle.handler.nwp_std,
                  scada_wind_mean=np.asarray(bundle.handler.scada_w_mean), scada_wind_std=np.asarray(bundle.handler.scada_w_std))
    edges = [dict(regime=sample["regime"], timestamp=sample["timestamp"], query_turbine=ids[i], source_turbine=ids[j], weight=float(case["attention"][i, j]))
             for sample, case in zip(samples, cases) for i in range(len(ids)) for j in range(len(ids))]
    parameters = dict(run_dir=str(record["run_dir"]), config=deepcopy(cfg), device=str(device),
                      actual_gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      checkpoints={label: str(record["source_dir"] / name) for label, name in CHECKPOINTS},
                      curve=dict(calibrated_wind_range_ms=[0, 30], step_ms=0.1, density_ratio=1,
                                 reference_density_kg_m3=float(model.reference_density),
                                 meaning="Actual dcn.power_at; per-unit nominal power versus calibrated wind at reference air density. Stage1 physical curve remains its initialization because pretraining trains the reconstruction branch. Local derivatives exclude interpretation of cut-in/cut-out discontinuities."),
                      attention=dict(layer="last graph layer", edge_direction="source j -> query i for A[i,j]",
                                     sample_rules=RULES, history_points=96, history_step_minutes=15,
                                     direction_wind_height_m=bundle.handler.nwp_derived_wind_height_m,
                                     target_speed_column=bundle.handler.nwp_target_speed_column),
                      samples=samples)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(args.output / "stage_curves.csv", index=False)
    pd.DataFrame(edges).to_csv(args.output / "attention_edges.csv", index=False)
    pd.DataFrame(samples).to_csv(args.output / "attention_samples.csv", index=False)
    np.savez_compressed(args.output / "attention_arrays.npz", **arrays)
    (args.output / "interpretation_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Exported three real stage curves and three rule-selected attention cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
