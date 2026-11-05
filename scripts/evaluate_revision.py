#!/usr/bin/env python3
"""Reuse completed revision runs for input faults, event strata, and paired inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pieef.core import calc_metrics
from pieef.data import build_data_bundle
from pieef.faults import apply_fault
from pieef.model_factory import build_main_model
from pieef.revision_data import require_verified_forecast_input
from pieef.revision_statistics import adjust_comparison_family, compare_prediction_groups, load_prediction


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _csv_set(value, cast=str):
    return None if value is None else {cast(x.strip()) for x in value.split(",") if x.strip()}


def _read_index(path):
    return pd.read_csv(path, keep_default_na=False,
                       dtype={"scenario_json": str, "capacity_json": str}).to_dict("records")


def _safe_name(value):
    name = str(value).replace("/", "_").replace("\\", "_").replace(":", "_")
    if name in {"", ".", ".."}:
        raise ValueError("An explicit nonempty output label is required")
    return name


def _run_record(directory):
    directory = _path(directory)
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    cfg = payload["config"]
    source = _path(payload.get("artifact_source", directory))
    model = payload.get("model", cfg["experiment"].get("model", "PI-EEF"))
    return dict(run_id=payload.get("run_id", directory.name), run_dir=directory,
                source_dir=source, config=cfg, model=model,
                seed=int(payload["seed"]), horizon=int(payload["horizon"]),
                data_version=str(cfg["data"]["data_version"]),
                group=cfg["experiment"].get("group", ""),
                variant=cfg["experiment"].get("variant", ""))


def _records(args):
    paths = [_path(p) for p in (args.run_dir or [])]
    if args.runs_root:
        paths.extend(p.parent for p in _path(args.runs_root).rglob("metrics.json"))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("No completed metrics.json runs found; specify --runs-root or --run-dir")
    models, horizons, seeds = _csv_set(args.models), _csv_set(args.horizons, int), _csv_set(args.seeds, int)
    result = []
    for path in paths:
        row = _run_record(path)
        if models is not None and row["model"] not in models:
            continue
        if horizons is not None and row["horizon"] not in horizons:
            continue
        if seeds is not None and row["seed"] not in seeds:
            continue
        if args.group is not None and row["group"] != args.group:
            continue
        result.append(row)
    if not result:
        raise ValueError("No completed runs match the requested model/horizon/seed/group")
    return result


def structured_scenarios():
    result = []
    for hours in (1, 4, 12, 24):
        for count in (1, 3, 6, 10):
            for channels in ("power", "both"):
                result.append(dict(name=f"missing_{channels}_{count}t_{hours}h", kind="missing",
                                   channels=channels, turbine_count=count, duration_hours=hours))
    for hours in (1, 4):
        for count in (1, 3):
            result.append(dict(name=f"frozen_both_{count}t_{hours}h", kind="frozen", channels="both",
                               turbine_count=count, duration_hours=hours))
    for level in (.5, 1., 2.):
        result.append(dict(name=f"wind_drift_{level:g}ms_4h", kind="drift", channels="wind",
                           drift_ms=level, duration_hours=4, turbine_count=1))
        result.append(dict(name=f"nwp_speed_{level:g}ms", kind="nwp_speed", sigma_ms=level))
    for degrees in (5, 10, 20):
        result.append(dict(name=f"nwp_direction_{degrees}deg", kind="nwp_direction", sigma_deg=degrees))
    return result


def _scenarios(args, bundle, cfg):
    if args.scenario_config:
        obj = yaml.safe_load(_path(args.scenario_config).read_text(encoding="utf-8"))
        scenarios = obj if isinstance(obj, list) else obj.get("faults", obj.get("scenarios"))
        if not isinstance(scenarios, list):
            raise ValueError("Scenario config must be a list, or contain faults/scenarios list")
    elif args.preset == "structured":
        scenarios = structured_scenarios()
    else:
        scenarios = []
    if args.only_scenarios:
        wanted = _csv_set(args.only_scenarios)
        found = {s.get("name") for s in scenarios} | {"normal"}
        if wanted - found:
            raise ValueError(f"Unknown scenario names: {sorted(wanted - found)}")
        scenarios = [s for s in scenarios if s.get("name") in wanted]
        include_normal = "normal" in wanted
    else:
        include_normal = True
    step = int(bundle.handler.protocol.get("step_minutes", 15))
    sequence = int(cfg["data"]["seq_len"])
    prepared = []
    names = set()
    if include_normal:
        prepared.append(("normal", None))
        names.add("normal")
    for original in scenarios:
        scenario = dict(original)
        name = _safe_name(scenario.pop("name", ""))
        if name in names:
            raise ValueError(f"Duplicate scenario name: {name}")
        names.add(name)
        if "step_minutes" in scenario and float(scenario["step_minutes"]) != step:
            raise ValueError("Fault sampling interval must equal the site's actual step_minutes")
        scenario["step_minutes"] = step
        if scenario.get("kind", "missing") in {"missing", "frozen", "drift"}:
            steps = scenario.get("duration_steps", float(scenario.get("duration_hours", 1)) * 60 / step)
            if not float(steps).is_integer() or not 1 <= steps <= sequence:
                raise ValueError(f"{name}: requested fault needs {steps} steps, trained history has {sequence}; use an adequately trained history window")
            if int(scenario.get("turbine_count", 1)) > bundle.num_turbines:
                raise ValueError(f"{name}: requested turbine count exceeds this site's actual turbines")
        prepared.append((name, scenario))
    if not prepared:
        raise ValueError("No requested evaluation scenarios")
    return prepared


def _bundle(record):
    cfg, source = record["config"], record["source_dir"]
    data = cfg["data"]
    protocol = data.get("revision_protocol")
    if protocol is None:
        raise ValueError("Formal evaluation requires a revision dataset with saved timestamps and observed targets")
    require_verified_forecast_input(protocol)
    if not (source / "preprocessing.npz").exists():
        raise FileNotFoundError(f"Missing saved preprocessing: {source / 'preprocessing.npz'}")
    saved = dict(np.load(source / "preprocessing.npz", allow_pickle=False))
    capacity = saved["capacity_mw"]
    if capacity.ndim == 0:
        capacity = float(capacity)
    bundle = build_data_bundle(
        _path(data["scada_path"]), _path(data["turbines_path"]),
        int(data["seq_len"]), record["horizon"], int(cfg["train"]["batch_size"]),
        float(data["split"][0]), float(data["split"][1]), capacity,
        revision_protocol=protocol,
    )
    h = bundle.handler
    actual = dict(nwp_mean=h.nwp_mean, nwp_std=h.nwp_std,
                  scada_wind_mean=h.scada_w_mean, scada_wind_std=h.scada_w_std,
                  capacity_mw=bundle.capacity_mw, coords=h.coords, turbine_ids=h.turbine_ids)
    for key, values in actual.items():
        if key not in saved or np.asarray(saved[key]).shape != np.asarray(values).shape:
            raise ValueError(f"Saved preprocessing mismatch: {key}")
        if key == "turbine_ids":
            matches = np.array_equal(saved[key].astype(str), np.asarray(values).astype(str))
        else:
            matches = np.allclose(saved[key], values, rtol=1e-6, atol=1e-7)
        if not matches:
            raise ValueError(f"Rebuilt data do not reproduce saved preprocessing: {key}")
    if cfg.get("model", {}).get("revin_missing_policy", "legacy") == "train_stats_20":
        from pieef.power_training_stats import ensure_training_power_statistics
        ensure_training_power_statistics(bundle, saved)
    return bundle


def _baseline_factory():
    spec = importlib.util.spec_from_file_location("revision_baseline_factory", ROOT / "scripts/train_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictor(record, bundle, device):
    name, source, cfg = record["model"], record["source_dir"], record["config"]
    if name == "Persistence":
        return None, "persistence", None
    if (source / "model.pkl").exists():
        import joblib
        return joblib.load(source / "model.pkl"), "tree", source / "model.pkl"
    weights = source / "weights_best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing validation-best model weights: {weights}")
    if name == "PI-EEF":
        model = build_main_model(bundle, cfg["data"]["seq_len"], cfg.get("model", {}), device)
        kind = "main"
    else:
        factory = _baseline_factory()
        turbines = _path(cfg["data"]["turbines_path"])
        meta = factory._load_turbine_meta(turbines, bundle.num_turbines)
        model = factory.build_model(name, int(cfg["data"]["seq_len"]), bundle=bundle,
                                    turbine_meta=meta, turbines_path=turbines).to(device)
        kind = "dl"
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    if name == "PI-EEF":
        from pieef.power_training_stats import check_model_training_statistics
        check_model_training_statistics(model, bundle)
    model.eval()
    return model, kind, weights


def _predict(model, kind, batch, power_mask, wind_mask, device):
    if kind == "persistence":
        return batch[0][:, :, -1, 0].numpy()
    if kind == "tree":
        p, w, hist, target, _ = batch
        bsz, turbines, sequence, _ = p.shape
        inputs = torch.cat([p, w, hist], dim=-1).reshape(bsz * turbines, sequence, -1)
        features = torch.cat([inputs.reshape(bsz * turbines, -1), target.reshape(bsz * turbines, -1)], dim=-1)
        return np.asarray(model.predict(features.numpy())).reshape(bsz, turbines)
    p, w, hist, target, _ = (x.to(device) for x in batch)
    if kind == "main":
        pm = power_mask.to(device) if bool(power_mask.any()) else None
        wm = wind_mask.to(device) if bool(wind_mask.any()) else None
        # Simulated training masks never invent a second sensor failure at evaluation.
        original_joint = model.joint_masking
        try:
            model.joint_masking = False
            _, features, _, _ = model.forward_stage1_reconstruct(p, w, hist, force_mask=pm, force_wind_mask=wm)
        finally:
            model.joint_masking = original_joint
        v, physics = model.forward_stage2_calibration(w, target)
        prediction = model.forward_stage3_forecast(features, v, physics, target,
                                                   hist_power=p, hist_power_mask=pm)
    else:
        prediction = model(p, x_w=w, x_n_hist=hist, x_nwp_target=target)
    return prediction.detach().cpu().numpy()


def _infer(model, kind, bundle, scenario, fault_seed, split, device):
    loader = getattr(bundle, f"{split}_loader")
    metadata = bundle.handler.metadata[split]
    predictions, truth = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            count = len(batch[0])
            if scenario is None:
                corrupted = batch
                pm, wm = torch.zeros_like(batch[0]), torch.zeros_like(batch[1])
            else:
                spec = dict(scenario, sample_ids=metadata["origin_timestamp"][offset:offset + count])
                if spec.get("kind") in {"nwp_speed", "nwp_direction"}:
                    spec.update(nwp_mean=bundle.handler.nwp_mean, nwp_std=bundle.handler.nwp_std)
                    spec.update(nwp_feature_columns=bundle.handler.feature_columns,
                                nwp_timing_rule=bundle.handler.protocol.get("nwp_timing_rule"))
                    spec.setdefault("u_index", bundle.handler.nwp_u_index)
                    spec.setdefault("v_index", bundle.handler.nwp_v_index)
                    spec.setdefault("speed_feature_indices", [7])
                corrupted, pm, wm = apply_fault(batch, spec, bundle.handler.scada_w_std,
                                                bundle.handler.coords, fault_seed)
            predictions.append(_predict(model, kind, corrupted, pm, wm, device))
            truth.append(batch[-1].numpy())
            offset += count
    if offset != len(metadata["timestamp"]):
        raise ValueError("Evaluation did not cover the complete indexed split")
    capacity = np.asarray(bundle.capacity_mw)
    return np.concatenate(truth, axis=0) * capacity, np.maximum(np.concatenate(predictions, axis=0) * capacity, 0)


def _save_predictions(path, record, bundle, truth, prediction, split, name, scenario, fault_seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {k: np.asarray(v) for k, v in bundle.handler.metadata[split].items()}
    fields = {k: v.astype(str) if v.dtype.kind == "O" else v for k, v in fields.items()}
    np.savez_compressed(
        path, y_true_mw=truth, y_pred_mw=prediction, turbine_ids=np.asarray(bundle.handler.turbine_ids, dtype=str),
        seed=record["seed"], horizon=record["horizon"], capacity_mw=bundle.capacity_mw,
        scenario=name, scenario_json=json.dumps(scenario, sort_keys=True), fault_seed=int(fault_seed), **fields,
    )


def infer_command(args):
    output = _path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "inference_metrics.csv"
    rows = _read_index(csv_path) if csv_path.exists() else []
    device = torch.device(args.device)
    for record in _records(args):
        if args.split == "train":
            raise ValueError("Training-loader shuffling cannot serve as indexed final evaluation")
        bundle = _bundle(record)
        scenarios = _scenarios(args, bundle, record["config"])
        model, kind, checkpoint = _predictor(record, bundle, device)
        checkpoint_mtime = checkpoint.stat().st_mtime_ns if checkpoint else 0
        for name, scenario in scenarios:
            destination = output / _safe_name(record["run_id"]) / name / f"{args.split}_predictions.npz"
            scenario_json = json.dumps(scenario, sort_keys=True)
            prior = [r for r in rows if r["run_id"] == record["run_id"] and r["scenario"] == name and r["split"] == args.split]
            if destination.exists() and prior and not args.overwrite:
                row = prior[-1]
                if row["scenario_json"] != scenario_json or int(row["fault_seed"]) != args.fault_seed or int(row["checkpoint_mtime_ns"]) != checkpoint_mtime:
                    raise ValueError("Saved inference corresponds to a different scenario or checkpoint; choose a new output or --overwrite")
                print(f"[reuse] {record['model']} h={record['horizon']} seed={record['seed']} {name}", flush=True)
                continue
            truth, prediction = _infer(model, kind, bundle, scenario, args.fault_seed, args.split, device)
            _save_predictions(destination, record, bundle, truth, prediction, args.split, name, scenario, args.fault_seed)
            nmae, nrmse, r2 = calc_metrics(truth, prediction, bundle.capacity_mw)
            row = dict(run_id=record["run_id"], source_dir=str(record["source_dir"]), model=record["model"],
                       variant=record["variant"], seed=record["seed"], horizon=record["horizon"],
                       data_version=record["data_version"], scenario=name, split=args.split,
                       scenario_json=scenario_json, fault_seed=args.fault_seed, checkpoint_mtime_ns=checkpoint_mtime,
                       prediction_path=str(destination), capacity_json=json.dumps(np.asarray(bundle.capacity_mw).tolist()),
                       nmae=nmae, nrmse=nrmse, r2=r2, n_time=len(truth),
                       n_valid_targets=int(np.isfinite(truth).sum()))
            rows = [r for r in rows if not (r["run_id"] == row["run_id"] and r["scenario"] == name and r["split"] == args.split)]
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[done] {record['model']} h={record['horizon']} seed={record['seed']} {name} NRMSE={nrmse:.6f}", flush=True)
        del model, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(csv_path)


def _prediction_entries(args):
    if args.index:
        rows = _read_index(_path(args.index))
        models, horizons, seeds = _csv_set(args.models), _csv_set(args.horizons, int), _csv_set(args.seeds, int)
        rows = [r for r in rows if (models is None or r["model"] in models)
                and (horizons is None or int(r["horizon"]) in horizons)
                and (seeds is None or int(r["seed"]) in seeds)]
        if args.scenario != "all":
            rows = [r for r in rows if r["scenario"] == args.scenario]
        return rows
    rows = []
    for record in _records(args):
        path = record["source_dir"] / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved test predictions: {path}")
        preprocessing = record["source_dir"] / "preprocessing.npz"
        if not preprocessing.exists():
            raise FileNotFoundError(f"Missing saved preprocessing: {preprocessing}")
        with np.load(preprocessing, allow_pickle=False) as data:
            capacity = data["capacity_mw"].tolist()
        rows.append(dict(run_id=record["run_id"], source_dir=str(record["source_dir"]), model=record["model"],
                         variant=record["variant"], seed=record["seed"], horizon=record["horizon"],
                         data_version=record["data_version"], scenario="normal", split="test",
                         prediction_path=str(path), capacity_json=json.dumps(capacity)))
    if args.scenario not in {"all", "normal"}:
        raise ValueError("Fault scenarios require an inference --index CSV")
    return rows


def events_command(args):
    rows = []
    for entry in _prediction_entries(args):
        with np.load(_path(entry["prediction_path"]), allow_pickle=False) as data:
            if "events" not in data or "target_valid" not in data:
                raise ValueError("Event stratification requires the saved event/target-valid arrays")
            truth, prediction = data["y_true_mw"], data["y_pred_mw"]
            events, valid = data["events"].astype(str), data["target_valid"].astype(bool)
        capacity = np.asarray(json.loads(entry["capacity_json"]))
        for label in sorted(set(events.reshape(-1))):
            mask = valid & (events == label)
            number = int(mask.sum())
            if number:
                values = calc_metrics(np.where(mask, truth, np.nan), prediction, capacity)
            else:
                values = (float("nan"),) * 3
            rows.append(dict(run_id=entry["run_id"], model=entry["model"], seed=entry["seed"],
                             horizon=entry["horizon"], data_version=entry["data_version"], scenario=entry["scenario"],
                             event_label=label, label_status="candidate" if label.endswith("candidate") else "rule_defined",
                             n_targets=number, n_time=int(mask.any(axis=1).sum()), nmae=values[0], nrmse=values[1], r2=values[2]))
    if not rows:
        raise ValueError("No prediction files matched the requested event scope")
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


def statistics_command(args):
    entries = _prediction_entries(args)
    baselines = _csv_set(args.baselines)
    expected_seeds = _csv_set(args.expected_seeds, int)
    groups = {}
    for row in entries:
        key = (row["data_version"], int(row["horizon"]), row["scenario"], row["split"])
        groups.setdefault(key, {}).setdefault(row["model"], []).append(row)
    if not groups:
        raise ValueError("No prediction groups to compare")
    expected_horizons = _csv_set(args.horizons, int)
    if expected_horizons is not None:
        found = {key[1] for key in groups}
        if expected_horizons - found:
            raise ValueError(f"Missing requested horizons: {sorted(expected_horizons - found)}")
    result = []
    for (version, horizon, scenario, split), models in sorted(groups.items()):
        if args.reference_model not in models:
            raise ValueError(f"No reference model for {version}, h={horizon}, {scenario}")
        needed = (set(models) - {args.reference_model}) if baselines is None else baselines
        if not needed or needed - models.keys():
            raise ValueError(f"Missing requested baselines: {sorted(needed - models.keys())}")
        reference = models[args.reference_model]
        capacity = np.asarray(json.loads(reference[0]["capacity_json"]))
        for name in {args.reference_model} | needed:
            runs = models[name]
            seeds = [int(r["seed"]) for r in runs]
            if len(set(seeds)) != len(seeds):
                raise ValueError("Multiple retained model variants share a seed/horizon; select one complete group")
            if expected_seeds is not None and set(seeds) != expected_seeds:
                raise ValueError(f"Incomplete seed set for {name}, h={horizon}: {sorted(seeds)}")
            for row in runs:
                if not np.array_equal(capacity, np.asarray(json.loads(row["capacity_json"]))):
                    raise ValueError("Compared runs use different nominal capacities")
                saved = load_prediction(_path(row["prediction_path"]))
                if saved.seed != int(row["seed"]) or saved.horizon != horizon:
                    raise ValueError("Prediction file metadata disagree with the selected run index")
        rows = compare_prediction_groups(
            [_path(r["prediction_path"]) for r in reference],
            {name: [_path(r["prediction_path"]) for r in models[name]] for name in sorted(needed)},
            capacity_mw=capacity, n_bootstrap=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed, block_length=args.block_length, hac_lag=args.hac_lag,
        )
        for row in rows:
            row.update(data_version=version, scenario=scenario, split=split, reference_model=args.reference_model)
        result.extend(rows)
    result = adjust_comparison_family(result)
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(output, index=False)
    print(f"{output}; BH family includes all {len(result)} retained comparisons")


def _source_arguments(parser):
    parser.add_argument("--runs-root")
    parser.add_argument("--run-dir", action="append")
    parser.add_argument("--models")
    parser.add_argument("--horizons")
    parser.add_argument("--seeds")
    parser.add_argument("--group")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="Reuse best checkpoints for normal and actual input-fault evaluation")
    _source_arguments(infer)
    infer.add_argument("--output", required=True)
    infer.add_argument("--preset", choices=["normal", "structured"], default="normal")
    infer.add_argument("--scenario-config")
    infer.add_argument("--only-scenarios")
    infer.add_argument("--fault-seed", type=int, default=2026)
    infer.add_argument("--split", choices=["val", "test"], default="test")
    infer.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=infer_command)
    events = commands.add_parser("events", help="Summarize measured targets by the saved rule/candidate labels")
    _source_arguments(events)
    events.add_argument("--index")
    events.add_argument("--scenario", default="normal")
    events.add_argument("--output", required=True)
    events.set_defaults(handler=events_command)
    statistics = commands.add_parser("statistics", help="Paired normalized-loss DM/HAC/HLN, block intervals, and one global BH family")
    _source_arguments(statistics)
    statistics.add_argument("--index")
    statistics.add_argument("--scenario", default="normal")
    statistics.add_argument("--reference-model", default="PI-EEF")
    statistics.add_argument("--baselines")
    statistics.add_argument("--expected-seeds")
    statistics.add_argument("--bootstrap-replicates", type=int, default=2000)
    statistics.add_argument("--bootstrap-seed", type=int, default=13)
    statistics.add_argument("--block-length", type=int)
    statistics.add_argument("--hac-lag", type=int)
    statistics.add_argument("--output", required=True)
    statistics.set_defaults(handler=statistics_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
