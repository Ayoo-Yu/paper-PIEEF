"""Independent, fail-closed h16 SCADA-fault recomputation; never trains a model.

The old delivery is read-only.  A fresh normal inference from the SAME h16
bundle must reproduce the original indexed normal prediction before any fault
is evaluated.  Source/data/checkpoint fingerprints are fixed by job_manifest.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SEEDS = (13, 21, 42, 87, 3407)
HORIZON = 16
FAULT_SEED = 2026
ATOL_PU = 1e-5
RTOL = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_horizon(meta, saved_horizon, expected_seed):
    """No default horizon, config-name parsing, or output-label correction."""
    cfg = meta["config"]
    values = (meta.get("horizon"), cfg["experiment"].get("horizon"), saved_horizon)
    if any(type(value) is not int or value != HORIZON for value in values):
        raise ValueError(f"Source metrics/config/indexed prediction must all be h16: {values}")
    seeds = cfg["experiment"].get("seeds")
    if meta.get("seed") != expected_seed or seeds != [expected_seed]:
        raise ValueError("Source seed differs from the frozen single-seed configuration")
    if meta.get("status") != "completed" or meta.get("evaluation_split") != "test":
        raise ValueError("Source must be a real completed TEST run")
    expected_id = f"energy_r3s2_main_PI-EEF_h16_seed{expected_seed}"
    if cfg["tracking"]["config_id"] != expected_id:
        raise ValueError("Source is not the frozen VF h16 main run")
    if cfg["tracking"].get("implementation_id") != "energy-r3-vf-F2F3F6":
        raise ValueError("Source is not the actual VF implementation")
    if cfg["experiment"].get("model") != "PI-EEF":
        raise ValueError("Only the actual PI-EEF source is permitted")
    return HORIZON


def verify_inventory(entries, root):
    for item in entries:
        path = Path(root) / item["relative_path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked immutable input: {path}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Immutable input differs: {path}")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def same_indices(expected, actual, *, count=None):
    required = ("timestamp", "origin_timestamp", "turbine_ids", "target_valid", "events")
    for key in required:
        if key not in expected or key not in actual:
            raise ValueError(f"Indexed inference lacks {key}")
        left = expected[key]
        if count is not None and key != "turbine_ids":
            left = left[:count]
        if left.shape != actual[key].shape or not np.array_equal(left, actual[key]):
            raise ValueError(f"Target/origin/turbine/valid/event index differs: {key}")


def check_h16_grid(fields):
    import pandas as pd
    targets = pd.DatetimeIndex(pd.to_datetime(fields["timestamp"], utc=True)).tz_localize(None).to_numpy()
    origins = pd.DatetimeIndex(pd.to_datetime(fields["origin_timestamp"], utc=True)).tz_localize(None).to_numpy()
    if np.isnat(targets).any() or np.isnat(origins).any():
        raise ValueError("Invalid indexed time")
    if not np.all(targets - origins == np.timedelta64(16 * 15, "m")):
        raise ValueError("Actual target-origin difference is not 240 minutes")
    if len(np.unique(targets)) != len(targets) or np.any(np.diff(targets) <= np.timedelta64(0, "ns")):
        raise ValueError("Duplicate or unordered target times")
    if len(np.unique(fields["turbine_ids"])) != len(fields["turbine_ids"]):
        raise ValueError("Duplicate turbine IDs")


def normal_equivalence(saved, actual, capacity, *, count=None):
    same_indices(saved, actual, count=count)
    check_h16_grid(actual)
    valid = actual["target_valid"]
    if valid.dtype.kind != "b" or not valid.any(axis=1).all():
        raise ValueError("Invalid target mask")
    old_truth = saved["y_true_mw"] if count is None else saved["y_true_mw"][:count]
    old_pred = saved["y_pred_mw"] if count is None else saved["y_pred_mw"][:count]
    truth, pred = actual["y_true_mw"], actual["y_pred_mw"]
    if not np.array_equal(np.isfinite(old_truth), valid) or not np.array_equal(np.isfinite(truth), valid):
        raise ValueError("Inference altered measured target validity")
    if not np.array_equal(old_truth[valid], truth[valid]):
        raise ValueError("Inference did not preserve the exact measured truth")
    if not np.isfinite(pred[valid]).all():
        raise ValueError("Non-finite normal prediction")
    cap = np.asarray(capacity, dtype=np.float64)
    if np.any(~np.isfinite(cap)) or np.any(cap <= 0):
        raise ValueError("Invalid actual capacity")
    # A priori FP32 cross-device tolerance, in per-unit capacity.  It is not
    # fitted to the new results and must never be increased automatically.
    normalized_delta = (pred - old_pred) / cap
    allowed = ATOL_PU + RTOL * np.abs(old_pred / cap)
    if np.any(np.abs(normalized_delta[valid]) > allowed[valid]):
        raise ValueError("Fresh normal inference does not reproduce the saved h16 prediction")
    return {"status": "PASS", "count_valid": int(valid.sum()),
            "atol_per_unit_capacity": ATOL_PU, "rtol": RTOL,
            "max_abs_delta_mw": float(np.max(np.abs(pred[valid] - old_pred[valid]))),
            "max_abs_delta_per_unit_capacity": float(np.max(np.abs(normalized_delta[valid]))),
            "truth_exact": True, "indices_exact": True,
            "tolerance_basis": "Predeclared tight FP32 numerical reproduction threshold (1e-5 pu + 1e-5 relative); no automatic relaxation, not a scientific effect tolerance."}


def index_fields(bundle, count=None):
    meta = bundle.handler.metadata["test"]
    fields = {}
    for key, values in meta.items():
        values = np.asarray(values if count is None else values[:count])
        fields[key] = values.astype(str) if values.dtype.kind == "O" else values
    fields["turbine_ids"] = np.asarray(bundle.handler.turbine_ids, dtype=str)
    return fields


def measured_metrics(truth, prediction, valid, capacity):
    # Match original production capacity-normalized definitions, with all
    # original valid measured targets retained rather than finite-pred filtering.
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Non-finite prediction on a valid measured target")
    err = (prediction - truth) / np.asarray(capacity)
    values = err[valid].astype(np.float64)
    truth_valid, pred_valid = truth[valid].astype(np.float64), prediction[valid].astype(np.float64)
    denominator = np.square(truth_valid - truth_valid.mean()).sum()
    return {"nrmse": float(np.sqrt(np.square(values).mean())),
            "nmae": float(np.abs(values).mean()),
            "r2": float(1 - np.square(pred_valid - truth_valid).sum() / denominator),
            "n": int(valid.sum())}


def save_prediction(path, fields, truth, pred, *, seed, scenario, capacity, source, scenario_spec=None):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **fields, y_true_mw=truth, y_pred_mw=pred,
                                seed=seed, horizon=HORIZON, capacity_mw=np.asarray(capacity),
                                scenario=scenario, fault_seed=FAULT_SEED,
                                scenario_json=json.dumps(scenario_spec, sort_keys=True, separators=(",", ":")),
                                source_config_id=source["config_id"], source_run_id=source["run_id"],
                                checkpoint_sha256=source["checkpoint_sha256"],
                                source_manifest_sha256=source["source_manifest_sha256"])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_runtime(args, manifest, seed):
    import torch
    torch.set_num_threads(2)
    project = args.project_root.resolve()
    verify_inventory(manifest["project_files"], project)
    verify_inventory(manifest["data_files"], args.data_root)
    entry = next(row for row in manifest["sources"] if row["seed"] == seed)
    directory = ((args.runs_root / entry["portable_run_dir"]) if args.runs_root
                 else Path(entry["original_run_directory"]))
    verify_inventory(entry["files"], directory)
    saved = load_npz(directory / "predictions.npz")
    meta = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    horizon = resolve_horizon(meta, int(saved["horizon"].item()), seed)
    if int(saved["seed"].item()) != seed:
        raise ValueError("Saved prediction seed differs")
    if any(name == "pieef" or name.startswith("pieef.") for name in sys.modules):
        raise ValueError("Run in a fresh process; an unverified pieef module is already imported")
    sys.path.insert(0, str(project / "src"))
    spec = importlib.util.spec_from_file_location("verified_h16_evaluation", project / "scripts/evaluate_revision.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    # Path rebasing changes only file locations, never data or scientific fields.
    cfg = deepcopy(meta["config"])
    cfg["data"]["scada_path"] = str(args.data_root.resolve() / "scada_ecmwf_15min.csv")
    cfg["data"]["turbines_path"] = str(args.data_root.resolve() / "turbines.csv")
    if cfg["model"].get("manufacturer_curve_path"):
        cfg["model"]["manufacturer_curve_path"] = str(args.data_root.resolve() / "manufacturer_curve.csv")
    record = dict(source_dir=directory, config=cfg, model="PI-EEF", seed=seed, horizon=horizon)
    bundle = evaluation._bundle(record)
    fields = index_fields(bundle)
    same_indices(saved, fields)
    check_h16_grid(fields)
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as pre:
        if not np.array_equal(pre["capacity_mw"], np.asarray(bundle.capacity_mw)):
            raise ValueError("Actual capacity differs from original preprocessing")
    model, kind, _ = evaluation._predictor(record, bundle, torch.device(args.device))
    if not all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise ValueError("Non-finite real checkpoint")
    source = dict(config_id=meta["config"]["tracking"]["config_id"], run_id=directory.name,
                  checkpoint_sha256=sha256(directory / "weights_best.pt"),
                  source_manifest_sha256=sha256(args.manifest), seed=seed,
                  original_config_sha256=entry["original_config_sha256"],
                  path_rebasing_only=True)
    return evaluation, bundle, model, kind, saved, fields, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--phase", choices=("cpu-gate", "cuda-gate", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.phase == "cpu-gate" and args.device != "cpu":
        parser.error("The preparatory gate must be CPU only")
    if args.phase == "cuda-gate" and args.device != "cuda":
        parser.error("The CUDA resource gate must explicitly select CUDA")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["horizon"] != HORIZON or manifest["fault_seed"] != FAULT_SEED:
        raise ValueError("Frozen h16/fault protocol changed")
    destination = args.output_root / f"seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    completion = destination / "completed.json"
    if args.phase == "run" and completion.exists():
        raise ValueError("A completed repair already exists; never overwrite or silently rerun it")
    started = datetime.now(timezone.utc).isoformat()
    if args.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    evaluation, bundle, model, kind, saved, fields, source = load_runtime(args, manifest, args.seed)
    import torch
    device = torch.device(args.device)
    if args.phase in ("cpu-gate", "cuda-gate"):
        batch = next(iter(bundle.test_loader))
        with torch.no_grad():
            pred = evaluation._predict(model, kind, batch, torch.zeros_like(batch[0]), torch.zeros_like(batch[1]), device)
        capacity = np.asarray(bundle.capacity_mw)
        pred = np.maximum(pred * capacity, 0)
        truth = batch[-1].numpy() * capacity
        actual = {**index_fields(bundle, len(truth)), "y_true_mw": truth, "y_pred_mw": pred}
        gate = normal_equivalence(saved, actual, capacity, count=len(truth))
        from pieef.revision_fault_jobs import scada_scenarios
        matrix = scada_scenarios("structured_39")
        if len(matrix) != 39 or len({row["name"] for row in matrix}) != 39:
            raise ValueError("Incomplete original SCADA scenario matrix")
        # Exercise the actual CPU corruption operator on the real full batch,
        # without evaluating any fault prediction during this preparatory gate.
        originals = tuple(value.clone() for value in batch)
        for original in matrix:
            spec = dict(original)
            spec.pop("name")
            spec["sample_ids"] = bundle.handler.metadata["test"]["origin_timestamp"][:len(truth)]
            corrupted, _, _ = evaluation.apply_fault(batch, spec, bundle.handler.scada_w_std,
                                                     bundle.handler.coords, FAULT_SEED)
            for index in (3, 4):
                if not torch.allclose(corrupted[index], batch[index], atol=0, rtol=0, equal_nan=True):
                    raise ValueError("SCADA fault altered target NWP or target labels")
            if any(not torch.allclose(before, after, atol=0, rtol=0, equal_nan=True)
                   for before, after in zip(originals, batch)):
                raise ValueError("Fault operator mutated the immutable input batch")
        if args.device == "cuda":
            torch.cuda.synchronize()
        gate.update(seed=args.seed, horizon=HORIZON, phase=f"{args.device}_real_batch_gate", source=source,
                    runner_sha256=sha256(Path(__file__)),
                    batch_shape=list(batch[0].shape), device=args.device, started_utc=started,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=="cuda" else None,
                    real_batch_all39_fault_operators_tested=True,
                    target_nwp_labels_original_inputs_unchanged=True,
                    full_normal_equivalence_required_before_faults=True,
                    formal_fault_inference_started=False)
        atomic_json(destination / f"{args.device}_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False))
        return
    from pieef.revision_fault_jobs import scada_scenarios
    scenarios = scada_scenarios("structured_39")
    if len(scenarios) != 39 or len({row["name"] for row in scenarios}) != 39:
        raise ValueError("The original fixed 39-scenario matrix is incomplete")
    capacity = np.asarray(bundle.capacity_mw)
    t0 = time.perf_counter()
    truth, pred = evaluation._infer(model, kind, bundle, None, FAULT_SEED, "test", device)
    actual = {**fields, "y_true_mw": truth, "y_pred_mw": pred}
    gate = normal_equivalence(saved, actual, capacity)
    atomic_json(destination / "normal_equivalence.json", {**gate, "source": source})
    save_prediction(destination / "normal.npz", fields, truth, pred, seed=args.seed,
                    scenario="normal", capacity=capacity, source=source)
    normal = measured_metrics(truth, pred, fields["target_valid"], capacity)
    rows = [dict(scenario="normal", **normal, seconds=time.perf_counter()-t0)]
    for original in scenarios:
        scenario = dict(original)
        name = scenario.pop("name")
        t1 = time.perf_counter()
        fault_truth, fault_pred = evaluation._infer(model, kind, bundle, scenario, FAULT_SEED, "test", device)
        # Same in-memory bundle and immutable test loader for every scenario;
        # also assert exact truth to prevent a hidden window/horizon mismatch.
        if not np.array_equal(truth, fault_truth, equal_nan=True):
            raise ValueError("Fault inference changed measured targets")
        metrics = measured_metrics(fault_truth, fault_pred, fields["target_valid"], capacity)
        save_prediction(destination / f"{name}.npz", fields, fault_truth, fault_pred,
                        seed=args.seed, scenario=name, capacity=capacity, source=source,
                        scenario_spec=scenario)
        rows.append(dict(scenario=name, scenario_json=scenario, **metrics,
                         nrmse_increase=(metrics["nrmse"]-normal["nrmse"])/normal["nrmse"],
                         nmae_increase=(metrics["nmae"]-normal["nmae"])/normal["nmae"],
                         seconds=time.perf_counter()-t1))
        print(f"[seed{args.seed} h16] {len(rows)-1}/39 {name}", flush=True)
    outputs = [destination / "normal_equivalence.json", destination / "normal.npz"]
    outputs += [destination / f"{row['name']}.npz" for row in scenarios]
    result = dict(version=1, status="completed", horizon=HORIZON, horizon_minutes=240,
                  seed=args.seed, evaluation_split="test", scenario_scope="structured_39", fault_seed=FAULT_SEED,
                  source=source, source_directory_original=next(r for r in manifest["sources"] if r["seed"]==args.seed)["original_run_directory"],
                  runner_sha256=sha256(Path(__file__)), job_manifest_sha256=sha256(args.manifest),
                  scientific_source_sha256=manifest["scientific_source_manifest_sha256"],
                  same_bundle_for_normal_and_faults=True, normal_reinferred=True,
                  full_normal_equivalence=gate, rows=rows, started_utc=started,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), process_command=sys.argv, device=args.device,
                  runtime={"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0) if device.type=="cuda" else None},
                  elapsed_full_inference_seconds=time.perf_counter()-t0,
                  outputs=[dict(path=p.name, bytes=p.stat().st_size, sha256=sha256(p)) for p in outputs],
                  old_h16_fault_outputs_reused=False, trained=False)
    atomic_json(completion, result)
    print(f"Completed repaired h16 seed{args.seed}: {completion}", flush=True)


if __name__ == "__main__":
    main()
