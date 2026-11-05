#!/usr/bin/env python3
"""Measure Table 6 inference on cloud 01 using completed main h16/seed42 weights."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pieef.artifacts import write_json_atomic
from pieef.baseline_protocol import validate_independent_baseline
from pieef.data import build_data_bundle
from pieef.model_factory import build_main_model
from pieef.revision_data import require_verified_forecast_input
from pieef.revision_dispatch import completed_run
from pieef.revision_paths import local_input_path
from pieef.revision_plan import training_trials
from pieef.revision_statistics import load_prediction
from pieef.training_timing import gpu_runtime

MODELS = ("PI-EEF", "GRU", "iTransformer", "PatchTST", "TimeMixer")
WORKER = "recorded-worker"


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def input_path(value, cfg):
    return Path(local_input_path(value, cfg.get("tracking", {}).get("execution_project_root")))


def formal_record(directory, final_config=None):
    directory = project_path(directory)
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    cfg = payload["config"]
    exp = cfg["experiment"]
    name = payload.get("model", exp["model"])
    if (name not in MODELS or exp.get("group") != "main" or
            payload.get("seed") != 42 or payload.get("horizon") != 16 or
            exp.get("seeds") != [42] or exp.get("horizon") != 16):
        raise ValueError("Table 6 requires one completed main h16/seed42 run of a specified neural model")
    if exp.get("variant") != name or (not exp.get("finalized") and not validate_independent_baseline(cfg)):
        raise ValueError("Table 6 requires the actual frozen main planner model label")
    if any(cfg["train"].get(key) is not None for key in ("max_batches_per_epoch", "max_test_batches")):
        raise ValueError("Partial-data checks are not formal efficiency sources")
    if payload.get("status") != "completed" or payload.get("evaluation_split") != "test":
        raise ValueError("Efficiency sources require completed formal test predictions")
    if final_config is None:
        final_config = yaml.safe_load((ROOT / "artifacts/energy_revision/final_model.yaml").read_text(encoding="utf-8"))
    if not isinstance(final_config, dict) or final_config.get("experiment", {}).get("finalized") is not True:
        raise ValueError("Formal efficiency sources require the actually finalized global model configuration")
    planned = [value for value in training_trials(final_config, "main", directory.parent.parent)
               if value["experiment"]["model"] == name and value["experiment"]["horizon"] == 16
               and value["experiment"]["seeds"] == [42]]
    if len(planned) != 1:
        raise ValueError("The frozen main planner does not identify exactly one requested efficiency source")
    expected = planned[0]
    verified = completed_run(expected)
    if verified is None or verified.resolve() != (directory / "metrics.json").resolve():
        raise ValueError("The supplied directory does not strictly match the actual finalized main scientific configuration")
    source = project_path(payload.get("artifact_source", str(directory)))
    return dict(run_dir=directory, source_dir=source, payload=payload, config=cfg, model=name)


def rebuild_bundle(record):
    cfg, source = record["config"], record["source_dir"]
    data = cfg["data"]
    protocol = data.get("revision_protocol")
    require_verified_forecast_input(protocol)
    with np.load(source / "preprocessing.npz", allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    capacity = saved["capacity_mw"]
    if capacity.ndim == 0:
        capacity = float(capacity)
    bundle = build_data_bundle(
        input_path(data["scada_path"], cfg), input_path(data["turbines_path"], cfg),
        int(data["seq_len"]), 16, int(cfg["train"]["batch_size"]),
        float(data["split"][0]), float(data["split"][1]), capacity,
        revision_protocol=protocol,
    )
    handler = bundle.handler
    actual = dict(nwp_mean=handler.nwp_mean, nwp_std=handler.nwp_std,
                  scada_wind_mean=handler.scada_w_mean, scada_wind_std=handler.scada_w_std,
                  capacity_mw=bundle.capacity_mw, coords=handler.coords, turbine_ids=handler.turbine_ids)
    for key, value in actual.items():
        value = np.asarray(value)
        if key not in saved or saved[key].shape != value.shape:
            raise ValueError(f"Saved preprocessing shape differs: {key}")
        matches = np.array_equal(saved[key].astype(str), value.astype(str)) if key == "turbine_ids" else np.allclose(saved[key], value, rtol=1e-6, atol=1e-7)
        if not matches:
            raise ValueError(f"Actual inputs do not reproduce saved preprocessing: {key}")
    prediction = load_prediction(source / "predictions.npz")
    metadata = handler.metadata["test"]
    if (not np.array_equal(prediction.timestamp.astype(str), np.asarray(metadata["timestamp"]).astype(str)) or
            not np.array_equal(prediction.target_valid, metadata["target_valid"])):
        raise ValueError("Efficiency source does not cover the complete genuine test sample index")
    return bundle


def baseline_factory():
    spec = importlib.util.spec_from_file_location("efficiency_baseline_factory", ROOT / "scripts/train_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Inference(torch.nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, power, wind, weather_history, weather_target):
        if self.name == "PI-EEF":
            _, features, _, _ = self.model.forward_stage1_reconstruct(power, wind, weather_history)
            speed, physical_power = self.model.forward_stage2_calibration(wind, weather_target)
            return self.model.forward_stage3_forecast(features, speed, physical_power, weather_target,
                                                      hist_power=power)
        return self.model(power, x_w=wind, x_n_hist=weather_history, x_nwp_target=weather_target)


def load_inference(record, bundle, device):
    cfg, name = record["config"], record["model"]
    if name == "PI-EEF":
        options = deepcopy(cfg.get("model", {}))
        if options.get("manufacturer_curve_path"):
            options["manufacturer_curve_path"] = str(input_path(options["manufacturer_curve_path"], cfg))
        model = build_main_model(bundle, cfg["data"]["seq_len"], options, device)
    else:
        model = baseline_factory().build_model(name, int(cfg["data"]["seq_len"]), bundle=bundle).to(device)
    weights = record["source_dir"] / "weights_best.pt"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return Inference(model, name).to(device).eval()


def input_batch(bundle, batch_size, device):
    batch = next(iter(bundle.test_loader))
    if len(batch) != 5 or batch[0].shape[0] < batch_size:
        raise ValueError("The genuine test split cannot provide the requested complete latency batch")
    inputs = tuple(value[:batch_size].to(device) for value in batch[:4])
    if any(not bool(torch.isfinite(value).all()) for value in inputs):
        raise ValueError("The latency batch contains unfilled/non-finite real inputs")
    return inputs


def ensure_cloud01(worker_id):
    if worker_id != WORKER:
        raise ValueError("Formal Table 6 measurements must run on recorded-worker")
    dispatch = ROOT / "artifacts/energy_revision/dispatch"
    found = []
    for filename in ("worker_state.json", "worker_queue.json"):
        path = dispatch / filename
        if path.is_file():
            found.append(json.loads(path.read_text(encoding="utf-8")).get("worker_id"))
    if not found or any(value != WORKER for value in found):
        raise ValueError("This workspace does not identify itself as cloud worker 01")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal efficiency measurement requires CUDA; CPU fallback is forbidden")
    environment = gpu_runtime(torch.device("cuda:0"))
    if not environment["gpu_uuid"]:
        raise RuntimeError("The actual GPU UUID must be available for a same-hardware comparison")
    line = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True, timeout=10).stdout
    for row in csv.reader(line.splitlines()):
        if len(row) == 2 and row[0].strip() == environment["gpu_uuid"] and int(row[1].strip()) != os.getpid():
            raise RuntimeError("Another CUDA process is using GPU 0; measure after that training task naturally finishes")
    return dict(environment, worker_id=worker_id, pid=os.getpid())


def measure_cuda(model, inputs, warmup, repeat):
    device = inputs[0].device
    with torch.no_grad():
        prediction = model(*inputs)
        if prediction.shape != inputs[0].shape[:2] or not bool(torch.isfinite(prediction).all()):
            raise ValueError("The restored real model has invalid prediction shape or values")
        del prediction
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize(device)
        started, finished = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Materialize CUDA events before collecting the timed samples.
        started.record()
        finished.record()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        cuda_ms, wall_ms = [], []
        for _ in range(repeat):
            torch.cuda.synchronize(device)
            wall_started = time.perf_counter()
            started.record()
            prediction = model(*inputs)
            finished.record()
            torch.cuda.synchronize(device)
            wall_ms.append((time.perf_counter() - wall_started) * 1000.)
            cuda_ms.append(float(started.elapsed_time(finished)))
            del prediction
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    def statistics(values):
        return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                    p95_ms=float(np.percentile(values, 95)), std_ms=float(np.std(values, ddof=1)))
    return dict(warmup=warmup, repeats=repeat, cuda_latency=statistics(cuda_ms), wall_latency=statistics(wall_ms),
                cuda_samples_ms=cuda_ms, wall_samples_ms=wall_ms,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                resident_model_and_input_bytes=int(baseline_allocated),
                incremental_peak_allocated_bytes=peak_allocated - int(baseline_allocated),
                scope="GPU-resident forward with CUDA-event and synchronized wall timing, all four genuine inputs; host loading/transfers and output postprocessing excluded.")


def measured_training(record, environment):
    path = record["source_dir"] / "training_timing.json"
    if not path.is_file():
        return dict(train_time_s=None, validation_time_s=None, timing_reason="No actual separated training timing was saved")
    timing = json.loads(path.read_text(encoding="utf-8"))
    cfg = record["config"]
    for key, expected in (("run_id", record["source_dir"].name), ("config_id", cfg["tracking"]["config_id"]),
                          ("model", record["model"]), ("seed", 42), ("horizon", 16)):
        if timing.get(key) != expected:
            raise ValueError(f"Training timing differs from its actual source run: {key}")
    runtime = timing.get("runtime", {})
    if timing.get("worker_id") != WORKER or runtime.get("gpu_uuid") != environment.get("gpu_uuid"):
        return dict(train_time_s=None, validation_time_s=None, timing_reason="Training did not use this same worker/GPU UUID")
    if not timing.get("cuda_synchronized_boundaries") or timing.get("status") != "completed":
        raise ValueError("Training timing lacks completed synchronized measurements")
    rows = timing.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training timing lacks actual epoch measurements")
    sums = {"train": 0., "validation": 0.}
    for row in rows:
        phase, seconds = row.get("phase"), row.get("wall_clock_s")
        if phase not in sums or type(seconds) not in (float, int) or not np.isfinite(seconds) or seconds < 0 or type(row.get("batches")) is not int or row["batches"] < 1:
            raise ValueError("Training timing contains invalid epoch work")
        sums[phase] += seconds
    if not np.isclose(sums["train"], timing.get("train_only_s", np.nan)) or not np.isclose(sums["validation"], timing.get("validation_only_s", np.nan)):
        raise ValueError("Training timing totals differ from actual epoch records")
    return dict(train_time_s=sums["train"], validation_time_s=sums["validation"], timing_reason=timing["scope"],
                training_timing_path=str(path), training_epochs=rows)


def count_flops(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return dict(flops=None, flops_reason="fvcore is unavailable; full-model FLOPs were not measured")
    try:
        analysis = FlopCountAnalysis(model, inputs).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = float(analysis.total())
        unsupported = dict(analysis.unsupported_ops())
        if unsupported:
            return dict(flops=None, flops_reason="The installed counter does not cover the complete model", unsupported_flop_ops=unsupported)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Nonpositive FLOPs measurement")
        return dict(flops=total, flops_reason="fvcore traced full forward; one fused multiply-add counts as one operation", flops_batch_size=inputs[0].shape[0])
    except Exception as exc:
        return dict(flops=None, flops_reason=f"Complete-model FLOPs tracing unavailable: {type(exc).__name__}: {exc}")


def run(args):
    requested = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(MODELS):
        raise ValueError("Efficiency models must be a nonempty subset of the five Table 6 models")
    if args.warmup < 1 or args.repeat < 2 or args.batch_size < 1:
        raise ValueError("Positive warmup/batch size and at least two repeated timings are required")
    paths = [project_path(value) for value in args.run_dir]
    if not paths:
        paths = [path.parent for path in project_path(args.runs_root).rglob("metrics.json")]
    records = {}
    final_config = yaml.safe_load(project_path(args.final_config).read_text(encoding="utf-8"))
    for path in sorted(set(paths)):
        payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        exp = cfg.get("experiment", {})
        if not args.run_dir and payload.get("status") != "completed":
            continue
        if exp.get("model") not in requested or exp.get("group") != "main" or payload.get("seed") != 42 or payload.get("horizon") != 16:
            if args.run_dir:
                raise ValueError(f"Explicit efficiency input is outside the requested formal comparison: {path}")
            continue
        record = formal_record(path, final_config)
        if record["model"] in records:
            raise ValueError(f"Duplicate real main run for {record['model']}; select an unambiguous source")
        records[record["model"]] = record
    if set(records) != set(requested):
        raise ValueError(f"Completed efficiency sources missing: {sorted(set(requested) - set(records))}")
    environment = ensure_cloud01(args.worker_id)
    output = project_path(args.output_root)
    if (output / "efficiency_metrics.json").exists():
        raise FileExistsError("A completed efficiency measurement already exists; use a new output-root for another measurement")
    output.mkdir(parents=True, exist_ok=True)
    rows, common_inputs, common_index = [], None, None
    for name in requested:
        record = records[name]
        bundle = rebuild_bundle(record)
        if args.batch_size > int(record["config"]["train"]["batch_size"]):
            raise ValueError("Latency batch cannot exceed the genuine saved loader batch")
        host_inputs = input_batch(bundle, args.batch_size, torch.device("cpu"))
        signature = tuple(value.numpy().copy() for value in host_inputs)
        index = (record["config"]["data"]["data_version"],
                 tuple(str(value) for value in bundle.handler.metadata["test"]["timestamp"][:args.batch_size]),
                 tuple(str(value) for value in bundle.handler.turbine_ids))
        if common_inputs is None:
            common_inputs, common_index = signature, index
        elif common_index != index or any(not np.array_equal(left, right) for left, right in zip(common_inputs, signature)):
            raise ValueError("The five models do not share exactly the same genuine inference inputs")
        model = load_inference(record, bundle, torch.device("cuda:0"))
        inputs = tuple(value.to("cuda:0") for value in host_inputs)
        measurements = measure_cuda(model, inputs, args.warmup, args.repeat)
        with torch.no_grad():
            flops = count_flops(model, inputs)
        result = dict(status="completed", operation="cuda_efficiency", model=name,
                      run_id=record["payload"]["run_id"], config_id=record["config"]["tracking"]["config_id"],
                      source_directory=str(record["source_dir"]), checkpoint="weights_best.pt", seed=42, horizon=16,
                      data_version=record["config"]["data"]["data_version"], environment=environment,
                      input_shapes=[list(value.shape) for value in inputs], input_dtype=str(inputs[0].dtype),
                      test_timestamps=[str(value) for value in bundle.handler.metadata["test"]["timestamp"][:args.batch_size]],
                      num_params=sum(parameter.numel() for parameter in model.parameters()),
                      training=measured_training(record, environment), inference=measurements, **flops)
        write_json_atomic(output / f"{name.replace('-', '_')}_efficiency.json", result)
        rows.append(dict(model=name, run_id=result["run_id"], config_id=result["config_id"], seed=42, horizon=16,
                         train_time_s=result["training"]["train_time_s"], validation_time_s=result["training"]["validation_time_s"],
                         inference_cuda_mean_ms=measurements["cuda_latency"]["mean_ms"], inference_wall_mean_ms=measurements["wall_latency"]["mean_ms"],
                         inference_wall_p50_ms=measurements["wall_latency"]["p50_ms"], inference_wall_p95_ms=measurements["wall_latency"]["p95_ms"],
                         inference_wall_ms_per_sample=measurements["wall_latency"]["mean_ms"] / args.batch_size,
                         peak_allocated_mb=measurements["peak_allocated_bytes"] / 1024**2,
                         peak_reserved_mb=measurements["peak_reserved_bytes"] / 1024**2, num_params=result["num_params"], flops=result["flops"],
                         flops_reason=result["flops_reason"], batch_size=args.batch_size, turbines=bundle.num_turbines,
                         sequence_length=inputs[0].shape[2], weather_features=bundle.nwp_dim, worker_id=WORKER,
                         gpu=environment["gpu"], gpu_uuid=environment["gpu_uuid"], torch=environment["torch"],
                         source_directory=result["source_directory"], timing_reason=result["training"]["timing_reason"]))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        del model, inputs, bundle, host_inputs
        gc.collect()
        torch.cuda.empty_cache()
    csv_path = output / "efficiency.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    write_json_atomic(output / "efficiency_metrics.json", dict(status="completed", operation="cuda_efficiency", models=list(requested),
                                                             environment=environment, rows=rows, recorded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--runs-root", default="artifacts/energy_revision/main")
    parser.add_argument("--final-config", default="artifacts/energy_revision/final_model.yaml")
    parser.add_argument("--output-root", default="artifacts/energy_revision/efficiency/table6_h16_seed42_cuda01")
    parser.add_argument("--worker-id", default=WORKER)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    print(f"Saved formal efficiency results: {run(args)}", flush=True)


if __name__ == "__main__":
    main()
