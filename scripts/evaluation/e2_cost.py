#!/usr/bin/env python3
"""Author-operated controlled GPU microbenchmark; this file does not run on import.

prepare: CPU data construction only. run: GPU inference and optimizer-step timing.
No checkpoint is loaded, no accuracy claim is made, and no model is saved.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
E = Path(r"results/paper")
P = Path(r".")
DEFAULT_METRICS = E / "补做重算/inputs/round3/stage2/main/energy_r3_stage2_PI-EEF_h16_s13_20261021_011452/metrics.json"
COUNTS = (10, 20, 29, 50, 72, 100, 200)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--code-project", type=Path, default=E / "next_batch/merged_v2/project")
    p.add_argument("--data-project", type=Path, default=P)
    p.add_argument("--final-metrics", type=Path, default=DEFAULT_METRICS,
                   help="Completed final h16 metrics.json, containing its actual config")
    p.add_argument("--out", type=Path, required=True, help="A NEW E2 result directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--train-batch", type=int, default=4,
                   help="Fixed for every N; default4 limits memory on 12GiB GPUs")
    p.add_argument("--epoch-windows", type=int, default=26500,
                   help="Stated equal workload used ONLY for derived epoch time")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def final_config(args):
    result = read_json(args.final_metrics)
    cfg = result["config"]
    if result.get("horizon") != 16 or cfg["experiment"].get("horizon") != 16:
        raise ValueError("E2 requires final h16 configuration")
    expected = {"d_model": 128, "encoder_depth": 3, "graph_depth": 1,
                "num_control_points": 15, "use_density": True, "use_terrain_bias": True,
                "use_mask_indicators": True, "revin_missing_policy": "train_stats_20",
                "forecast_head_policy": "target_nwp_physics_mlp",
                "graph_score_policy": "cosine_temperature", "forecast_skip_policy": "last_obs_linear"}
    for key, value in expected.items():
        if cfg["model"].get(key) != value:
            raise ValueError(f"Unexpected final model {key}: {cfg['model'].get(key)!r}")
    if cfg["data"]["seq_len"] != 96 or cfg["train"].get("stage3_mode") != "joint":
        raise ValueError("Expected seq96 and joint stage3")
    return cfg


def load_preparer(args):
    source = args.code_project / "scripts/prepare_revision_scalability.py"
    spec = importlib.util.spec_from_file_location("e2_project_scalability", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    """Extend the *same* nested seed42 translated-block layout, without noise."""
    import numpy as np
    import pandas as pd
    cfg = final_config(args)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new E2 directory; preserved existing {out}")
    prep = load_preparer(args)
    frame, turbines, prefixes, source, static_source = prep.load_source(cfg, args.data_project)
    existing_path = args.data_project / "data/raw/scalability/scalability_manifest.json"
    original = read_json(existing_path)
    if original.get("seed") != 42 or original.get("donor_mapping") != "original_29_then_seed42_permutation_per_translated_block":
        raise ValueError("Unexpected existing scalability construction")
    layout100, donors100, offset = prep.fixed_layout(turbines)
    if abs(offset - original["layout_rule"]["block_offset_m"]) > 1e-6:
        raise ValueError("Existing block offset differs")
    # fixed_layout consumes an entire permutation for each block (even its truncated
    # last block). Continuing full permutations preserves every original first100 donor.
    rng = np.random.default_rng(42)
    donors, blocks = list(range(29)), [0] * 29
    block = 1
    while len(donors) < 200:
        selected = rng.permutation(29)[:min(29, 200 - len(donors))]
        donors.extend(selected.tolist())
        blocks.extend([block] * len(selected))
        block += 1
    donors = np.asarray(donors)
    if not np.array_equal(donors[:100], donors100):
        raise ValueError("Extended donor map does not preserve original N100")
    layout = turbines.iloc[donors].reset_index(drop=True).copy()
    layout["donor_turbine_id"] = layout["turbine_id"].astype(str)
    layout["donor_node_index"] = donors + 1
    layout["donor_power_column"] = [f"power{i+1}" for i in donors]
    layout["layout_block"] = blocks
    layout["offset_x_m"] = np.asarray(blocks, dtype=float) * offset
    layout["offset_y_m"] = 0.0
    layout["x_coord"] = layout["x_coord"].to_numpy(dtype=float) + layout["offset_x_m"].to_numpy()
    layout["is_synthetic_node"] = np.asarray(blocks) > 0
    layout["turbine_id"] = [str(turbines.iloc[i]["turbine_id"]) if i < 29 else f"S{i+1:03d}" for i in range(200)]
    for column in ("latitude", "longitude"):
        if column in layout:
            layout["donor_" + column] = layout[column]
            layout.loc[layout["is_synthetic_node"], column] = np.nan
    layout["coordinate_unit"] = "m"
    layout["elevation_unit"] = "m"
    layout["power_capacity_unit"] = "MW"
    layout["elevation_role"] = np.where(layout["is_synthetic_node"], "copied_donor_attribute", "actual_wind_farm_A_elevation")
    layout["donor_mapping_seed"] = 42
    if layout[["x_coord", "y_coord"]].duplicated().any():
        raise ValueError("Overlapping replicated nodes")
    # Verify the existing N10/20/50/100 are precisely the declared nested layouts.
    points = {}
    for n in (10, 20, 29, 50, 100):
        point = original["points"][str(n)]
        scada, static = (args.data_project / point[k] for k in ("scada_path", "turbines_path"))
        actual = pd.read_csv(static)
        if n != 29:
            for key in ("donor_node_index", "layout_block", "x_coord", "y_coord", "elevation_m", "capacity_mw"):
                if not np.allclose(actual[key].to_numpy(dtype=float), layout.iloc[:n][key].to_numpy(dtype=float), rtol=0, atol=1e-6):
                    raise ValueError(f"Existing N{n}/{key} differs from the original construction")
        points[str(n)] = {"scada_path": str(scada.resolve()), "turbines_path": str(static.resolve()),
                          "data_kind": point["data_kind"], "scada_sha256": sha(scada), "turbines_sha256": sha(static)}
    out.mkdir(parents=True)
    per_node = {f"{prefix}{i}" for prefix in prefixes for i in range(1, 30)}
    shared = [column for column in frame if column not in per_node | {"Total_Power"}]
    for n in (72, 200):
        folder = out / "data" / f"N{n}"
        folder.mkdir(parents=True)
        columns = {f"{prefix}{i+1}": frame[f"{prefix}{donor+1}"].to_numpy(copy=True)
                   for prefix in prefixes for i, donor in enumerate(donors[:n])}
        expanded = pd.concat([frame[shared].reset_index(drop=True), pd.DataFrame(columns)], axis=1)
        scada, static = folder / "SCADA.csv", folder / "turbines.csv"
        expanded.to_csv(scada, index=False, lineterminator="\n")
        layout.iloc[:n].to_csv(static, index=False, lineterminator="\n")
        points[str(n)] = {"scada_path": str(scada), "turbines_path": str(static),
                          "data_kind": "synthetic_donor_expansion", "scada_sha256": sha(scada), "turbines_sha256": sha(static)}
        print(f"Prepared NEW N{n}: {len(expanded)} rows, no interpolation/no Gaussian perturbation", flush=True)
    manifest = {"counts": list(COUNTS), "horizon": 16, "donor_seed": 42, "block_offset_m": offset,
                "construction": "original29 then seed42 permuted x-translated blocks; first100 preserved",
                "source_manifest": str(existing_path), "source_manifest_sha256": sha(existing_path),
                "final_metrics": str(args.final_metrics.resolve()), "final_metrics_sha256": sha(args.final_metrics),
                "code_project": str(args.code_project.resolve()), "data_project": str(args.data_project.resolve()),
                "N72_note": "Synthetic matched workload; not the real Wind Farm B72", "points": points}
    dump(out / "e2_data_manifest.json", manifest)
    dump(out / "final_h16_config.json", cfg)
    print(f"Data ready: {out / 'e2_data_manifest.json'}", flush=True)


def analytical_count(n, cfg, nwp_dim=12):
    """Count algebraically from module definitions, independent of named_parameters."""
    m = cfg["model"]
    d, t, w, k, ff = m["d_model"], cfg["data"]["seq_len"], nwp_dim, m["num_control_points"], 512
    # Linear(a,b)=ab+b; attention QKV+out=4d²+4d; LayerNorms=4d.
    components = {
        "pi_embedding": (w + 5) * d + d,
        "position_embedding": t * d,
        "transformer": m["encoder_depth"] * (4*d*d + 2*d*ff + 9*d + ff),
        "mae_decoder": d*(d//2) + d//2 + d//2 + 1,
        "revin": 2*n,
        "calibration_mlp": w*32 + 32 + 32 + 1,
        "spline": n*((k-1) + 1 + 1),
        "graph": m["graph_depth"] * (3*(d*d+d) + 2 + (16+16+16+1) + (2*16+16+16+1)),
        "forecast_head": w*16+16 + (d+1+16+1)*64+64 + 64+1,
        "forecast_skip": 1+t+1,
    }
    return sum(components.values()), components


def build_bundle(cfg, point, batch_size):
    from pieef.revision_data import build_revision_bundle
    data = cfg["data"]
    protocol = copy.deepcopy(data["revision_protocol"])
    if "scalability" in point["data_kind"] or point["data_kind"] != "actual_wind_farm_A_full_layout":
        protocol["capacity_column"] = "capacity_mw"
    return build_revision_bundle(
        Path(point["scada_path"]), Path(point["turbines_path"]), int(data["seq_len"]), 16,
        batch_size, float(data["split"][0]), float(data["split"][1]), data["capacity_mw"], None, protocol)


def fixed_batch(bundle, split, count, expected_indices):
    import numpy as np
    from torch.utils.data._utils.collate import default_collate
    dataset = getattr(bundle, split + "_loader").dataset
    if not np.array_equal(dataset.indices[:count], expected_indices[:count]):
        raise ValueError(f"N layouts do not share {split} forecast origins")
    return tuple(default_collate([dataset[i] for i in range(count)]))


def to_gpu(batch, device):
    return tuple(t.to(device).contiguous() for t in batch)


def infer(model, batch):
    p, w, hn, tn, _ = batch
    _, features, _, _ = model.forward_stage1_reconstruct(p, w, hn)
    v, physical = model.forward_stage2_calibration(w, tn)
    return model.forward_stage3_forecast(features, v, physical, tn, hist_power=p)


def measured(fn, args, device, context):
    import numpy as np
    import torch
    with context():
        for i in range(args.warmup):
            fn(i)
        torch.cuda.synchronize(device)
        gc.collect()
        # The allocation baseline includes resident model, current batch, and optimizer
        # state (for training). Cached-but-unallocated bytes are excluded by design.
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for i in range(args.iterations):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            fn(i)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - begin) * 1000)
        peak = torch.cuda.max_memory_allocated(device)
    q1, median, q3 = (float(v) for v in np.percentile(times, [25, 50, 75]))
    return {"median_ms": median, "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3-q1,
            "peak_allocated_mib": peak/(1024**2), "baseline_allocated_mib": baseline/(1024**2)}, times


def training_setup(model, bundle, clean_cpu, cfg, args, device):
    import torch
    from pieef.forecast_corruption import ForecastCorruption
    from pieef.train_main import TrainConfig, _configure_stage3, _forecast_loss, _stage3_features
    accepted = inspect.signature(TrainConfig).parameters
    tc = TrainConfig(**{k: v for k, v in cfg["train"].items() if k in accepted})
    _configure_stage3(model, tc.stage3_mode, tc.stage3_decoder_policy, tc.reconstruction_weight)
    model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=tc.lr_stage3)
    clean = to_gpu(clean_cpu, device)
    sampler = ForecastCorruption("f2_block", bundle.handler.coords, bundle.handler.scada_w_std,
                                 args.seed, bundle.num_turbines, cfg["data"]["seq_len"])
    # Materialise one corruption on CPU outside GPU compute timing. Alternate a
    # clean and this corrupted history to make the F2 batch frequency exactly 0.5.
    for _ in range(100):
        corrupted_cpu, pm, wm = sampler.corrupt(clean_cpu)
        if pm is not None:
            break
    else:
        raise RuntimeError("Could not construct a representative F2 batch")
    corrupted = to_gpu(corrupted_cpu, device)
    pm, wm = pm.to(device), wm.to(device)

    def step(index):
        p, w, hn, tn, y = corrupted if index % 2 else clean
        power_mask, wind_mask = (pm, wm) if index % 2 else (None, None)
        optimizer.zero_grad()
        feat = _stage3_features(model, p, w, hn, tc.stage3_mode,
                                force_power_mask=power_mask, force_wind_mask=wind_mask)
        v, physical = model.forward_stage2_calibration(w, tn)
        pred = model.forward_stage3_forecast(feat, v, physical, tn, hist_power=p, hist_power_mask=power_mask)
        loss = _forecast_loss(pred, y, tc)
        # Epoch 1: exp(-decay*0)=1, as in the actual final trainer.
        rec, _, _, _ = model.forward_stage1_reconstruct(*clean[:3])
        _, aux_power, delta = model.forward_stage2_training(tn)
        valid = torch.isfinite(y) & (y > 0.01)
        calibration = 0.05 * delta.square().mean()
        if valid.any():
            calibration = calibration + torch.nn.functional.mse_loss(aux_power.squeeze(-1)[valid], y[valid])
        loss = loss + tc.reconstruction_weight*rec + tc.calibration_weight*calibration
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite benchmark loss; session is incomplete")
        loss.backward()
        optimizer.step()

    return step, optimizer, sampler.last_scenarios


def run(args):
    if args.warmup < 10 or args.iterations < 100 or args.iterations % 2:
        raise ValueError("Require warmup>=10, iterations>=100 and an even timed count")
    if args.train_batch < 1 or args.epoch_windows < 1 or args.cpu_threads < 1:
        raise ValueError("Batch, epoch windows, and CPU thread counts must be positive")
    if not args.device.startswith("cuda"):
        raise ValueError("E2 is a controlled one-GPU benchmark; a CPU run is not comparable")
    out = args.out.resolve()
    manifest = read_json(out / "e2_data_manifest.json")
    if manifest["final_metrics_sha256"] != sha(args.final_metrics):
        raise ValueError("Final config changed after data preparation")
    cfg = final_config(args)
    for name in ("E2_cost_vs_turbines.csv", "E2_session.json", "E2_iterations.csv"):
        if (out / name).exists():
            raise FileExistsError(f"Preserved previous session: {out/name}; use another prepared --out")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(args.code_project.resolve() / "src"))
    import numpy as np
    import torch
    from pieef.model_factory import build_main_model
    import pieef.models.e2e as e2e_module
    if Path(e2e_module.__file__).resolve() != (args.code_project.resolve()/"src/pieef/models/e2e.py"):
        raise RuntimeError("Wrong pieef code imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot record NVIDIA driver via nvidia-smi; fix PATH before benchmark") from exc
    source_hashes = {str(p.relative_to(args.code_project)): sha(p)
                     for p in (args.code_project / "src/pieef").rglob("*.py")}
    orders = [random.Random(args.seed+i).sample(list(COUNTS), len(COUNTS)) for i in (1, 2)]
    metadata = {"started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device), "gpu_total_memory": torch.cuda.get_device_properties(device).total_memory,
                "device": str(device), "nvidia_smi": driver, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "amp": False,
                "inference_batches": [1, 16], "training_batch": args.train_batch,
                "warmup": args.warmup, "timed_iterations": args.iterations,
                "sweep_orders": orders, "seed": args.seed, "cpu_threads": args.cpu_threads,
                "config": cfg, "source_sha256": source_hashes, "random_weights": True,
                "timed_scope": "GPU-resident float32; Python+GPU compute+sync; excludes I/O, preprocessing, transfers, CPU F2 sampling",
                "training_scope": "Actual epoch1 joint Huber+0.05rec+0.05cal, frozen decoder, Adam; alternating clean/F2-corrupted forecast branch; random reconstruction masks; no gradient-conflict probe",
                "epoch_estimate": "ceil(epoch_windows/train_batch)*median_step; no validation or data-loader time",
                "memory_scope": "max_memory_allocated: includes resident input, parameters and training optimizer state; not cached/reserved/device total"}
    dump(out / "E2_session.json", metadata)
    reference = build_bundle(cfg, manifest["points"]["29"], max(16, args.train_batch))
    ref_test = reference.test_loader.dataset.indices[:16].copy()
    ref_train = reference.train_loader.dataset.indices[:args.train_batch].copy()
    metadata["reference_train_windows"] = len(reference.train_loader.dataset)
    metadata["reference_test_origins"] = ref_test.tolist()
    metadata["reference_train_origins"] = ref_train.tolist()
    del reference
    gc.collect()
    rows, raw_rows, count_rows = [], [], []
    begin_session = time.perf_counter()
    for sweep, order in enumerate(orders, 1):
        for position, n in enumerate(order, 1):
            point = manifest["points"][str(n)]
            for key in ("scada", "turbines"):
                if sha(point[key+"_path"]) != point[key+"_sha256"]:
                    raise ValueError(f"Input changed after preparation: N{n}/{key}")
            bundle = build_bundle(cfg, point, max(16, args.train_batch))
            torch.manual_seed(args.seed+n)
            torch.cuda.manual_seed_all(args.seed+n)
            opts = copy.deepcopy(cfg["model"])
            opts["manufacturer_curve_path"] = str((args.data_project / opts["manufacturer_curve_path"]).resolve())
            model = build_main_model(bundle, cfg["data"]["seq_len"], opts, device)
            analytical, components = analytical_count(n, cfg, bundle.nwp_dim)
            actual = sum(p.numel() for p in model.parameters())
            expected = 678131 + 18*(n-29)
            if actual != analytical or analytical != expected:
                raise ValueError(f"Parameter FAIL N={n}: actual={actual} analytical={analytical} expected={expected}; wrong code/config")
            count_rows.append({"sweep": sweep, "n_turbines": n, "actual": actual, "analytical": analytical, "components": components})
            row = {"sweep": sweep, "order": position, "n_turbines": n, "data_kind": point["data_kind"],
                   "parameters": actual, "analytical_parameters": analytical, "parameter_check": "PASS",
                   "horizon": 16, "seq_len": 96, "nwp_dim": bundle.nwp_dim,
                   "inference_batch_1": 1, "inference_batch_16": 16, "train_batch": args.train_batch,
                   "epoch_windows": args.epoch_windows, "epoch_steps": math.ceil(args.epoch_windows/args.train_batch),
                   "actual_train_windows": len(bundle.train_loader.dataset), "gpu": metadata["gpu"],
                   "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "session_pid": os.getpid(), "warmup": args.warmup, "iterations": args.iterations}
            model.eval()
            for batch_size in (1, 16):
                batch = to_gpu(fixed_batch(bundle, "test", batch_size, ref_test), device)
                stats, samples = measured(lambda i: infer(model, batch), args, device, torch.inference_mode)
                for key, value in stats.items():
                    row[f"infer_b{batch_size}_{key}"] = value
                row[f"infer_b{batch_size}_per_window_median_ms"] = stats["median_ms"]/batch_size
                row[f"infer_b{batch_size}_per_window_iqr_ms"] = stats["iqr_ms"]/batch_size
                raw_rows.extend({"sweep": sweep, "n_turbines": n, "phase": f"infer_b{batch_size}", "iteration": i+1, "ms": ms} for i, ms in enumerate(samples))
                del batch
                gc.collect()
                torch.cuda.empty_cache()
            clean_cpu = fixed_batch(bundle, "train", args.train_batch, ref_train)
            step, optimizer, f2_scenarios = training_setup(model, bundle, clean_cpu, cfg, args, device)
            train_stats, samples = measured(step, args, device, torch.enable_grad)
            for key, value in train_stats.items():
                row["train_"+key] = value
            row["derived_epoch_median_s"] = row["epoch_steps"]*train_stats["median_ms"]/1000
            row["derived_epoch_q1_s"] = row["epoch_steps"]*train_stats["q1_ms"]/1000
            row["derived_epoch_q3_s"] = row["epoch_steps"]*train_stats["q3_ms"]/1000
            raw_rows.extend({"sweep": sweep, "n_turbines": n, "phase": "train", "iteration": i+1, "ms": ms} for i, ms in enumerate(samples))
            dump(out / f"E2_sweep{sweep}_N{n}_f2.json", f2_scenarios)
            rows.append(row)
            write_csv(out / "E2_cost_vs_turbines.csv", rows)
            write_csv(out / "E2_iterations.csv", raw_rows)
            dump(out / "E2_parameter_audit.json", count_rows)
            print(f"sweep={sweep} {position}/7 N={n}: parameters={actual} PASS, inferB1={row['infer_b1_median_ms']:.4f}ms, inferB16/window={row['infer_b16_per_window_median_ms']:.4f}ms, train={train_stats['median_ms']:.4f}ms", flush=True)
            del step, optimizer, model, bundle, clean_cpu
            gc.collect()
            torch.cuda.empty_cache()
    if len(rows) != 14 or {(r['sweep'], r['n_turbines']) for r in rows} != {(s,n) for s in (1,2) for n in COUNTS}:
        raise RuntimeError("Incomplete E2 session")
    metadata["elapsed_compute_and_loading_s"] = time.perf_counter()-begin_session
    metadata["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["status"] = "PASS: 2 sweeps x 7 counts; analytical parameters agree"
    dump(out / "E2_session.json", metadata)
    print(metadata["status"], flush=True)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
