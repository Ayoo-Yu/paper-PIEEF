"""Gate and report nine real Energy LR validation runs; default generates report.

No training, inference, gradient computation, test-score reading or model selection.
--check-only runs read-only gates. Source CSVs are the plotting evidence chain.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_key] = "2"
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
DIAG = ROOT / "diagnostics"
RESULTS = DIAG / "results"
OUT = ROOT / "report"
SEEDS = (13, 42, 3407)
SCHEDULES = {"LR0": (0.001, 0.001, 0.0003), "LR1": (0.001, 0.001, 0.001), "LR2": (0.0003, 0.0003, 0.0003)}
REQUIRED = (
    "gradient_probe.csv", "loss_components.csv", "probe_within_run_summary.csv",
    "stage2_best_displacement.csv", "pretraining_pair_audit.csv", "validation_trajectories.csv",
    "training_first_batch_cosines.csv", "validation_runs.csv", "validation_groups.csv",
    "validation_paired_differences.csv", "fixed_probe_batches.json", "probe_audit.json", "validation_pairing_gate.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {x}")))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        result = list(csv.DictReader(f))
    require(result, f"Empty required CSV: {path}")
    for row in result:
        for key, value in row.items():
            require(value is not None, f"Malformed CSV: {path}/{key}")
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            require(math.isfinite(number), f"Non-finite CSV: {path}/{key}")
    return result


def num(row, key):
    value = float(row[key])
    require(math.isfinite(value), f"Non-finite {key}")
    return value


def integer(row, key):
    value = num(row, key)
    require(value == int(value), f"Noninteger {key}")
    return int(value)


def same_number(a, b, label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)), f"Non-finite {label}")
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12), f"Arithmetic/source mismatch: {label}")


def truth(value):
    require(value in (True, False, "True", "False", "true", "false"), f"Invalid boolean: {value}")
    return value in (True, "True", "true")


def file_item(path, name=None):
    path = Path(path)
    return {"name": name or path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic(path, content):
    path = Path(path)
    require(OUT in path.resolve().parents, "Report output escaped report/")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_csv(path, data):
    keys = list(dict.fromkeys(k for row in data for k in row))
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def verify(args):
    done_path = RESULTS / "completed.json"
    require(done_path.is_file(), "Formal diagnostic completed.json is not ready")
    done = read(done_path)
    require(done.get("status") == "completed" and done.get("run_count") == 9, "Need nine formal diagnostic runs; smoke/incomplete rejected")
    require(done.get("full_training_batches") == 5 and done.get("probe_seed") == 20261022, "Unexpected fixed probe protocol")
    require(done.get("optimizer_steps") == 0, "Diagnostic mutated model")
    require(done.get("script_sha256") == sha(DIAG / "offline_probe.py"), "Diagnostic script SHA changed after completion")
    manifest = done.get("output_files", [])
    by_name = {x["name"]: x for x in manifest}
    require(len(by_name) == len(manifest) and set(REQUIRED).issubset(by_name), "Incomplete/duplicate diagnostic output SHA manifest")
    for name, item in by_name.items():
        path = (RESULTS / name).resolve()
        require(RESULTS.resolve() in path.parents and path.is_file(), f"Invalid diagnostic output path: {name}")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"Diagnostic output changed: {name}")

    spec = importlib.util.spec_from_file_location("lr_report_readonly_gate", DIAG / "offline_probe.py")
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    diagnostic.load_source(ROOT / "project")
    config_manifest = diagnostic.verify_frozen_inputs(ROOT / "project")
    mapping = args.run_index or (ROOT / "run_index.json" if (ROOT / "run_index.json").is_file() else None)
    records, missing = diagnostic.ready_records(config_manifest, mapping)
    require(not missing and len(records) == 9, f"Strict runs not ready: {missing}")
    source = {r["id"]: r for r in records}
    completed_sources = {r["id"]: r for r in done["sources"]}
    require(len(done["sources"]) == 9 and set(completed_sources) == set(source), "Diagnostic source identities differ")
    for cid, rec in source.items():
        evidence = completed_sources[cid]
        require(evidence["receipt_sha256"] == rec["receipt_sha256"], f"Receipt changed since diagnostic: {cid}")
        require(Path(evidence["source_dir"]).resolve() == rec["source_dir"].resolve(), f"Source path differs: {cid}")
        require(evidence["schedule"] == rec["schedule"] and evidence["seed"] == rec["seed"], "Source schedule/seed differs")
        require(evidence["hardware_audit"] == rec["hardware_audit"], "Hardware identity evidence differs")
        require(evidence["validation_pairing_audit"] == rec["validation_pairing_audit"], "Validation pairing evidence differs")
        require(rec["wall_clock_s"] is not None and float(rec["wall_clock_s"]) > 0, "Missing actual complete child wall clock")
        same_number(evidence["wall_clock_s"], rec["wall_clock_s"], cid + " wallclock")
    pairing_gate = read(RESULTS / "validation_pairing_gate.json")
    require(pairing_gate.get("status") == "PASS" and pairing_gate.get("expected_runs") == pairing_gate.get("compared_runs") == 9, "Nine-run validation pairing gate is not PASS")
    require(pairing_gate["runs"] == [r["validation_pairing_audit"] for r in sorted(records, key=lambda r: (r["seed"], r["schedule"]))], "Pairing gate records do not match current strict sources")

    data = {name: rows(RESULTS / name) for name in REQUIRED if name.endswith(".csv")}
    validation = data["validation_runs.csv"]
    require(len(validation) == 9 and {r["config_id"] for r in validation} == set(source), "Validation run table is not the exact nine runs")
    require({(r["schedule"], integer(r, "seed")) for r in validation} == {(s, seed) for s in SCHEDULES for seed in SEEDS}, "Invalid/duplicate schedule-seed identity")
    trajectories = data["validation_trajectories.csv"]
    for row in validation:
        rec = source[row["config_id"]]
        require(row["site"] == "wind_farm_A" and integer(row, "horizon_minutes") == 240 and row["split"] == "val", "Unexpected evaluation data/protocol")
        require(row["schedule"] == rec["schedule"] and integer(row, "seed") == rec["seed"], "CSV/source identity differs")
        for metric in ("nrmse", "nmae", "r2"):
            same_number(num(row, metric), rec["metrics"][metric], metric)
        require(num(row, "nrmse") > 0 and num(row, "nmae") > 0, "Relative-error comparison requires positive control errors")
        same_number(num(row, "wall_clock_s"), rec["wall_clock_s"], "wallclock")
        hardware = rec["hardware_audit"]
        require(row["gpu_name"] == hardware["runtime_gpu_name"] and row["gpu_uuid"] == hardware["effective_gpu_uuid"], "CSV hardware does not match actual evidence")
        require(row["uuid_evidence"] == hardware["uuid_evidence"], "UUID evidence level changed")
        history = read(rec["source_dir"] / "loss_history.json")
        require(integer(row, "best_epoch") == history["best_epoch"] and integer(row, "epochs_observed") == len(history["stage3"]), "Best epoch/observed epochs differs")
        curve = sorted((r for r in trajectories if r["config_id"] == rec["id"]), key=lambda r: integer(r, "epoch"))
        require(len(curve) == len(history["stage3"]), "Missing/extra trajectory epochs")
        require([integer(r, "epoch") for r in curve] == list(range(1, len(curve) + 1)), "Noncontiguous/duplicate epochs")
        for observed, actual in zip(curve, history["stage3"]):
            require(observed["schedule"] == rec["schedule"] and integer(observed, "seed") == rec["seed"], "Trajectory identity mismatch")
            for key in ("epoch", "train_loss", "val_loss", "val_nrmse", "elapsed_s", "num_train_batches", "num_val_batches"):
                same_number(num(observed, key), actual[key], "trajectory." + key)
    require({r["config_id"] for r in trajectories} == set(source), "Extra trajectory source")

    audits = read(RESULTS / "probe_audit.json")
    require(len(audits) == 18 and {(r["config_id"], r["checkpoint"]) for r in audits} == {(cid, tag) for cid in source for tag in ("stage2", "best")}, "Need all 18 actual checkpoint probes")
    for item in audits:
        require(item["weights_unchanged"] is True and item["optimizer_steps"] == 0, "Probe changed checkpoint")
        require(item["checkpoint_sha256"] == sha(source[item["config_id"]]["source_dir"] / ("weights_" + item["checkpoint"] + ".pt")), "Checkpoint changed since probe")
    for row in data["gradient_probe.csv"]:
        require(row["config_id"] in source and row["checkpoint"] in ("stage2", "best"), "Unknown gradient probe source")
        if row["kind"] == "cosine":
            zero = truth(row["zero_a"]) or truth(row["zero_b"])
            require((row["cosine"] == "") == zero, "Null cosine must correspond to zero gradient norm")
            if row["cosine"]:
                require(abs(num(row, "cosine")) <= 1.000001, "Cosine out of range")
    pretraining = data["pretraining_pair_audit.csv"]
    require(len(pretraining) == 6 and {(integer(r, "seed"), integer(r, "stage")) for r in pretraining} == {(s, stage) for s in SEEDS for stage in (1, 2)}, "Incomplete pretraining audit")
    equality = all(truth(r["all_tensors_equal"]) for r in pretraining)
    require(done["LR0_LR1_pretraining_bitwise_equal"] is equality, "Pretraining equality flags disagree")
    for row in pretraining:
        if truth(row["all_tensors_equal"]):
            require(num(row, "state_l2_difference") == 0 and num(row, "max_abs_difference") == 0, "Equal-state audit has nonzero difference")

    groups, pairs, comparisons, worst = summarize(validation)
    observed_groups = {r["schedule"]: r for r in data["validation_groups.csv"]}
    require(len(observed_groups) == len(data["validation_groups.csv"]) == 3, "Incorrect group count")
    for group in groups:
        check = observed_groups[group["schedule"]]
        require(integer(check, "training_seeds") == 3, "Not three training seeds")
        for key, value in group.items():
            if key not in ("schedule", "training_seeds"):
                same_number(check[key], value, key)
    observed_pairs = {(r["schedule"], integer(r, "seed")): r for r in data["validation_paired_differences.csv"]}
    require(len(observed_pairs) == len(data["validation_paired_differences.csv"]) == 6, "Incorrect matched-pair count")
    for pair in pairs:
        check = observed_pairs[pair["schedule"], pair["seed"]]
        require(check["control"] == "LR0", "Incorrect control")
        for key, value in pair.items():
            if key.endswith("_difference") or key.endswith("_relative_improvement_percent"):
                same_number(check[key], value, key)
    inputs = [file_item(p, str(p.relative_to(ROOT)).replace("\\", "/")) for p in
              [Path(__file__), ROOT / "report_design.md", ROOT / "protocol.json", ROOT / "config_manifest.json", ROOT / "source_data_manifest.json", DIAG / "offline_probe.py", done_path]
              + [RESULTS / name for name in sorted(by_name)]]
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return {"done": done, "records": records, "source": source, "data": data, "runs": validation,
            "groups": groups, "pairs": pairs, "comparisons": comparisons, "worst": worst,
            "inputs": inputs, "fingerprint": fingerprint}


def summarize(runs):
    groups, pairs, comparisons, worst = [], [], [], []
    for schedule in SCHEDULES:
        subset = [r for r in runs if r["schedule"] == schedule]
        group = {"schedule": schedule, "training_seeds": 3}
        for metric in ("nrmse", "nmae", "r2"):
            values = [num(r, metric) for r in subset]
            group[metric + "_mean"] = statistics.mean(values)
            group[metric + "_sample_sd"] = statistics.stdev(values)
        groups.append(group)
        for metric in ("nrmse", "nmae"):
            maximum = max(num(r, metric) for r in subset)
            worst.append({"schedule": schedule, "metric": metric,
                          "worst_seeds": ";".join(r["seed"] for r in subset if num(r, metric) == maximum), "value": maximum})
        if schedule == "LR0":
            continue
        for row in subset:
            control = next(r for r in runs if r["schedule"] == "LR0" and r["seed"] == row["seed"])
            pair = {"schedule": schedule, "seed": int(row["seed"]), "control": "LR0", "worker": row["worker"]}
            for metric in ("nrmse", "nmae", "r2"):
                pair[metric + "_difference"] = num(row, metric) - num(control, metric)
                if metric != "r2":
                    pair[metric + "_relative_improvement_percent"] = 100 * (num(control, metric) - num(row, metric)) / num(control, metric)
            pairs.append(pair)
        comparison = {"schedule": schedule, "control": "LR0", "matched_seeds": 3}
        selected = [r for r in pairs if r["schedule"] == schedule]
        for metric in ("nrmse", "nmae", "r2"):
            differences = [r[metric + "_difference"] for r in selected]
            comparison[metric + "_difference_mean"] = statistics.mean(differences)
            comparison[metric + "_difference_sample_sd"] = statistics.stdev(differences)
            better = [(x < 0 if metric != "r2" else x > 0) for x in differences]
            comparison[metric + "_improved_seeds"] = sum(better)
            comparison[metric + "_tied_seeds"] = sum(x == 0 for x in differences)
            comparison[metric + "_worse_seeds"] = sum(not b and x != 0 for b, x in zip(better, differences))
            if metric != "r2":
                relative = [r[metric + "_relative_improvement_percent"] for r in selected]
                comparison[metric + "_paired_relative_improvement_mean_percent"] = statistics.mean(relative)
                comparison[metric + "_paired_relative_improvement_sample_sd_percent"] = statistics.stdev(relative)
                comparison[metric + "_group_mean_relative_improvement_percent"] = 100 * (groups[0][metric + "_mean"] - group[metric + "_mean"]) / groups[0][metric + "_mean"]
        comparisons.append(comparison)
    return groups, pairs, comparisons, worst


def fmt(value, scale=1, signed=False):
    if value is None or value == "":
        return "—"
    return format(float(value) * scale, "+.6f" if signed else ".6f")


def table(headers, records):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in records) + "\n"


def link(label, path):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def gradient_summary(context):
    rows_out = []
    for r in context["data"]["probe_within_run_summary.csv"]:
        if r["group"] == "all_shared" and r["weighting"] == "common_epoch1" and r["kind"] == "cosine" and r["term_a"] == "forecast" and r["term_b"] == "auxiliary_total":
            require(integer(r, "full_batches") == 5, "Expected five within-run probes")
            rows_out.append({k: r[k] for k in ("config_id", "schedule", "seed", "checkpoint", "full_batches", "defined_batches", "zero_norm_undefined_batches", "mean", "probe_batch_sd")})
    require(len(rows_out) == 18, "Missing compact gradient rows")
    return sorted(rows_out, key=lambda r: (r["schedule"], int(r["seed"]), r["checkpoint"]))


def documents(context):
    run_rows = [["PI-EEF / " + r["schedule"], "wind_farm_A", 240, r["seed"], "val", fmt(r["nrmse"], 100), fmt(r["nmae"], 100), fmt(r["r2"])] for r in sorted(context["runs"], key=lambda r: (r["schedule"], int(r["seed"])))]
    group_rows = [[r["schedule"], 3] + [fmt(r[m + "_mean"], 100 if m != "r2" else 1) + " ± " + fmt(r[m + "_sample_sd"], 100 if m != "r2" else 1) for m in ("nrmse", "nmae", "r2")] for r in context["groups"]]
    pair_rows = [[r["schedule"], r["seed"], fmt(r["nrmse_difference"], 100, True), fmt(r["nrmse_relative_improvement_percent"], signed=True), fmt(r["nmae_difference"], 100, True), fmt(r["nmae_relative_improvement_percent"], signed=True), fmt(r["r2_difference"], signed=True)] for r in context["pairs"]]
    aggregate_rows = []
    for r in context["comparisons"]:
        for metric in ("nrmse", "nmae"):
            aggregate_rows.append([r["schedule"], metric.upper(), fmt(r[metric + "_difference_mean"], 100, True) + " ± " + fmt(r[metric + "_difference_sample_sd"], 100),
                                   fmt(r[metric + "_paired_relative_improvement_mean_percent"], signed=True) + " ± " + fmt(r[metric + "_paired_relative_improvement_sample_sd_percent"]),
                                   fmt(r[metric + "_group_mean_relative_improvement_percent"], signed=True),
                                   f"{r[metric + '_improved_seeds']}/{r[metric + '_worse_seeds']}/{r[metric + '_tied_seeds']}"])
    timing_rows, hardware_rows, artifact_rows = [], [], []
    for r in sorted(context["runs"], key=lambda r: (int(r["seed"]), r["schedule"])):
        rec = context["source"][r["config_id"]]
        hw = rec["hardware_audit"]
        timing_rows.append([r["schedule"], r["seed"], r["best_epoch"], r["epochs_observed"], fmt(r["wall_clock_s"]), fmt(num(r, "wall_clock_s") / 60), r["gpu_name"]])
        hardware_rows.append([r["schedule"], r["seed"], r["gpu_name"], hw["runtime_gpu_uuid"] or "not recorded", hw["launch_gpu_uuid"], hw["uuid_evidence"]])
        folder = rec["source_dir"]
        artifact_rows.append([r["schedule"], r["seed"], link("metrics", folder / "metrics.json"), link("indexed val predictions", folder / "val_predictions.npz"), link("preprocessing", folder / "preprocessing.npz"), link("all four checkpoints / run", folder)])
    pre_rows = [[r["seed"], r["stage"], r["all_tensors_equal"], fmt(r["state_l2_difference"]), fmt(r["relative_state_l2_difference"]), fmt(r["max_abs_difference"])] for r in context["data"]["pretraining_pair_audit.csv"]]
    compact = gradient_summary(context)
    grad_rows = [[r["schedule"], r["seed"], r["checkpoint"], r["defined_batches"] + "/5", fmt(r["mean"]), fmt(r["probe_batch_sd"])] for r in compact]
    displacement = [r for r in context["data"]["stage2_best_displacement.csv"] if r["group"] == "all_trainable"]
    require(len(displacement) == 9, "Missing displacement rows")
    disp_rows = [[r["schedule"], r["seed"], fmt(r["stage2_norm"]), fmt(r["best_norm"]), fmt(r["displacement_l2"]), fmt(r["relative_displacement"])] for r in displacement]
    equality = context["done"]["LR0_LR1_pretraining_bitwise_equal"]
    response_en = ["We added a controlled learning-rate validation experiment using the final VF implementation at a 240-min horizon on wind_farm_A. We trained all three predefined schedules with seeds 13, 42 and 3407, retaining the batch size, precision, 5/5/20 maximum epoch budget, patience, data processing, losses and validation-NRMSE checkpoint rule. The three schedules for each seed ran on the same recorded GPU. This supplementary diagnostic does not select a replacement model or use test scores."]
    response_zh = ["我们补充了最终 VF 的受控学习率验证：wind_farm_A、240 分钟提前量，三个固定方案各训练种子 13、42、3407，保持 batch、精度、5/5/20 最大轮数、patience、数据处理、损失与验证 NRMSE 检查点规则不变。每个种子的三个方案均使用同一实际记录 GPU。本次补充验证不重新选择模型，也不使用测试分数。"]
    for r in context["comparisons"]:
        response_en.append(f"For {r['schedule']} versus LR0, the mean paired NRMSE difference was {fmt(r['nrmse_difference_mean'], 100, True)} percentage points (sample SD {fmt(r['nrmse_difference_sample_sd'], 100)}), with {r['nrmse_improved_seeds']} improved, {r['nrmse_worse_seeds']} worse and {r['nrmse_tied_seeds']} tied seeds. The mean paired NMAE difference was {fmt(r['nmae_difference_mean'], 100, True)} percentage points (sample SD {fmt(r['nmae_difference_sample_sd'], 100)}), with {r['nmae_improved_seeds']} improved, {r['nmae_worse_seeds']} worse and {r['nmae_tied_seeds']} tied seeds. Positive candidate-minus-control differences indicate worse errors.")
        response_zh.append(f"相对 LR0，{r['schedule']} 的配对 NRMSE 差均值为 {fmt(r['nrmse_difference_mean'], 100, True)} 个百分点，样本标准差 {fmt(r['nrmse_difference_sample_sd'], 100)}；改善/退步/持平种子数为 {r['nrmse_improved_seeds']}/{r['nrmse_worse_seeds']}/{r['nrmse_tied_seeds']}。NMAE 差均值为 {fmt(r['nmae_difference_mean'], 100, True)} 个百分点，样本标准差 {fmt(r['nmae_difference_sample_sd'], 100)}；改善/退步/持平为 {r['nmae_improved_seeds']}/{r['nmae_worse_seeds']}/{r['nmae_tied_seeds']}。候选减控制的正差代表误差变大。")
    response_en.append("The LR0/LR1 Stage-1 and Stage-2 checkpoint tensors were " + ("equal for every matched seed." if equality else "not equal in every matched comparison; the accompanying audit lists the exact discrepancies, so we do not assume an identical Stage-3 starting state.") + " LR1 changes only the Stage-3 rate, whereas LR2 jointly changes the pretraining rates. We report the complete learning curves and fixed-training-batch gradient diagnostics as descriptive evidence. Three paired seeds do not establish optimality or equivalence; no significance test is claimed. A scalar learning rate does not directly rotate a gradient, and checkpoint displacement is not the optimization path length.")
    response_zh.append("LR0/LR1 的 Stage-1、Stage-2 检查点逐张量核验" + ("在全部匹配种子均相等。" if equality else "并非全部相等，实际偏差完整列出，不能假定 Stage-3 起点严格相同。") + "LR1 仅改变 Stage-3 学习率，LR2 同时改变预训练阶段学习率。完整收敛曲线与固定训练批次梯度仅作描述性诊断；三个配对种子不能证明最优性或等效性，不声称显著性。标量学习率不会直接旋转梯度；检查点间位移不是优化路径长度。")
    fragments = "# English response fragment\n\n" + "\n\n".join(response_en) + "\n\n# 中文回复片段\n\n" + "\n\n".join(response_zh) + "\n"
    outputs = {}
    for zh in (True, False):
        sections = ["# " + ("Energy 学习率补充验证完整报告" if zh else "Energy learning-rate validation: complete report"),
                    ("仅验证集；9 次独立完整训练；全部候选与不利结果保留。数值来自已回传并通过严格完成核验的实际产物。" if zh else "Validation only; nine complete independent training runs; all schedules and unfavorable results retained. Values come from transferred artifacts that passed strict completion checks."),
                    "## " + ("固定方案" if zh else "Fixed schedules"),
                    table(["Schedule", "Stage 1 LR", "Stage 2 LR", "Stage 3 LR"], [[s, *rates] for s, rates in SCHEDULES.items()]),
                    ("wind_farm_A；h16=240 分钟；seed=13/42/3407；batch16、FP32、最大轮数5/5/20、patience5；按验证NRMSE选checkpoint。LR0是新训练对照。本报告不据此重新选择模型。" if zh else "wind_farm_A; h16 = 240 min; seeds 13/42/3407; batch 16, FP32, maximum epochs 5/5/20, patience 5; validation-NRMSE checkpoint selection. LR0 is a newly trained control. This report does not reselect the model."),
                    "## " + ("逐运行验证结果" if zh else "Every validation run"),
                    table(["Model / variant", "Farm", "Horizon (min)", "Seed", "Split", "NRMSE (%)", "NMAE (%)", "R²"], run_rows),
                    "## " + ("三共同种子汇总" if zh else "Three matched-seed summaries"),
                    table(["Schedule", "n seeds", "NRMSE (%) mean ± SD", "NMAE (%) mean ± SD", "R² mean ± SD"], group_rows),
                    ("SD 为训练种子的样本标准差（ddof=1），不是置信区间。三个种子构成配对硬件块，汇总不是同硬件效率比较。" if zh else "SD is the sample SD across training seeds (ddof = 1), not a confidence interval. Seeds define paired hardware blocks; pooled results are not a same-hardware efficiency comparison."),
                    "## " + ("逐种子配对差与全部退步" if zh else "Every paired difference and regression"),
                    table(["Vs LR0", "Seed", "ΔNRMSE (pp)", "NRMSE improvement (%)", "ΔNMAE (pp)", "NMAE improvement (%)", "ΔR²"], pair_rows),
                    ("Δ=候选−LR0；误差Δ负值更好，R²Δ正值更好；相对改善正值更好。所有退步、持平均显示，无种子筛选。" if zh else "Δ = candidate − LR0: negative error differences and positive R² differences are better. Positive relative improvement is better. All regressions and ties are retained; no seed is selected."),
                    table(["Vs LR0", "Metric", "Paired Δ mean ± SD (pp)", "Paired relative improvement mean ± SD (%)", "Group-mean relative improvement (%)", "Better / worse / tie seeds"], aggregate_rows),
                    "## " + ("最坏种子" if zh else "Worst seeds"),
                    table(["Schedule", "Metric", "Worst seed(s)", "Error (%)"], [[r["schedule"], r["metric"].upper(), r["worst_seeds"], fmt(r["value"], 100)] for r in context["worst"]]),
                    "## " + ("实际收敛与完整运行耗时" if zh else "Observed convergence and complete run wall time"),
                    table(["Schedule", "Seed", "Best Stage-3 epoch", "Observed Stage-3 epochs", "Child wall time (s)", "Child wall time (min)", "Actual GPU"], timing_rows),
                    ("墙钟以实际训练子进程启动至成功退出计，含初始化、训练、验证、最终预测和保存，不含回传与CPU诊断。不得与仅epoch计时混称，也不把不同GPU速度视为学习率的单独效应。" if zh else "Whole wall time spans the actual training child process to successful exit, including initialization, training, validation, final prediction and saving; transfer and CPU diagnostics are excluded. Epoch-only timing is distinct, and different GPU speeds are not attributed solely to learning rate."),
                    "![Validation convergence](validation_convergence.png)",
                    "## " + ("真实硬件与预训练起点核验" if zh else "Actual hardware and pretraining starting-state audit"),
                    table(["Schedule", "Seed", "Runtime GPU name", "Runtime UUID", "Launch UUID", "UUID evidence"], hardware_rows),
                    ("运行期未记录UUID时保持缺失；launch receipt-only仅说明启动回执的UUID与运行GPU名唯一匹配，不伪称运行期UUID已记录。" if zh else "An absent runtime UUID remains absent. Receipt-only UUID evidence denotes a unique match of launch identity and runtime GPU name, not an independently recorded runtime UUID."),
                    table(["Seed", "Stage", "All tensors equal", "State L2 difference", "Relative state L2 difference", "Max absolute difference"], pre_rows),
                    "## " + ("固定训练批次梯度诊断" if zh else "Fixed-training-batch gradient diagnostics"),
                    ("以下为共享可训练参数固定并集、统一epoch1损失权重下，预测梯度与辅助总梯度的余弦。每行5个完整训练批次是run内探针；批次SD不是训练随机性SD。零范数导致余弦未定义，保留空值。完整raw/common/actual权重、全部参数组和各loss见CSV。" if zh else "The compact table uses the fixed union of shared trainable parameters and common epoch-1 loss weights: forecast versus total auxiliary gradient cosine. Five full training batches are within-run probes; batch SD is not training-seed SD. Zero-norm undefined cosines remain missing. CSVs retain all raw/common/actual weightings, parameter groups and loss terms."),
                    table(["Schedule", "Seed", "Checkpoint", "Defined probes", "Mean cosine", "Within-run probe SD"], grad_rows),
                    table(["Schedule", "Seed", "Stage-2 norm", "Best norm", "Endpoint L2 displacement", "Relative endpoint displacement"], disp_rows),
                    ("统一权重用来区分状态差异与早停轮数引起的辅助损失衰减；actual-checkpoint权重仍单独保存。学习率标量不会直接改变梯度方向。位移不是路径长度或Adam单步更新。原训练每epoch首batch余弦使用非空梯度交集，不能与本表固定并集直接混用。" if zh else "Common weights separate state differences from auxiliary-loss decay at different stopping epochs; actual-checkpoint weights remain separately available. A scalar learning rate does not directly alter gradient direction. Endpoint displacement is neither path length nor an Adam update. Original first-batch epoch cosines use non-None gradient intersections and must not be pooled with this fixed-union table."),
                    "## " + ("核验和解释范围" if zh else "Verification and interpretation limits"),
                    ("九项均已通过有限数值、索引、真实权重、配置/源码、成功退出、逐文件SHA和配对硬件核验。该门禁不等于证明所有预测均无任何科学异常。三种子结果为描述性证据；未进行显著性或等效检验。不读取旧测试择优，不声明学习率最优，不自动替换现有模型。" if zh else "All nine runs passed finite-value, index, actual-checkpoint, config/source, successful-exit, per-file SHA and paired-hardware gates. These gates do not establish that every prediction is scientifically free of anomalies. Three-seed results are descriptive; no significance or equivalence test was performed. No old-test selection, learning-rate optimality claim, or automatic model replacement is made."),
                    "## " + ("可复核产物" if zh else "Auditable artifacts"),
                    table(["Schedule", "Seed", "Metrics", "Predictions", "Preprocessor", "Weights / run"], artifact_rows),
                    "\n".join([link("Frozen source", ROOT / "project"), link("Frozen configurations", ROOT / "configs"), link("Protocol", ROOT / "protocol.json"), link("All diagnostic CSVs", OUT / "diagnostic_source"), link("Input and output SHA manifest", OUT / "completed.json")]),
                    ("本脚本仅保存D盘report；E盘备份是否完成应以实际回传/备份清单为准，不能由报告生成代替。" if zh else "This script writes the report directory on D only. E-drive backup completion must be established by the actual backup manifest, not inferred from report generation."),
                    "## " + ("回复片段" if zh else "Response fragment"),
                    "\n\n".join(response_zh if zh else response_en)]
        outputs["report_zh.md" if zh else "report_en.md"] = "\n\n".join(sections) + "\n"
    outputs["response_fragments.md"] = fragments
    return outputs, compact


def plot(context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in installed else "DejaVu Sans"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family], "font.size": 7,
                         "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "savefig.facecolor": "white"})
    width_mm = 183
    height_mm = 76
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, height_mm / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.27, top=0.80, wspace=0.12)
    colors = {"LR0": "#4C78A8", "LR1": "#B279A2", "LR2": "#D99000"}
    styles = {"LR0": "-", "LR1": "--", "LR2": ":"}
    trajectories = context["data"]["validation_trajectories.csv"]
    plotted, marks, line_count = [], 0, 0
    values = [100 * num(r, "val_nrmse") for r in trajectories]
    pad = max(0.1, (max(values) - min(values)) * 0.08)
    ymax = max(integer(r, "epoch") for r in trajectories)
    for ax, seed, letter in zip(axes, SEEDS, "abc"):
        for schedule in SCHEDULES:
            run = next(r for r in context["runs"] if r["schedule"] == schedule and int(r["seed"]) == seed)
            curve = sorted((r for r in trajectories if r["config_id"] == run["config_id"]), key=lambda r: integer(r, "epoch"))
            epochs = [integer(r, "epoch") for r in curve]
            errors = [100 * num(r, "val_nrmse") for r in curve]
            ax.plot(epochs, errors, color=colors[schedule], linestyle=styles[schedule], linewidth=1, label=schedule)
            best = next(r for r in curve if integer(r, "epoch") == integer(run, "best_epoch"))
            ax.plot(integer(best, "epoch"), 100 * num(best, "val_nrmse"), marker="o", markersize=3,
                    markerfacecolor="white", markeredgecolor=colors[schedule], linestyle="None")
            plotted.extend(curve)
            marks += 1
            line_count += 1
        gpu = run["gpu_name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        ax.set_title(f"{letter}  Seed {seed}\n{gpu}", loc="left", pad=7)
        ax.set_xlabel("Stage-3 epoch")
        ax.set_xlim(0.5, ymax + 0.5)
        ax.set_ylim(min(values) - pad, max(values) + pad)
        ax.set_xticks(sorted(set([1] + list(range(5, ymax + 1, 5)) + [ymax])))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=2)
    axes[0].set_ylabel("Validation NRMSE (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox = text.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    require(not outside, f"Figure text extends outside canvas: {outside}")
    fig.savefig(OUT / "validation_convergence.png", dpi=600)
    fig.savefig(OUT / "validation_convergence.pdf")
    fig.savefig(OUT / "validation_convergence.svg")
    plt.close(fig)
    require(line_count == marks == 9 and len(plotted) == len(trajectories), "Figure lost a run/epoch")
    caption = ("Validation convergence for three predefined learning-rate schedules at a 240-min horizon on wind_farm_A. Panels show the three paired training seeds; all observed Stage-3 epochs are plotted without smoothing or extrapolation. Open circles mark the actual validation-NRMSE-selected checkpoint. Maximum Stage-3 epochs = 20; patience = 5. LR0/LR1/LR2 stage-wise rates are (0.001,0.001,0.0003), (0.001,0.001,0.001), and (0.0003,0.0003,0.0003). Each seed's schedules share the same recorded GPU; seed 13 uses an RTX 4060 and seeds 42/3407 an RTX 4070 SUPER. Epochs are repeated observations within runs, not independent experimental replicates. No confidence interval or significance claim is shown. Source: validation_trajectories.csv, cross-checked against each completed run's loss_history.json.\n")
    atomic(OUT / "validation_convergence_caption.txt", caption)
    qa = {"status": "machine_checks_passed_visual_review_pending", "width_mm": width_mm, "height_mm": height_mm,
          "font": family, "minimum_font_pt": 7, "png_dpi": 600, "editable_svg_text": True, "pdf_fonttype": 42,
          "curves": line_count, "checkpoint_markers": marks, "observed_epochs": len(trajectories), "plotted_epochs": len(plotted),
          "smoothing": False, "extrapolation": False, "sampled_rows": False, "outside_canvas_labels": outside,
          "human_visual_review": "pending; inspect exported PNG/PDF at final size before submission", "statistical_unit": "training seed; n=3 paired blocks",
          "source_sha256": sha(OUT / "validation_trajectories.csv"),
          "exports": [file_item(OUT / ("validation_convergence." + ext)) for ext in ("png", "pdf", "svg")]}
    atomic(OUT / "figure_qa.json", qa)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-index", type=Path)
    args = parser.parse_args()
    context = verify(args)
    if args.check_only:
        print(json.dumps({"status": "ready", "strict_completed_runs": 9, "diagnostic_gate": "passed", "report_generated": False}, ensure_ascii=False))
        return
    prior = OUT / "completed.json"
    if prior.is_file():
        saved = read(prior)
        require(saved["input_fingerprint"] == context["fingerprint"], "A finalized report exists with different inputs; preserve it and review before generating a new version")
        for item in saved["output_files"]:
            path = (OUT / item["name"]).resolve()
            require(OUT.resolve() in path.parents and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Previously finalized report file changed")
        print(json.dumps({"status": "completed_existing_verified", "report": str(OUT)}, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    copied = OUT / "diagnostic_source"
    copied.mkdir(exist_ok=True)
    for name in REQUIRED:
        shutil.copyfile(RESULTS / name, copied / name)
    shutil.copyfile(RESULTS / "completed.json", copied / "completed.json")
    for name in ("validation_runs.csv", "validation_groups.csv", "validation_paired_differences.csv", "validation_trajectories.csv", "pretraining_pair_audit.csv"):
        shutil.copyfile(RESULTS / name, OUT / name)
    write_csv(OUT / "paired_group_summary.csv", context["comparisons"])
    write_csv(OUT / "worst_seeds.csv", context["worst"])
    docs, compact = documents(context)
    write_csv(OUT / "gradient_compact_summary.csv", compact)
    for name, text in docs.items():
        atomic(OUT / name, text)
    qa = plot(context)
    atomic(OUT / "summary.json", {"experiment_id": "energy_learning_rate_20261022", "split": "val", "selection": "none; diagnostic only",
                                   "groups": context["groups"], "paired_differences": context["pairs"], "paired_summaries": context["comparisons"],
                                   "worst_seeds": context["worst"], "significance_tests": None,
                                   "LR0_LR1_pretraining_bitwise_equal": context["done"]["LR0_LR1_pretraining_bitwise_equal"]})
    outputs = [file_item(p, p.relative_to(OUT).as_posix()) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "completed.json"]
    outputs.append(file_item(copied / "completed.json", "diagnostic_source/completed.json"))
    atomic(prior, {"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(), "run_count": 9,
                   "input_fingerprint": context["fingerprint"], "inputs": context["inputs"], "output_files": outputs,
                   "source_receipts": [{"config_id": r["id"], "receipt_path": r["receipt_path"], "receipt_sha256": r["receipt_sha256"], "artifact_directory": str(r["source_dir"])} for r in context["records"]],
                   "figure_status": qa["status"], "training_or_inference_started": False,
                   "backup_status": "not asserted by report generator; check external backup manifest"})
    print(json.dumps({"status": "completed", "report": str(OUT), "runs": 9, "figure_visual_review": "pending"}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "report_completion_claimed": False}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
