"""Strict saved-prediction reanalysis. Never imports a training entry point.

Time inference and training-seed inference have distinct, explicit estimands.
All output writers are restricted to the independently created repair directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np
from scipy import stats
import revision_statistics_frozen as established

HERE = Path(__file__).resolve().parent
PLAN = HERE.parents[1]
PROJECT = Path(".")
SEEDS = (13, 21, 42, 87, 3407)
S3_SEEDS = (13, 42, 3407)
S3_CELLS = (("ACFormer", 1), ("ACFormer", 4), ("ACFormer", 16), ("ACFormer", 48),
            ("ACFormer", 96), ("LightGBM", 48), ("TCN", 96), ("TimesNet", 96))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def output_json(name, value):
    target = HERE / name
    require(target.resolve().is_relative_to(HERE), "Output outside repair directory")
    target.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def output_csv(name, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (HERE / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else clean(v) for k, v in row.items()})


@dataclass
class Run:
    directory: Path
    metadata: dict
    config: dict
    seed: int
    horizon: int
    model: str
    group: str
    variant: str


@dataclass
class Saved:
    run: Run
    timestamp: np.ndarray
    origin: np.ndarray
    turbines: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    valid: np.ndarray
    capacity: np.ndarray
    mse_time: np.ndarray
    mae_time: np.ndarray
    metric: dict
    evidence: dict


def discover(root, mode, issues=None):
    """Metadata only; duplicates are errors, never filesystem-order selection."""
    result, conflicted = {}, set()
    for mp in sorted(Path(root).rglob("metrics.json")):
        m = load_json(mp)
        cfg = m.get("config", {})
        ex = cfg.get("experiment", {})
        if m.get("status") != "completed" or not (mp.parent / "predictions.npz").is_file():
            continue
        cid = cfg.get("tracking", {}).get("config_id", "")
        group, variant = ex.get("group", ""), ex.get("variant", "")
        model, horizon, seed = m.get("model", ex.get("model")), int(m["horizon"]), int(m["seed"])
        if mode == "VF" and ("_main_" not in cid or group != "main"):
            continue
        if mode == "variants" and (horizon != 16 or group not in ("ablation", "physics", "masking", "training")):
            continue
        if mode == "original" and (group != "main" or model == "PI-EEF"):
            continue
        key = (group, variant, seed) if mode == "variants" else (model, horizon, seed)
        if key in result or key in conflicted:
            if mode == "S3" and issues is not None:
                previous = result.pop(key, None)
                issues.append(dict(status="duplicate_source_identity_quarantined", identity=list(key),
                                   paths=([str(previous.directory)] if previous else [])+[str(mp.parent)]))
                conflicted.add(key)
                continue
            raise ValueError(f"Duplicate completed identity {mode}: {key}")
        result[key] = Run(mp.parent, m, cfg, seed, horizon, model, group, variant)
    return result


def protocol(run):
    data = run.config.get("data", {})
    return dict(revision_protocol=data.get("revision_protocol"), seq_len=data.get("seq_len"),
                split=data.get("split"), capacity_mw=data.get("capacity_mw"))


def load_saved(run):
    p = run.directory / "predictions.npz"
    with np.load(p, allow_pickle=False) as z:
        required = {"timestamp", "origin_timestamp", "turbine_ids", "y_true_mw", "y_pred_mw", "target_valid", "seed", "horizon"}
        require(required <= set(z.files), f"Missing real prediction index: {p}: {required-set(z.files)}")
        a = {k: np.array(z[k], copy=True) for k in required}
    y, pred, valid = a["y_true_mw"].astype(float), a["y_pred_mw"].astype(float), a["target_valid"]
    require(y.ndim == 2 and pred.shape == valid.shape == y.shape and valid.dtype.kind == "b", "Invalid prediction axes/mask")
    ids = a["turbine_ids"].astype(str)
    require(ids.ndim == 1 and len(ids) == y.shape[1] and len(set(ids)) == len(ids), "Invalid/duplicate turbine IDs")
    ts, origin = a["timestamp"].astype("datetime64[ns]"), a["origin_timestamp"].astype("datetime64[ns]")
    require(ts.ndim == 1 and ts.shape == origin.shape == (len(y),), "Invalid temporal index axes")
    require(not np.isnat(ts).any() and not np.isnat(origin).any(), "NaT temporal index")
    step = np.timedelta64(15, "m")
    require(len(ts) > run.horizon and np.all(np.diff(ts) == step), "Non-contiguous/unsorted/duplicate 15-minute timeline")
    require(np.all(ts-origin == run.horizon*step), "Forecast origin/horizon mismatch")
    require(int(a["seed"].item()) == run.seed and int(a["horizon"].item()) == run.horizon, "Saved scalar identity mismatch")
    require(np.isfinite(y[valid]).all() and np.isfinite(pred[valid]).all(), "Nonfinite prediction/truth at valid target")
    counts = valid.sum(axis=1)
    require(np.all(counts > 0), "All-invalid timestamp: explicit separate scope needed, never silently remove time")
    capacity = np.asarray(run.config["data"]["capacity_mw"], dtype=float)
    require(capacity.ndim <= 1 and (capacity.ndim == 0 or capacity.shape == (len(ids),)) and np.isfinite(capacity).all() and (capacity > 0).all(), "Invalid capacity")
    preprocess = run.directory / "preprocessing.npz"
    if preprocess.is_file():
        with np.load(preprocess, allow_pickle=False) as z:
            if "capacity_mw" in z.files:
                require(np.array_equal(z["capacity_mw"], capacity), "Saved/config capacity mismatch")
            if "turbine_ids" in z.files:
                require(np.array_equal(z["turbine_ids"].astype(str), ids), "Preprocessing turbine order mismatch")
    require(protocol(run)["revision_protocol"] is not None, "Unverified legacy training/data protocol")
    dp = run.directory / "data_protocol.json"
    if dp.is_file():
        require(load_json(dp) == protocol(run)["revision_protocol"], "Config/actual protocol differs")
    err = np.where(valid, (pred-y)/capacity, 0.)
    mse, mae = (err**2).sum(axis=1)/counts, np.abs(err).sum(axis=1)/counts
    cell_error = (pred-y)[valid]
    metric = dict(nrmse=float(np.sqrt(np.mean((err[valid])**2))), nmae=float(np.mean(np.abs(err[valid]))),
                  r2=float(1-np.sum(cell_error**2)/np.sum((y[valid]-np.mean(y[valid]))**2)), n_valid_cells=int(valid.sum()))
    for key in ("nrmse", "nmae", "r2"):
        require(np.isclose(metric[key], run.metadata["metrics"][key], rtol=1e-5, atol=1e-7), f"Saved metric differs from actual prediction {key}: {p}")
    evidence = dict(directory=str(run.directory), config_id=run.config.get("tracking", {}).get("config_id"),
                    seed=run.seed, horizon=run.horizon, model=run.model, group=run.group, variant=run.variant,
                    protocol=protocol(run), metric=metric, n_time=len(ts), n_turbines=len(ids), files=[])
    for f in (run.directory/"metrics.json", p, preprocess, dp):
        if f.is_file():
            evidence["files"].append(dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f)))
    return Saved(run, ts, origin, ids, y, pred, valid, capacity, mse, mae, metric, evidence)


def assert_paired(reference, actual):
    require(reference.run.horizon == actual.run.horizon and reference.run.seed == actual.run.seed, "Seed/horizon mismatch")
    for field in ("timestamp", "origin", "turbines", "valid", "capacity"):
        require(np.array_equal(getattr(reference, field), getattr(actual, field)), f"Paired mismatch: {field}")
    require(np.array_equal(reference.truth, actual.truth, equal_nan=True), "Paired mismatch: raw truth")
    require(protocol(reference.run) == protocol(actual.run), "Paired data protocol differs")


def inference(reference_loss, candidate_loss, horizon):
    """Difference = candidate minus reference; positive gain = candidate is better."""
    result = established.dm_hac_hln(candidate_loss, reference_loss, horizon)
    result.update(established.paired_block_bootstrap_ci(candidate_loss, reference_loss,
                  horizon=horizon, n_bootstrap=2000, confidence=.95, seed=13))
    result.update(relative_loss_gain_pct=float((1-np.mean(candidate_loss)/np.mean(reference_loss))*100),
                  inference_unit="timestamp_mean_loss_over_identical_valid_turbines", estimand="conditional_on_saved_trained_models",
                  bootstrap_seed=13, alternative="two_sided", loss_difference_direction="candidate_minus_reference")
    return result


def seed_supplement(differences):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    if n > 1 and sd > 0:
        t = stats.ttest_1samp(values, 0.)
        half = float(stats.t.ppf(.975, n-1)*sd/np.sqrt(n))
        tp, tv, interval = float(t.pvalue), float(t.statistic), [float(values.mean()-half), float(values.mean()+half)]
    else:
        tp, tv, interval = None, None, None
    nz = values[values != 0]
    sp = float(stats.binomtest(int((nz > 0).sum()), len(nz), .5, alternative="two-sided").pvalue) if len(nz) else None
    return dict(n_seed=n, mean=float(values.mean()), sample_sd=sd, differences=values.tolist(), paired_t_statistic=tv,
                paired_t_p_raw=tp, across_seed_t_interval95=interval, exact_sign_p_two_sided_raw=sp,
                n_positive=int((values > 0).sum()), n_negative=int((values < 0).sum()), n_zero=int((values == 0).sum()),
                sign_nonzero_n=len(nz), multiplicity="supplementary_reanalysis_no_new_BH_family_no_significance_flag",
                uncertainty_scope="training_seed_difference_conditional_on_this_observed_test_period")


def apply_complete_family(rows, expected, label):
    require(len(rows) == expected and len({x["hypothesis_id"] for x in rows}) == expected, f"Incomplete/duplicate family {label}")
    values = established.bh_adjust([x["p_value"] if x["p_value"] is not None else np.nan for x in rows])
    for row, value in zip(rows, values):
        row.update(bh_p_value=float(value), declared_family=label, declared_family_size=expected,
                   bh_alpha=.05, bh_reject=bool(np.isfinite(value) and value <= .05))


def summarize_group(label, pairs):
    seeds = sorted(pairs)
    output = dict(comparison=label, seeds=seeds, n_seeds=len(seeds), supplementary_reanalysis=True)
    for side in ("full", "candidate"):
        rows = [pairs[s][0 if side == "full" else 1].metric for s in seeds]
        output[side] = {key: dict(mean=float(np.mean([x[key] for x in rows])), sample_sd=float(np.std([x[key] for x in rows], ddof=1)) if len(rows)>1 else None) for key in ("nrmse", "nmae", "r2")}
    output["relative_nrmse_gain_pct"] = 100*(1-output["candidate"]["nrmse"]["mean"]/output["full"]["nrmse"]["mean"])
    output["relative_nmae_gain_pct"] = 100*(1-output["candidate"]["nmae"]["mean"]/output["full"]["nmae"]["mean"])
    output["gain_direction"] = "positive_means_candidate_lower_error; for a removed component this is counter_full evidence"
    output["training_seed_tests"] = {}
    for family, attr in (("mse", "mse_time"), ("mae", "mae_time")):
        diff = [float(getattr(pairs[s][1], attr).mean()-getattr(pairs[s][0], attr).mean()) for s in seeds]
        output["training_seed_tests"][family] = seed_supplement(diff)
    return output


def main():
    start = utc()
    design = load_json(HERE/"design_fixed.json")
    print(json.dumps(dict(event="start", utc=start, pid=os.getpid(), design_sha256=sha(HERE/"design_fixed.json"))), flush=True)
    vf = discover(design["inputs"]["VF_main"], "VF")
    variants = discover(design["inputs"]["variants"], "variants")
    original = discover(design["inputs"]["original_revision_baselines"], "original")
    s3_discovery_issues=[]
    s3 = discover(design["inputs"]["S3_revision"], "S3",s3_discovery_issues)
    output_json("S3_source_collision_audit.json",dict(checked_utc=utc(),issues=s3_discovery_issues,
                policy="quarantine_duplicate_seed_identity_without_choosing_latest_or_best; do_not_block_unrelated_P3_or_original350"))
    require(len(vf) == 25 and len(variants) == 75 and len(original) == 350, "Frozen VF25/variant75/original350 inventory incomplete")
    old_p3 = load_json(PLAN/"交付_20261021/02_实验结果/p3_variant_tests.json")
    family_keys = {(x["group"], x["variant"], int(x["seed"])) for x in old_p3["rows"]}
    require(set(variants) == family_keys and len(family_keys) == 75, "P3 legacy family membership changed")
    membership = dict(P3=[list(x) for x in sorted(family_keys)], original_main=sorted({f"{model}:h{h}" for model,h,s in original}),
                      legacy_P3_source_sha256=sha(PLAN/"交付_20261021/02_实验结果/p3_variant_tests.json"))
    output_json("family_membership_frozen.json", membership)
    inputs, cache = [], {}
    def get(run):
        key = str(run.directory)
        if key not in cache:
            cache[key] = load_saved(run)
            inputs.append(cache[key].evidence)
        return cache[key]
    vf_saved = {(h,s):get(run) for (model,h,s),run in vf.items()}
    p3_rows, p3_groups = [], []
    for group, variant in sorted({(g,v) for g,v,s in variants}):
        pairs = {}
        for seed in SEEDS:
            base, candidate = vf_saved[(16,seed)], get(variants[(group,variant,seed)])
            assert_paired(base, candidate)
            pairs[seed] = (base,candidate)
            for family, attr in (("mse", "mse_time"),("mae", "mae_time")):
                result = inference(getattr(base,attr), getattr(candidate,attr), 16)
                result.update(hypothesis_id=f"{group}:{variant}:seed{seed}", group=group, variant=variant, seed=seed,
                              family=family, loss_unit="normalized_squared_error" if family=="mse" else "normalized_absolute_error",
                              n_valid_cells=base.metric["n_valid_cells"])
                p3_rows.append(result)
        p3_groups.append(summarize_group(f"{group}:{variant}", pairs))
        for seed in SEEDS:
            cache.pop(str(variants[(group,variant,seed)].directory), None)
        print(json.dumps(dict(event="P3_group_done",group=group,variant=variant,utc=utc())),flush=True)
    for family in ("mse","mae"):
        apply_complete_family([x for x in p3_rows if x["family"]==family],75,"inherited_P3_75_"+family)
    output_json("p3_conditional_corrected.json",dict(status="complete",rows=p3_rows,unit="75_saved_variant_seed_models_per_loss_family",method_design_sha256=sha(HERE/"design_fixed.json")))
    output_json("p3_training_seed_supplement.json",dict(status="complete_15_groups",groups=p3_groups,no_new_BH_family=True,no_mean_p=True,no_average_predictions=True))
    output_csv("p3_conditional_corrected.csv",p3_rows)
    summary_rows=[]
    for group in p3_groups:
        row={key:group[key] for key in ("comparison","seeds","n_seeds","relative_nrmse_gain_pct","relative_nmae_gain_pct")}
        for side in ("full","candidate"):
            for metric in ("nrmse","nmae","r2"):
                for stat in ("mean","sample_sd"):row[f"{side}_{metric}_{stat}"]=group[side][metric][stat]
        for fam in ("mse","mae"):
            for key,value in group["training_seed_tests"][fam].items():row[f"{fam}_seed_{key}"]=value
        summary_rows.append(row)
    output_csv("p3_training_seed_supplement.csv",summary_rows)
    main_rows,main_groups=[] ,[]
    for model,horizon in sorted({(m,h) for m,h,s in original}):
        pairs={}
        for seed in SEEDS:
            base,candidate=vf_saved[(horizon,seed)],get(original[(model,horizon,seed)])
            assert_paired(base,candidate)
            pairs[seed]=(base,candidate)
        group=summarize_group(f"original_revision:{model}:h{horizon}",pairs)
        group.update(reference="VF",candidate=model,source_group="original_revision_350",horizon=horizon)
        main_groups.append(group)
        for fam,attr in (("mse","mse_time"),("mae","mae_time")):
            # Average independent trained model LOSSES, never predictions.
            full_loss=np.mean([getattr(pairs[s][0],attr) for s in SEEDS],axis=0)
            baseline_loss=np.mean([getattr(pairs[s][1],attr) for s in SEEDS],axis=0)
            # Main table direction: candidate is VF, reference is the baseline.
            result=inference(baseline_loss,full_loss,horizon)
            result.update(hypothesis_id=f"{model}:h{horizon}",family=fam,baseline=model,horizon=horizon,seeds=list(SEEDS),
                          source_group="original_revision_350",candidate="VF",loss_unit="normalized_squared_error" if fam=="mse" else "normalized_absolute_error",
                          multiplicity="original_main_70_MSE" if fam=="mse" else "supplementary_MAE_raw_no_new_BH_family")
            main_rows.append(result)
        print(json.dumps(dict(event="original_main_cell_done",model=model,horizon=horizon,utc=utc())),flush=True)
        for seed in SEEDS:
            cache.pop(str(original[(model,horizon,seed)].directory), None)
    apply_complete_family([x for x in main_rows if x["family"]=="mse"],70,"original_main_70_MSE")
    output_json("original_revision_main_70_corrected.json",dict(status="complete",model_horizon_comparisons=70,matched_original_baseline_artifacts=350,
                  rows=main_rows,descriptive_groups=main_groups,reference_model="VF",MSE_BH_family_size=70,MAE_BH_applied=False,
                  inference_conditional_on_trained_models=True,no_new_training=True))
    output_csv("original_revision_main_70_corrected.csv",main_rows)
    s3_rows=[]
    for model,horizon in S3_CELLS:
        available=[s for s in S3_SEEDS if (model,horizon,s) in s3]
        row=dict(model=model,horizon=horizon,source_group="S3_baselines_revision",required_common_seeds=list(S3_SEEDS),available_seeds=available,
                 missing_seeds=sorted(set(S3_SEEDS)-set(available)),same_loss_training_implementation_certified=False,
                 training_fairness_caveat="Requires independent training source/loss audit; matching predictions do not prove identical training objective.",BH_applied=False)
        if set(available)!=set(S3_SEEDS):
            row.update(status="waiting_incomplete_three_common_seeds");s3_rows.append(row);continue
        pairs={}
        try:
            for seed in S3_SEEDS:
                base,candidate=vf_saved[(horizon,seed)],get(s3[(model,horizon,seed)])
                assert_paired(base,candidate);pairs[seed]=(base,candidate)
        except ValueError as exc:
            row.update(status="waiting_pairing_or_protocol_rejected",reason=str(exc));s3_rows.append(row);continue
        row.update(status="complete_exact_evaluation_pairing_conditional_not_training_fairness_certified",summary=summarize_group(f"S3_revision:{model}:h{horizon}",pairs),time_tests={})
        for fam,attr in (("mse","mse_time"),("mae","mae_time")):
            full_loss=np.mean([getattr(pairs[s][0],attr) for s in S3_SEEDS],axis=0)
            baseline_loss=np.mean([getattr(pairs[s][1],attr) for s in S3_SEEDS],axis=0)
            row["time_tests"][fam]=inference(baseline_loss,full_loss,horizon)
        s3_rows.append(row)
    output_json("S3_revision_readiness_and_conditional.json",dict(checked_utc=utc(),rows=s3_rows,source_group_separate_from_original70=True,
                  global_status="waiting_entire_S3_24_and_independent_training_audit",legacy_huber_7493_excluded=True,source_discovery_issues=s3_discovery_issues))
    output_json("input_provenance.json",dict(checked_utc=utc(),sources=inputs,count=len(inputs),design_sha256=sha(HERE/"design_fixed.json"),frozen_helper_sha256=sha(HERE/"revision_statistics_frozen.py")))
    output_json("run_receipt.json",dict(status="completed_with_explicit_S3_waiting",started_utc=start,completed_utc=utc(),pid=os.getpid(),
                  P3_actual_variants=75,P3_conditional_rows=150,P3_seed_groups=15,original_revision_baseline_artifacts=350,original_main_cells=70,
                  S3_complete_evaluation_cells=sum(x["status"].startswith("complete_") for x in s3_rows),S3_waiting_cells=sum(x["status"].startswith("waiting_") for x in s3_rows),
                  original_deliverables_modified=False,training_or_inference_executed=False,score_selection_performed=False,
                  local_CUDA=os.environ["CUDA_VISIBLE_DEVICES"],local_threads=2))
    print(json.dumps(dict(event="completed",utc=utc(),sources=len(inputs))),flush=True)


if __name__ == "__main__":
    main()
