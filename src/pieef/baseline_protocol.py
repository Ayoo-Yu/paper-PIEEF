"""Fixed effective baseline settings, independent of PI-EEF validation selection.

Pure standard-library code so the queue publisher does not initialize Torch.
The constructors/trainers in scripts/train_baselines.py consume these settings;
PI-EEF's width, layers, losses and stage learning rates are not baseline inputs.
"""
from __future__ import annotations

from copy import deepcopy

ARCHITECTURES = {
    "Persistence": {"rule": "last_causal_power_observation"},
    "GRU": {"hidden_dim": 64},
    "TCN": {"channels": 32},
    "DLinear": {"seq_len": "data.seq_len"},
    "iTransformer": {"seq_len": "data.seq_len", "d_model": 128, "n_heads": 4},
    "PatchTST": {"seq_len": "data.seq_len", "patch_len": 16, "stride": 8, "d_model": 64},
    "ACFormer": {"seq_len": "data.seq_len", "d_model": 96, "patch_kernel": 10, "stride": 2,
                 "n_heads": 8, "n_layers": 1},
    "STAN": {"seq_len": "data.seq_len", "hidden": 64, "gru_hidden": 200, "gat_hidden": 64,
             "gravity_alpha": 0.35, "gravity_beta": 0.37, "gravity_r": 30.0,
             "node_static": "data.turbines_path", "dist_matrix": "actual_data_bundle"},
    "Wake-GNN": {"seq_len": "data.seq_len", "hidden": 64, "kw": 0.05,
                 "coords_xy": "data.turbines_path", "rotor_d": "data.turbines_path"},
    "Jensen-Net": {"seq_len": "data.seq_len", "hidden": 64, "kw": 0.05,
                   "coords_xy": "data.turbines_path", "rotor_d": "data.turbines_path"},
    "TimesNet": {"seq_len": "data.seq_len", "d_model": 64},
    "TimeMixer": {"seq_len": "data.seq_len", "hidden": 64},
    "XGBoost": {"objective": "reg:squarederror", "n_estimators": 200, "max_depth": 6,
                "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                "tree_method": "hist", "random_state": "experiment.seeds[0]"},
    "LightGBM": {"objective": "regression", "n_estimators": 300, "learning_rate": 0.05,
                 "num_leaves": 64, "subsample": 0.8, "colsample_bytree": 0.8,
                 "random_state": "experiment.seeds[0]"},
}
TREE_MODELS = {"XGBoost", "LightGBM"}
DL_MODELS = set(ARCHITECTURES) - TREE_MODELS - {"Persistence"}
FIXED_DL_TRAIN = {"batch_size": 16, "epochs": 50, "lr": 0.001, "patience": 5, "min_delta": 0.0}


def effective_settings(model, group="main"):
    if model not in ARCHITECTURES:
        raise ValueError("Independent baseline protocol supports only the 14 submitted baselines")
    options = {"baseline_factory": model, "architecture": deepcopy(ARCHITECTURES[model]),
               "weather_dimension": "declared_site_fields_plus_3", "weather_semantics": "actual_data_bundle"}
    train = deepcopy(FIXED_DL_TRAIN) if model in DL_MODELS else {"batch_size": 16}
    protocol = {"version": 1, "scope": "fixed_multifarm_baseline" if group == "multifarm" else "fixed_submitted_baseline", "model": model,
                "factory": "scripts/train_baselines.py", "model_settings": deepcopy(options),
                "training_settings": deepcopy(train),
                "loss": "l1" if model == "ACFormer" else "mse" if model in DL_MODELS else "fixed_estimator_or_persistence",
                "checkpoint_selection": "validation_loss" if model in DL_MODELS else "fixed_protocol",
                "pi_selection_dependency": False}
    return options, train, protocol


def independent_baseline_config(cfg):
    result = deepcopy(cfg)
    model = result["experiment"]["model"]
    options, train, protocol = effective_settings(model, result["experiment"].get("group", "main"))
    # A changed effective baseline budget must not be silently reset by a PI scan.
    for key, expected in train.items():
        actual = result.get("train", {}).get(key, expected)
        if actual is not None and actual != expected:
            raise ValueError(f"Fixed baseline protocol changed its effective {key}: {actual}")
    if any(result.get("train", {}).get(key) is not None for key in ("max_batches_per_epoch", "max_test_batches")):
        raise ValueError("Independent baselines require the full actual dataset")
    result["model"], result["train"], result["baseline_protocol"] = options, train, protocol
    result["experiment"].update(independent_baseline_protocol=True, finalized=False)
    result.setdefault("baselines", {})["models"] = [model]
    validate_independent_baseline(result)
    return result


def validate_independent_baseline(cfg):
    """Return True for a fully verified independent protocol; a bare flag fails."""
    exp = cfg.get("experiment", {})
    if exp.get("independent_baseline_protocol") is not True:
        return False
    group = exp.get("group")
    if group not in {"main", "multifarm"} or exp.get("selection_scope") is not None:
        raise ValueError("Independent baseline protocol is only for fixed main baseline runs")
    model = exp.get("model")
    options, train, protocol = effective_settings(model, group)
    if cfg.get("model") != options or cfg.get("train") != train or cfg.get("baseline_protocol") != protocol:
        raise ValueError("Independent baseline effective architecture/training contract differs")
    if exp.get("models") != [model] or cfg.get("baselines", {}).get("models") != [model]:
        raise ValueError("An independent baseline task must identify exactly its fixed model")
    seeds = exp.get("seeds")
    allowed_seeds = (13, 42, 3407) if group == "multifarm" else (13, 21, 42, 87, 3407)
    if not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int or seeds[0] not in allowed_seeds:
        raise ValueError("Independent baseline requires one original main seed")
    if exp.get("horizon") not in ((1, 4, 16) if group == "multifarm" else (1, 4, 16, 48, 96)):
        raise ValueError("Independent baseline requires one original main horizon")
    if group == "multifarm" and (exp.get("site") not in {"wind_farm_B", "kelmarsh"} or exp.get("variant") != "fixed_submitted_baseline"):
        raise ValueError("Independent multi-farm baseline requires the original site and fixed variant")
    if exp.get("finalized") is not False:
        raise ValueError("Independent baseline must not claim PI-EEF/global model finalization")
    return True
