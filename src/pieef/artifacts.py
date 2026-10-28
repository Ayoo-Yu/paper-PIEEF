from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_preprocessing(run_dir, bundle, *, revin_missing_policy="legacy"):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    h = bundle.handler
    extra = {}
    if revin_missing_policy == "train_stats_20":
        from .power_training_stats import preprocessing_fields
        extra = preprocessing_fields(bundle)
    elif revin_missing_policy != "legacy":
        raise ValueError("Unsupported RevIN preprocessing policy")
    np.savez_compressed(
        run_dir / "preprocessing.npz", nwp_mean=h.nwp_mean, nwp_std=h.nwp_std,
        scada_wind_mean=h.scada_w_mean, scada_wind_std=h.scada_w_std,
        capacity_mw=bundle.capacity_mw,
        coords=getattr(h, "coords", bundle.rel_coords_tensor[:, 0].cpu().numpy()),
        turbine_ids=np.asarray(getattr(h, "turbine_ids", [f"T{i+1:02d}" for i in range(bundle.num_turbines)]), dtype=str),
        **extra,
    )
    if hasattr(h, "protocol"):
        (run_dir / "data_protocol.json").write_text(json.dumps(h.protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def save_run_predictions(run_dir, bundle, y_true, y_pred, seed, horizon, split="test"):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    count = len(y_true)
    fields = {}
    if hasattr(bundle.handler, "metadata"):
        metadata = bundle.handler.metadata[split]
        if count > len(metadata["timestamp"]):
            raise ValueError("Predictions exceed the saved sample index.")
        for name, values in metadata.items():
            value = np.asarray(values[:count])
            fields[name] = value.astype(str) if value.dtype.kind == "O" else value
    else:
        # Legacy/demo-only data lack timestamps. They cannot support paired formal statistics.
        fields["sample_index"] = np.arange(count)
    np.savez_compressed(
        run_dir / ("predictions.npz" if split == "test" else f"{split}_predictions.npz"),
        y_true_mw=y_true, y_pred_mw=y_pred,
        turbine_ids=np.asarray(getattr(bundle.handler, "turbine_ids", [f"T{i+1:02d}" for i in range(bundle.num_turbines)]), dtype=str),
        seed=int(seed), horizon=int(horizon), **fields,
    )
