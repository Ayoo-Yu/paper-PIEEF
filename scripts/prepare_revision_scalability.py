#!/usr/bin/env python3
"""Prepare the original N=10/20/29/50/100 computational scale comparison.

No training or checkpoint copying occurs. The 50/100 layouts are explicitly
synthetic translations of donor turbine observations; they are not new sites
or independent observations. Every weather value and raw SCADA target comes
from the original wind_farm_A CSV without interpolation or target filling.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (10, 20, 29, 50, 100)
NEW_TRAIN_COUNTS = (10, 20, 50, 100)
DONOR_SEED = 42
HORIZON = 1
MAIN_REFERENCE_ID = "energy_main_PI-EEF_h1_seed42"
WORKER_ID = "recorded-worker"
WEATHER_UNITS = {
    "nwp_direct_solar_radiation": "J/m^2", "nwp_total_sky_direct_shortwave": "J/m^2",
    "Geopotential_10": "m^2/s^2", "nwp_surface_pressure_pa": "Pa", "Mean_sea_level_pressure_10": "Pa",
    "Total_column_vertically-integrated_water_vapour_10": "kg/m^2", "Total_cloud_cover_10": "fraction",
    "nwp_temperature_2m_k": "K", "2_metre_dewpoint_temperature_10": "K", "Skin_temperature_10": "K",
    "Surface_short-wave_(solar)_radiation_downwards_10": "J/m^2", "Medium_cloud_cover_10": "fraction", "High_cloud_cover_10": "fraction",
    "nwp_u10_ms": "m/s", "nwp_v10_ms": "m/s",
    "nwp_u100_ms": "m/s", "nwp_v100_ms": "m/s",
    "200_metre_U_wind_component_10": "m/s", "200_metre_V_wind_component_10": "m/s",
    "nwp_wind_speed10_ms": "m/s", "nwp_wind_speed100_ms": "m/s", "nwp_wind_speed200_ms": "m/s",
}


def within(path, boundary, *, allow_equal=False):
    path, boundary = Path(path).resolve(), Path(boundary).resolve()
    if not path.is_relative_to(boundary) or path == boundary and not allow_equal:
        raise ValueError(f"Path is outside the explicit preparation boundary: {path}")
    return path


def project_path(value, workspace=ROOT):
    path = Path(value)
    if not path.is_absolute():
        path = Path(workspace) / path
    return within(path, workspace)


def relative(path, workspace=ROOT):
    return within(path, workspace).relative_to(Path(workspace).resolve()).as_posix()


def atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        if path.is_file():
            # Preserve an already issued dataset. Never silently change its
            # synthetic layout or donor observations on a second invocation.
            with path.open("rb") as original, temporary.open("rb") as candidate:
                while True:
                    a, b = original.read(1024 * 1024), candidate.read(1024 * 1024)
                    if a != b:
                        raise FileExistsError(f"Prepared scale dataset differs; existing file preserved: {path}")
                    if not a:
                        break
            return
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path, text, *, require_same=False):
    path = Path(path)
    if path.is_file():
        previous = path.read_text(encoding="utf-8")
        if previous == text:
            return
        if require_same:
            raise FileExistsError(f"Previously prepared configuration differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def portable_data(value, workspace=ROOT):
    """Normalize scientific input paths for equality/remote use, nothing else."""
    if isinstance(value, dict):
        return {key: relative(project_path(item, workspace), workspace)
                if key.endswith("_path") and item is not None else portable_data(item, workspace)
                for key, item in value.items()}
    if isinstance(value, list):
        return [portable_data(item, workspace) for item in value]
    return value


def load_source(base, workspace=ROOT, rated_power_unit="kW"):
    data = base["data"]
    protocol = data.get("revision_protocol", {})
    if protocol.get("nwp_timing_rule") != "author_prepared_ecmwf" or protocol.get("nwp_preparation_author_confirmed") is not True:
        raise ValueError("Scale data must retain the author's confirmed original wind_farm_A ECMWF input protocol")
    if protocol.get("power_unit") not in {"kW", "MW"}:
        raise ValueError("The original SCADA power unit must be explicitly declared")
    if protocol.get("input_filling") != "forward-only" or protocol.get("target_policy") != "raw-measured-valid-power":
        raise ValueError("Scale tests must keep causal input filling and raw measured targets")
    source_path = project_path(data["scada_path"], workspace)
    turbine_path = project_path(data["turbines_path"], workspace)
    frame, turbines = pd.read_csv(source_path), pd.read_csv(turbine_path)
    columns = sorted((column for column in frame if re.fullmatch(r"power[1-9]\d*", column)), key=lambda column: int(column[5:]))
    if columns != [f"power{i}" for i in range(1, 30)] or len(turbines) != 29:
        raise ValueError("This original scale matrix requires the actual 29-node wind_farm_A source, not an implicit expansion")
    required_static = {"turbine_id", "x_coord", "y_coord", "elevation_m"}
    if not required_static.issubset(turbines):
        raise ValueError("Every donor needs its actual turbine ID, meter coordinates and elevation_m")
    if turbines["turbine_id"].astype(str).duplicated().any():
        raise ValueError("Original turbine IDs must be unique")
    coordinates = turbines[["x_coord", "y_coord", "elevation_m"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or turbines[["x_coord", "y_coord"]].duplicated().any():
        raise ValueError("Donor coordinates/elevations must be finite and spatially distinct")
    features = protocol.get("nwp_feature_columns")
    if not features or len(set(features)) != len(features) or not set(features).issubset(frame):
        raise ValueError("Declared original weather fields are absent or ambiguous")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Original required NWP fields must remain finite; do not impute them for a scale test")
    timestamp = pd.DatetimeIndex(pd.to_datetime(frame["Timestamp"], errors="raise"))
    if timestamp.hasnans or not timestamp.is_monotonic_increasing or timestamp.has_duplicates:
        raise ValueError("Original SCADA grid must be unique and chronological")
    step_ns = int(protocol.get("step_minutes", 15)) * 60_000_000_000
    if len(timestamp) < 2 or not (np.diff(timestamp.asi8) == step_ns).all():
        raise ValueError("Original timestamps do not match the declared sampling grid")
    if int(data["seq_len"]) + HORIZON >= len(frame):
        raise ValueError("Original dataset is too short for the scale model's history window")
    prefixes = ["power", "turbine_ws"]
    input_prefix = protocol.get("input_power_prefix")
    if input_prefix and input_prefix not in prefixes:
        prefixes.append(input_prefix)
    for prefix in prefixes:
        if not all(f"{prefix}{index}" in frame for index in range(1, 30)):
            raise ValueError(f"Original per-turbine {prefix} fields are incomplete")
    capacity_column = protocol.get("capacity_column")
    if capacity_column:
        if capacity_column not in turbines:
            raise ValueError("Declared original per-turbine capacity field is missing")
        capacity = pd.to_numeric(turbines[capacity_column], errors="raise").to_numpy(dtype=float)
    elif "capacity_mw" in turbines:
        capacity = pd.to_numeric(turbines["capacity_mw"], errors="raise").to_numpy(dtype=float)
    elif "rated_power" in turbines:
        if rated_power_unit not in {"kW", "MW"}:
            raise ValueError("Explicit rated_power unit must be kW or MW")
        capacity = pd.to_numeric(turbines["rated_power"], errors="raise").to_numpy(dtype=float) / (1000 if rated_power_unit == "kW" else 1)
        declared = np.asarray(data.get("capacity_mw"), dtype=float)
        if declared.ndim != 0 or not np.isfinite(declared) or not np.allclose(capacity, declared, rtol=0, atol=1e-10):
            raise ValueError("Static rated_power and the original model's MW capacity disagree")
    else:
        raise ValueError("Actual rated_power or per-turbine MW capacity is required; do not invent it")
    if not np.isfinite(capacity).all() or not (capacity > 0).all():
        raise ValueError("Actual turbine capacities must be positive finite MW")
    turbines = turbines.copy()
    turbines["capacity_mw"] = capacity
    return frame, turbines, prefixes, source_path, turbine_path


def fixed_layout(turbines):
    """One nested 100-node donor map; 50 is exactly its first 50 nodes."""
    if len(turbines) != 29:
        raise ValueError("The donor layout must have exactly 29 actual wind_farm_A nodes")
    coordinates = turbines[["x_coord", "y_coord"]].to_numpy(dtype=float)
    rotor = pd.to_numeric(turbines.get("rotor_diameter", pd.Series(np.zeros(29))), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(rotor).all() or (rotor < 0).any():
        raise ValueError("Donor rotor diameters must be finite nonnegative meters")
    # A whole translated block is separated by at least four donor rotor diameters
    # (one meter minimum if no diameter is supplied). This is computational
    # geometry, not a proposed wind-farm design or new topographic measurement.
    block_offset_m = float(np.ptp(coordinates[:, 0])) + max(4 * float(rotor.max()), 1.0)
    rng = np.random.default_rng(DONOR_SEED)
    donor = list(range(29))
    blocks = [0] * 29
    block = 1
    while len(donor) < 100:
        selected = rng.permutation(29)[:min(29, 100 - len(donor))]
        donor.extend(selected.tolist())
        blocks.extend([block] * len(selected))
        block += 1
    layout = turbines.iloc[donor].reset_index(drop=True).copy()
    layout["donor_turbine_id"] = layout["turbine_id"].astype(str)
    layout["donor_node_index"] = np.asarray(donor) + 1
    layout["donor_power_column"] = [f"power{i + 1}" for i in donor]
    layout["layout_block"] = blocks
    layout["offset_x_m"] = np.asarray(blocks, dtype=float) * block_offset_m
    layout["offset_y_m"] = 0.0
    layout["x_coord"] = layout["x_coord"].to_numpy(dtype=float) + layout["offset_x_m"].to_numpy(dtype=float)
    layout["is_synthetic_node"] = np.asarray(blocks) > 0
    layout["turbine_id"] = [str(turbines.iloc[index]["turbine_id"]) if index < 29 else f"S{index + 1:03d}" for index in range(100)]
    # Preserve actual geolocation only for the original nodes. Donor geolocation
    # is provenance, not the translated synthetic node's latitude/longitude.
    for column in ("latitude", "longitude"):
        if column in layout:
            layout["donor_" + column] = layout[column]
            layout.loc[layout["is_synthetic_node"], column] = np.nan
    layout["coordinate_unit"] = "m"
    layout["elevation_unit"] = "m"
    layout["power_capacity_unit"] = "MW"
    layout["elevation_role"] = np.where(layout["is_synthetic_node"], "copied_donor_attribute", "actual_wind_farm_A_elevation")
    layout["donor_mapping_seed"] = DONOR_SEED
    if layout[["x_coord", "y_coord"]].duplicated().any():
        raise ValueError("The explicit translated synthetic layout contains overlapping nodes")
    return layout, np.asarray(donor), block_offset_m


def prepare_data(base, output_dir, workspace=ROOT, *, rated_power_unit="kW"):
    output_dir = within(output_dir, Path(workspace) / "data/raw")
    frame, turbines, prefixes, source_path, turbine_path = load_source(base, workspace, rated_power_unit)
    if source_path.is_relative_to(output_dir) or turbine_path.is_relative_to(output_dir):
        raise ValueError("Scale output directory must not contain the original source files")
    layout, donors, offset = fixed_layout(turbines)
    per_node = {f"{prefix}{index}" for prefix in prefixes for index in range(1, 30)}
    # Farm-wide Total_Power belongs to the original 29-node farm. It is not used
    # by the model and must not masquerade as a smaller/larger layout's total.
    shared = [column for column in frame if column not in per_node | {"Total_Power"}]
    manifest = {"version": 1, "counts": list(COUNTS), "horizon": HORIZON, "seed": DONOR_SEED,
                "required_worker_id": WORKER_ID, "scope": "computational_scalability_only",
                "source_scada": relative(source_path, workspace), "source_turbines": relative(turbine_path, workspace),
                "source_node_count": 29, "timestamp_policy": "unchanged_original_grid",
                "weather_policy": "all_original_shared_fields_unchanged",
                "target_policy": "original_raw_donor_values_and_missingness_no_filling",
                "donor_mapping": "original_29_then_seed42_permutation_per_translated_block",
                "layout_rule": {"translation_axis": "x", "block_offset_m": offset, "synthetic_elevation": "copied_donor_attribute_not_new_topography"},
                "units": {"x_coord": "m", "y_coord": "m", "elevation_m": "m", "capacity_mw": "MW",
                          "power_columns": base["data"]["revision_protocol"]["power_unit"], "turbine_ws_columns": "m/s",
                          "rated_power": rated_power_unit, "weather_fields": {column: WEATHER_UNITS.get(column, "inherited_original_source_unit_not_reinterpreted") for column in shared if column != "Timestamp"}},
                "points": {}}
    for count in COUNTS:
        if count == 29:
            manifest["points"][str(count)] = {"data_kind": "actual_wind_farm_A_full_layout", "scada_path": relative(source_path, workspace),
                                            "turbines_path": relative(turbine_path, workspace), "new_training": False,
                                            "reuse_reference_config_id": MAIN_REFERENCE_ID, "reuse_condition": "exact_actual_frozen_main_h1_seed42_on_required_worker"}
            continue
        directory = output_dir / f"N{count}"
        selected = donors[:count]
        output = frame[shared].copy()
        node_fields = {f"{prefix}{index + 1}": frame[f"{prefix}{donor + 1}"].to_numpy(copy=True)
                       for prefix in prefixes for index, donor in enumerate(selected)}
        output = pd.concat([output.reset_index(drop=True), pd.DataFrame(node_fields)], axis=1)
        static = layout.iloc[:count].copy()
        scada_path, static_path = directory / "SCADA.csv", directory / "turbines.csv"
        atomic_csv(scada_path, output)
        atomic_csv(static_path, static)
        kind = "actual_wind_farm_A_first_nodes" if count < 29 else "synthetic_donor_expansion"
        manifest["points"][str(count)] = {"data_kind": kind, "scada_path": relative(scada_path, workspace),
                                        "turbines_path": relative(static_path, workspace), "new_training": True,
                                        "synthetic_nodes": max(0, count - 29), "timestamp_count": len(output),
                                        "missing_raw_power_values": int(output[[f"power{i}" for i in range(1, count + 1)]].isna().sum().sum())}
        print(f"Prepared N={count}: {len(output)} original times, {max(0, count - 29)} explicit synthetic nodes; no training", flush=True)
    manifest_path = output_dir / "scalability_manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", require_same=True)
    return manifest


def prepare_configs(base, final, manifest, config_dir, workspace=ROOT):
    if final.get("experiment", {}).get("finalized") is not True:
        raise ValueError("Only the actual validation-selected finalized model can issue formal scale configurations")
    if portable_data(final.get("data", {}), workspace) != portable_data(base.get("data", {}), workspace):
        raise ValueError("Frozen model data differs from the original actual prepared wind_farm_A data")
    config_dir = within(config_dir, Path(workspace) / "artifacts")
    configs = []
    for count in NEW_TRAIN_COUNTS:
        cfg = deepcopy(final)
        point = manifest["points"][str(count)]
        cfg["experiment"].update(exp_type="scalability", group="scalability", model="PI-EEF", models=["PI-EEF"],
                                  variant=f"N{count}_{point['data_kind']}", horizon=HORIZON, seeds=[DONOR_SEED],
                                  eval_scope="computational-scale-only", finalized=True)
        cfg["experiment"].pop("selection_scope", None)
        cfg["data"] = portable_data(cfg["data"], workspace)
        cfg["data"].update(scada_path=point["scada_path"], turbines_path=point["turbines_path"],
                             data_version=str(base["data"]["data_version"]) + f"_scale_N{count}_{point['data_kind']}")
        cfg["data"]["revision_protocol"]["capacity_column"] = "capacity_mw"
        cfg["data"]["revision_protocol"].update(scalability_scope="computational-only", scale_node_count=count,
                                                  synthetic_layout=count > 29, donor_mapping_seed=DONOR_SEED)
        cfg["model"] = portable_data(cfg.get("model", {}), workspace)
        cfg["train"] = portable_data(cfg.get("train", {}), workspace)
        tracking = cfg.setdefault("tracking", {})
        for key in ("control_reference", "control_artifact_root", "required_input", "worker_id", "execution_environment", "execution_project_root"):
            tracking.pop(key, None)
        tracking.update(config_id=f"energy_scalability_N{count}_h1_seed42", artifact_root=f"artifacts/energy_revision/scalability/N{count}",
                        ledger_dir="artifacts/energy_revision/summary", scalability_prepared=True,
                        required_worker_id=WORKER_ID, num_turbines=count, data_kind=point["data_kind"],
                        layout_kind="actual_subsample" if count < 29 else "synthetic_compute",
                        computation_scope="synthetic_computational_scaling" if count > 29 else "real_node_subset_computational_scaling")
        destination = config_dir / f"{tracking['config_id']}.yaml"
        atomic_text(destination, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), require_same=True)
        configs.append(cfg)
    return configs


def prepare_exact_29_alias(final, config_dir, workspace=ROOT):
    """Declare reuse only after the same-cloud actual main source is complete.

    The alias has no new data, weights, or reported results. Its scientific
    inputs/settings are copied exactly from the verified source configuration.
    """
    if final.get("experiment", {}).get("finalized") is not True:
        raise ValueError("N29 exact reuse requires the actual finalized model")
    main_root = Path(workspace) / "artifacts/energy_revision/main"
    candidates = []
    for path in main_root.glob("*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config", {}).get("tracking", {}).get("config_id") == MAIN_REFERENCE_ID:
            candidates.append((within(path, Path(workspace) / "artifacts"), payload))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Multiple main h1/seed42 sources exist; do not choose an arbitrary scale checkpoint")
    path, row = candidates[0]
    source = row["config"]
    if row.get("status") != "completed" or row.get("horizon") != HORIZON or row.get("seed") != DONOR_SEED:
        return None
    if source.get("tracking", {}).get("worker_id") != WORKER_ID or row.get("metrics", {}).get("num_turbines") != 29:
        raise ValueError("N29 reuse requires the actual 29-node main source executed on the same cloud worker 01")
    # Compare with the actual main matrix, including the h1 gradient-recording
    # setting. Output paths and machine tracking do not change that identity.
    specification = importlib.util.spec_from_file_location("scale_reference_plan", ROOT / "src/pieef/revision_plan.py")
    plan = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(plan)
    expected = next(cfg for cfg in plan.training_trials(final, "main", str(Path(workspace) / "artifacts/energy_revision"))
                    if cfg["tracking"]["config_id"] == MAIN_REFERENCE_ID)
    for section in ("data", "model", "train"):
        if portable_data(source.get(section, {}), workspace) != portable_data(expected.get(section, {}), workspace):
            raise ValueError("N29 actual main source differs from the frozen model; exact checkpoint reuse is unavailable")
    # Lazy import only when a real completed reference exists. Prepare-only data
    # generation never loads the Torch stack or fabricates completion metadata.
    sys.path.insert(0, str(ROOT / "src"))
    from pieef.revision_dispatch import completed_run
    local_check = deepcopy(source)
    local_check["tracking"]["artifact_root"] = str(main_root)
    verified = completed_run(local_check)
    if verified is None or verified.resolve() != path:
        raise ValueError("N29 main source does not have strict completed predictions/preprocessing/weights")
    alias = deepcopy(source)
    alias["experiment"].update(exp_type="scalability", group="scalability", variant="N29_exact_main_reuse",
                               finalized=True, eval_scope="computational-scale-only")
    alias["tracking"].update(config_id="energy_scalability_N29_h1_seed42", artifact_root="artifacts/energy_revision/scalability/N29",
                             ledger_dir="artifacts/energy_revision/summary", control_reference=MAIN_REFERENCE_ID,
                             control_artifact_root="artifacts/energy_revision/main", actual_source=relative(path.parent, workspace),
                             scalability_prepared=True, num_turbines=29, layout_kind="actual_full_reuse",
                             required_worker_id=WORKER_ID, new_training=False)
    config_dir = within(config_dir, Path(workspace) / "artifacts")
    atomic_text(config_dir / "energy_scalability_N29_h1_seed42.yaml", yaml.safe_dump(alias, sort_keys=False, allow_unicode=True), require_same=True)
    return alias


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "artifacts/energy_revision/data_ready.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="Prepare data/layouts only; no provisional formal training YAML")
    mode.add_argument("--final-config", type=Path, help="Actual finalized model; prepare the four new scale training configurations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/scalability")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "artifacts/energy_revision/configs/scalability")
    parser.add_argument("--rated-power-unit", choices=("kW", "MW"), default="kW", help="Explicit original turbines.csv rated_power unit (wind_farm_A: kW)")
    args = parser.parse_args(argv)
    base = yaml.safe_load(project_path(args.base_config).read_text(encoding="utf-8"))
    final = None
    if args.final_config is not None:
        final = yaml.safe_load(project_path(args.final_config).read_text(encoding="utf-8"))
        if final.get("experiment", {}).get("finalized") is not True or portable_data(final.get("data", {})) != portable_data(base.get("data", {})):
            raise ValueError("Actual frozen model must be finalized and match prepared wind_farm_A data before issuing scale configurations")
    manifest = prepare_data(base, project_path(args.output_dir), rated_power_unit=args.rated_power_unit)
    if final is not None:
        configs = prepare_configs(base, final, manifest, project_path(args.config_dir))
        alias = prepare_exact_29_alias(final, project_path(args.config_dir))
        print(f"Prepared {len(configs)} new formal configurations for CUDA worker {WORKER_ID}", flush=True)
        if alias is not None:
            print("N29 actual same-worker checkpoint alias prepared; no checkpoint or scientific result was copied", flush=True)
        else:
            print(f"N29 awaits exact completed {MAIN_REFERENCE_ID} on {WORKER_ID}", flush=True)
    else:
        print("Prepared data/layouts only. N50/100 are synthetic computational tests; no final configuration or checkpoint has been created.", flush=True)


if __name__ == "__main__":
    main()
