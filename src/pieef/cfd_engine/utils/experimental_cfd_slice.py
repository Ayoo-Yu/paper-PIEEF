#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental CFD slice backend.

Purpose:
1. Read authoritative CFD results from an OpenFOAM `.foam` file when present,
   otherwise fall back to the exported `internal.vtu`.
2. Build a lightweight vector-volume cache on the server.
3. Serve arbitrary plane slices sampled from that cache for isolated web labs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
import fcntl
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from visualization_contract import json_safe, positive_number, source_fingerprint, write_json_atomic


CACHE_VERSION = 3
DEFAULT_TARGET_CELLS = 1_500_000
DEFAULT_MIN_DIMS = (64, 64, 24)
DEFAULT_MAX_DIMS = (220, 220, 96)
DEFAULT_SLICE_RESOLUTION = (220, 160)
DEFAULT_PARTICLE_COUNT = 24_000
MIN_PARTICLE_COUNT = 4_000
MAX_PARTICLE_COUNT = 60_000


def log(message: str) -> None:
    print(message, file=sys.stderr)


def json_dump(payload: dict) -> None:
    json.dump(json_safe(payload), sys.stdout, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("零长度法向量无效。")
    return vector / norm


def choose_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    tangent = None
    for axis in candidates:
        cross = np.cross(axis, normal)
        if np.linalg.norm(cross) > 1e-6:
            tangent = normalize(cross)
            break
    if tangent is None:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    bitangent = normalize(np.cross(normal, tangent))
    return tangent, bitangent


def generate_bbox_corners(bounds: list[float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ], dtype=np.float64)


def compute_projected_extents(bounds_m: list[float], origin_m: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray) -> tuple[float, float, float, float]:
    corners = generate_bbox_corners(bounds_m)
    relative = corners - origin_m[None, :]
    u_coords = relative @ tangent
    v_coords = relative @ bitangent
    pad_u = max(1.0, 0.03 * max(1.0, float(u_coords.max() - u_coords.min())))
    pad_v = max(1.0, 0.03 * max(1.0, float(v_coords.max() - v_coords.min())))
    return (
        float(u_coords.min() - pad_u),
        float(u_coords.max() + pad_u),
        float(v_coords.min() - pad_v),
        float(v_coords.max() + pad_v),
    )


@dataclass
class SourceSelection:
    kind: str
    path: Path


def numeric_time(value: str) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else -math.inf
    except ValueError:
        return -math.inf


def select_source(run_dir: Path) -> SourceSelection:
    foam_files = sorted(run_dir.glob("*.foam")) + sorted(run_dir.glob("*.OpenFOAM"))
    if foam_files:
        return SourceSelection("foam", foam_files[0])

    processed_internal = run_dir / "VTK" / "processed" / "internal.vtu"
    if processed_internal.exists():
        return SourceSelection("internal_vtu", processed_internal)

    vtk_run_dirs = sorted((run_dir / "VTK").glob("run_*"), key=lambda item: numeric_time(item.name[4:]), reverse=True)
    for vtk_run_dir in vtk_run_dirs:
        candidate = vtk_run_dir / "internal.vtu"
        if candidate.exists():
            return SourceSelection("internal_vtu", candidate)

    raise FileNotFoundError("未找到可用的 .foam 或 internal.vtu 源文件。")


def load_case_info(case_dir: Path) -> dict:
    info_path = case_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到 info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def resolve_scale(info: dict) -> float:
    return positive_number(info.get("mesh", {}).get("scale"), "mesh.scale")


def source_files(source: SourceSelection) -> list[Path]:
    files = [source.path]
    if source.kind == "foam":
        run = source.path.parent
        times = [item for item in run.iterdir() if item.is_dir() and math.isfinite(numeric_time(item.name))]
        if times:
            latest = max(times, key=lambda item: float(item.name))
            files.extend(item for item in latest.rglob('*') if item.is_file())
        constant = run / 'constant'
        if constant.exists():
            files.extend(item for item in constant.rglob('*') if item.is_file())
        control = run / 'system' / 'controlDict'
        if control.exists():
            files.append(control)
    return files


def cache_storage_valid(metadata: dict, cache_dir: Path) -> bool:
    try:
        nx, ny, nz = metadata['dims']
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (nx, ny, nz)):
            return False
        volume_path = Path(metadata['cacheFile'])
        texture_path = Path(metadata['volumeTextureFile'])
        if volume_path.parent.resolve() != cache_dir.resolve() or texture_path.parent.resolve() != cache_dir.resolve():
            return False
        if texture_path.stat().st_size != nx * ny * nz:
            return False
        with np.load(volume_path, allow_pickle=False) as payload:
            return (payload['vectors'].shape == (nz, ny, nx, 3)
                    and payload['valid'].shape == payload['speed'].shape == (nz, ny, nx)
                    and np.array_equal(payload['dims'], [nx, ny, nz])
                    and np.allclose(payload['spacing_m'], metadata['spacing_m'])
                    and np.allclose(payload['origin_m'], metadata['origin_m']))
    except (ValueError, KeyError, OSError, EOFError):
        return False


def read_dataset(source: SourceSelection) -> pv.DataSet:
    if source.kind == "foam":
        reader_cls = getattr(pv, "POpenFOAMReader", None) or getattr(pv, "OpenFOAMReader", None)
        if reader_cls is None:
            raise RuntimeError("当前 PyVista 环境不支持 OpenFOAM Reader。")

        reader = reader_cls(str(source.path))
        for attr, value in (
            ("cell_to_point_creation", True),
            ("decompose_polyhedra", True),
            ("skip_zero_time", False),
        ):
            if hasattr(reader, attr):
                try:
                    setattr(reader, attr, value)
                except Exception:
                    pass

        times = [float(value) for value in reader.time_values if np.isfinite(value)]
        if not times:
            raise ValueError("OpenFOAM source has no readable time steps")
        reader.set_active_time_value(max(times))
        data = reader.read()
        if isinstance(data, pv.MultiBlock):
            for key in ("internalMesh", "internalMesh-region0"):
                try:
                    block = data[key]
                except Exception:
                    block = None
                if block is not None:
                    data = block
                    break
            else:
                for block in data:
                    if block is not None:
                        data = block
                        break
        if data is None:
            raise RuntimeError(f"无法从 OpenFOAM 结果中解析内部网格: {source.path}")
    else:
        data = pv.read(source.path)

    if "U" not in getattr(data, "point_data", {}):
        if "U" in getattr(data, "cell_data", {}):
            data = data.cell_data_to_point_data(pass_cell_data=True)
        else:
            raise KeyError("结果数据中未找到速度矢量场 U。")

    vectors = np.asarray(data.point_data["U"])
    if not data.n_points or vectors.shape != (data.n_points, 3):
        raise ValueError("U must contain one 3-component vector per point")
    return data


def compute_volume_dims(bounds_model: list[float], target_cells: int) -> np.ndarray:
    lengths = np.array([
        max(1e-6, bounds_model[1] - bounds_model[0]),
        max(1e-6, bounds_model[3] - bounds_model[2]),
        max(1e-6, bounds_model[5] - bounds_model[4]),
    ], dtype=np.float64)
    scale = float((target_cells / float(np.prod(lengths))) ** (1.0 / 3.0))
    dims = np.round(lengths * scale).astype(np.int32) + 1
    dims = np.maximum(dims, np.array(DEFAULT_MIN_DIMS, dtype=np.int32))
    dims = np.minimum(dims, np.array(DEFAULT_MAX_DIMS, dtype=np.int32))
    return dims


def ensure_cache(case_dir: Path, *, target_cells: int = DEFAULT_TARGET_CELLS, force_rebuild: bool = False) -> dict:
    cache_dir = case_dir / 'flow_lab_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    # HTTP handlers launch separate Python processes. Serialize builders here.
    with (cache_dir / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ensure_cache(case_dir, target_cells=target_cells, force_rebuild=force_rebuild)


def _ensure_cache(case_dir: Path, *, target_cells: int = DEFAULT_TARGET_CELLS, force_rebuild: bool = False) -> dict:
    info = load_case_info(case_dir)
    scale = resolve_scale(info)
    run_dir = case_dir / "run"
    source = select_source(run_dir)
    dependencies = source_files(source)
    fingerprint = source_fingerprint(dependencies, {"scale": scale})

    cache_dir = case_dir / "flow_lab_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "vector_volume_metadata.json"
    generation = uuid.uuid4().hex
    volume_path = cache_dir / f"vector_volume_{generation}.npz"
    volume_texture_path = cache_dir / f"speed_volume_{generation}.u8.bin"
    case_id = case_dir.name

    if not force_rebuild and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                int(metadata.get("cacheVersion", -1)) == CACHE_VERSION
                and metadata.get("sourceKind") == source.kind
                and metadata.get("sourcePath") == str(source.path)
                and metadata.get("targetCells") == int(target_cells)
                and metadata.get("sourceFingerprint") == fingerprint
                and cache_storage_valid(metadata, cache_dir)
            ):
                metadata["cacheReady"] = True
                metadata["cacheDir"] = str(cache_dir)
                return metadata
        except Exception:
            pass

    started = time.time()
    dataset = read_dataset(source)
    bounds_model = [float(value) for value in dataset.bounds]
    if not np.isfinite(bounds_model).all() or any(bounds_model[i + 1] <= bounds_model[i] for i in (0, 2, 4)):
        raise ValueError("CFD source must have a finite nonzero 3D extent")
    dims = compute_volume_dims(bounds_model, target_cells)
    lengths_model = np.array([
        bounds_model[1] - bounds_model[0],
        bounds_model[3] - bounds_model[2],
        bounds_model[5] - bounds_model[4],
    ], dtype=np.float64)
    spacing_model = lengths_model / np.maximum(dims - 1, 1)

    image = pv.ImageData(
        dimensions=tuple(int(value) for value in dims),
        spacing=tuple(float(value) for value in spacing_model),
        origin=(bounds_model[0], bounds_model[2], bounds_model[4]),
    )
    sampled = image.sample(dataset)

    vectors = np.asarray(sampled["U"], dtype=np.float32)
    valid_mask = np.asarray(sampled["vtkValidPointMask"], dtype=np.float32)

    nz, ny, nx = int(dims[2]), int(dims[1]), int(dims[0])
    vectors_wind_farm_B = vectors.reshape((nz, ny, nx, 3))
    valid_wind_farm_B = ((valid_mask.reshape((nz, ny, nx)) > 0.5) & np.isfinite(vectors_wind_farm_B).all(axis=-1)).astype(np.float32)
    vectors_wind_farm_B[valid_wind_farm_B == 0] = np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        speed_wind_farm_B = np.linalg.norm(vectors_wind_farm_B.astype(np.float64), axis=-1).astype(np.float32)
    valid_wind_farm_B *= np.isfinite(speed_wind_farm_B)
    vectors_wind_farm_B[valid_wind_farm_B == 0] = np.nan
    speed_wind_farm_B[valid_wind_farm_B == 0] = np.nan

    origin_m = np.array([
        bounds_model[0] / scale,
        bounds_model[2] / scale,
        bounds_model[4] / scale,
    ], dtype=np.float64)
    spacing_m = np.array([
        spacing_model[0] / scale,
        spacing_model[1] / scale,
        spacing_model[2] / scale,
    ], dtype=np.float64)
    bounds_m = [
        bounds_model[0] / scale,
        bounds_model[1] / scale,
        bounds_model[2] / scale,
        bounds_model[3] / scale,
        bounds_model[4] / scale,
        bounds_model[5] / scale,
    ]
    valid_points = valid_wind_farm_B > 0.5
    valid_speed = speed_wind_farm_B[valid_points]
    if valid_speed.size == 0:
        raise ValueError("CFD sampling produced no valid finite vector samples")
    speed_min = float(np.nanmin(valid_speed))
    speed_max = float(np.nanmax(valid_speed))

    texture_u8 = np.zeros(speed_wind_farm_B.shape, dtype=np.uint8)
    if speed_max - speed_min < 1e-6:
        texture_u8[valid_points] = 255
    else:
        normalized = np.clip((speed_wind_farm_B[valid_points] - speed_min) / (speed_max - speed_min), 0.0, 1.0)
        texture_u8[valid_points] = np.clip(np.round(normalized * 254.0) + 1.0, 1.0, 255.0).astype(np.uint8)

    published = False
    try:
        np.savez(
            volume_path,
            vectors=vectors_wind_farm_B.astype(np.float32),
            speed=speed_wind_farm_B.astype(np.float32),
            valid=valid_wind_farm_B.astype(np.float32),
            origin_m=origin_m.astype(np.float64),
            spacing_m=spacing_m.astype(np.float64),
            dims=np.array([nx, ny, nz], dtype=np.int32),
            bounds_m=np.array(bounds_m, dtype=np.float64),
            center_m=np.array([
                0.5 * (bounds_m[0] + bounds_m[1]),
                0.5 * (bounds_m[2] + bounds_m[3]),
                0.5 * (bounds_m[4] + bounds_m[5]),
            ], dtype=np.float64),
            scale=np.array([scale], dtype=np.float64),
        )
        volume_texture_path.write_bytes(texture_u8.tobytes(order="C"))

        metadata = {
            "cacheVersion": CACHE_VERSION,
            "cacheGeneration": generation,
            "sourceFingerprint": fingerprint,
            "cacheReady": True,
            "cacheDir": str(cache_dir),
            "cacheFile": str(volume_path),
            "volumeTextureFile": str(volume_texture_path),
            "volumeTextureUrl": f"/uploads/{case_id}/flow_lab_cache/{volume_texture_path.name}",
            "sourceKind": source.kind,
            "sourcePath": str(source.path),
            "targetCells": int(target_cells),
            "dims": [int(nx), int(ny), int(nz)],
            "voxelCount": int(nx * ny * nz),
            "origin_m": [float(value) for value in origin_m],
            "spacing_m": [float(value) for value in spacing_m],
            "bounds_m": [float(value) for value in bounds_m],
            "scale": float(scale),
            "validRatio": float(np.mean(valid_wind_farm_B)),
            "speedRange": [
                speed_min,
                speed_max,
            ],
            "volumeTextureRange": [1.0 / 255.0, 1.0],
            "speedP95": float(np.nanpercentile(valid_speed, 95.0)),
            "speedP995": float(np.nanpercentile(valid_speed, 99.5)),
            "buildSeconds": round(time.time() - started, 3),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if source_fingerprint(source_files(source), {"scale": scale}) != fingerprint:
            volume_path.unlink(missing_ok=True)
            volume_texture_path.unlink(missing_ok=True)
            raise RuntimeError("CFD source changed while building the cache")
        write_json_atomic(metadata_path, metadata)
        published = True
    finally:
        if not published:
            volume_path.unlink(missing_ok=True)
            volume_texture_path.unlink(missing_ok=True)
    return metadata


def load_cached_volume(case_dir: Path, *, target_cells: int = DEFAULT_TARGET_CELLS, ensure: bool = True) -> tuple[dict, dict]:
    if ensure:
        metadata = ensure_cache(case_dir, target_cells=target_cells)
    else:
        metadata_path = case_dir / "flow_lab_cache" / "vector_volume_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("矢量体缓存不存在。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    cache_file = Path(metadata["cacheFile"])
    if not cache_storage_valid(metadata, case_dir / "flow_lab_cache"):
        raise ValueError("Vector cache dimensions or files are invalid")
    with np.load(cache_file, allow_pickle=False) as payload:
        volume = {
            "vectors": payload["vectors"].astype(np.float32),
            "speed": payload["speed"].astype(np.float32),
            "valid": payload["valid"].astype(np.float32),
            "origin_m": payload["origin_m"].astype(np.float64),
            "spacing_m": payload["spacing_m"].astype(np.float64),
            "dims": payload["dims"].astype(np.int32),
            "bounds_m": payload["bounds_m"].astype(np.float64),
            "center_m": payload["center_m"].astype(np.float64),
        }
    return metadata, volume


def build_particle_cloud(case_dir: Path, metadata: dict, volume: dict, *, particle_count: int, force_rebuild: bool = False) -> dict:
    cache_dir = case_dir / "flow_lab_cache"
    sample_count = int(np.clip(int(particle_count), MIN_PARTICLE_COUNT, MAX_PARTICLE_COUNT))
    generation = metadata["cacheGeneration"]
    particle_path = cache_dir / f"particle_cloud_{generation}_{sample_count}.f32.bin"
    particle_meta_path = cache_dir / f"particle_cloud_{generation}_{sample_count}.json"
    case_id = case_dir.name

    if not force_rebuild and particle_path.exists() and particle_meta_path.exists():
        try:
            payload = json.loads(particle_meta_path.read_text(encoding="utf-8"))
            if payload.get("cacheGeneration") != generation or particle_path.stat().st_size != sample_count * 8 * 4:
                raise ValueError("Particle cache does not match the current vector cache")
            payload["particleCloudFile"] = str(particle_path)
            payload["particleCloudUrl"] = f"/uploads/{case_id}/flow_lab_cache/{particle_path.name}"
            return payload
        except Exception:
            pass

    started = time.time()
    valid_mask = (np.asarray(volume["valid"] > 0.5, dtype=bool) & np.isfinite(volume["speed"]) & np.isfinite(volume["vectors"]).all(axis=-1))
    flat_valid = np.flatnonzero(valid_mask.reshape(-1))
    if flat_valid.size == 0:
        raise ValueError("当前体缓存里没有可用于粒子云的有效体素。")

    flat_speed = volume["speed"].reshape(-1).astype(np.float32)
    flat_vectors = volume["vectors"].reshape((-1, 3)).astype(np.float32)
    candidate_speed = flat_speed[flat_valid]
    candidate_vectors = flat_vectors[flat_valid]
    candidate_speed_max = float(np.nanmax(candidate_speed))
    candidate_speed_p95 = float(np.nanpercentile(candidate_speed, 95.0))
    speed_scale = max(1e-6, candidate_speed_p95)

    # Favor energetic regions, but retain a baseline weight so slower recirculation zones still appear.
    weights = 0.14 + np.power(np.clip(candidate_speed / speed_scale, 0.0, 1.4), 1.2)
    weights = np.where(np.isfinite(weights), weights, 0.14).astype(np.float64)
    weights_sum = float(np.sum(weights))
    if weights_sum < 1e-9:
        weights = np.full_like(weights, 1.0 / max(1, weights.size), dtype=np.float64)
    else:
        weights /= weights_sum

    replace = flat_valid.size < sample_count
    seed = int((metadata["voxelCount"] * 1315423911 + sample_count * 2654435761) % (2 ** 32))
    rng = np.random.default_rng(seed=seed)
    chosen_indices = rng.choice(flat_valid.size, size=sample_count, replace=replace, p=weights)
    chosen_flat = flat_valid[chosen_indices]

    nx, ny, nz = (int(value) for value in volume["dims"])
    z_idx, y_idx, x_idx = np.unravel_index(chosen_flat, (nz, ny, nx))
    origin_m = volume["origin_m"]
    spacing_m = volume["spacing_m"]

    positions = np.column_stack((
        origin_m[0] + (x_idx.astype(np.float32) * spacing_m[0]),
        origin_m[1] + (y_idx.astype(np.float32) * spacing_m[1]),
        origin_m[2] + (z_idx.astype(np.float32) * spacing_m[2]),
    )).astype(np.float32)
    jitter = (rng.random((sample_count, 3), dtype=np.float32) - 0.5) * spacing_m.astype(np.float32)[None, :]
    positions += jitter
    positions = np.clip(positions, volume["bounds_m"][[0, 2, 4]], volume["bounds_m"][[1, 3, 5]])

    directions = flat_vectors[chosen_flat].copy()
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    valid_dir = direction_norm[:, 0] > 1e-6
    directions[valid_dir] /= direction_norm[valid_dir]
    directions[~valid_dir] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    speeds = flat_speed[chosen_flat].astype(np.float32)
    phase = rng.random(sample_count, dtype=np.float32)

    records = np.column_stack((
        positions,
        directions.astype(np.float32),
        speeds,
        phase,
    )).astype(np.float32)
    temporary_particle = particle_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary_particle.write_bytes(records.astype("<f4").tobytes(order="C"))
        os.replace(temporary_particle, particle_path)
    finally:
        temporary_particle.unlink(missing_ok=True)

    payload = {
        "success": True,
        "cacheGeneration": generation,
        "particleCount": int(sample_count),
        "particleStrideFloats": 8,
        "particleFields": [
            "x_m",
            "y_m",
            "z_m",
            "dir_x",
            "dir_y",
            "dir_z",
            "speed_mps",
            "phase01",
        ],
        "particleCloudFile": str(particle_path),
        "particleCloudUrl": f"/uploads/{case_id}/flow_lab_cache/{particle_path.name}",
        "particleCloudByteLength": int(records.nbytes),
        "particleSpeedRange": [
            float(np.nanmin(speeds)),
            float(np.nanmax(speeds)),
        ],
        "particleSpeedP95": float(np.nanpercentile(speeds, 95.0)),
        "sampleBias": "weighted_by_speed",
        "buildSeconds": round(time.time() - started, 3),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(particle_meta_path, payload)
    return payload


def build_normal_from_mode(mode: str, azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    if mode == "xy":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if mode == "xz":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if mode == "yz":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    az = math.radians(float(azimuth_deg))
    tilt = math.radians(float(tilt_deg))
    return normalize(np.array([
        math.sin(tilt) * math.cos(az),
        math.sin(tilt) * math.sin(az),
        math.cos(tilt),
    ], dtype=np.float64))


def interpolate_trilinear(volume: dict, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = volume["vectors"]
    valid = volume["valid"]
    origin = volume["origin_m"]
    spacing = volume["spacing_m"]
    nx, ny, nz = (int(value) for value in volume["dims"])

    fx = (points_m[:, 0] - origin[0]) / spacing[0]
    fy = (points_m[:, 1] - origin[1]) / spacing[1]
    fz = (points_m[:, 2] - origin[2]) / spacing[2]

    inside = (
        (fx >= 0.0) & (fx <= (nx - 1)) &
        (fy >= 0.0) & (fy <= (ny - 1)) &
        (fz >= 0.0) & (fz <= (nz - 1))
    )

    result_vectors = np.full((points_m.shape[0], 3), np.nan, dtype=np.float32)
    result_valid = np.zeros((points_m.shape[0],), dtype=np.float32)
    if not np.any(inside):
        return result_vectors, result_valid

    fx_i = fx[inside]
    fy_i = fy[inside]
    fz_i = fz[inside]

    x0 = np.floor(fx_i).astype(np.int32)
    y0 = np.floor(fy_i).astype(np.int32)
    z0 = np.floor(fz_i).astype(np.int32)

    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    z1 = np.clip(z0 + 1, 0, nz - 1)
    x0 = np.clip(x0, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1)
    z0 = np.clip(z0, 0, nz - 1)

    wx = (fx_i - x0).astype(np.float32)
    wy = (fy_i - y0).astype(np.float32)
    wz = (fz_i - z0).astype(np.float32)

    def sample(arr, xi, yi, zi):
        return arr[zi, yi, xi]

    weighted = np.zeros((len(fx_i), 3), dtype=np.float64)
    all_valid = np.ones(len(fx_i), dtype=bool)
    for xi, xw in ((x0, 1 - wx), (x1, wx)):
        for yi, yw in ((y0, 1 - wy), (y1, wy)):
            for zi, zw in ((z0, 1 - wz), (z1, wz)):
                weight = xw * yw * zw
                needed = weight > 0
                corner = sample(vectors, xi, yi, zi)
                usable = (sample(valid, xi, yi, zi) > 0.5) & np.isfinite(corner).all(axis=1)
                all_valid &= ~needed | usable
                weighted += np.where((needed & usable)[:, None], np.nan_to_num(corner) * weight[:, None], 0)
    weighted[~all_valid] = np.nan
    result_vectors[inside] = weighted.astype(np.float32)
    result_valid[inside] = all_valid.astype(np.float32)
    return result_vectors, result_valid


def read_optional_text(file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8")


def parse_performance_rows(content: str | None, source_name: str) -> list[dict]:
    if not content or not content.strip():
        return []
    rows = []
    for index, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = re.split(r"\s+", line)
        if len(tokens) < 4:
            raise ValueError(f"{source_name} 第 {index} 行列数不足。")
        values = [float(tokens[i]) for i in range(4)]
        if not np.isfinite(values).all():
            raise ValueError(f"{source_name} 第 {index} 行含非有限值。")
        rows.append({
            "speed": values[0],
            "power": values[1],
            "ct": values[2],
            "fn": values[3],
        })
    return rows


def parse_curve_rows(content: str | None, source_name: str) -> list[dict]:
    if not content or not content.strip():
        return []
    rows = []
    for index, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = re.split(r"\s+", line)
        if len(tokens) < 3:
            raise ValueError(f"{source_name} 第 {index} 行列数不足。")
        wind_speed, power, ct = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
        rows.append({
            "windSpeed": wind_speed,
            "power": power,
            "ct": ct,
        })
    rows.sort(key=lambda item: item["windSpeed"])
    if any(not np.isfinite(list(row.values())).all() for row in rows) or any(right["windSpeed"] <= left["windSpeed"] for left, right in zip(rows, rows[1:])):
        raise ValueError(f"{source_name} requires finite samples and unique increasing speeds")
    return rows


def parse_real_high_rows(content: str | None, scale: float) -> list[dict]:
    if not content or not content.strip():
        return []
    rows = []
    for index, raw_line in enumerate(content.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([\w-]+)\s+on\s+([\w-]+)", line)
        if not match:
            raise ValueError(f"Output02-realHigh 第 {index + 1} 行格式无效")
        numeric_tokens = re.split(r"\s+", line.replace(match.group(0), "", 1).strip())
        if len(numeric_tokens) < 5:
            raise ValueError(f"Output02-realHigh 第 {index + 1} 行列数不足")
        dxy, solver_x, solver_y, terrain_z, actual_hub_z = (float(value) for value in numeric_tokens[:5])
        if not np.isfinite([dxy, solver_x, solver_y, terrain_z, actual_hub_z]).all():
            raise ValueError(f"Output02-realHigh 第 {index + 1} 行含非有限坐标")
        rows.append({
            "solverIndex": index,
            "solverLabel": match.group(1),
            "node": match.group(2),
            "dxyModel": dxy,
            "solverXModel": solver_x,
            "solverYModel": solver_y,
            "terrainZModel": terrain_z,
            "actualHubZModel": actual_hub_z,
            "dxy_m": dxy / scale,
            "solverX_m": solver_x / scale,
            "solverY_m": solver_y / scale,
            "terrainZ_m": terrain_z / scale,
            "actualHubZ_m": actual_hub_z / scale,
        })
    return rows


def interpolate_curve_by_speed(rows: list[dict] | None, wind_speed: float | None) -> dict:
    if not rows or wind_speed is None or not np.isfinite(wind_speed):
        return {"inRange": False, "power": None, "ct": None}
    if len(rows) == 1:
        row = rows[0]
        if abs(float(wind_speed) - row["windSpeed"]) <= 1e-9:
            return {"inRange": True, "power": row["power"], "ct": row["ct"]}
        return {"inRange": False, "power": 0.0, "ct": 0.0}

    if wind_speed < rows[0]["windSpeed"] or wind_speed > rows[-1]["windSpeed"]:
        return {"inRange": False, "power": 0.0, "ct": 0.0}

    if abs(wind_speed - rows[-1]["windSpeed"]) <= 1e-9:
        return {"inRange": True, "power": rows[-1]["power"], "ct": rows[-1]["ct"]}

    for left, right in zip(rows[:-1], rows[1:]):
        if not (left["windSpeed"] <= wind_speed < right["windSpeed"]):
            continue
        span = right["windSpeed"] - left["windSpeed"]
        ratio = 0.0 if abs(span) < 1e-12 else (wind_speed - left["windSpeed"]) / span
        return {
            "inRange": True,
            "power": left["power"] + (right["power"] - left["power"]) * ratio,
            "ct": left["ct"] + (right["ct"] - left["ct"]) * ratio,
        }
    return {"inRange": False, "power": 0.0, "ct": 0.0}


def build_disk_offsets(sample_resolution: int) -> np.ndarray:
    resolution = max(3, int(sample_resolution))
    offsets: list[tuple[float, float]] = []
    for iy in range(-resolution, resolution + 1):
        for iz in range(-resolution, resolution + 1):
            lateral = iy / resolution
            vertical = iz / resolution
            if lateral * lateral + vertical * vertical > 1.0:
                continue
            offsets.append((lateral, vertical))
    return np.asarray(offsets, dtype=np.float64)


def average_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def std_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.std(values))


def compute_alignment_deg(mean_vector: np.ndarray) -> float | None:
    magnitude = float(np.linalg.norm(mean_vector))
    if magnitude < 1e-9:
        return None
    cosine = float(np.clip(mean_vector[0] / magnitude, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_vector_disk_metrics(volume: dict, center_point_m: np.ndarray, rotor_diameter_m: float, sample_resolution: int) -> dict:
    radius = float(rotor_diameter_m) * 0.5
    if radius <= 0:
        return {
            "sampleCount": 0,
            "validSampleCount": 0,
            "coverageRatio": 0.0,
        }

    offsets = build_disk_offsets(sample_resolution)
    points_m = np.column_stack((
        np.full(offsets.shape[0], center_point_m[0], dtype=np.float64),
        center_point_m[1] + offsets[:, 0] * radius,
        center_point_m[2] + offsets[:, 1] * radius,
    ))
    vectors, valid = interpolate_trilinear(volume, points_m)
    valid_mask = valid >= 0.5
    if not np.any(valid_mask):
        return {
            "sampleCount": int(offsets.shape[0]),
            "validSampleCount": 0,
            "coverageRatio": 0.0,
        }

    valid_vectors = vectors[valid_mask].astype(np.float64)
    ux = valid_vectors[:, 0]
    uy = valid_vectors[:, 1]
    uz = valid_vectors[:, 2]
    speed_mag = np.linalg.norm(valid_vectors, axis=1)
    positive_ux = np.clip(ux, 0.0, None)
    top_half = ux[offsets[valid_mask, 1] > 0]
    bottom_half = ux[offsets[valid_mask, 1] < 0]
    mean_vector = np.mean(valid_vectors, axis=0)

    return {
        "sampleCount": int(offsets.shape[0]),
        "validSampleCount": int(valid_vectors.shape[0]),
        "coverageRatio": float(np.mean(valid_mask)),
        "meanUx": average_or_none(ux),
        "equivalentUx": float(np.cbrt(np.mean(positive_ux ** 3))),
        "meanSpeedMag": average_or_none(speed_mag),
        "stdUx": std_or_none(ux),
        "stdSpeedMag": std_or_none(speed_mag),
        "meanUy": average_or_none(uy),
        "meanUz": average_or_none(uz),
        "reverseFlowRatio": float(np.mean(ux < 0.0)),
        "misalignmentDeg": compute_alignment_deg(mean_vector),
        "topBottomUxDelta": (
            float(np.mean(top_half) - np.mean(bottom_half))
            if top_half.size > 0 and bottom_half.size > 0
            else None
        ),
        "speedMinusUx": (
            float(np.mean(speed_mag) - np.mean(ux))
            if speed_mag.size > 0 and ux.size > 0
            else None
        ),
    }


def compute_vector_window_metrics(
    volume: dict,
    hub_point_m: np.ndarray,
    rotor_diameter_m: float,
    dx_m: float,
    sample_resolution: int,
) -> dict:
    if rotor_diameter_m <= 0 or dx_m <= 0:
        return {
            "sampleCount": 0,
            "validSampleCount": 0,
            "coverageRatio": 0.0,
        }

    radial_limit = rotor_diameter_m * 0.25
    axial_half_span = 3.0 * dx_m
    center_x = hub_point_m[0] - rotor_diameter_m
    disk_offsets = build_disk_offsets(sample_resolution)
    nominal_spacing = max(float(volume["spacing_m"][0]), 1.0)
    axial_count = max(3, min(15, int(round((2.0 * axial_half_span) / nominal_spacing)) + 1))
    axial_axis = np.linspace(-axial_half_span, axial_half_span, axial_count, dtype=np.float64)

    points: list[list[float]] = []
    for axial_offset in axial_axis:
        for lateral_norm, vertical_norm in disk_offsets:
            points.append([
                center_x + axial_offset,
                hub_point_m[1] + lateral_norm * radial_limit,
                hub_point_m[2] + vertical_norm * radial_limit,
            ])

    points_m = np.asarray(points, dtype=np.float64)
    vectors, valid = interpolate_trilinear(volume, points_m)
    valid_mask = valid >= 0.5
    if not np.any(valid_mask):
        return {
            "sampleCount": int(points_m.shape[0]),
            "validSampleCount": 0,
            "coverageRatio": 0.0,
        }

    valid_vectors = vectors[valid_mask].astype(np.float64)
    ux = valid_vectors[:, 0]
    speed_mag = np.linalg.norm(valid_vectors, axis=1)
    positive_ux = np.clip(ux, 0.0, None)
    mean_vector = np.mean(valid_vectors, axis=0)

    return {
        "sampleCount": int(points_m.shape[0]),
        "validSampleCount": int(valid_vectors.shape[0]),
        "coverageRatio": float(np.mean(valid_mask)),
        "meanUx": average_or_none(ux),
        "equivalentUx": float(np.cbrt(np.mean(positive_ux ** 3))),
        "meanSpeedMag": average_or_none(speed_mag),
        "stdUx": std_or_none(ux),
        "stdSpeedMag": std_or_none(speed_mag),
        "reverseFlowRatio": float(np.mean(ux < 0.0)),
        "misalignmentDeg": compute_alignment_deg(mean_vector),
        "speedMinusUx": (
            float(np.mean(speed_mag) - np.mean(ux))
            if speed_mag.size > 0 and ux.size > 0
            else None
        ),
    }


def build_slice(case_dir: Path, args: argparse.Namespace) -> dict:
    metadata, volume = load_cached_volume(case_dir, target_cells=args.target_cells, ensure=True)

    normal = build_normal_from_mode(args.mode, args.azimuth_deg, args.tilt_deg)
    tangent, bitangent = choose_basis(normal)
    center_m = volume["center_m"]
    bounds_m = volume["bounds_m"].tolist()

    corners = generate_bbox_corners(bounds_m)
    projected_n = (corners - center_m[None, :]) @ normal
    min_offset = float(projected_n.min())
    max_offset = float(projected_n.max())
    offset_m = float(np.clip(args.offset_m, min_offset, max_offset))
    origin_m = center_m + normal * offset_m

    min_u, max_u, min_v, max_v = compute_projected_extents(bounds_m, origin_m, tangent, bitangent)
    res_x = max(32, int(args.resolution_x))
    res_y = max(32, int(args.resolution_y))
    u_axis = np.linspace(min_u, max_u, res_x, dtype=np.float64)
    v_axis = np.linspace(min_v, max_v, res_y, dtype=np.float64)
    uu, vv = np.meshgrid(u_axis, v_axis)
    points_m = (
        origin_m[None, None, :]
        + uu[..., None] * tangent[None, None, :]
        + vv[..., None] * bitangent[None, None, :]
    ).reshape((-1, 3))

    vectors_flat, valid_flat = interpolate_trilinear(volume, points_m)
    vectors_grid = vectors_flat.reshape((res_y, res_x, 3))
    valid_grid = valid_flat.reshape((res_y, res_x))
    plane_u = np.tensordot(vectors_grid, tangent, axes=([2], [0])).astype(np.float32)
    plane_v = np.tensordot(vectors_grid, bitangent, axes=([2], [0])).astype(np.float32)
    speed_grid = np.linalg.norm(vectors_grid, axis=2).astype(np.float32)

    valid_mask = valid_grid >= 0.5
    coverage = float(np.mean(valid_mask))
    if np.any(valid_mask):
        speed_min = float(np.nanmin(speed_grid[valid_mask]))
        speed_max = float(np.nanmax(speed_grid[valid_mask]))
        speed_p95 = float(np.nanpercentile(speed_grid[valid_mask], 95.0))
    else:
        speed_min = None
        speed_max = None
        speed_p95 = None

    return json_safe({
        "success": True,
        "sourceKind": metadata["sourceKind"],
        "sourcePath": metadata["sourcePath"],
        "cacheVersion": metadata["cacheVersion"],
        "cacheDims": metadata["dims"],
        "plane": {
            "mode": args.mode,
            "normal": [float(value) for value in normal],
            "tangent": [float(value) for value in tangent],
            "bitangent": [float(value) for value in bitangent],
            "origin_m": [float(value) for value in origin_m],
            "offset_m": offset_m,
            "offsetRange_m": [min_offset, max_offset],
            "uRange_m": [float(min_u), float(max_u)],
            "vRange_m": [float(min_v), float(max_v)],
            "resolution": [res_x, res_y],
        },
        "bounds_m": bounds_m,
        "uAxis_m": [float(value) for value in u_axis],
        "vAxis_m": [float(value) for value in v_axis],
        "speed": speed_grid.tolist(),
        "planeU": plane_u.tolist(),
        "planeV": plane_v.tolist(),
        "validMask": valid_mask.astype(np.uint8).tolist(),
        "stats": {
            "coverage": coverage,
            "speedMin": speed_min,
            "speedMax": speed_max,
            "speedP95": speed_p95,
        },
    })


def build_particles_payload(case_dir: Path, args: argparse.Namespace) -> dict:
    metadata, volume = load_cached_volume(case_dir, target_cells=args.target_cells, ensure=True)
    particle_payload = build_particle_cloud(
        case_dir,
        metadata,
        volume,
        particle_count=args.particle_count,
        force_rebuild=bool(args.force_rebuild),
    )
    return {
        "success": True,
        "sourceKind": metadata["sourceKind"],
        "sourcePath": metadata["sourcePath"],
        "cacheVersion": metadata["cacheVersion"],
        "cacheDims": metadata["dims"],
        "bounds_m": metadata["bounds_m"],
        "scale": metadata["scale"],
        "speedRange": metadata["speedRange"],
        "speedP95": metadata["speedP95"],
        "speedP995": metadata["speedP995"],
        **particle_payload,
    }


def build_turbine_payload(case_dir: Path, args: argparse.Namespace) -> dict:
    info = load_case_info(case_dir)
    scale = resolve_scale(info)
    if bool(getattr(args, "force_rebuild", False)):
        ensure_cache(case_dir, target_cells=args.target_cells, force_rebuild=True)
    metadata, volume = load_cached_volume(case_dir, target_cells=args.target_cells, ensure=True)

    turbines = list(info.get("turbines", []))
    if not turbines:
        raise ValueError("info.json 中缺少风机信息。")

    run_dir = case_dir / "run"
    init_rows = parse_performance_rows(
        read_optional_text(run_dir / "Output" / "Output04-U-P-Ct-fn(INIT)"),
        "Output04-U-P-Ct-fn(INIT)",
    )
    adjust_rows = parse_performance_rows(
        read_optional_text(run_dir / "Output" / "Output06-U-P-Ct-fn(ADJUST)"),
        "Output06-U-P-Ct-fn(ADJUST)",
    )
    real_high_rows = parse_real_high_rows(
        read_optional_text(run_dir / "Output" / "Output02-realHigh"),
        scale,
    )
    for rows, name in ((init_rows, "INIT"), (adjust_rows, "ADJUST"), (real_high_rows, "realHigh")):
        if rows and len(rows) != len(turbines):
            raise ValueError(f"{name} row count does not match info.turbines")
    run_input_path = run_dir / "Input" / "input.json"
    run_input = json.loads(run_input_path.read_text(encoding="utf-8")) if run_input_path.exists() else {}
    lc2_raw = run_input.get("mesh", {}).get("lc2", info.get("mesh", {}).get("lc2", []))
    if isinstance(lc2_raw, (int, float)):
        lc2_values = [float(lc2_raw)]
    else:
        lc2_values = [float(value) for value in lc2_raw if np.isfinite(float(value))]

    warnings: list[str] = []
    curve_cache: dict[str, list[dict] | None] = {}

    def resolve_dx_m(turbine: dict) -> float | None:
        if not lc2_values:
            return None
        model_index = int(turbine.get("model", 1) or 1) - 1
        if model_index < 0 or model_index >= len(lc2_values):
            return float(lc2_values[0])
        return float(lc2_values[model_index])

    def load_curve(model_id: object) -> list[dict] | None:
        key = str(model_id or "").strip()
        if not key:
            return None
        if key in curve_cache:
            return curve_cache[key]
        candidates = [
            run_dir / "Input" / f"{key}-U-P-Ct.txt",
            case_dir / "customCurves" / f"{key}-U-P-Ct.txt",
        ]
        for candidate in candidates:
            content = read_optional_text(candidate)
            if not content:
                continue
            curve_cache[key] = parse_curve_rows(content, candidate.name)
            return curve_cache[key]
        warnings.append(f"缺少模型 {key} 的性能曲线文件，矢量功率映射将为空。")
        curve_cache[key] = None
        return None

    turbine_rows: list[dict] = []
    for index, turbine in enumerate(turbines):
        real_row = real_high_rows[index] if index < len(real_high_rows) else None
        if real_row is None:
            warnings.append(f"风机 {turbine.get('name', turbine.get('id', index + 1))} 缺少 Output02-realHigh 行，无法做矢量诊断。")
            continue

        rotor_diameter_m = float(turbine.get("d", 0) or 0)
        dx_m = resolve_dx_m(turbine)
        hub_point_m = np.array([
            real_row["solverX_m"],
            real_row["solverY_m"],
            real_row["actualHubZ_m"],
        ], dtype=np.float64)

        hub_vector, hub_valid = interpolate_trilinear(volume, hub_point_m.reshape((1, 3)))
        hub_vec = hub_vector[0].astype(np.float64)
        hub_valid_ratio = float(hub_valid[0])
        hub_speed_mag = float(np.linalg.norm(hub_vec)) if hub_valid_ratio >= 0.5 else None
        hub_ux = float(hub_vec[0]) if hub_valid_ratio >= 0.5 else None

        disk_metrics = compute_vector_disk_metrics(volume, hub_point_m, rotor_diameter_m, args.sample_resolution)
        window_metrics = compute_vector_window_metrics(volume, hub_point_m, rotor_diameter_m, float(dx_m or 0.0), args.sample_resolution)

        curve_rows = load_curve(turbine.get("model"))
        window_curve = interpolate_curve_by_speed(curve_rows, window_metrics.get("meanUx"))
        rotor_curve = interpolate_curve_by_speed(curve_rows, disk_metrics.get("equivalentUx"))
        adjust_row = adjust_rows[index] if index < len(adjust_rows) else None
        init_row = init_rows[index] if index < len(init_rows) else None

        adjust_power = adjust_row["power"] if adjust_row else None
        vector_window_power = window_curve.get("power")
        vector_rotor_power = rotor_curve.get("power")
        vector_window_gap = (
            float(vector_window_power - adjust_power)
            if vector_window_power is not None and adjust_power is not None
            else None
        )
        vector_rotor_gap = (
            float(vector_rotor_power - adjust_power)
            if vector_rotor_power is not None and adjust_power is not None
            else None
        )

        turbine_rows.append({
            "solverIndex": index + 1,
            "id": turbine.get("id", f"WT-{index + 1}"),
            "name": turbine.get("name", turbine.get("id", f"WT-{index + 1}")),
            "modelId": turbine.get("model"),
            "init": init_row,
            "adjust": adjust_row,
            "solverX_m": real_row["solverX_m"],
            "solverY_m": real_row["solverY_m"],
            "terrainZ_m": real_row["terrainZ_m"],
            "actualHubZ_m": real_row["actualHubZ_m"],
            "dx_m": dx_m,
            "hubUx": hub_ux,
            "hubSpeedMag": hub_speed_mag,
            "hubCoverageRatio": hub_valid_ratio,
            "diskMeanUx": disk_metrics.get("meanUx"),
            "diskEquivalentUx": disk_metrics.get("equivalentUx"),
            "diskMeanSpeedMag": disk_metrics.get("meanSpeedMag"),
            "diskMisalignmentDeg": disk_metrics.get("misalignmentDeg"),
            "diskReverseFlowRatio": disk_metrics.get("reverseFlowRatio"),
            "diskCoverageRatio": disk_metrics.get("coverageRatio"),
            "diskTopBottomUxDelta": disk_metrics.get("topBottomUxDelta"),
            "diskSpeedMinusUx": disk_metrics.get("speedMinusUx"),
            "windowMeanUx": window_metrics.get("meanUx"),
            "windowEquivalentUx": window_metrics.get("equivalentUx"),
            "windowMeanSpeedMag": window_metrics.get("meanSpeedMag"),
            "windowMisalignmentDeg": window_metrics.get("misalignmentDeg"),
            "windowReverseFlowRatio": window_metrics.get("reverseFlowRatio"),
            "windowCoverageRatio": window_metrics.get("coverageRatio"),
            "windowSpeedMinusUx": window_metrics.get("speedMinusUx"),
            "curvePowerAtWindowUx": vector_window_power,
            "curveCtAtWindowUx": window_curve.get("ct"),
            "curvePowerAtDiskEquivalentUx": vector_rotor_power,
            "curveCtAtDiskEquivalentUx": rotor_curve.get("ct"),
            "vectorWindowPowerGapToSolver": vector_window_gap,
            "vectorRotorPowerGapToSolver": vector_rotor_gap,
        })

    def avg_from(field: str) -> float | None:
        values = np.asarray([row[field] for row in turbine_rows if row.get(field) is not None], dtype=np.float64)
        return average_or_none(values)

    def sum_from(field: str) -> float | None:
        values = [row[field] for row in turbine_rows if row.get(field) is not None]
        if len(values) != len(turbine_rows) or not values:
            return None
        return float(np.sum(np.asarray(values, dtype=np.float64)))

    summary = {
        "turbineCount": len(turbine_rows),
        "averageWindowMeanUx": avg_from("windowMeanUx"),
        "averageDiskEquivalentUx": avg_from("diskEquivalentUx"),
        "averageWindowMeanSpeedMag": avg_from("windowMeanSpeedMag"),
        "averageDiskMeanSpeedMag": avg_from("diskMeanSpeedMag"),
        "averageWindowMisalignmentDeg": avg_from("windowMisalignmentDeg"),
        "averageDiskMisalignmentDeg": avg_from("diskMisalignmentDeg"),
        "averageWindowReverseFlowRatio": avg_from("windowReverseFlowRatio"),
        "averageDiskReverseFlowRatio": avg_from("diskReverseFlowRatio"),
        "totalCurvePowerAtWindowUx": sum_from("curvePowerAtWindowUx"),
        "totalCurvePowerAtDiskEquivalentUx": sum_from("curvePowerAtDiskEquivalentUx"),
        "averageAbsoluteWindowPowerGap": None,
        "averageAbsoluteDiskPowerGap": None,
        "highWindowReverseFlowCount": int(sum((row.get("windowReverseFlowRatio") or 0.0) >= 0.1 for row in turbine_rows)),
        "highDiskReverseFlowCount": int(sum((row.get("diskReverseFlowRatio") or 0.0) >= 0.1 for row in turbine_rows)),
        "highWindowMisalignmentCount": int(sum((row.get("windowMisalignmentDeg") or 0.0) >= 15.0 for row in turbine_rows)),
        "highDiskMisalignmentCount": int(sum((row.get("diskMisalignmentDeg") or 0.0) >= 15.0 for row in turbine_rows)),
        "windowCloserOnPowerCount": int(sum(
            row.get("vectorWindowPowerGapToSolver") is not None
            and row.get("vectorRotorPowerGapToSolver") is not None
            and abs(row["vectorWindowPowerGapToSolver"]) < abs(row["vectorRotorPowerGapToSolver"])
            for row in turbine_rows
        )),
        "diskCloserOnPowerCount": int(sum(
            row.get("vectorWindowPowerGapToSolver") is not None
            and row.get("vectorRotorPowerGapToSolver") is not None
            and abs(row["vectorRotorPowerGapToSolver"]) < abs(row["vectorWindowPowerGapToSolver"])
            for row in turbine_rows
        )),
    }

    summary["averageAbsoluteWindowPowerGap"] = average_or_none(
        np.asarray([abs(row["vectorWindowPowerGapToSolver"]) for row in turbine_rows if row.get("vectorWindowPowerGapToSolver") is not None], dtype=np.float64)
    )
    summary["averageAbsoluteDiskPowerGap"] = average_or_none(
        np.asarray([abs(row["vectorRotorPowerGapToSolver"]) for row in turbine_rows if row.get("vectorRotorPowerGapToSolver") is not None], dtype=np.float64)
    )

    return {
        "success": True,
        "sourceKind": metadata["sourceKind"],
        "sourcePath": metadata["sourcePath"],
        "cacheVersion": metadata["cacheVersion"],
        "cacheDims": metadata["dims"],
        "sampleResolution": int(args.sample_resolution),
        "sourceFrame": "solver_coordinates_from_internal_vtu",
        "summary": summary,
        "warnings": list(dict.fromkeys(warnings)),
        "turbines": turbine_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental CFD slice backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser("metadata", help="Build or inspect vector-volume cache")
    metadata_parser.add_argument("--case-dir", required=True, help="Case directory path")
    metadata_parser.add_argument("--target-cells", type=int, default=DEFAULT_TARGET_CELLS)
    metadata_parser.add_argument("--force-rebuild", action="store_true")

    slice_parser = subparsers.add_parser("slice", help="Extract an arbitrary plane slice from cached vector volume")
    slice_parser.add_argument("--case-dir", required=True, help="Case directory path")
    slice_parser.add_argument("--target-cells", type=int, default=DEFAULT_TARGET_CELLS)
    slice_parser.add_argument("--mode", choices=("xy", "xz", "yz", "oblique"), default="xy")
    slice_parser.add_argument("--offset-m", type=float, default=0.0)
    slice_parser.add_argument("--azimuth-deg", type=float, default=35.0)
    slice_parser.add_argument("--tilt-deg", type=float, default=55.0)
    slice_parser.add_argument("--resolution-x", type=int, default=DEFAULT_SLICE_RESOLUTION[0])
    slice_parser.add_argument("--resolution-y", type=int, default=DEFAULT_SLICE_RESOLUTION[1])

    particle_parser = subparsers.add_parser("particles", help="Build a sparse 3D particle cloud from cached vector volume")
    particle_parser.add_argument("--case-dir", required=True, help="Case directory path")
    particle_parser.add_argument("--target-cells", type=int, default=DEFAULT_TARGET_CELLS)
    particle_parser.add_argument("--particle-count", type=int, default=DEFAULT_PARTICLE_COUNT)
    particle_parser.add_argument("--force-rebuild", action="store_true")

    turbine_parser = subparsers.add_parser("turbines", help="Sample turbine diagnostics from cached vector volume")
    turbine_parser.add_argument("--case-dir", required=True, help="Case directory path")
    turbine_parser.add_argument("--target-cells", type=int, default=DEFAULT_TARGET_CELLS)
    turbine_parser.add_argument("--sample-resolution", type=int, default=11)
    turbine_parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"工况目录不存在: {case_dir}")

    if args.command == "metadata":
        payload = ensure_cache(
            case_dir,
            target_cells=max(250_000, int(args.target_cells)),
            force_rebuild=bool(args.force_rebuild),
        )
        payload["success"] = True
        json_dump(payload)
        return 0

    if args.command == "slice":
        payload = build_slice(case_dir, args)
        json_dump(payload)
        return 0

    if args.command == "particles":
        payload = build_particles_payload(case_dir, args)
        json_dump(payload)
        return 0

    if args.command == "turbines":
        payload = build_turbine_payload(case_dir, args)
        json_dump(payload)
        return 0

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"[experimental_cfd_slice] {type(exc).__name__}: {exc}")
        json_dump({
            "success": False,
            "error": str(exc),
            "errorType": type(exc).__name__,
        })
        sys.exit(1)
