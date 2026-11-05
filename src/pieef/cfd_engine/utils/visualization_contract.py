"""Shared scalar-grid and JSON contracts; no solver execution or implicit defaults."""
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def load_speed_data(binfile, meta):
    size = meta.get('size') if isinstance(meta, dict) else None
    if not isinstance(size, list) or len(size) != 3 or any(
        isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n != int(n) or n <= 0 for n in size
    ):
        raise ValueError("size must contain three positive integers [X,Y,Z]")
    nx, ny, nz = map(int, size)
    expected = nx * ny * nz
    if Path(binfile).stat().st_size != expected * 4:
        raise ValueError(f"speed.bin must contain exactly {expected * 4} bytes of float32 data")
    data = np.fromfile(binfile, dtype='<f4').reshape(nz, ny, nx)
    data[~np.isfinite(data)] = np.nan
    return data, nx, ny, nz


def grid_axes(meta, info, dimensions):
    nx, ny, nz = dimensions
    length = positive_number(info.get('domain', {}).get('lt'), 'domain.lt')
    dh = positive_number(meta.get('dh'), 'dh')
    return np.arange(1, nz + 1, dtype=float) * dh, np.linspace(-length / 2, length / 2, ny), np.linspace(-length / 2, length / 2, nx)


class StrictGridInterpolator:
    """Trilinear interpolation that never treats missing contributors as zero.

    A missing corner with exactly zero weight does not invalidate an exact node.
    Axes and query points are ordered Z,Y,X; returned samples retain query order.
    """
    def __init__(self, axes, data):
        self.axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        self.data = np.asarray(data)
        if self.data.shape != tuple(len(axis) for axis in self.axes):
            raise ValueError('Grid shape does not match its axes')
        if any(not len(axis) or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any() for axis in self.axes):
            raise ValueError('Grid axes must be finite and strictly increasing')

    def __call__(self, points):
        query = np.asarray(points, dtype=float)
        if query.shape[-1:] != (len(self.axes),):
            raise ValueError('Query points must use Z,Y,X coordinates')
        shape = query.shape[:-1]
        query = query.reshape(-1, len(self.axes))
        valid = np.isfinite(query).all(axis=1)
        brackets = []
        for index, axis in enumerate(self.axes):
            coordinate = query[:, index]
            valid &= (coordinate >= axis[0]) & (coordinate <= axis[-1])
            if len(axis) == 1:
                lower = upper = np.zeros(len(query), dtype=int)
                weight = np.zeros(len(query))
            else:
                lower = np.clip(np.searchsorted(axis, coordinate, side='right') - 1, 0, len(axis) - 2)
                upper = lower + 1
                weight = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
            brackets.append((lower, upper, weight))
        values = np.zeros(len(query))
        for corner in product((0, 1), repeat=len(self.axes)):
            indices = tuple(brackets[i][bit] for i, bit in enumerate(corner))
            weight = np.ones(len(query))
            for i, bit in enumerate(corner):
                weight *= brackets[i][2] if bit else 1 - brackets[i][2]
            sample = self.data[indices]
            needed = weight != 0
            valid &= ~needed | np.isfinite(sample)
            values += np.multiply(sample, weight, out=np.zeros(len(query)), where=needed & np.isfinite(sample) & np.isfinite(weight))
        values[~valid] = np.nan
        return values.reshape(shape)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(filename, value):
    filename = Path(filename)
    temporary = filename.with_name(f'.{filename.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf8')
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def source_fingerprint(files, context=None):
    records = []
    for filename in sorted(set(Path(name).resolve() for name in files)):
        stat = filename.stat()
        records.append([str(filename), stat.st_size, stat.st_mtime_ns])
    payload = json.dumps([records, context], sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
