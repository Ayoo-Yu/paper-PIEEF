"""Shared geometry operations for tiny CLI previews and production AGL clipping."""
import argparse
import gzip
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyvista as pv


def clip_height_range(mesh, heights, lower, upper):
    """Clip line segments, interpolating their point arrays at slab boundaries."""
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Height bounds must be finite and increasing")
    heights = np.asarray(heights, dtype=float)
    if heights.shape != (mesh.n_points,) or not np.isfinite(heights).all():
        raise ValueError("Every vertex must have a finite height")
    working = mesh.copy()
    name = "__clip_height"
    while name in working.point_data:
        name += "_"
    working.point_data[name] = heights
    clipped = working.clip_scalar(scalars=name, value=lower, invert=False)
    clipped = clipped.clip_scalar(scalars=name, value=upper, invert=True)
    if not isinstance(clipped, pv.PolyData):
        clipped = clipped.extract_geometry()
    if name in clipped.point_data:
        del clipped.point_data[name]
    return clipped


def limit_line_points(mesh, target):
    """Keep complete line identities and endpoints; sample only interior vertices.

    If even two endpoints per line exceed the budget, retain evenly spaced
    whole lines first. No new connections between distinct lines are created.
    """
    if target < 2:
        raise ValueError("Point target must be at least 2")
    lines = []
    offsets = []
    cursor = 0
    line_index = 0
    while cursor < len(mesh.lines):
        count = int(mesh.lines[cursor])
        ids = mesh.lines[cursor + 1:cursor + count + 1]
        if count >= 2:
            lines.append(ids)
            offsets.append(mesh.n_verts + line_index)
        cursor += count + 1
        line_index += 1
    if not lines:
        raise ValueError("Input contains no polyline segments")
    if len(lines) * 2 > target:
        keep = np.linspace(0, len(lines) - 1, target // 2, dtype=int)
        lines = [lines[i] for i in keep]
        offsets = [offsets[i] for i in keep]
    capacity = np.array([len(ids) - 2 for ids in lines], dtype=int)
    budget = min(target - 2 * len(lines), int(capacity.sum()))
    interior = np.zeros(len(lines), dtype=int)
    if budget and capacity.sum():
        quotas = capacity * (budget / capacity.sum())
        interior = np.floor(quotas).astype(int)
        remainder = budget - int(interior.sum())
        order = np.argsort(-(quotas - interior), kind="stable")
        interior[order[:remainder]] += 1
    chosen = [ids[np.linspace(0, len(ids) - 1, int(n) + 2, dtype=int)]
              for ids, n in zip(lines, interior)]
    point_ids = np.concatenate(chosen)
    connectivity = []
    offset = 0
    for ids in chosen:
        connectivity.extend([len(ids), *range(offset, offset + len(ids))])
        offset += len(ids)
    result = pv.PolyData(mesh.points[point_ids], lines=np.asarray(connectivity))
    for name, values in mesh.point_data.items():
        result.point_data[name] = values[point_ids]
    for name, values in mesh.cell_data.items():
        result.cell_data[name] = values[np.asarray(offsets)]
    for name, values in mesh.field_data.items():
        result.field_data[name] = values
    return result


def plane_cli(compress=False):
    parser = argparse.ArgumentParser(description="Clip existing polylines to an absolute-Z slab; this does not generate flow trajectories.")
    parser.add_argument("-i", "--infile", required=True)
    parser.add_argument("-H", "--height", type=float, required=True, help="Absolute Z height in input model units")
    parser.add_argument("-d", "--delta", type=float, default=0.15, help="Positive slab half-thickness in input model units")
    parser.add_argument("--target", type=int, default=500_000, help="Maximum output vertices; preserves endpoints of retained lines")
    parser.add_argument("-o", "--outfile")
    args = parser.parse_args()
    if not np.isfinite(args.height) or not np.isfinite(args.delta) or args.delta <= 0 or args.target < 2:
        parser.error("height must be finite, delta positive and target at least 2")
    source = Path(args.infile).expanduser().resolve()
    mesh = pv.read(source)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_geometry()
    if not mesh.n_lines:
        raise ValueError("Input has no existing polylines; provide streamline VTP data")
    clipped = clip_height_range(mesh, mesh.points[:, 2], args.height - args.delta, args.height + args.delta)
    result = limit_line_points(clipped, args.target)
    base = Path(args.outfile).expanduser() if args.outfile else source.with_stem(f"{source.stem}_{args.height:g}m")
    if base.suffix == ".gz":
        base = base.with_suffix("")
    destination = base.with_suffix(".vtp")
    if compress:
        destination = Path(str(destination) + ".gz")
    if destination.resolve() == source:
        raise ValueError("Output must differ from the input file")
    # Staging avoids leaving half-written output or deleting an existing VTP.
    with tempfile.TemporaryDirectory(prefix="streamline-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "result.vtp"
        result.save(staged)
        if compress:
            packed = Path(tmp) / "result.vtp.gz"
            with staged.open("rb") as src, gzip.open(packed, "wb") as dst:
                shutil.copyfileobj(src, dst)
            packed.replace(destination)
        else:
            staged.replace(destination)
    print(f"Wrote {destination}: {result.n_lines} lines, {result.n_points} vertices (from {mesh.n_lines}/{mesh.n_points})")
