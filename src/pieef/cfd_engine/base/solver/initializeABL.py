"""Initialize the actual v2212 mesh with a scaled neutral log profile.

Run AFTER mesh/boundary generation and BEFORE decomposePar/roughFoam. Only
initial U/k/epsilon/nut internal fields are changed; p is never changed. Ground
is interpolated on the actual triangulated bot patch, not on the source DSM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from writeBoundaryFields import CMU, KAPPA, resolve_physical_scaling


def run_native(command, case, log_name):
    with (case / log_name).open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=case, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=300)


def read_vector_internal(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise ValueError(f"Expected native ASCII nonuniform vector field: {path}")
    count = int(match.group(1))
    values = np.fromstring(match.group(2).replace("(", " ").replace(")", " "), sep=" ")
    if values.size != count * 3:
        raise ValueError(f"Native vector count mismatch: {path}")
    return values.reshape(count, 3)


def ground_on_triangles(points, triangles, queries):
    """Vertical height on native bot triangles, without nearest extrapolation."""
    from matplotlib.tri import Triangulation
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64).copy()
    a, b, c = (points[triangles[:, i], :2] for i in range(3))
    cross = (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])-(b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
    if np.any(np.abs(cross) < 1e-20):
        raise ValueError("bot has a vertical/degenerate triangle; vertical AGL initialization is unsupported")
    clockwise = cross < 0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    tri = Triangulation(points[:, 0], points[:, 1], triangles)
    found = tri.get_trifinder()(queries[:, 0], queries[:, 1])
    if np.any(found < 0):
        raise ValueError(f"{int((found < 0).sum())} cell centres lie outside actual bot footprint; no extrapolation used")
    selected = points[triangles[found]]
    av, bv, cv = selected[:, 0], selected[:, 1], selected[:, 2]
    den = (bv[:, 1]-cv[:, 1])*(av[:, 0]-cv[:, 0])+(cv[:, 0]-bv[:, 0])*(av[:, 1]-cv[:, 1])
    wa = ((bv[:, 1]-cv[:, 1])*(queries[:, 0]-cv[:, 0])+(cv[:, 0]-bv[:, 0])*(queries[:, 1]-cv[:, 1]))/den
    wb = ((cv[:, 1]-av[:, 1])*(queries[:, 0]-cv[:, 0])+(av[:, 0]-cv[:, 0])*(queries[:, 1]-cv[:, 1]))/den
    return wa*av[:, 2]+wb*bv[:, 2]+(1-wa-wb)*cv[:, 2]


def replace_internal(path, values):
    text = path.read_text(encoding="utf-8")
    pattern = r"internalField\s+(?:uniform\s+[^;]+|nonuniform\s+List<\w+>\s+\d+\s*\(.*?\))\s*;"
    if values.ndim == 2:
        content = "\n".join("("+" ".join(f"{v:.12g}" for v in row)+")" for row in values)
        kind = "vector"
    else:
        content = "\n".join(f"{v:.12g}" for v in values)
        kind = "scalar"
    internal = f"internalField nonuniform List<{kind}>\n{len(values)}\n(\n{content}\n);"
    new, count = re.subn(pattern, lambda _: internal, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f"Cannot replace initial internalField: {path}")
    temporary = path.with_name(path.name + ".abl.tmp")
    temporary.write_text(new, encoding="utf-8")
    temporary.replace(path)


def initialize(case, info, flat_ground_m=None):
    scaling = resolve_physical_scaling(info)
    if scaling["profile"] != "abl_log":
        return {"profile": scaling["profile"], "required": False}
    case = Path(case).resolve()
    zero = case / "0"
    if not (case / "constant/polyMesh/boundary").is_file():
        raise ValueError("Initialize ABL only after native mesh/boundary generation")
    if not all((zero / field).is_file() for field in ("U", "k", "epsilon", "nut")):
        raise ValueError("Generate ABL boundary fields before spatial initialization")
    run_native(["postProcess", "-case", str(case), "-func", "writeCellCentres", "-time", "0"],
               case, "log.ablCellCentres")
    centres = read_vector_internal(zero / "C")
    scale = scaling["length_scale"]
    if flat_ground_m is None:
        import pyvista as pv
        run_native(["foamToVTK", "-case", str(case), "-time", "0", "-no-internal", "-no-point-data",
                    "-fields", "(U)", "-patches", "(bot)"], case, "log.ablGroundVTK")
        candidates = list((case / "VTK").glob("*/boundary/bot.vtp"))
        if not candidates:
            raise ValueError("Native bot patch VTK was not generated")
        # Only the just-written time-0 export is used; no stale solved field.
        bot_file = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        bot = pv.read(bot_file).clean().triangulate()
        faces = np.asarray(bot.faces).reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise ValueError("Triangulated bot must contain triangles")
        ground = ground_on_triangles(bot.points, faces[:, 1:], centres[:, :2])
        method = "vertical_interpolation_on_native_bot_triangles"
    else:
        ground = np.full(len(centres), float(flat_ground_m)*scale)
        method = "explicit_flat_ground_for_native_validation_only"
    agl = centres[:, 2]-ground
    if not np.isfinite(agl).all() or np.any(agl <= 0):
        raise ValueError("Non-positive/non-finite actual cell-centre AGL; refusing log-profile clipping")
    abl = scaling["abl"]
    z0 = abl["roughness_length_model_m"]
    friction = abl["friction_velocity_mps"]
    k = scaling["k_model_m2ps2"]
    speed = friction/KAPPA * np.log((agl+z0)/z0)
    epsilon = friction**3/(KAPPA*(agl+z0))
    nut = CMU*k*k/epsilon
    vectors = np.column_stack((speed, np.zeros(len(speed)), np.zeros(len(speed))))
    replace_internal(zero / "U", vectors)
    replace_internal(zero / "epsilon", epsilon)
    replace_internal(zero / "nut", nut)
    # k is analytically height-independent and already emitted uniformly.
    receipt = {"schema_version": 1, "profile": "abl_log", "cell_count": len(centres),
               "ground_method": method, "length_scale": scale,
               "agl_min_physical_m": float(agl.min()/scale),
               "agl_max_physical_m": float(agl.max()/scale),
               "velocity_min_mps": float(speed.min()), "velocity_max_mps": float(speed.max()),
               "k_m2ps2": k, "epsilon_min_model": float(epsilon.min()),
               "epsilon_max_model": float(epsilon.max()),
               "pressure_modified": False, "no_nearest_extrapolation": True,
               "centres_sha256": hashlib.sha256((zero / "C").read_bytes()).hexdigest()}
    (case / "constant/ablInitialization.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path.cwd())
    parser.add_argument("--info", type=Path)
    parser.add_argument("--flat-ground-m", type=float)
    args = parser.parse_args()
    case = args.case.resolve()
    info = json.loads((args.info or case.parent / "info.json").read_text(encoding="utf-8"))
    print(json.dumps(initialize(case, info, args.flat_ground_m), allow_nan=False), flush=True)
