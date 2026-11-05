"""Validate the existing solver file contract before meshing or removing old output."""

import json
import math
import sys


def number(value, name, *, positive=False, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and > 0' if positive else ''}")
    if integer and not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result) if integer else result


def turbine_type(turbine):
    """A missing legacy type means 1; explicitly invalid types must not be truncated."""
    ids = [number(turbine[key], f"turbine.{key}", integer=True)
           for key in ("type", "model") if turbine.get(key) is not None]
    if any(not 1 <= value <= 10 for value in ids):
        raise ValueError("turbine type/model must be an integer in 1..10")
    if len(set(ids)) > 1:
        raise ValueError("turbine type and model must identify the same curve")
    return ids[0] if ids else 1


def resolve_post_heights(post):
    """Resolve optional explicit AGL layers without changing the legacy default.

    The native postFoam already accepts a strictly increasing array. This
    bridge permits real hub heights such as 68.5 and 78.5 m in isolated cases.
    """
    if "heights" not in post:
        count = number(post["numh"], "post.numh", positive=True, integer=True)
        spacing = number(post["dh"], "post.dh", positive=True, integer=True)
        # Preserve int/float representation in the old Input/input.json too.
        spacing = post["dh"] if isinstance(post["dh"], (int, float)) else spacing
        return [spacing * (index + 1) for index in range(count)]
    raw = post["heights"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 1000:
        raise ValueError("post.heights must contain 1..1000 explicit AGL heights")
    heights = [number(value, "post.heights[]", positive=True) for value in raw]
    if any(current <= previous for previous, current in zip(heights, heights[1:])):
        raise ValueError("post.heights must be strictly increasing")
    if "numh" in post and number(post["numh"], "post.numh", positive=True, integer=True) != len(heights):
        raise ValueError("post.numh must equal the explicit post.heights length")
    if post.get("streamlines", False):
        raise ValueError("explicit post.heights requires streamlines=false: the web streamline shell path uses legacy spacing")
    return heights


def validate_input(info):
    domain, mesh, wind, post = (info[key] for key in ("domain", "mesh", "wind", "post"))
    for key in ("lt", "h"):
        number(domain[key], f"domain.{key}", positive=True)
    for key in ("centerLon", "centerLat"):
        number(domain[key], f"domain.{key}")
    for key in ("h1", "lc1", "lc2", "lc3", "wakeL", "wakeB"):
        number(mesh[key], f"mesh.{key}", positive=True)
    number(mesh["ceng"], "mesh.ceng", positive=True, integer=True)
    if not float(mesh["h1"]) < float(domain["h"]):
        raise ValueError("mesh.h1 must be less than domain.h")
    mapping = mesh.get("verticalTerrainMapping", "linearCompressed")
    if mapping not in ("linearCompressed", "preserveNearGround"):
        raise ValueError("mesh.verticalTerrainMapping must be linearCompressed or preserveNearGround")
    if "terrainFollowingHeight" in mesh:
        follow = number(mesh["terrainFollowingHeight"], "mesh.terrainFollowingHeight", positive=True)
        if follow >= float(domain["h"]):
            raise ValueError("mesh.terrainFollowingHeight must be below domain.h")
    if "maxLayerHeight" in mesh:
        number(mesh["maxLayerHeight"], "mesh.maxLayerHeight", positive=True)
    if number(mesh["q1"], "mesh.q1") <= 1:
        raise ValueError("mesh.q1 must be > 1 (graded layers divide by log(q1))")
    if number(mesh["scale"], "mesh.scale") != 0.001:
        raise ValueError("mesh.scale must be 0.001 for the existing sampling pipeline")
    for inner, outer in (("r1", "r2"), ("tr1", "tr2")):
        lo = number(mesh[inner], f"mesh.{inner}")
        hi = number(mesh[outer], f"mesh.{outer}", positive=True)
        if not 0 <= lo < hi:
            raise ValueError(f"mesh.{inner}/{outer} must satisfy 0 <= inner < outer")
    number(wind["angle"], "wind.angle")
    number(wind["speed"], "wind.speed", positive=True)
    profile = str(wind.get("profile") or "legacy_fixed").strip().lower()
    if profile not in ("legacy_fixed", "uniform", "abl_log"):
        raise ValueError(f"Unsupported wind.profile '{profile}'")
    if profile == "abl_log":
        number(wind.get("referenceHeight"), "wind.referenceHeight", positive=True)
        number(wind.get("roughnessLength"), "wind.roughnessLength", positive=True)
        if number(wind.get("displacementHeight", 0), "wind.displacementHeight") != 0:
            raise ValueError("abl_log currently requires displacementHeight=0; canopy displacement is not validated")
        if float(wind["referenceHeight"]) >= float(domain["h"]):
            raise ValueError("wind.referenceHeight must be below domain.h")
        if number(info.get("roughness", {}).get("vege_times", 0), "roughness.vege_times") != 0:
            raise ValueError("abl_log requires roughness.vege_times=0; additional canopy sources are not verified")
    if profile == "uniform":
        for keys in (("turbulenceIntensity", "TI"), ("turbulenceLengthScale", "lengthScale", "L")):
            present = next((wind[key] for key in keys if wind.get(key) is not None), None)
            if present is not None:
                number(present, f"wind.{keys[0]}", positive=True)
    heights = resolve_post_heights(post)
    number(post["width"], "post.width", positive=True, integer=True)
    if "heights" in post and heights[-1] >= float(domain["h"]):
        raise ValueError("explicit post.heights must remain below domain.h")
    if "streamlines" in post and not isinstance(post["streamlines"], bool):
        raise ValueError("post.streamlines must be a boolean")
    if "height" in post and number(post["height"], "post.height", positive=True, integer=True) != float(post["width"]):
        raise ValueError("post.height must equal post.width: the solver exports a square sampling grid")
    simulation = info["simulation"]
    if simulation.get("advectionScheme", "upwind") not in ("upwind", "linearUpwind"):
        raise ValueError("simulation.advectionScheme must be upwind or linearUpwind")
    for key in ("core", "step_count"):
        number(simulation[key], f"simulation.{key}", positive=True, integer=True)
    if "writeInterval" in simulation:
        number(simulation["writeInterval"], "simulation.writeInterval", positive=True, integer=True)
    if "purgeWrite" in simulation:
        purge = number(simulation["purgeWrite"], "simulation.purgeWrite", integer=True)
        if purge < 0:
            raise ValueError("simulation.purgeWrite must be >= 0")
    timestep = simulation.get("pseudoTimeStep")
    number(simulation.get("deltaT") if timestep is None else timestep, "simulation.pseudoTimeStep", positive=True)
    turbines = info["turbines"]
    if not isinstance(turbines, list) or len(turbines) > 200:
        raise ValueError("turbines must be a list of at most 200 entries (roughFoam capacity)")
    types = []
    for turbine in turbines:
        for key in ("x", "y", "hub", "d"):
            number(turbine[key], f"turbine.{key}", positive=key in ("hub", "d"))
        if float(turbine["hub"]) <= float(turbine["d"]) / 2:
            raise ValueError("turbine.hub must exceed the rotor radius (blade tip reaches terrain)")
        types.append(turbine_type(turbine))
    return max(types, default=0)


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            print(validate_input(json.load(handle)))
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"[ERROR] Invalid solver configuration: {error}", file=sys.stderr)
        sys.exit(1)
