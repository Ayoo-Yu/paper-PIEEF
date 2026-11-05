"""Translate saved case parameters to the C++ solver's Input/ contract."""

import json
from pathlib import Path

from validateInput import resolve_post_heights, turbine_type, validate_input


def make_input(info):
    max_type = validate_input(info)
    # UI provides one spacing; roughFoam indexes this array by model ID - 1.
    mesh = dict(info["mesh"], lc2=[info["mesh"]["lc2"]] * max(1, max_type))
    output = {key: info[key] for key in ("domain", "wind", "terrain", "roughness")}
    output["mesh"] = mesh
    heights = resolve_post_heights(info["post"])
    output["post"] = {
        "num_udh": len(heights) if "heights" in info["post"] else info["post"]["numh"],
        "udh": heights,
        "meshSize": info["post"]["width"],
    }
    rows = ["\t".join(map(str, [t["x"], t["y"], t["hub"], t["d"], turbine_type(t)]))
            for t in info["turbines"]]
    Path("Input").mkdir(exist_ok=True)
    Path("Input/Turbines.txt").write_text("\n".join(rows), encoding="utf-8")
    Path("Input/input.json").write_text(json.dumps(output, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    make_input(json.loads(Path("../info.json").read_text(encoding="utf-8")))
