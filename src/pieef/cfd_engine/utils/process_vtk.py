# @Author: joe contact withheld
# @Date: 2025-01-19 20:17:24
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-02-16 16:17:18
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\utils\process_vtk.py
# @Description: 
# 
# Copyright (c) 2025 by joe, All Rights Reserved.

#!/usr/bin/env python3
import pyvista as pv
import json
import os
import sys
from pathlib import Path
import traceback
import tempfile
from visualization_contract import write_json_atomic

def convert_multiblock_to_polydata(input_path, output_dir):
    try:
        print("Processing file:", input_path)
        print("Output directory:", output_dir)

        if not os.path.exists(input_path):
            raise FileNotFoundError("Input file not found: " + input_path)

        print("Reading VTK file...")
        data = pv.read(input_path)
        print("Data type:", type(data))

        os.makedirs(output_dir, exist_ok=True)
        
        groups = {"bottom": [], "mesh": []}

        def collect(block, names=()):
            if block is None:
                return
            if isinstance(block, pv.MultiBlock):
                for index in range(len(block)):
                    collect(block[index], names + (str(block.get_block_name(index) or "").lower(),))
                return
            surface = block if isinstance(block, pv.PolyData) else block.extract_geometry()
            if not surface.n_points or not surface.n_cells:
                return
            role = "bottom" if any(name in {"bot", "bottom", "ground", "terrain"} for name in names) else "mesh"
            groups[role].append(surface)

        collect(data)
        if not any(groups.values()):
            raise ValueError("VTK input contains no nonempty surface geometry")
        metadata = {"blocks": []}
        # Stage files first, so malformed input never overwrites valid output.
        with tempfile.TemporaryDirectory(prefix=".vtk-build-", dir=output_dir) as staging:
            for role, surfaces in groups.items():
                if not surfaces:
                    continue
                surface = surfaces[0]
                for additional in surfaces[1:]:
                    surface = surface.merge(additional)
                filename = "bot.vtp" if role == "bottom" else "mesh.vtp"
                surface.save(Path(staging) / filename)
                metadata["blocks"].append({
                    "filename": filename, "type": role,
                    "n_points": surface.n_points, "n_cells": surface.n_cells,
                    "bounds": list(surface.bounds), "scalar_fields": list(surface.array_names),
                })
            write_json_atomic(Path(staging) / "metadata.json", metadata)
            for block in metadata["blocks"]:
                os.replace(Path(staging) / block["filename"], Path(output_dir) / block["filename"])
            os.replace(Path(staging) / "metadata.json", Path(output_dir) / "metadata.json")
        # These two generated filenames are owned by this converter. Other VTK
        # artifacts are untouched; stale named surfaces must not appear current.
        emitted = {block["filename"] for block in metadata["blocks"]}
        for filename in {"bot.vtp", "mesh.vtp"} - emitted:
            (Path(output_dir) / filename).unlink(missing_ok=True)

        print("Processing completed successfully")
        return True, metadata
        
    except Exception as e:
        print("Error:", str(e))
        print("Traceback:")
        traceback.print_exc()
        return False, str(e)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_vtk.py <input_path> <output_dir>")
        sys.exit(1)
    
    print("Starting VTK processing script")
    print("Arguments:", sys.argv)
    
    success, result = convert_multiblock_to_polydata(sys.argv[1], sys.argv[2])
    if success:
        print(json.dumps(result))
        sys.exit(0)
    else:
        print("Error:", result, file=sys.stderr)
        sys.exit(1)