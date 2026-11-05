# @Author: joe contact withheld
# @Date: 2024-09-28 21:36:52
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-02-16 20:39:58
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\base\solver\buildTerrain.py
# @Description:
#
# Copyright (c) 2025 by joe, All Rights Reserved.

import math
import json
import numpy
from pathlib import Path
import rasterio


def require_section(handle, section):
    for line in handle:
        if line.strip() == section:
            return
    raise ValueError(f"flat.msh is missing {section} (incomplete or unsupported mesh)")


def interpolate_dem(geodata, px, py, bounds, mean_elevation):
    """Bilinear interpolation inside the DEM; mean-height padding outside it."""
    xmin, xmax, ymin, ymax = bounds
    if not (xmin <= px <= xmax and ymin <= py <= ymax):
        return 0.0
    height, width = geodata.shape
    # GeoTIFF bounds enclose pixel areas, whereas samples sit at pixel centres.
    # Clamp only the outer half-pixel band, which is still inside the raster.
    col = min(max((px - xmin) / (xmax - xmin) * width - 0.5, 0.0), width - 1)
    row = min(max((ymax - py) / (ymax - ymin) * height - 0.5, 0.0), height - 1)
    ix, iy = math.floor(col), math.floor(row)
    fx, fy = col - ix, row - iy
    ix1, iy1 = min(ix + 1, width - 1), min(iy + 1, height - 1)
    return float(
        geodata[iy, ix] * (1 - fx) * (1 - fy)
        + geodata[iy, ix1] * fx * (1 - fy)
        + geodata[iy1, ix] * (1 - fx) * fy
        + geodata[iy1, ix1] * fx * fy
        - mean_elevation
    )


def validate_terrain_origin(inp, longitude, latitude, tolerance_m=1.0):
    """Turbines, DEM, and roughness must use the same local coordinate origin."""
    metre_per_degree = 6371000 * math.pi / 180
    dx = (float(inp["domain"]["centerLon"]) - longitude) * metre_per_degree * math.cos(math.radians(latitude))
    dy = (float(inp["domain"]["centerLat"]) - latitude) * metre_per_degree
    offset = math.hypot(dx, dy)
    if not math.isfinite(offset) or offset > tolerance_m:
        raise ValueError("domain.centerLon/centerLat must match the terrain.tif bounds midpoint within 1 m; use one origin for turbine and roughness coordinates")
    return offset


def vertical_mapping_configuration(inp):
    """Keep attachment meshes reproducible; opt in to fixed near-ground layers.

    The legacy map compresses every layer by (roof - terrain)/roof.  Comparing
    roof heights then also changes the mesh at the rotor.  The new map follows
    terrain rigidly below a physical AGL height and smoothly removes its
    displacement above it, leaving the same flat roof and near-ground layers.
    """
    mesh, roof = inp["mesh"], float(inp["domain"]["h"])
    method = mesh.get("verticalTerrainMapping", "linearCompressed")
    if method not in ("linearCompressed", "preserveNearGround"):
        raise ValueError("mesh.verticalTerrainMapping must be linearCompressed or preserveNearGround")
    follow = float(mesh.get("terrainFollowingHeight", mesh.get("h1", 0)))
    if method == "preserveNearGround" and not (math.isfinite(follow) and 0 < follow < roof):
        raise ValueError("mesh.terrainFollowingHeight must be finite and between 0 and domain.h")
    return method, follow


def map_vertical_coordinate(original_z, terrain_elevation, roof, method, follow_height):
    """Map one mesh column in physical metres, rejecting inverted mappings."""
    if terrain_elevation >= roof:
        raise ValueError("terrain elevation reaches/exceeds domain.h; vertical cells would be inverted")
    if method == "linearCompressed":
        return original_z * (roof - terrain_elevation) / roof + terrain_elevation
    transition = roof - follow_height
    # For positive terrain, dz/doz >= 1 - 1.5*terrain/transition.  Check the
    # continuous map, not just nodes that could miss its steepest derivative.
    if terrain_elevation > 0 and 1 - 1.5 * terrain_elevation / transition <= 0:
        raise ValueError("preserveNearGround terrain mapping would invert cells; enlarge domain.h or lower terrainFollowingHeight")
    if original_z <= follow_height:
        return original_z + terrain_elevation
    t = min(1.0, max(0.0, (original_z - follow_height) / transition))
    weight = 1 - 3 * t * t + 2 * t * t * t
    return original_z + terrain_elevation * weight


def validate_turbine_clearance(inp, bounds, elevation_at):
    """Reject unsupported turbine geometry before writing a deformed mesh.

    Check projected cylinder support and the fixed -1D reference window in
    solver coordinates, then rotate their corners back into DEM coordinates.
    The domain roof is relative to the DEM mean, while hub is above terrain.
    """
    xmin, xmax, ymin, ymax = bounds
    roof = float(inp["domain"]["h"])
    half_domain = float(inp["domain"]["lt"]) / 2
    terrain_inner_radius = float(inp["mesh"]["tr1"])
    theta = math.radians(float(inp["wind"]["angle"]) + 90)
    cos_angle, sin_angle = math.cos(theta), math.sin(theta)
    checked = []
    for index, turbine in enumerate(inp["turbines"]):
        label = str(turbine.get("name") or turbine.get("id") or index + 1)
        x, y = float(turbine["x"]), float(turbine["y"])
        hub, radius = float(turbine["hub"]), float(turbine["d"]) / 2
        if hub <= radius:
            raise ValueError(f"Turbine {label}: blade tip reaches terrain (hub <= rotor radius)")
        solver_x = x * cos_angle - y * sin_angle
        solver_y = x * sin_angle + y * cos_angle
        diameter = 2 * radius
        # Keep these physical support definitions in agreement with
        # roughFoam/actuatorMath.H. A projected axial cylinder is a rectangle.
        support_rectangles = (
            ("rotor support", solver_x - 0.25 * diameter, solver_x + 0.25 * diameter,
             solver_y - radius, solver_y + radius),
            ("upstream reference window", solver_x - 1.25 * diameter, solver_x - 0.75 * diameter,
             solver_y - 0.25 * diameter, solver_y + 0.25 * diameter),
        )
        for support_name, low_x, high_x, low_y, high_y in support_rectangles:
            if low_x <= -half_domain or high_x >= half_domain or low_y <= -half_domain or high_y >= half_domain:
                raise ValueError(f"Turbine {label}: {support_name} reaches/exceeds the rotated solver-domain boundary; enlarge domain.lt")
            for sx, sy in ((low_x, low_y), (low_x, high_y), (high_x, low_y), (high_x, high_y)):
                px = sx * cos_angle + sy * sin_angle
                py = sy * cos_angle - sx * sin_angle
                if not (xmin <= px <= xmax and ymin <= py <= ymax):
                    raise ValueError(f"Turbine {label}: {support_name} extends outside terrain.tif coverage")
                if math.hypot(px, py) > terrain_inner_radius:
                    raise ValueError(f"Turbine {label}: {support_name} reaches the artificial terrain flattening region (mesh.tr1)")
        terrain_elevation = float(elevation_at(x, y))
        clearance = roof - terrain_elevation - hub - radius
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError(f"Turbine {label}: blade tip reaches/exceeds domain roof; enlarge domain.h")
        # A hub above its own ground does not rule out a nearby ridge touching
        # the rotor. Screen 17 points on the lower semicircle, including its
        # centre and lateral tips; this is a discrete check, not a full geometry
        # intersection proof for unresolved terrain between DEM samples.
        sampled_ground_clearances = []
        for sample_index in range(-8, 9):
            offset = radius * sample_index / 8
            sx, sy = solver_x, solver_y + offset
            px = sx * cos_angle + sy * sin_angle
            py = sy * cos_angle - sx * sin_angle
            ground = float(elevation_at(px, py))
            lower_rotor_z = terrain_elevation + hub - math.sqrt(max(0.0, radius * radius - offset * offset))
            ground_clearance = lower_rotor_z - ground
            if not math.isfinite(ground_clearance) or ground_clearance <= 0:
                raise ValueError(f"Turbine {label}: terrain intersects the discretely sampled lower rotor envelope")
            sampled_ground_clearances.append(ground_clearance)
        checked.append({
            "id": label,
            "terrain_relative_elevation_m": terrain_elevation,
            "upper_blade_tip_roof_clearance_m": clearance,
            "rotor_and_reference_support_checked": True,
            "lower_rotor_terrain_check": "discrete_lower_semicircle_not_full_geometry",
            "lower_rotor_terrain_sample_count": len(sampled_ground_clearances),
            "minimum_sampled_lower_rotor_terrain_clearance_m": min(sampled_ground_clearances),
        })
    return checked

# @jit
def buildTerrain(inp):
    EARTH_RADIUS = 6371
    RAD_PER_DEG = math.pi / 180

    # 输入文件参数
    # lat = inp['domain']['lat']
    # lon = inp['domain']['long']
    lt = inp['domain']['lt']
    h = inp['domain']['h']
    r1 = inp['mesh']['tr1']
    r2 = inp['mesh']['tr2']
    windAngle = (inp['wind']['angle'] + 90) * RAD_PER_DEG
    cos_angle, sin_angle = math.cos(windAngle), math.sin(windAngle)
    scale = inp['mesh']['scale']
    num_center = len(inp['turbines'])
    vertical_mapping, following_height = vertical_mapping_configuration(inp)

    with rasterio.open("../terrain.tif") as georaster:
        # The solver's local metre conversion below is explicitly WGS84/north-up.
        # Reject unsupported inputs instead of interpreting projected metres as degrees.
        if georaster.crs is None or georaster.crs.to_epsg() != 4326:
            raise ValueError("terrain.tif must use EPSG:4326; reproject the terrain before solving")
        transform = georaster.transform
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError("terrain.tif must be a north-up, unrotated raster")
        masked = georaster.read(1, masked=True)
        if numpy.ma.getmaskarray(masked).any() or not numpy.isfinite(masked.data).all():
            raise ValueError("terrain.tif contains NoData/non-finite elevations; crop or repair the DEM first")
        geodata = numpy.asarray(masked.data, dtype=float)
        bounds = georaster.bounds
        xnum, ynum = georaster.width, georaster.height
    if xnum < 2 or ynum < 2:
        raise ValueError("terrain.tif needs at least 2 x 2 samples")
    lon = (bounds.left + bounds.right) / 2.0
    lat = (bounds.bottom + bounds.top) / 2.0
    if not (-180 <= bounds.left < bounds.right <= 180 and -90 < bounds.bottom < bounds.top < 90):
        raise ValueError("terrain.tif has invalid WGS84 bounds")
    origin_offset = validate_terrain_origin(inp, lon, lat)

    # 地形
    METER_PER_DEG_LAT = 1000 * EARTH_RADIUS * RAD_PER_DEG
    LONG_LAT_RATIO = math.cos(lat * RAD_PER_DEG)

    zeroElevation = numpy.average(geodata)
    xmin, xmax = [(v - lon) * METER_PER_DEG_LAT * LONG_LAT_RATIO
                  for v in [bounds.left, bounds.right]]
    ymin, ymax = [(v - lat) * METER_PER_DEG_LAT
                  for v in [bounds.bottom, bounds.top]]
    def elevation_at(px, py):
        # Outside the DEM the historical attachment pads the peripheral domain
        # to its mean elevation. This is permitted only away from turbines and
        # is counted explicitly below; it is not measured terrain coverage.
        if not (xmin <= px <= xmax and ymin <= py <= ymax):
            return 0.0
        return interpolate_dem(geodata, px, py, (xmin, xmax, ymin, ymax), zeroElevation)

    turbine_clearance = validate_turbine_clearance(
        inp, (xmin, xmax, ymin, ymax), elevation_at
    )
    outside_dem_bottom_nodes = 0
    # print('loaded terrian:', [xnum, ynum], [xmin, xmax], [ymin, ymax], zeroElevation)

    # 读写文件
    with open('output.neu.tmp', 'w+', newline='') as out, open('flat.msh') as msh:
        head = ''.join(['        CONTROL INFO 2.4.6\n',
                        '** GAMBIT NEUTRAL FILE\n',
                        'example\n',
                        'PROGRAM :                Gambit     VERSION :  2.4.6\n\n',
                        '     NUMNP     NELEM     NGRPS    NBSETS     NDFCD     NDFVL\n'])
        out.write(head)
        out.writelines(['%30d%10d%10d%10d\n' % (1, 6, 3, 3),
                        'ENDOFSECTION\n'])
        require_section(msh, '$MeshFormat')
        mesh_format = msh.readline().split()
        if len(mesh_format) != 3 or not mesh_format[0].startswith('2.') or mesh_format[1] != '0':
            raise ValueError("flat.msh must use ASCII MSH 2.x (gmsh -format msh2)")
        require_section(msh, '$Nodes')
        out.write('   NODAL COORDINATES 2.4.6\n')
        np = int(msh.readline()) - num_center
        if np <= 0:
            raise ValueError('flat.msh has no volume mesh nodes after refinement points')
        node_ids = {}
        for i in range(num_center):
            msh.readline()  # 跳过加密区中心
        p_bot = {}   # 底层节点 { (x, y) ： (el, flr) }
        aa = lt / 2  # 半边长，最大（边界）xy坐标值
        # 边界编号（原msh文件）
        p_inlet, p_outlet, p_front, p_back = (set() for _ in range(4))
        # 建筑编号（原msh文件）
        p_building = []
        for ip in range(0, np):
            if ip % 10000 == 0: print('writting points %dw/%dw' % (ip / 10000, np / 10000), end='\r')
            row = msh.readline().split()
            if len(row) != 4:
                raise ValueError("flat.msh contains a missing/malformed node row")
            i = int(row[0])
            x, y, oz = map(float, row[1:])
            if i <= 0 or i in node_ids or not all(math.isfinite(v) for v in (x, y, oz)):
                raise ValueError("flat.msh contains invalid/duplicate node IDs or coordinates")
            node_ids[i] = ip + 1
            if oz == 0:
                distance = math.hypot(x, y)
                px = x * cos_angle + y * sin_angle
                py = y * cos_angle - x * sin_angle
                if not (xmin <= px <= xmax and ymin <= py <= ymax):
                    outside_dem_bottom_nodes += 1
                # 计算高程
                if distance < r2:
                    el = elevation_at(px, py)
                else:
                    el = 0
                if distance > r1 and distance < r2:
                    el = el * (r2 - distance) / (r2 - r1)
                # 建筑层数
                flr = 0
                p_bot.update({(float(x), float(y)):  (el, flr)})   # 统计底层节点
            else:
                if (x, y) not in p_bot:
                    raise ValueError("flat.msh must list a bottom node before each vertical column")
                el, flr = p_bot[(x, y)]
            # 保存边界点编号
            if x == -aa:
                p_inlet.add(i)
            if x == aa:
                p_outlet.add(i)
            if y == aa:
                p_back.add(i)
            if y == -aa:
                p_front.add(i)
            z = map_vertical_coordinate(oz, el, h, vertical_mapping, following_height)
            # 保存建筑编号
            if flr > 0 and z - el < flr * FLOOR_HEIGHT:
                p_building.append(i)
            # 写入节点
            out.write('%10d%20.11e%20.11e%20.11e\n' %
                      (node_ids[i], *[v * scale for v in (x, y, z)]))
        out.write('ENDOFSECTION\n')
        print('written points:', np, '. ')

        # 写单元
        require_section(msh, '$Elements')
        out.write('      ELEMENTS/CELLS 2.4.6\n')
        nele = int(msh.readline())
        numtri = 0  # 统计单元个数
        # 边界单元（原msh文件）
        e_inlet, e_outlet, e_front, e_back = [], [], [], []
        for iele in range(0, nele):
            if iele % 10000 == 0: print('writting elemnets %dw/%dw' % (iele / 10000, nele / 10000), end='\r')
            line = msh.readline()
            if not line:
                raise RuntimeError("flat.msh 在读取 Elements 时意外结束（文件可能不完整：gmsh 被系统终止或写入失败）")
            data = [*map(int, line.split())]
            if len(data) < 3:
                raise RuntimeError(f"flat.msh 元素行格式错误（列数不足）：{line.strip()[:200]}")

            if data[1] != 6:
                continue

            num_tags = data[2]
            node_start = 3 + num_tags
            if len(data) < node_start + 6:
                raise RuntimeError(
                    "flat.msh 元素行不完整（可能 gmsh 被系统终止或网格文件损坏）。"
                    f" 期望至少 {node_start + 6} 列，实际 {len(data)} 列。行内容：{line.strip()[:200]}"
                )

            nodes = data[node_start:node_start + 6]
            if any(v not in node_ids for v in nodes):
                raise ValueError('flat.msh prism references an unknown or discarded refinement node')

            numtri = numtri + 1
            out.write('%8d%3d%3d' % (numtri, 5, 6) +
                      ''.join(['%8d' % node_ids[v] for v in (nodes[3], nodes[5], nodes[4], nodes[0], nodes[2], nodes[1])]) + '\n')
            # 判断单元是否在边界上
            for p_bound, e_bound in [(p_inlet, e_inlet), (p_outlet, e_outlet),
                                     (p_front, e_front), (p_back, e_back)]:
                a1, a2, a3 = [v in p_bound for v in (nodes[3], nodes[5], nodes[4])]
                numface = 1 if a1 and a2 else \
                    3 if a1 and a3 else \
                    2 if a2 and a3 else 0
                if numface > 0:
                    e_bound.append((numtri, numface))

        out.write('ENDOFSECTION\n')
        print('written elements:', numtri, '. ')

        # 回到文件开头，更新节点数、单元数
        out.seek(len(head), 0)
        out.write(' %9d %9d' % (np, numtri))
        out.seek(0, 2)

        # 输出domain1
        out.writelines([
            '       ELEMENT GROUP 2.4.6\n',
            'GROUP:          1 ELEMENTS: %10d MATERIAL:          2 NFLAGS:          1\n' % numtri,
            '%32s\n' % 'all',
            '%8d' % 0
        ])
        for i in range(0, numtri):
            if i % 10 == 0:
                out.write('\n')
            out.write('%8d' % (i+1))
        out.write('\nENDOFSECTION\n')

        # 输出侧边
        for e_tag, e_bound in [('inlet', e_inlet), ('outlet', e_outlet),
                               ('front', e_front), ('back', e_back)]:
            out.writelines([' BOUNDARY CONDITIONS 2.4.6\n',
                            '%32s%8d%8d%8d%8d\n' % (e_tag, 1, len(e_bound), 0, 6)])
            for e_index, e_face in e_bound:
                out.write('%10d %4d %4d\n' % (e_index, 5, e_face))
            out.write('ENDOFSECTION\n')

        # 输出bottom
        # 单元层数 = 节点数 / 底层节点数 - 1
        num_bot_p = len(p_bot)
        if not num_bot_p or np % num_bot_p or np <= num_bot_p or not numtri:
            raise ValueError("flat.msh must contain complete, uniformly extruded prism layers")
        num_ceng = np // num_bot_p - 1
        if numtri % num_ceng:
            raise ValueError("flat.msh prism count is inconsistent with its vertical layers")
        num_bot = numtri // num_ceng
        # print(numtri, num_bot)
        out.writelines([' BOUNDARY CONDITIONS 2.4.6\n',
                        '%32s%8d%8d%8d%8d\n' % ('bot', 1, num_bot, 0, 6)])
        for e_index in range(1, num_bot + 1):
            out.write('%10d %4d %4d\n' % (e_index, 5, 5))
        out.write('ENDOFSECTION\n')

        # 输出top
        out.writelines([' BOUNDARY CONDITIONS 2.4.6\n',
                        '%32s%8d%8d%8d%8d\n' % ('top', 1, num_bot, 0, 6)])
        for e_index in range(numtri - num_bot + 1, numtri + 1):
            out.write('%10d %4d %4d\n' % (e_index, 5, 4))
        out.write('ENDOFSECTION\n')
    Path('output.neu.tmp').replace('output.neu')
    quality = {
        "schema_version": 1,
        "origin_longitude": lon,
        "origin_latitude": lat,
        "configured_origin_offset_m": origin_offset,
        "origin_tolerance_m": 1.0,
        "dem_mean_elevation_m": float(zeroElevation),
        "dem_local_bounds_m": [xmin, xmax, ymin, ymax],
        "interpolation": "bilinear_pixel_centres_with_outer_half_pixel_clamp",
        "outside_dem_policy": "flat_at_dem_mean_away_from_turbines",
        "outside_dem_bottom_node_count": outside_dem_bottom_nodes,
        "full_terrain_coverage": outside_dem_bottom_nodes == 0,
        "turbine_clearance": turbine_clearance,
        "vertical_terrain_mapping": vertical_mapping,
        "terrain_following_height_m": following_height if vertical_mapping == "preserveNearGround" else None,
        "near_ground_layer_heights_preserved": vertical_mapping == "preserveNearGround",
        "nominal_first_uniform_layer_height_m": float(inp["mesh"]["h1"]) / int(inp["mesh"]["ceng"])
        if "h1" in inp["mesh"] and "ceng" in inp["mesh"] else None,
        "nominal_upper_layer_height_cap_m": inp["mesh"].get("maxLayerHeight"),
        "upper_layer_cap_definition": "before_terrain_deformation_not_a_bound_on_actual_vertical_edges"
        if inp["mesh"].get("maxLayerHeight") is not None else None,
    }
    Path("terrain-quality.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if outside_dem_bottom_nodes:
        print(f"[WARNING] {outside_dem_bottom_nodes} bottom mesh nodes lack DEM coverage; peripheral terrain is padded flat at the DEM mean. This is a modeling assumption, not surveyed terrain.")
    print("底层结点数:",num_bot_p,"层数:", num_ceng, "底层网格数:",numtri / num_ceng)

if __name__ == '__main__':
    import json
    with open("../info.json", 'r') as f:
        buildTerrain(json.load(f))

