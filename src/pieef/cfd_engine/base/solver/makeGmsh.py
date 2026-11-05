# @Author: joe contact withheld
# @Date: 2024-09-28 21:36:52
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-05-26 20:21:45
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\base\solver\makeGmsh.py
# @Description: 修复空风机列表的网格生成脚本
#
# Copyright (c) 2025 by joe, All Rights Reserved.

import math
import json
from validateInput import validate_input


def graded_layer_heights(info):
    """Reproduce the attachment's upper layers, optionally subdividing them.

    Subdivision retains every historical layer boundary and the entire first
    uniform part.  Increasing the roof therefore cannot silently coarsen the
    rotor layers, and a small horizontal turbine mesh is not extruded through
    kilometre-thick upper cells.  The cap is in physical metres before terrain
    deformation; terrain mapping may subsequently change upper thicknesses.
    """
    mesh = info["mesh"]
    roof, first = float(info["domain"]["h"]), float(mesh["h1"])
    spacing, growth = first / int(mesh["ceng"]), float(mesh["q1"])
    remaining = roof - first
    count = max(0, math.ceil(math.log(remaining * (growth - 1) / spacing / growth + 1 + 1e-9)
                             / math.log(growth)) - 1)
    layers = [spacing * growth ** i for i in range(1, count + 1)]
    last = remaining - sum(layers)
    if last > 1e-6 * roof:
        layers.append(last)
    cap = mesh.get("maxLayerHeight")
    if cap is None:
        return layers
    cap = float(cap)
    if not math.isfinite(cap) or cap < spacing:
        raise ValueError("mesh.maxLayerHeight must be finite and at least the unchanged initial uniform layer height")
    divided = []
    for thickness in layers:
        subdivisions = max(1, math.ceil(thickness / cap))
        divided.extend([thickness / subdivisions] * subdivisions)
    return divided


def writeGeo(input):
    validate_input(input)
    # EARTH_RADIUS = 6371
    # RAD_PER_DEG = math.pi / 180

    # georaster = rasterio.open("../terrain.tif")

    # lon = (georaster.bounds.left + georaster.bounds.right) / 2.0
    # lat = (georaster.bounds.bottom + georaster.bounds.top) / 2.0

    # METER_PER_DEG_LAT = 1000 * EARTH_RADIUS * RAD_PER_DEG
    # LONG_LAT_RATIO = math.cos(lat * RAD_PER_DEG)

    # xmin, xmax = [(v - lon) * METER_PER_DEG_LAT * LONG_LAT_RATIO
    #               for v in [georaster.bounds.left, georaster.bounds.right]]
    # ymin, ymax = [(v - lat) * METER_PER_DEG_LAT
    #               for v in [georaster.bounds.bottom, georaster.bounds.top]]

    data = {
        'h': input['domain']['h'],
        'lt': input['domain']['lt'],
        'lc1': input['mesh']['lc1'],
        'lc2': input['mesh']['lc2'],
        'lc3': input['mesh']['lc3'],
        'h1': input['mesh']['h1'],
        'ceng': input['mesh']['ceng'],
        'r1': input['mesh']['r1'],
        'r2': input['mesh']['r2'],
        'q1': input['mesh']['q1']
    }

    windAngle = -(input['wind']['angle'] + 90) * math.pi / 180
    wakeL = input['mesh']['wakeL']
    wakeB = input['mesh']['wakeB']

    turbines = [dict(turbine) for turbine in input['turbines']]
    numTurbine = len(turbines)

    with open('modeling.geo', 'w', encoding='utf-8') as geo:
        geo.write('\n'.join(['%s=%s;' % (k, data[k])
                 for k in data]) + "\n")

        # 处理风机点（如果有的话）
        # Gmsh 实体 tag 必须为正整数，避免出现 curve 0 / point 0 相关的 Coherence 错误。
        first_point_id = 1
        turbine_point_ids = []
        for i in range(numTurbine):
            tempX = turbines[i]["x"]
            tempY = turbines[i]["y"]
            turbines[i]["x"] = tempX*math.cos(windAngle)+tempY*math.sin(windAngle)
            turbines[i]["y"] = -tempX*math.sin(windAngle)+tempY*math.cos(windAngle)
            pid = first_point_id + i
            turbine_point_ids.append(pid)
            geo.write('Point(%d) = {%f,%f,0,lc2};\n' %
                      (pid, turbines[i]["x"], turbines[i]["y"]))

        # 创建域的四个角点（紧接在风机点之后）
        corner_point_start = first_point_id + numTurbine  # 第一个角点的索引
        geo.write('Point(%d) = {-lt/2,-lt/2,0,lc1};\n' % corner_point_start)
        geo.write('Point(%d) = {lt/2,-lt/2,0,lc1};\n' % (corner_point_start + 1))
        geo.write('Point(%d) = {lt/2,lt/2,0,lc1};\n' % (corner_point_start + 2))
        geo.write('Point(%d) = {-lt/2,lt/2,0,lc1};\n' % (corner_point_start + 3))

        # 创建域的边界线
        line_start = corner_point_start  # 线的索引与点相同（不同实体类型 namespace 不冲突）
        geo.write('Line(%d) = {%d,%d};\n' % (line_start, corner_point_start, corner_point_start + 1))
        geo.write('Line(%d) = {%d,%d};\n' % (line_start + 1, corner_point_start + 1, corner_point_start + 2))
        geo.write('Line(%d) = {%d,%d};\n' % (line_start + 2, corner_point_start + 2, corner_point_start + 3))
        geo.write('Line(%d) = {%d,%d};\n' % (line_start + 3, corner_point_start + 3, corner_point_start))

        # 创建线环和平面
        geo.write('Line Loop(5) = {%d, %d, %d, %d};\n' % (line_start, line_start + 1, line_start + 2, line_start + 3))
        geo.write('Plane Surface(6) = {5};\n')

        # 只有在有风机时才创建风机相关的场
        if numTurbine > 0:
            # 为每个风机创建Box场
            for turbine_index in range(numTurbine):
                # Gmsh 4 uses positive field tags and Distance replaces Attractor.
                j = turbine_index + 1
                geo.write("Field[%d] = Box;\n" % j)
                geo.write("Field[%d].VIn = lc3;\n" % j)   # 尾流区网格尺寸
                geo.write("Field[%d].VOut = lc1;\n" % j)
                geo.write("Field[%d].XMin = %f;\n" % (j, turbines[turbine_index]["x"]))
                geo.write("Field[%d].XMax = %f;\n" % (j, turbines[turbine_index]["x"]+wakeL))
                geo.write("Field[%d].YMin = %f;\n" % (j, turbines[turbine_index]["y"]-wakeB*0.5))
                geo.write("Field[%d].YMax = %f;\n" % (j, turbines[turbine_index]["y"]+wakeB*0.5))

            # 创建吸引子场
            attractor_field_id = numTurbine + 1
            geo.write('Field[%d] = Distance;\n' % attractor_field_id)
            sequence1 = ",".join([str(pid) for pid in turbine_point_ids])
            geo.write('Field[%d].PointsList = {%s};\n' % (attractor_field_id, sequence1))

            # 创建阈值场
            threshold_field_id = attractor_field_id + 1
            geo.write("Field[%d] = Threshold;\n" % threshold_field_id)
            geo.write("Field[%d].InField = %d;\n" % (threshold_field_id, attractor_field_id))
            geo.write("Field[%d].SizeMin = lc2;\n" % threshold_field_id)
            geo.write("Field[%d].SizeMax = lc1;\n" % threshold_field_id)
            geo.write("Field[%d].DistMin = r1;\n" % threshold_field_id)
            geo.write("Field[%d].DistMax = r2;\n" % threshold_field_id)

            # 创建最小场
            sequence2 = ",".join([str(i) for i in range(1, numTurbine + 1)])
            min_field_id = threshold_field_id + 1
            geo.write("Field[%d] = Min;\n" % min_field_id)
            geo.write("Field[%d].FieldsList = {%s};\n" % (min_field_id, sequence2+","+str(threshold_field_id)))
            geo.write("Background Field = %d;\n" % min_field_id)
        else:
            # 无风机情况：使用均匀网格
            print("无风机配置，使用均匀网格尺寸")
            # 不设置Background Field，使用默认的lc1尺寸

        # --- 开始挤出操作 ---
        geo.write("// --- Start Extrusion Operations ---\n")
        geo.write("m = 6; // Initial surface ID to extrude\n")
        geo.write("h0 = h1 / ceng; // Calculate height of the first uniform layers part\n")

        # --- First extrusion layer ---
        geo.write("// First extrusion layer (layer 1)\n")
        geo.write("V[] = Extrude {0, 0, h0} {\n")
        geo.write("  Surface{m}; Layers{1}; Recombine;\n")
        geo.write("};\n")
        geo.write("m = V[0]; // Update m with the new top surface ID\n")

        # --- Uniform layers loop ---
        geo.write("// Uniform layers loop (layers 2 to ceng)\n")
        geo.write("For i In {2 : ceng}\n")
        geo.write("  V[] = Extrude {0, 0, h0} {\n")
        geo.write("    Surface{m}; Layers{1}; Recombine;\n")
        geo.write("  };\n")
        geo.write("  m = V[0]; // Update m inside the loop\n")
        geo.write("EndFor\n")

        if input["mesh"].get("maxLayerHeight") is not None:
            geo.write("// Upper layers subdivided to an explicit physical-height cap.\n")
            for thickness in graded_layer_heights(input):
                geo.write("G[] = Extrude {0, 0, %.17g} {\n" % thickness)
                geo.write("  Surface{m}; Layers{1}; Recombine;\n")
                geo.write("};\n")
                geo.write("m = G[0];\n")
            geo.write("// --- End of Extrusion ---\n")
            return

        # --- Second part: Graded height extrusion ---
        geo.write("// --- Second part: Graded height extrusion ---\n")
        geo.write("h2 = h - h1; // Calculate remaining height for graded layers\n")
        geo.write("h0_graded = h0; // Start graded height from last uniform height\n")
        geo.write("sum = 0; // Initialize sum of graded heights\n")

        # --- Calculate number of graded layers N ---
        geo.write("// Calculate number of graded layers N, with safety checks for Log\n")
        geo.write("N = Ceil(Log(h2 * (q1 - 1) / h0_graded / q1 + 1 + 1e-9) / Log(q1)) - 1;\n")
        geo.write("If (N < 0) // Ensure N is not negative\n")
        geo.write("  N = 0;\n")
        geo.write("EndIf\n")

        # --- Graded layers loop ---
        geo.write("// Graded layers loop (N layers)\n")
        geo.write("For i In {1 : N}\n")
        geo.write("  h0_graded = h0_graded * q1; // Calculate current layer height\n")
        geo.write("  G[] = Extrude {0, 0, h0_graded} {\n")
        geo.write("    Surface{m}; Layers{1}; Recombine;\n")
        geo.write("  };\n")
        geo.write("  m = G[0]; // Update m inside the loop\n")
        geo.write("  sum = sum + h0_graded; // Accumulate graded height\n")
        geo.write("EndFor\n")

        # --- Final layer to reach total height h ---
        geo.write("// --- Final layer to reach total height h ---\n")
        geo.write("last_h = h2 - sum; // Calculate the very last height needed\n")

        # --- Conditional extrusion for the last layer ---
        geo.write("// Add tolerance check and use Gmsh 2.9.1 specific If syntax\n")
        geo.write("If (last_h > 1e-6 * h) // Only extrude if height is significant\n")
        geo.write("  // For Gmsh 2.9.1: No semicolon after the Extrude block inside If\n")
        geo.write("  Extrude {0, 0, last_h} {\n")
        geo.write("    Surface{m}; Layers{1}; Recombine;\n")
        geo.write("  }\n")
        geo.write("EndIf // End the conditional statement\n")

        # --- End of Extrusion ---
        geo.write("// --- End of Geometry Definition ---\n")


if __name__ == '__main__':
    with open('../info.json', 'r') as f:
        data = json.load(f)
    writeGeo(data)
