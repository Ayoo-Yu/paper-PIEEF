# @Author: joe contact withheld
# @Date: 2025-07-08 12:41:21
# @LastEditors: joe contact withheld
# @LastEditTime: 2025-07-13 20:13:32
# @FilePath: \\wsl.localhost\Ubuntu-22.04\home\joe\wind_project\WindSimProj\backend\utils\preStreamLines.py
# @Description:
#
# Copyright (c) 2025 by joe, All Rights Reserved.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complete_streamline_processor.py  --  完整的流线处理：生成 → 裁剪 → 压缩
- 通过修改顶部的 TEST_LEVEL 变量，可以方便地在不同步长级别之间切换。
- 目标：系统性地找到能够生成“长且弯曲”流线的最佳参数。
- 新增：采用地形跟随或任意表面顶点的种子生成和裁剪，以适应非平坦地表。
"""

import pyvista as pv
import numpy as np
import sys
import pathlib
import argparse
from scipy.spatial import cKDTree
from streamline_geometry import clip_height_range

# =======================================================
#               ↓↓↓ 在这里修改测试级别 ↓↓↓
# =======================================================
TEST_LEVEL = 3  # 可选值: 1 (保守), 2 (中等), 3 (精细)
# =======================================================


def terrain_seeds(bounds, ground_kd, ground_z, z_rel, nx=60, ny=60):
    """
    在规则的XY网格上播种，但每个种子的Z坐标是
    ground(x,y) + z_rel（所有值均为模型单位）。
    """
    xmin,xmax, ymin,ymax, *_ = bounds
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    xy     = np.column_stack([xx.ravel(), yy.ravel()])

    _, idx = ground_kd.query(xy)
    z0     = ground_z[idx]
    zz     = z0 + z_rel
    pts    = np.column_stack([xy, zz])
    print(f"生成了 {nx}×{ny} 的随地形种子网格，距地 {z_rel:.3f} (模型单位)")
    return pv.PolyData(pts)


def surface_vertices_seeds(path, sample=None, every=1):
    """Sample at most N surface vertices without requesting more than exist."""
    if sample is not None and sample <= 0:
        raise ValueError("采样数必须为正整数")
    if every <= 0:
        raise ValueError("采样步长必须为正整数")
    surf = pv.read(path).clean()
    pts = surf.points
    if not len(pts) or not np.isfinite(pts).all():
        raise ValueError("种子表面没有有效的有限坐标")
    if sample is not None:
        count = min(sample, len(pts))
        idx = np.random.default_rng(0).choice(len(pts), count, replace=False)
        pts = pts[idx]
        print(f"[seed] 采样: {len(pts)} / {surf.n_points} 个顶点")
    else:
        pts = pts[::every]
    return pv.PolyData(pts)


def build_ground_lookup(mesh, z_rel, seed_surface=None):
    """Use the translated terrain seed surface, or the mesh's lower XY envelope.

    The production caller passes the sampled surface translated by z_rel from
    bot.stl. Its vertices therefore give ground Z after subtracting z_rel.
    Full-volume exterior vertices also contain the roof: a raw nearest-XY tree
    cannot distinguish roof and floor when their XY coordinates coincide.
    """
    if seed_surface:
        points = np.asarray(pv.read(seed_surface).points, dtype=float).copy()
        points[:, 2] -= z_rel
    else:
        points = np.asarray(mesh.extract_surface().points, dtype=float)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("无法从空数据或非有限坐标构建地面高度")
    xy, inverse = np.unique(points[:, :2], axis=0, return_inverse=True)
    ground_z = np.full(len(xy), np.inf)
    np.minimum.at(ground_z, inverse, points[:, 2])
    return cKDTree(xy), ground_z


def clip_following_terrain(stream, ground_kd, ground_z, z_rel, delta):
    """Clip polyline segments at the relative-height bounds, interpolating data.

    extract_points(adjacent_cells=True) selects complete polylines when just
    one vertex matches; it does not clip them to the requested height slab.
    """
    if stream is None or stream.n_cells == 0:
        return None
    if not np.isfinite(z_rel) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("相对高度必须有限，半厚度必须为正数")
    _, idx = ground_kd.query(stream.points[:, :2])
    relative_height = stream.points[:, 2] - ground_z[idx]
    clipped = clip_height_range(stream, relative_height, z_rel - delta, z_rel + delta)
    if clipped.n_cells == 0:
        return None
    print(f"裁剪后: {clipped.n_cells} 条线 / {clipped.n_points} 点")
    return clipped


def analyze_streamlines_enhanced(stream):
    """增强版流线分析，包含更详细的诊断"""
    print("=== 增强流线轨迹分析 ===")
    if not stream or stream.n_cells == 0:
        print("警告: 没有生成流线!")
        return
    line_lengths, max_displacement, line_points_count = [], [], []
    for i in range(stream.n_cells):
        line = stream.get_cell(i)
        if line.n_points > 1:
            line_points = line.points
            line_points_count.append(line.n_points)
            segments = np.diff(line_points, axis=0)
            length = np.sum(np.linalg.norm(segments, axis=1))
            line_lengths.append(length)
            displacement = np.linalg.norm(line_points[-1] - line_points[0])
            max_displacement.append(displacement)
    if not line_lengths:
        print("没有有效的流线长度数据。")
        return
    print(f"分析流线数量: {len(line_lengths)} / {stream.n_cells}")
    print(f"流线长度: 平均={np.mean(line_lengths):.3f}, 最大={np.max(line_lengths):.3f}, 最小={np.min(line_lengths):.3f}")
    print(f"端到端位移: 平均={np.mean(max_displacement):.3f}, 最大={np.max(max_displacement):.3f}")
    print(f"每条流线点数: 平均={np.mean(line_points_count):.1f}, 最大={np.max(line_points_count)}, 最小={np.min(line_points_count)}")
    all_points = stream.points
    x_range = all_points[:, 0].max() - all_points[:, 0].min()
    y_range = all_points[:, 1].max() - all_points[:, 1].min()
    z_range = all_points[:, 2].max() - all_points[:, 2].min()
    print(f"流线覆盖范围: X={x_range:.3f}, Y={y_range:.3f}, Z={z_range:.3f}")
    curvature_ratios = [disp/length for disp, length in zip(max_displacement, line_lengths) if length > 1e-3]
    if not curvature_ratios:
        print("无法计算流线直线度。")
        return
    avg_curvature = np.mean(curvature_ratios)
    print(f"流线直线度: 平均={avg_curvature:.3f} (1.0=完全直线, <1.0=有弯曲)")
    if avg_curvature > 0.995:
        print("⚠️  警告: 流线过度直线化，建议减小步长或检查湍流模型。")


def generate_streamlines_enhanced(mesh, ground_kd, ground_z, z_rel, args):
    """
    增强版流线生成，使用随地形或表面顶点的种子和RK45积分器
    """
    print("=" * 50)
    print(f"第一步：生成流线 (测试 Level {TEST_LEVEL})")
    print("=" * 50)

    # --- 选择播种策略 --------------------------------------
    if args.seed_surface:
        try:
            seeds = surface_vertices_seeds(args.seed_surface,
                                           sample=args.seed_sample,
                                           every=1)
            print(f"使用表面 {args.seed_surface} 的 {seeds.n_points} 个顶点作种子")
        except FileNotFoundError:
            print(f"✗ 错误: 种子表面文件未找到: {args.seed_surface}")
            sys.exit(1)
        except ValueError as e:
            print(f"✗ 错误: {e}")
            sys.exit(1)
    else:
        # 回退到地形网格播种
        if z_rel is None:
            raise ValueError("z_rel (相对地形高度) 必须提供才能使用地形种子。")
        seeds = terrain_seeds(mesh.bounds,
                              ground_kd,
                              ground_z,
                              z_rel,
                              nx=int(np.sqrt(args.seeds)),
                              ny=int(np.sqrt(args.seeds)))

    bounds = mesh.bounds
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

    test_params = {
        1: {"factor": 0.005, "name": "保守"},
        2: {"factor": 0.002, "name": "中等"},
        3: {"factor": 0.0005, "name": "精细"}
    }

    if TEST_LEVEL not in test_params:
        raise ValueError(f"无效的 TEST_LEVEL: {TEST_LEVEL}。请选择 1, 2, 或 3。")

    params = test_params[TEST_LEVEL]
    step_factor = params["factor"]

    step_length = max_dim * step_factor
    max_steps = int((max_dim * 1.5) / step_length) * (4 if TEST_LEVEL == 3 else 1)
    max_time = 10000

    print(f"=== 增强积分参数 (RK45 - {params['name']}) ===")
    print(f"积分器类型: RK45 (自适应步长)")
    print(f"初始步长: {step_length:.6f}")
    print(f"最大步数: {max_steps}")
    print(f"终端速度阈值: 1e-12")

    stream = None
    try:
        stream = mesh.streamlines_from_source(
            seeds,
            vectors=args.vec,
            integrator_type=45,
            integration_direction='forward',
            initial_step_length=step_length,
            step_unit='l',
            max_steps=max_steps,
            max_time=max_time,
            terminal_speed=1e-12
        )
        print(f'✓ RK45积分成功: {stream.n_cells} 条流线')

    except Exception as e:
        print(f"✗ RK45积分失败: {e}")
        return None

    if stream and stream.n_cells > 0:
        analyze_streamlines_enhanced(stream)
    else:
        print("警告: 流线生成失败!")
        return None

    return stream


def main():
    ap = argparse.ArgumentParser(description='完整的流线处理：从网格生成流线并进行随地形裁剪')
    ap.add_argument('infile', help='输入网格文件 (*.vtu, *.foam, etc)')
    ap.add_argument('-H', '--height', type=float, required=True, help='目标相对高度 (m，实际尺寸)')
    ap.add_argument('-d', '--delta', type=float, default=0.5, help='半厚度 Δ (m，实际尺寸，建议0.5-2.0)')
    ap.add_argument('--scale', type=float, default=1000.0, help='缩尺比例')
    ap.add_argument('--seeds', type=int, default=4000, help='种子点数量 (近似值) - 仅用于地形网格模式')
    ap.add_argument('--vec', default='U', help='速度矢量场名称')
    ap.add_argument('-o', '--outfile', help='输出文件名 (自动添加后缀)')
    ap.add_argument('--save-intermediate', action='store_true', help='保存中间结果 (裁剪前)')

    # --- 新增命令行参数 ---
    ap.add_argument('--seed-surface', help='路径: 用该 VTP/STL 表面顶点作种子')
    ap.add_argument('--seed-sample',  type=int,
                    help='若提供，则从表面随机抽取 N 个顶点作种子')

    args = ap.parse_args()

    try:
        if not np.isfinite(args.height) or args.height < 0:
            raise ValueError("目标相对高度必须是非负有限数")
        if not np.isfinite(args.scale) or args.scale <= 0:
            raise ValueError("缩尺比例必须是正有限数")
        if not np.isfinite(args.delta) or args.delta <= 0:
            raise ValueError("半厚度必须是正有限数")
        if args.seeds <= 0 or (args.seed_sample is not None and args.seed_sample <= 0):
            raise ValueError("种子数量必须为正整数")
        height_model = args.height / args.scale
        delta_model = args.delta / args.scale

        print("=" * 60)
        print("参数配置:")
        print(f"  输入相对高度: {args.height} m | 模型相对高度 (z_rel): {height_model:.6f}")
        print(f"  输入半厚度: {args.delta} m | 模型半厚度 (delta): {delta_model:.6f}")
        print(f"  缩尺比例: 1:{args.scale}")
        print(f"  测试级别: {TEST_LEVEL}")
        print("=" * 60)

        infile = pathlib.Path(args.infile)
        mesh = pv.read(infile)

        mesh = mesh.clean()

        if args.vec in mesh.cell_data and args.vec not in mesh.point_data:
            mesh = mesh.cell_data_to_point_data()
            print('[info] 已将单元格矢量转换为点矢量')

        print('[info] 正在构建地表高度查找树...')
        if args.vec not in mesh.point_data or mesh.point_data[args.vec].shape != (mesh.n_points, 3):
            raise ValueError(f"未找到三分量点速度场: {args.vec}")
        ground_kd, ground_z = build_ground_lookup(mesh, height_model, args.seed_surface)
        print('[info] 地表查找树构建完成。')

        # --- 修改函数调用以传递所有参数 ---
        stream = generate_streamlines_enhanced(mesh,
                                               ground_kd, ground_z,
                                               height_model,
                                               args)

        if args.save_intermediate and stream:
            intermediate_file = infile.with_stem(f"{infile.stem}_level{TEST_LEVEL}_dense").with_suffix('.vtp')
            stream.save(intermediate_file)
            print(f'保存中间流线到: {intermediate_file}')

        if stream:
            final_stream = clip_following_terrain(stream,
                                                  ground_kd, ground_z,
                                                  height_model, delta_model)
        else:
            print("流线生成失败，无法进行后续处理。")
            sys.exit(1)

        if final_stream and final_stream.n_cells > 0:
            if args.outfile:
                out_path = pathlib.Path(args.outfile).with_suffix('.vtp')
            else:
                out_path = infile.with_stem(f'{infile.stem}_{args.height:.0f}m-agl_level{TEST_LEVEL}_web').with_suffix('.vtp')

            final_stream.save(out_path)
            print("=" * 60)
            print(f'✓ 处理完成！最终文件: {out_path}')
            print(f'  最终流线: {final_stream.n_cells} 条线 / {final_stream.n_points} 顶点')
            print(f'  目标相对高度范围: {args.height-args.delta:.1f} → {args.height+args.delta:.1f} m (AGL)')
            print("=" * 60)
        else:
            print("❌ 最终没有生成有效的流线文件。")
            sys.exit(1)

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
