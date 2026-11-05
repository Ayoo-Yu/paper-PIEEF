"""Read-only quality evidence from a roughFoam log, independent of job exit status.

The outer SIMPLE criterion uses the *initial* residual of each linear solve.
Small final linear residuals and a normal process exit do not establish it.
Only the solver's own convergence message, corroborated by the reported
residual controls and the last complete outer iteration, is called converged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TIME = re.compile(r"^Time = (" + _NUMBER + r")\s*$")
_CRITERION = re.compile(r'^\s*field\s+("[^"]+"|\S+)\s+tolerance\s+(' + _NUMBER + r")\s*$")
_RESIDUAL = re.compile(r"Solving for (\w+),\s*Initial residual = (" + _NUMBER + r"),\s*Final residual = (" + _NUMBER + r"),\s*No Iterations\s+(\d+)")
_CONTINUITY = re.compile(r"time step continuity errors\s*:\s*sum local = (" + _NUMBER + r"),\s*global = (" + _NUMBER + r"),\s*cumulative = (" + _NUMBER + r")")
_EXEC = re.compile(r"^Exec\s*:\s*(\S+)")
_CONVERGED = re.compile(r"\bSIMPLE(?:\s+solution|:)?\s+converged\b", re.IGNORECASE)


def parse_mesh_log(text, source=None):
    """Read native checkMesh evidence: an exit code alone is not a pass.

    OpenCFD v2212 can return zero while printing ``Failed 1 mesh checks``.
    Preserve the extended geometry/topology scope, failing counts and native
    messages instead of treating a previous/basic ``Mesh OK`` as decisive.
    """
    failure_counts = [int(value) for value in re.findall(r"Failed\s+(\d+)\s+(?:mesh\s+)?checks?", text, re.IGNORECASE)]
    # The normal banner says "trapFpe: Floating point exception trapping
    # enabled"; it is not a numerical crash.
    failed = any(failure_counts) or bool(re.search(
        r"FOAM FATAL|^\s*Segmentation fault|^\s*Floating point exception(?:\s|$)",
        text, re.IGNORECASE | re.MULTILINE))
    normal_end = bool(re.search(r"^\s*End\s*$", text, re.MULTILINE))
    okay = bool(re.search(r"^\s*Mesh OK\.?\s*$", text, re.MULTILINE))
    all_geometry = bool(re.search(r"Enabling all geometry checks", text))
    all_topology = bool(re.search(r"Enabling all .*topology checks", text))
    diagnostics = [line.strip() for line in text.splitlines()
                   if re.search(r"\*\*\*|FOAM FATAL|Failed\s+\d+\s+(?:mesh\s+)?checks?", line, re.IGNORECASE)]
    metrics = {}
    patterns = {
        "cell_count": r"^\s*cells:\s*(\d+)",
        "max_non_orthogonality_deg": r"Mesh non-orthogonality Max:\s*(" + _NUMBER + r")",
        "max_skewness": r"Max skewness\s*=\s*(" + _NUMBER + r")",
        "min_cell_determinant": r"Cell determinant[^\n]*minimum:\s*(" + _NUMBER + r")",
        "underdetermined_cell_count": r"Cells with small determinant[^\n]*number of cells:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metrics[key] = int(match.group(1)) if key.endswith("count") else float(match.group(1))
    if "underdetermined_cell_count" not in metrics and "Cell determinant check OK" in text:
        metrics["underdetermined_cell_count"] = 0
    return {"status": "failed" if failed else "passed" if okay else "not_assessed",
            "source": source, "all_geometry": all_geometry, "all_topology": all_topology,
            "extended_scope": all_geometry and all_topology,
            "normal_end": normal_end, "failed_check_count": max(failure_counts, default=0),
            "diagnostics": diagnostics, "metrics": metrics}


def add_mesh_quality(quality, case_path):
    run = Path(case_path) / "run"
    evidence = []
    for name in ("checkMesh.log", "checkMesh-all.log", "checkMesh-extended.log"):
        path = run / name
        if path.is_file():
            evidence.append(parse_mesh_log(path.read_text(encoding="utf-8", errors="replace"), "run/" + name))
    # Failure in either check must survive a later/basic pass.
    failed = any(item["status"] == "failed" for item in evidence)
    extended = [item for item in evidence if item["extended_scope"]]
    if failed:
        status = "failed"
        quality["quality_acceptable"] = False
        warning = "原生网格检查存在失败项；即使求解器正常结束或 SIMPLE 收敛，也不能据此判定网格或工程结果可接受。"
        quality["warnings"] = [*quality.get("warnings", []), warning]
        quality["message"] += " " + warning
    else:
        status = "passed" if any(item["status"] == "passed" for item in evidence) else "not_assessed"
    chosen = next((item for item in evidence if item["status"] == "failed"), (extended or evidence or [{}])[-1])
    standard = [item for item in evidence if not item["extended_scope"]]
    quality["mesh_check"] = {"status": status, "source": chosen.get("source"),
                             "standard_status": "failed" if any(item["status"] == "failed" for item in standard)
                             else "passed" if any(item["status"] == "passed" for item in standard)
                             or any(item["status"] == "passed" for item in extended) else "not_assessed",
                             "extended_status": "failed" if any(item["status"] == "failed" for item in extended)
                             else "passed" if any(item["status"] == "passed" for item in extended) else "not_assessed",
                             "evidence": evidence}
    return quality


def parse_solver_log(lines):
    """Parse an iterable, scoping combined pipeline logs to roughFoam only.

    A bare solver log with no Exec header is also accepted.  A new solver
    invocation resets evidence; other OpenFOAM utilities cannot supply its
    time, residual, normal-end or convergence evidence.
    """
    active = True
    criteria, iterations, current = {}, [], None
    explicit_scope = False
    ended = False
    convergence_message = False
    malformed = False

    def reset():
        nonlocal criteria, iterations, current, ended, convergence_message, malformed
        criteria, iterations, current = {}, [], None
        ended = convergence_message = malformed = False

    for raw in lines:
        line = raw.strip()
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                event = {}
            if event.get("action") == "taskStart":
                if event.get("taskId") == "run_roughFoam":
                    reset()
                    active, explicit_scope = True, True
                elif explicit_scope:
                    active = False
            continue
        executable = _EXEC.match(line)
        if executable:
            is_solver = Path(executable.group(1)).name == "roughFoam"
            if is_solver:
                reset()
            active, explicit_scope = is_solver, True
            continue
        if not active:
            continue
        criterion = _CRITERION.match(line)
        if criterion:
            value = float(criterion.group(2))
            if math.isfinite(value) and value > 0:
                criteria[criterion.group(1).strip('"')] = value
            continue
        time = _TIME.match(line)
        if time:
            current = {"iteration": float(time.group(1)), "initial_residuals": {}, "linear_final_residuals": {}}
            continue
        residual = _RESIDUAL.search(line)
        if residual and current is not None:
            field, initial, final = residual.group(1), float(residual.group(2)), float(residual.group(3))
            if not all(math.isfinite(value) and value >= 0 for value in (initial, final)):
                malformed = True
                continue
            # Multiple pressure corrections must not hide an earlier, larger
            # outer residual behind the final correction's smaller one.
            current["initial_residuals"][field] = max(initial, current["initial_residuals"].get(field, 0))
            current["linear_final_residuals"][field] = final
            continue
        if "Solving for " in line and "Initial residual" in line:
            malformed = True
        continuity = _CONTINUITY.search(line)
        if continuity and current is not None:
            values = [float(value) for value in continuity.groups()]
            if all(math.isfinite(value) for value in values):
                current["continuity_errors"] = dict(zip(("sum_local", "global", "cumulative"), values))
            else:
                malformed = True
            continue
        if line.startswith("ExecutionTime =") and current and current["initial_residuals"]:
            iterations.append(current)
            current = None
        if _CONVERGED.search(line):
            convergence_message = True
        if line == "End":
            ended = True
            if explicit_scope:
                active = False

    last = iterations[-1] if iterations else {}
    initial = last.get("initial_residuals", {})
    observed = {field for iteration in iterations for field in iteration["initial_residuals"]}
    checks = {}
    missing = []
    for pattern, tolerance in criteria.items():
        try:
            fields = {field for field in observed if (field in {"Ux", "Uy", "Uz"} if pattern == "U" else re.fullmatch(pattern, field))}
        except re.error:
            fields = set()
        if pattern == "U":
            fields.update({"Ux", "Uy", "Uz"})
        absent = sorted(fields - set(initial))
        if not fields or absent:
            missing.extend(absent or [pattern])
            checks[pattern] = None
        else:
            checks[pattern] = all(initial[field] < tolerance for field in fields)
    complete = bool(iterations and criteria and not missing and not malformed and "continuity_errors" in last)
    criteria_met = all(value is True for value in checks.values()) if complete else None
    convergence = "converged" if complete and criteria_met and convergence_message and ended else "not_converged" if complete else "not_assessed"
    message = {
        "converged": "求解器报告 SIMPLE 残差收敛，末次完整迭代的初始残差满足其报告阈值；连续性误差另列，尚未完成网格独立性或工程精度验证。",
        "not_converged": "已完成求解流程，但尚未确认 SIMPLE 收敛；不能以退出码为零或线性方程最终残差较小作为收敛证明。",
        "not_assessed": "缺少完整的求解日志或残差阈值证据，尚未评价 SIMPLE 收敛；结果不代表工程精度。",
    }[convergence]
    return {
        "convergence": convergence, "convergence_assessed": complete,
        "solver_finished": ended, "solver_convergence_message": convergence_message,
        "residual_definition": "maximum_initial_linear_residual_per_field_per_outer_SIMPLE_iteration",
        "residual_control": criteria, "initial_residual_criteria_met": criteria_met,
        "criteria_checks": checks, "missing_residual_fields": sorted(set(missing)),
        "iteration_count": len(iterations), "last_iteration": last.get("iteration"),
        "last_initial_residuals": initial,
        "last_linear_final_residuals": last.get("linear_final_residuals", {}),
        "continuity_errors": last.get("continuity_errors"),
        "continuity_assessed": False, "grid_independence_assessed": False,
        "quality_acceptable": False if convergence == "not_converged" else None,
        "recent_iterations": iterations[-20:], "message": message,
    }


def add_terrain_quality(quality, case_path):
    """Retain the complete terrain evidence and expose synthetic padding clearly."""
    terrain_path = Path(case_path) / "run" / "terrain-quality.json"
    terrain = None
    status = "not_assessed"
    warnings = list(quality.get("warnings", []))
    if terrain_path.is_file():
        try:
            if terrain_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("terrain quality metadata exceeds the supported size")
            def reject_nonfinite(value):
                raise ValueError("non-finite terrain quality metadata: " + value)
            terrain = json.loads(terrain_path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
            if not isinstance(terrain, dict):
                raise ValueError("terrain quality metadata must be an object")
            status = "available"
            count = terrain.get("outside_dem_bottom_node_count", 0)
            if isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count) and count > 0:
                warnings.append(f"外围有{int(count)}个底面网格节点缺少 DEM 覆盖，已用 DEM 平均高程人工平地填补；这部分不是实测地形，须结合地形范围敏感性评估。")
        except (OSError, ValueError, TypeError):
            terrain, status = None, "invalid"
            warnings.append("地形质量元数据无法解析，尚不能核查 DEM 覆盖、坐标原点与风轮净空。")
    quality.update(terrain_quality=terrain, terrain_quality_status=status, warnings=warnings)
    if warnings:
        quality["message"] += " " + " ".join(warnings)
    return quality


def add_agl_sampling_quality(quality, case_path):
    sampling_path = Path(case_path) / "run" / "agl-sampling-quality.json"
    sampling, status = None, "not_assessed"
    warnings = []
    if sampling_path.is_file():
        try:
            if sampling_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("AGL sampling metadata exceeds the supported size")
            def reject_nonfinite(value):
                raise ValueError("non-finite AGL sampling metadata: " + value)
            sampling = json.loads(sampling_path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
            if not isinstance(sampling, dict) or not isinstance(sampling.get("layers"), list):
                raise ValueError("Invalid AGL sampling quality metadata")
            status = "available"
            for layer in sampling["layers"]:
                outside, invalid = layer.get("outside_domain_count", 0), layer.get("non_finite_count", 0)
                if outside or invalid:
                    warnings.append(f"离地{layer.get('height_m')}米采样有{outside}个目标点位于实际计算域外、{invalid}个目标点数值无效，已保留为缺测，未用最近单元替代。")
        except (OSError, ValueError, TypeError, AttributeError):
            sampling, status = None, "invalid"
            warnings.append("离地高度采样质量元数据无法解析，尚不能核查各层实际覆盖。")
    quality.update(agl_sampling_quality=sampling, agl_sampling_quality_status=status)
    quality["warnings"] = [*quality.get("warnings", []), *warnings]
    if warnings:
        quality["message"] += " " + " ".join(warnings)
    return quality


def add_numerics_quality(quality, case_path):
    path = Path(case_path) / "run" / "constant" / "numerics.json"
    numerics, status = None, "not_assessed"
    warnings = []
    if path.is_file():
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Numerics metadata exceeds the supported size")
            def reject_nonfinite(value):
                raise ValueError("non-finite numerics metadata: " + value)
            numerics = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
            if not isinstance(numerics, dict):
                raise ValueError("Numerics metadata must be an object")
            status = "available"
            if numerics.get("velocity_formal_order") == 1:
                warnings.append("当前速度对流项采用一阶迎风格式以改善稳定性，存在数值扩散，可能使尾流更平滑；收敛不代表预测精度提高，须继续进行格式与网格敏感性验证。")
            elif numerics.get("turbulence_advection_scheme") == "upwind":
                warnings.append("湍流输运采用一阶迎风格式，存在数值扩散；仍须进行格式与网格敏感性验证。")
        except (OSError, ValueError, TypeError):
            numerics, status = None, "invalid"
            warnings.append("数值格式元数据无法解析，尚不能核对本次计算实际使用的对流格式。")
    quality.update(numerics=numerics, numerics_status=status)
    quality["warnings"] = [*quality.get("warnings", []), *warnings]
    if warnings:
        quality["message"] += " " + " ".join(warnings)
    return quality


def assess_case_quality(log_path, case_path):
    log_path, case_path = Path(log_path), Path(case_path)
    if log_path.is_file():
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            quality = parse_solver_log(handle)
    else:
        quality = parse_solver_log(())
    add_mesh_quality(quality, case_path)
    return add_numerics_quality(add_agl_sampling_quality(add_terrain_quality(quality, case_path), case_path), case_path)
