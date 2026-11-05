#!/usr/bin/env python3
"""Check E2 completeness and render both sweeps as one Markdown table (stdlib only)."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    session_path = args.csv.parent / "E2_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    if not session.get("status", "").startswith("PASS:"):
        raise SystemExit("FAIL E2 session has no completed PASS status")
    expected = {(s, n) for s in (1, 2) for n in (10, 20, 29, 50, 72, 100, 200)}
    actual = {(int(r["sweep"]), int(r["n_turbines"])) for r in rows}
    if len(rows) != 14 or actual != expected:
        raise SystemExit(f"FAIL incomplete/duplicate E2: rows={len(rows)}, missing={expected-actual}")
    def value(row, key):
        number = float(row[key])
        if not math.isfinite(number):
            raise ValueError(f"Non-finite result {key}")
        return number
    def spread(row, prefix, divisor=1):
        return f"{value(row,prefix+'median_ms')/divisor:.3f} [{value(row,prefix+'q1_ms')/divisor:.3f}, {value(row,prefix+'q3_ms')/divisor:.3f}]"
    lines = ["各单元为中位数 [Q1, Q3]；两轮保留，不合并为伪重复。", "",
             "| 轮次 | N | 参数量 | 推理 B1 ms/窗口 | 推理 B16 ms/窗口 | 推理峰值 MiB B1/B16 | 训练步 ms | 训练峰值 MiB | 推导每 epoch s |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: (int(r["sweep"]), int(r["n_turbines"]))):
        n = int(row["n_turbines"])
        if int(row["parameters"]) != 678131+18*(n-29) or int(row["analytical_parameters"]) != int(row["parameters"]) or row["parameter_check"] != "PASS":
            raise SystemExit(f"FAIL parameter count N={n}")
        peak = f"{value(row,'infer_b1_peak_allocated_mib'):.1f}/{value(row,'infer_b16_peak_allocated_mib'):.1f}"
        lines.append(f"| {row['sweep']} | {n} | {row['parameters']} | {spread(row,'infer_b1_')} | {spread(row,'infer_b16_',16)} | {peak} | {spread(row,'train_')} | {value(row,'train_peak_allocated_mib'):.1f} | {value(row,'derived_epoch_median_s'):.2f} |")
    batches = {r["train_batch"] for r in rows}
    windows = {r["epoch_windows"] for r in rows}
    if len(batches) != 1 or len(windows) != 1 or {r["session_pid"] for r in rows} != {str(session["pid"])} or len({r["gpu"] for r in rows}) != 1:
        raise SystemExit("FAIL inconsistent session/training batch/epoch workload")
    lines += ["", f"训练 batch={next(iter(batches))}，每 epoch={next(iter(windows))} 个窗口，步数向上取整；epoch 时间由步中位数推导，未含验证、数据读取或主机到 GPU 传输。",
              "推理 B16 同时保存完整批次延迟，表中除以 16 为吞吐等价的每窗口耗时；它不是 B1 响应延迟。",
              "N72 为合成布局；真实 Wind Farm B 的效率结果另报。"]
    args.out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"PASS 14/14 E2 rows; wrote {args.out}")


if __name__ == "__main__":
    main()
