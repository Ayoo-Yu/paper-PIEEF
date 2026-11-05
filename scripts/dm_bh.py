#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pieef.analysis import bh_correction, dm_test, load_main_runs, load_loss_series_from_npz
from pieef.ledger import LEDGER_SPECS, base_row, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DM test + BH correction from saved prediction artifacts.")
    p.add_argument("--seed", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--config-id", "--config_id", default="dm_bh_v1")
    p.add_argument("--artifact-path", "--artifact_path", default="NA")
    p.add_argument("--ledger-dir", default=str(ROOT / "ledger"))
    p.add_argument("--ref-model", default="PI-EEF")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_rows(csv_path: Path, fields: list[str], rows: list[dict], overwrite: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    need_header = overwrite or (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "NA") for k in fields})


def main() -> None:
    args = parse_args()
    ledger_dir = Path(args.ledger_dir)

    df = load_main_runs(ledger_dir / "main_benchmark_runs.csv")
    if args.horizon is not None:
        df = df[df["horizon"] == args.horizon]
    if args.seed is not None:
        df = df[df["seed"] == args.seed]
    if df.empty:
        print("No successful main runs found.")
        return

    by_key = {}
    for _, row in df.iterrows():
        key = (str(row["model"]), int(row["horizon"]), int(row["seed"]))
        by_key[key] = row

    models = sorted(set(df["model"].astype(str)))
    horizons = sorted(set(df["horizon"].dropna().astype(int)))

    results = []
    grouped_for_bh = defaultdict(list)

    for horizon in horizons:
        for baseline in models:
            if baseline == args.ref_model:
                continue

            ref_rows = df[(df["model"] == args.ref_model) & (df["horizon"] == horizon)]
            base_rows = df[(df["model"] == baseline) & (df["horizon"] == horizon)]
            ref_seeds = set(ref_rows["seed"].astype(int).tolist())
            base_seeds = set(base_rows["seed"].astype(int).tolist())
            common_seeds = sorted(ref_seeds & base_seeds)
            if not common_seeds:
                continue

            loss_ref_all = []
            loss_base_all = []
            err_ids = []

            for seed in common_seeds:
                ref_art = Path(str(by_key[(args.ref_model, horizon, seed)]["artifact_path"]))
                base_art = Path(str(by_key[(baseline, horizon, seed)]["artifact_path"]))
                ref_npz = ref_art / "predictions.npz"
                base_npz = base_art / "predictions.npz"
                if not (ref_npz.exists() and base_npz.exists()):
                    continue

                ref_loss = load_loss_series_from_npz(ref_npz)
                base_loss = load_loss_series_from_npz(base_npz)
                n = min(len(ref_loss), len(base_loss))
                if n < 3:
                    continue

                loss_ref_all.append(ref_loss[:n])
                loss_base_all.append(base_loss[:n])
                err_ids.append(f"{horizon}:{seed}")

            if not loss_ref_all:
                continue

            dm = dm_test(
                loss_ref=np.concatenate(loss_ref_all),
                loss_base=np.concatenate(loss_base_all),
            )
            grouped_for_bh[horizon].append((baseline, dm, err_ids))

    for horizon, items in grouped_for_bh.items():
        pvals = [x[1].p_value for x in items]
        bh_vals = bh_correction(pvals)
        for (baseline, dm, err_ids), bh_p in zip(items, bh_vals):
            ts = now_iso()
            row = base_row(
                run_id=f"significance_{baseline}_h{horizon}_{ts.replace(':', '').replace('-', '')}",
                exp_type="significance",
                model=args.ref_model,
                variant="DM+BH",
                seed=0,
                horizon=horizon,
                data_version="NA",
                config_id=args.config_id,
                artifact_path=args.artifact_path,
                status="success",
                start_time=ts,
                end_time=ts,
                fail_reason="NA",
                eval_scope="all-sample",
            )
            row.update(
                {
                    "ref_model": args.ref_model,
                    "baseline_model": baseline,
                    "dm_stat": dm.dm_stat,
                    "dm_pvalue": dm.p_value,
                    "bh_pvalue": bh_p,
                    "significant": "Yes" if (bh_p <= args.alpha) else "No",
                    "error_series_id": json.dumps(err_ids, ensure_ascii=False),
                }
            )
            results.append(row)

    if not results:
        print("No comparable prediction artifacts found for DM/BH.")
        return

    spec = LEDGER_SPECS["significance"]
    out_csv = ledger_dir / spec.file_name
    write_rows(out_csv, list(spec.fields), results, overwrite=args.overwrite)
    write_rows(
        ledger_dir / LEDGER_SPECS["run_registry"].file_name,
        list(LEDGER_SPECS["run_registry"].fields),
        results,
        overwrite=False,
    )

    print(f"Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
