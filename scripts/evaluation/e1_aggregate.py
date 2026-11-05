"""Validate all 70 completed tasks, then merge 2800 metric rows."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import e1_common as c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root', type=Path, default=c.OUTPUT)
    ap.add_argument('--horizon', type=int, choices=(16, 96), default=16)
    ap.add_argument('--tasks', type=Path)
    args = ap.parse_args()
    import numpy as np
    tasks = c.load_tasks(args.horizon, args.tasks)
    root = args.output_root / f'h{args.horizon}'
    # A failed new audit must never leave a formerly valid final CSV looking current.
    final_csv = root / 'E1_fault_all_models.csv'
    final_csv.unlink(missing_ok=True)
    gate = c.read_json(root / 'verification/gate_PASS.json')
    if gate['status'] != 'PASS' or gate['horizon'] != args.horizon:
        raise ValueError('No matching verification PASS')
    c.require_gate(args.output_root, args.horizon, gate['identity'])
    reference_hashes = c.read_json(root / 'verification/l1/completed.json')['realization_sha256']
    reference = root / 'verification/l1/predictions.npz'
    rows, errors = [], []
    for task in tasks:
        directory = root / 'tasks' / task['task_id']
        try:
            complete = c.read_json(directory / 'completed.json')
            if (complete['task'] != task or complete['status'] != 'completed' or complete['cases'] != 40
                    or complete['identity'] != gate['identity'] or complete['realization_sha256'] != reference_hashes
                    or complete['predictions_sha256'] != c.sha(directory / 'predictions.npz')
                    or complete['metrics_sha256'] != c.sha(directory / 'metrics.csv')):
                raise ValueError('Completion/protocol/input-realization/checksum differs')
            part = c.read_csv(directory / 'metrics.csv')
            with np.load(directory / 'predictions.npz', allow_pickle=False) as pred, np.load(reference, allow_pickle=False) as ref:
                c.assert_indices(ref, pred, args.horizon)
                names = pred['scenario'].astype(str).tolist()
                if (len(names) != 40 or len(set(names)) != 40 or names != ref['scenario'].astype(str).tolist()
                        or len(part) != 40 or [r['scenario'] for r in part] != names
                        or not np.array_equal(pred['capacity_mw'], ref['capacity_mw'])
                        or not np.array_equal(pred['scenario_json'], ref['scenario_json'])
                        or int(pred['fault_seed']) != c.FAULT_SEED
                        or int(pred['horizon']) != args.horizon or int(pred['seed']) != task['seed']
                        or str(pred['model']) != task['model']
                        or pred['y_pred_mw'].shape != (40, *ref['y_true_mw'].shape)):
                    raise ValueError('Archive scenario/index/model dimensions differ')
                normal = None
                for i, row in enumerate(part):
                    if (row['model'], int(row['seed']), int(row['horizon'])) != (task['model'], task['seed'], args.horizon):
                        raise ValueError('Wrong model/seed/horizon in metrics')
                    actual = c.metrics(pred['y_true_mw'], pred['y_pred_mw'][i], pred['target_valid'], pred['capacity_mw'])
                    if i == 0:
                        normal = actual
                    if int(row['n_targets']) != actual['n_targets']:
                        raise ValueError('Wrong valid target count')
                    for metric in ('nrmse', 'nmae'):
                        if not np.isfinite(float(row[metric + '_pct'])) or abs(float(row[metric + '_pct']) - actual[metric + '_pct']) > 1e-9:
                            raise ValueError('CSV metric differs from NPZ')
                        increase = 100 * (actual[metric + '_pct'] / normal[metric + '_pct'] - 1)
                        if not np.isfinite(float(row[metric + '_increase_pct'])) or abs(float(row[metric + '_increase_pct']) - increase) > 1e-9:
                            raise ValueError('Wrong degradation reference')
            rows.extend(part)
            print(f'PASS {task["task_id"]}: 40 cases, targets, metrics, identical fault inputs')
        except Exception as error:
            errors.append(dict(task=task['task_id'], error=str(error)))
            print(f'FAIL {task["task_id"]}: {error}')
    report = dict(status='FAIL' if errors else 'PASS', expected_tasks=70, expected_rows=2800,
                  actual_rows=len(rows), errors=errors, horizon=args.horizon)
    c.dump_json(root / 'completeness.json', report)
    if errors or len(rows) != 2800:
        raise SystemExit('Incomplete matrix: final CSV was not created/updated')
    path = final_csv
    c.write_csv(path, rows)
    print(f'PASS 70 tasks x40 cases =2800 rows: {path}')


if __name__ == '__main__':
    main()
