"""Read-only prediction/metric comparison, then issue the author-run E1 gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import e1_common as c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    c.add_paths(ap)
    args = ap.parse_args()
    import numpy as np
    root = c.gate_dir(args.output_root, args.horizon)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'gate_PASS.json').unlink(missing_ok=True)
    refs = c.references(args.reference_root, args.horizon)
    reports, failures, completion_hashes = [], [], {}
    identity = c.protocol_identity(args.project_root, args.data_root)
    for kind in ('l1', 'huber'):
        destination = root / kind
        try:
            completed = c.read_json(destination / 'completed.json')
            if (completed['status'] != 'completed' or completed['identity'] != identity
                    or completed['cases'] != 40
                    or completed['task']['verification_kind'] != kind
                    or completed['task']['model'] != 'ACFormer'
                    or completed['task']['seed'] != 13
                    or completed['task']['horizon'] != args.horizon
                    or completed['predictions_sha256'] != c.sha(destination / 'predictions.npz')
                    or completed['metrics_sha256'] != c.sha(destination / 'metrics.csv')):
                raise ValueError('Completion identity/checksum mismatch')
            reference_rows = c.read_csv(refs[kind + '_csv'])
            if kind == 'l1':
                reference_rows = [r for r in reference_rows if r['model'] == 'ACFormer_L1'
                                  and int(r['horizon_steps']) == args.horizon and int(r['seed']) == 13]
            else:
                reference_rows = [r for r in reference_rows if r['model'] == 'ACFormer'
                                  and int(r['horizon']) == args.horizon and int(r['seed']) == 13]
            csv_rows = c.read_csv(destination / 'metrics.csv')
            if len(reference_rows) != 40 or len(csv_rows) != 40:
                raise ValueError('Expected 40 reference and new metric rows')
            old = {r['scenario']: r for r in reference_rows}
            new = {r['scenario']: r for r in csv_rows}
            if len(old) != 40 or set(old) != set(new):
                raise ValueError('Scenario matrix mismatch')
            with np.load(destination / 'predictions.npz', allow_pickle=False) as archive:
                names = archive['scenario'].astype(str).tolist()
                if len(names) != 40 or set(names) != set(old) or names[0] != 'normal':
                    raise ValueError('Archive must cover all40 cases exactly once, normal first')
                if archive['y_pred_mw'].shape != (40, *archive['y_true_mw'].shape):
                    raise ValueError('Archive prediction dimensions differ')
                for i, name in enumerate(names):
                    record = dict(kind=kind, scenario=name, status='FAIL')
                    try:
                        reference_file = refs[kind] / (name + '.npz')
                        with np.load(reference_file, allow_pickle=False) as ref:
                            c.assert_indices(ref, archive, args.horizon)
                            if (int(ref['fault_seed']) != c.FAULT_SEED or int(ref['horizon']) != args.horizon
                                    or int(ref['seed']) != 13
                                    or not np.array_equal(ref['capacity_mw'], archive['capacity_mw'])
                                    or json.loads(str(ref['scenario_json'])) != json.loads(str(archive['scenario_json'][i]))):
                                raise ValueError('Reference scenario/seed/capacity identity mismatch')
                            pred = archive['y_pred_mw'][i]
                            record.update(max_abs_diff_mw=float(np.abs(pred-ref['y_pred_mw']).max()),
                                          max_abs_diff_pu=float((np.abs(pred-ref['y_pred_mw'])/archive['capacity_mw']).max()))
                            record.update(c.assert_prediction_equal(ref['y_pred_mw'], pred, archive['capacity_mw']))
                            calculated = c.metrics(archive['y_true_mw'], pred, archive['target_valid'], archive['capacity_mw'])
                            differences = []
                            for metric in ('nrmse', 'nmae'):
                                key = metric + '_pct'
                                reference_metric = float(old[name][key] if kind == 'l1' else old[name][metric]) * (1 if kind == 'l1' else 100)
                                differences.append(abs(calculated[key] - reference_metric))
                                if not np.isfinite(float(new[name][key])) or abs(calculated[key] - float(new[name][key])) > 1e-9:
                                    raise ValueError('New CSV differs from its saved predictions')
                            record['max_abs_metric_diff_pp'] = max(differences)
                            if not np.isfinite(differences).all() or max(differences) > c.METRIC_ATOL_PP:
                                raise ValueError(f'Metric mismatch exceeds {c.METRIC_ATOL_PP} percentage points')
                            expected_n = int(old[name]['n_targets' if kind == 'l1' else 'n'])
                            if expected_n != calculated['n_targets'] or expected_n != int(new[name]['n_targets']):
                                raise ValueError('n_targets mismatch')
                            for metric in ('nrmse', 'nmae'):
                                baseline = float(new['normal'][metric + '_pct'])
                                derived = 100 * (calculated[metric + '_pct'] / baseline - 1)
                                value = float(new[name][metric + '_increase_pct'])
                                if not np.isfinite(value) or abs(derived - value) > 1e-9:
                                    raise ValueError('New degradation not relative to matched normal')
                                reference_degradation = old[name].get(metric + '_relative_degradation_pct', '') if kind == 'l1' else old[name].get(metric + '_increase', '')
                                reference_degradation = float(reference_degradation or 0) * (1 if kind == 'l1' else 100)
                                if not np.isfinite(reference_degradation) or abs(value - reference_degradation) > 0.05:
                                    raise ValueError('Relative degradation mismatch >0.05 percentage points')
                            record.update(status='PASS', reference_sha256=c.sha(reference_file))
                    except Exception as error:
                        record['error'] = str(error)
                        failures.append(f'{kind}/{name}: {error}')
                    reports.append(record)
                    print(f"{record['status']} {kind}/{name} max_abs_diff_MW={record.get('max_abs_diff_mw', 'NA')} "
                          f"max_abs_diff_pu={record.get('max_abs_diff_pu', 'NA')} max_metric_diff_pp={record.get('max_abs_metric_diff_pp', 'NA')}", flush=True)
            completion_hashes[kind] = c.sha(destination / 'completed.json')
        except Exception as error:
            failures.append(f'{kind}: {error}')
            print(f'FAIL {kind}: {error}', flush=True)
    if not failures:
        left = c.read_json(root / 'l1/completed.json')['realization_sha256']
        right = c.read_json(root / 'huber/completed.json')['realization_sha256']
        if len(left) != 39 or left != right:
            failures.append('L1 and Huber did not receive exactly identical corrupted inputs/missing flags')
    report = dict(status='FAIL' if failures else 'PASS', horizon=args.horizon,
                  prediction_atol_pu=c.ATOL_PU, prediction_rtol=c.RTOL,
                  metric_atol_percentage_points=c.METRIC_ATOL_PP,
                  reports=reports, failures=failures, identity=identity,
                  verified_completion_sha256=completion_hashes,
                  reference_csv_sha256={k: c.sha(refs[k + '_csv']) for k in ('l1', 'huber')})
    c.dump_json(root / 'comparison.json', report)
    if failures or len(reports) != 80:
        print('E1 GATE FAIL: full run remains disabled. Preserve logs; do not loosen tolerances.')
        raise SystemExit(1)
    c.dump_json(root / 'gate_PASS.json', report)
    print('E1 GATE PASS: both ACFormer losses x40 cases; full run enabled for these exact code/data hashes.')


if __name__ == '__main__':
    main()
