"""Shared IO/protocol helpers. Importing this module does not import torch or run a model."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
E = Path('results/paper')
PROJECT = E / 'next_batch/merged_v2/project'
MAIN = Path('results/paper')
DATA = Path('data/wind_farm_A')
OUTPUT = E / '补充验证_20261027/E1_faults_all_models'
MODELS = ('Persistence', 'LightGBM', 'XGBoost', 'GRU', 'TCN', 'DLinear',
          'iTransformer', 'TimesNet', 'TimeMixer', 'PatchTST', 'ACFormer',
          'STAN', 'Wake-GNN', 'Jensen-Net')
SEEDS = (13, 21, 42, 87, 3407)
FAULT_SEED = 2026
ATOL_PU = 1e-5
RTOL = 1e-5
METRIC_ATOL_PP = 0.002
COLUMNS = ('model', 'seed', 'horizon', 'scenario', 'n_targets', 'nrmse_pct',
           'nmae_pct', 'nrmse_increase_pct', 'nmae_increase_pct')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity(project):
    project = Path(project)
    files = sorted((project / 'src').rglob('*.py'))
    files += [project / 'scripts/evaluate_revision.py', project / 'scripts/train_baselines.py']
    if not files:
        raise ValueError('No project code found')
    return {str(p.relative_to(project)).replace('\\', '/'): sha(p) for p in files}


def protocol_identity(project, data):
    # Paths are deliberately excluded so identical packages can move machines.
    return dict(code=code_identity(project),
                data={name: sha(Path(data) / name) for name in ('scada_ecmwf_15min.csv', 'turbines.csv')},
                runner={name: sha(HERE / name) for name in
                        ('e1_common.py', 'e1_run.py', 'e1_compare.py')},
                fault_seed=FAULT_SEED, scenario_scope='structured_39')


def load_evaluation(project):
    if any(k == 'pieef' or k.startswith('pieef.') for k in sys.modules):
        raise RuntimeError('Use a fresh Python process; pieef is already imported')
    sys.path.insert(0, str(Path(project).resolve() / 'src'))
    spec = importlib.util.spec_from_file_location('e1_project_evaluation',
                                                 Path(project) / 'scripts/evaluate_revision.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from pieef.revision_fault_jobs import scada_scenarios
    scenarios = scada_scenarios('structured_39')
    if len(scenarios) != 39 or len({s['name'] for s in scenarios}) != 39:
        raise ValueError('Expected exactly 39 SCADA scenarios')
    return ev, scenarios


def task_file(horizon):
    return PACKAGE / 'tasks' / f'e1_h{horizon}.json'


def load_tasks(horizon, path=None):
    payload = read_json(path or task_file(horizon))
    tasks = payload['tasks']
    expected = {(m, s) for m in MODELS for s in SEEDS}
    if (len(tasks) != 70 or {(t['model'], t['seed']) for t in tasks} != expected
            or len({t['task_id'] for t in tasks}) != 70
            or any(t['task_id'] != f"{t['model']}_h{horizon}_seed{t['seed']}"
                   or Path(t['run_directory']).name != t['run_directory']
                   or t['horizon'] != horizon or t['cases'] != 40 for t in tasks)):
        raise ValueError('Task manifest must contain exactly 14 models x 5 seeds x 40 cases')
    return tasks


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in COLUMNS} for row in rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(truth, pred, valid, capacity):
    import numpy as np
    if (valid.dtype.kind != 'b' or not valid.any() or not np.array_equal(np.isfinite(truth), valid)
            or not np.isfinite(pred).all() or not np.isfinite(capacity).all() or not (capacity > 0).all()):
        raise ValueError('Invalid predictions or changed target validity')
    error = ((pred - truth) / capacity)[valid].astype(np.float64)
    return dict(n_targets=int(valid.sum()),
                nrmse_pct=float(100 * np.sqrt(np.square(error).mean())),
                nmae_pct=float(100 * np.abs(error).mean()))


def assert_prediction_equal(reference, actual, capacity):
    import numpy as np
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Prediction shape/nonfinite mismatch')
    delta = np.abs((actual - reference) / capacity)
    maximum = float(delta.max())
    if np.any(delta > ATOL_PU + RTOL * np.abs(reference / capacity)):
        raise ValueError(f'Prediction FAIL max_abs_diff_pu={maximum:.12g}')
    return dict(max_abs_diff_mw=float(np.abs(actual - reference).max()), max_abs_diff_pu=maximum)


def assert_indices(reference, actual, horizon):
    import numpy as np
    import pandas as pd
    for key in ('timestamp', 'origin_timestamp', 'turbine_ids', 'target_valid', 'events'):
        if not np.array_equal(reference[key].astype(str) if reference[key].dtype.kind == 'O' else reference[key], actual[key]):
            raise ValueError(f'Index mismatch: {key}')
    if not np.array_equal(reference['y_true_mw'], actual['y_true_mw'], equal_nan=True):
        raise ValueError('Measured targets differ')
    t = pd.to_datetime(actual['timestamp'], utc=True).asi8
    o = pd.to_datetime(actual['origin_timestamp'], utc=True).asi8
    if not np.all(t - o == horizon * 15 * 60 * 10**9) or not np.all(np.diff(t) == 15 * 60 * 10**9):
        raise ValueError('Incorrect horizon or discontinuous 15-minute test grid')


def add_paths(parser):
    parser.add_argument('--project-root', type=Path, default=PROJECT,
                        help='Final merged_v2 project CODE root (not the old P/src)')
    parser.add_argument('--main-root', type=Path, default=MAIN,
                        help='Original main baseline run directories')
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--reference-root', type=Path, default=E)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--horizon', type=int, choices=(16, 96), default=16)


def gate_dir(output, horizon):
    return Path(output) / f'h{horizon}' / 'verification'


def references(root, horizon):
    root = Path(root)
    return dict(l1=root / f'补充验证_20261022/l1_fault_events/runs/ACFormer_L1_h{horizon}_seed13',
                huber=root / f'补做重算/final/F2/ACFormer_h{horizon}_seed13',
                l1_csv=root / '补充验证_20261022/l1_fault_events/fault_per_run.csv',
                huber_csv=root / '补做重算/final/G3_all_baseline_metrics.csv')


def require_gate(output, horizon, identity):
    path = gate_dir(output, horizon) / 'gate_PASS.json'
    gate = read_json(path)
    if gate.get('status') != 'PASS' or gate.get('horizon') != horizon or gate.get('identity') != identity:
        raise ValueError('Missing/stale E1 verification PASS. Run both verification tasks and e1_compare.py first')
    for name in ('l1', 'huber'):
        summary = gate_dir(output, horizon) / name / 'completed.json'
        archive = summary.parent / 'predictions.npz'
        completed = read_json(summary)
        if (sha(summary) != gate['verified_completion_sha256'][name] or sha(archive) != completed['predictions_sha256']
                or sha(summary.parent / 'metrics.csv') != completed['metrics_sha256']):
            raise ValueError('Verification evidence changed after PASS')
    return gate
