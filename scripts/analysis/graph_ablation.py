"""E4 read-only artifact validation and paired-seed statistics; never imports a model.

No output is published until all five source training/fault jobs and the root
archive/D/E acceptance receipt pass. Original predictions/results stay unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import zipfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 21, 42, 87, 3407)
CONTROL_SHA = '1716c3fe8d05184e398ab6fe076a4720ebfe2ff2f350298c10799e6ed9b5c1c2'
ADDENDUM_SHA = '87bc861c3bee2eae42ce36215d9b0d80d6021df9b33caf7c990290885f3f778d'
METRIC_RECALC_ATOL_PP = 0.002  # Existing E1 metric serialization check, not effect selection.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def within(root, relative):
    p = (root / relative).resolve()
    require(root.resolve() in p.parents, 'Path escaped E4 root: ' + str(p))
    return p


def check_archive(path):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'Archive CRC failed: ' + str(path))


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_indices(reference, actual):
    import numpy as np
    for k in ('timestamp', 'origin_timestamp', 'turbine_ids', 'target_valid', 'events'):
        require(np.array_equal(reference[k], actual[k]), 'Index differs: ' + k)
    require(np.array_equal(reference['y_true_mw'], actual['y_true_mw'], equal_nan=True),
            'Measured targets differ')
    # Parse each timestamp rather than relying on suffix/array equality alone.
    def ns(a):
        times = [datetime.fromisoformat(str(s).replace('Z', '+00:00')) for s in a]
        require(all(t.tzinfo is not None for t in times), 'Timestamp timezone absent')
        return np.array([round(t.timestamp()) * 10**9 for t in times], dtype='int64')
    t, o = ns(actual['timestamp']), ns(actual['origin_timestamp'])
    require(np.all(t - o == 240 * 60 * 10**9), 'Actual lead time differs from 240 min')
    require(np.all(np.diff(t) == 15 * 60 * 10**9), 'Test time grid differs')


def metrics(truth, prediction, valid, capacity):
    import numpy as np
    require(valid.dtype.kind == 'b' and valid.any(), 'Invalid target mask')
    require(np.array_equal(np.isfinite(truth), valid), 'Measured-target validity changed')
    require(np.isfinite(prediction).all(), 'Nonfinite prediction retained as failure')
    require(np.isfinite(capacity).all() and (capacity > 0).all(), 'Invalid capacity')
    e = ((prediction - truth) / capacity)[valid].astype('float64')
    return dict(n_targets=int(valid.sum()), nrmse_pct=float(100 * np.sqrt((e * e).mean())),
                nmae_pct=float(100 * np.abs(e).mean()))


def subset(expected, actual, where='config'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), where + ' differs in type')
        for k, v in expected.items():
            require(k in actual, where + '.' + k + ' absent')
            subset(v, actual[k], where + '.' + k)
    else:
        require(expected == actual, where + ' differs from frozen configuration')


def portable_cfg(cfg):
    cfg = json.loads(json.dumps(cfg))
    # Only actual file-location fields are normalized; no scientific field is ignored.
    for section, key in [('data', 'scada_path'), ('data', 'turbines_path'),
                         ('model', 'manufacturer_curve_path')]:
        cfg[section][key] = cfg[section][key].replace('\\', '/').split('/')[-1]
    return cfg


def paired(variant, control):
    from scipy.stats import ttest_rel
    diffs = [a - b for a, b in zip(variant, control)]
    require(len(diffs) == 5 and all(math.isfinite(x) for x in diffs), 'Incomplete paired seeds')
    if all(x == 0 for x in diffs):
        t, p, special = 0.0, 1.0, 'all five paired differences exactly zero'
    elif statistics.stdev(diffs) == 0:
        t, p, special = None, 0.0, 'nonzero constant paired differences; limiting t distribution'
    else:
        res = ttest_rel(variant, control, alternative='two-sided')
        t, p, special = float(res.statistic), float(res.pvalue), None
        require(math.isfinite(t) and math.isfinite(p), 'Invalid paired test result')
    return dict(n_pairs=5, df=4, t_statistic=t, p_value=p, special_case=special,
                difference_variant_minus_control_pp=statistics.mean(diffs),
                paired_difference_sample_sd_pp=statistics.stdev(diffs),
                relative_change_of_means_pct=100 * (statistics.mean(variant) / statistics.mean(control) - 1),
                mean_seed_relative_change_pct=statistics.mean(100 * (a / b - 1) for a, b in zip(variant, control)),
                control_better_seed_count=sum(b < a for a, b in zip(variant, control)),
                variant_better_seed_count=sum(a < b for a, b in zip(variant, control)),
                tied_seed_count=sum(a == b for a, b in zip(variant, control)))


def bh_two(tests):
    require(len(tests) == 2, 'Each primary BH family must contain exactly two tests')
    order = sorted(range(2), key=lambda i: tests[i]['p_value'])
    running = 1.0
    for rank in (2, 1):
        i = order[rank - 1]
        running = min(running, 2 * tests[i]['p_value'] / rank)
        tests[i]['q_value'] = running
        tests[i]['BH_family_size'] = 2


def summary(values):
    require(len(values) == 5, 'All five seeds required')
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n_seeds=5)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(root, acceptance_path, backup):
    import numpy as np
    root = root.resolve()
    acceptance = read(acceptance_path)
    require(acceptance.get('status') == 'PASS' and acceptance.get('expected_seeds') == list(SEEDS),
            'Root final acceptance missing or not PASS for all five seeds')
    require(set(acceptance['seeds']) == set(map(str, SEEDS)), 'Acceptance seed set differs')
    deploy_path = root / 'launch_manifest.json'
    deploy = read(deploy_path)
    require(sha(deploy_path) == acceptance['manifest_sha256'], 'Deployment acceptance SHA mismatch')
    require(sha(root / 'analysis_addendum.json') == ADDENDUM_SHA, 'Pre-training NMAE decision clarification changed')
    protocol = read(root / 'protocol.json')
    require(sha(root / 'protocol.json') == deploy['protocol_sha256'], 'Frozen protocol changed')
    require(protocol['seeds'] == list(SEEDS) and protocol['n_trainings'] == 5 and protocol['horizon'] == 16,
            'Experiment scope changed')
    expected_stats = protocol['statistics']
    require(expected_stats['P1']['BH_family_size'] == expected_stats['P2']['BH_family_size'] == 2,
            'BH family changed')
    cs_path = root / 'ops/control_sources.json'
    require(sha(cs_path) == CONTROL_SHA, 'Frozen existing-control source contract changed')
    cs = read(cs_path)
    require(sha(cs['audit_path']) == cs['audit_sha256'], 'Control audit changed')
    evidence = []

    def file_verified(path, digest=None, size=None, mirrored=False):
        p = Path(path)
        require(p.is_file() and not p.is_symlink(), 'Missing or symlinked artifact: ' + str(p))
        h = sha(p)
        require(digest is None or h == digest, 'SHA mismatch: ' + str(p))
        require(size is None or p.stat().st_size == size, 'Size mismatch: ' + str(p))
        if mirrored:
            require(backup is not None, 'Backup root required')
            target = backup / p.relative_to(root)
            require(target.is_file() and sha(target) == h, 'D/E mirror differs: ' + str(p))
        evidence.append(dict(path=str(p), bytes=p.stat().st_size, sha256=h, D_E_checked=mirrored))
        return h

    for f in deploy['files']:
        file_verified(within(root, f['path']), f['sha256'], f['bytes'], mirrored=True)
    for f in cs['files']:
        file_verified(f['path'], f['sha256'], f['bytes'])
    file_verified(acceptance_path, mirrored=True)
    file_verified(root / 'analysis_addendum.json', ADDENDUM_SHA, mirrored=True)
    file_verified(cs_path, CONTROL_SHA, mirrored=True)
    all_rows, per_seed, all_realizations = [], [], None
    code_paths = sorted((root / 'project/src').rglob('*.py')) + [root / 'project/scripts/evaluate_revision.py', root / 'project/scripts/train_baselines.py']
    code_identity = {p.relative_to(root / 'project').as_posix(): sha(p) for p in code_paths}
    expected_runner = {n: protocol['official_e1_copy_sha256'][n] for n in ('e1_common.py', 'e1_run.py', 'e1_compare.py')}
    for rel in ('src/pieef/faults.py', 'src/pieef/revision_fault_jobs.py',
                'src/pieef/revision_data.py', 'scripts/evaluate_revision.py'):
        old = next(f for f in cs['project_files'] if Path(f['path']).as_posix().endswith('/' + rel))
        require(sha(root / 'project' / rel) == old['sha256'], 'Control/new fault or data kernel differs: ' + rel)

    for task in protocol['tasks']:
        seed, owner = task['seed'], task['owner']
        accepted = acceptance['seeds'][str(seed)]
        require(accepted['owner'] == owner and accepted['archive_verified'] is True and accepted['D_E_verified'] is True,
                'Incomplete archive/source acceptance')
        require(all(accepted[k] == 0 for k in ('train_exit_code', 'fault_exit_code', 'gate_exit_code')), 'Nonzero scientific/gate exit')
        runtime = root / 'ops/runtime' / owner
        terminal = read(runtime / 'terminal.json')
        require(terminal['status'] == 'completed' and terminal['seed'] == seed and terminal['owner'] == owner,
                'Owner not actually completed')
        for stage in ('gate', 'train', 'fault'):
            wpath = runtime / (stage + '_wait.json')
            w = read(wpath)
            require(w['native_wait'] is True and w['exit_code'] == 0 and w['seed'] == seed and w['owner'] == owner,
                    'Actual native wait absent/nonzero')
            file_verified(wpath, mirrored=True)
        verified = read(runtime / 'training_artifacts_verified.json')
        run = within(root, accepted['source_run'])
        require(run.parent == root / 'project/artifacts' / ('seed' + str(seed)), 'Source directory is not assigned seed')
        require(len(list(run.parent.glob('*/metrics.json'))) == 1, 'Ambiguous/repeated candidate training')
        for name, info in verified['files'].items():
            p = run / name
            file_verified(p, info['sha256'], info['bytes'], mirrored=True)
            if p.suffix in ('.pt', '.npz'):
                check_archive(p)
        cfgpath = root / task['config']
        file_verified(cfgpath, task['config_sha256'], mirrored=True)
        cfg, meta = read(cfgpath), read(run / 'metrics.json')
        require((meta['status'], meta['seed'], meta['horizon'], meta['evaluation_split']) == ('completed', seed, 16, 'test'), 'Training identity differs')
        subset(portable_cfg(cfg), portable_cfg(meta['config']))
        require(meta['config']['model']['graph_neighbour_policy'] == 'self_only', 'Wrong variant')
        ctrl = next(x for x in cs['rows'] if x['seed'] == seed)
        control_meta = read(Path(ctrl['source_directory']) / 'metrics.json')
        control_pred = load_npz(ctrl['predictions']['path'])
        train_pred = load_npz(run / 'predictions.npz')
        check_indices(control_pred, train_pred)
        source_measures = metrics(train_pred['y_true_mw'], train_pred['y_pred_mw'], train_pred['target_valid'], np.asarray(2.0))
        for name in ('nrmse', 'nmae'):
            require(abs(source_measures[name + '_pct'] - 100 * meta['metrics'][name]) <= METRIC_RECALC_ATOL_PP,
                    'Saved training metric disagrees with indexed normal predictions')
        hist = read(run / 'loss_history.json')
        require(len(hist['stage1']) == len(hist['stage2']) == 5 and 1 <= len(hist['stage3']) <= 20,
                'Training budget differs')
        require(hist['best_epoch'] == min(hist['stage3'], key=lambda x: x['val_nrmse'])['epoch'], 'Checkpoint not minimum validation NRMSE')
        gatepath = root / 'gates' / owner / 'gate_PASS.json'
        gate = read(gatepath)
        file_verified(gatepath, mirrored=True)
        require(gate['status'] == 'PASS' and gate['identity_attention_exact'] is True and gate['state_dict_key_shapes_equal'] is True,
                'No real default/self-only gate')
        require(gate['source_checkpoint_sha256'] == ctrl['weights']['sha256'] and gate['parameter_count_all'] == gate['parameter_count_self_only'], 'Gate control/parameter count differs')
        require(gate['code'] == code_identity and gate['data_sha256'] == protocol['frozen_data_sha256'], 'Gate source/data differs')
        fdir = root / 'faults' / ('seed' + str(seed))
        done = read(fdir / 'completed.json')
        require((done['status'], done['seed'], done['horizon'], done['cases']) == ('completed', seed, 16, 40), 'Fault run incomplete')
        require(done['identity']['code'] == code_identity and done['identity']['runner'] == expected_runner and done['identity']['fault_seed'] == 2026,
                'Fault scientific code/runner differs')
        require(done['identity']['data'] == {k: protocol['frozen_data_sha256'][k] for k in ('scada_ecmwf_15min.csv', 'turbines.csv')}
                and done['identity']['scenario_scope'] == 'structured_39', 'Fault data/scope differs')
        require(set(done['source_sha256']) == {'metrics.json', 'preprocessing.npz', 'predictions.npz',
                'weights_best.pt', 'weights_stage1.pt', 'weights_stage2.pt', 'weights_last.pt'},
                'Fault source artifact set incomplete')
        for name, digest in done['source_sha256'].items():
            file_verified(run / name, digest, mirrored=True)
        file_verified(fdir / 'completed.json', mirrored=True)
        file_verified(fdir / 'predictions.npz', done['predictions_sha256'], mirrored=True)
        file_verified(fdir / 'metrics.csv', done['metrics_sha256'], mirrored=True)
        check_archive(fdir / 'predictions.npz')
        a = load_npz(fdir / 'predictions.npz')
        check_indices(control_pred, a)
        require(int(a['seed']) == seed and int(a['horizon']) == 16 and int(a['fault_seed']) == 2026, 'Fault NPZ identity differs')
        require(np.all(a['capacity_mw'] == 2) and a['y_pred_mw'].shape == (40, 7604, 29), 'Fault prediction layout/capacity differs')
        names = [str(s) for s in a['scenario']]
        specs = {'normal': None, **{r['scenario']: r['parameters'] for r in ctrl['fault_scenarios']}}
        require(len(names) == 40 and len(set(names)) == 40 and set(names) == set(specs), 'Exact frozen scenario set differs')
        require(done['identity_attention_checked_cases'] == names, 'Not all cases identity-attention checked')
        require(set(done['realization_sha256']) == set(names) - {'normal'}, 'Missing realization hashes')
        for hashes in done['realization_sha256'].values():
            require(set(hashes) == {'power', 'wind', 'power_mask', 'wind_mask'} and all(len(v) == 64 for v in hashes.values()), 'Invalid realization hash schema')
        if all_realizations is None:
            all_realizations = done['realization_sha256']
        else:
            require(all_realizations == done['realization_sha256'], 'Actual variant corruption differs across seeds')
        with (fdir / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            csv_rows = list(csv.DictReader(f))
        require(len(csv_rows) == 40 and [r['scenario'] for r in csv_rows] == names, 'CSV scenarios/order differs')
        normal_i = names.index('normal')
        delta = np.abs((a['y_pred_mw'][normal_i] - train_pred['y_pred_mw']) / 2)
        require(np.all(delta <= 1e-5 + 1e-5 * np.abs(train_pred['y_pred_mw'] / 2)), 'Fresh normal reproduction failed original strict tolerance')
        variant_metrics, control_metrics = {}, {}
        for i, name in enumerate(names):
            require(json.loads(str(a['scenario_json'][i])) == specs[name], 'Exact fault parameters differ')
            m = metrics(a['y_true_mw'], a['y_pred_mw'][i], a['target_valid'], a['capacity_mw'])
            row = csv_rows[i]
            require(int(row['seed']) == seed and int(row['horizon']) == 16 and int(row['n_targets']) == m['n_targets'], 'Metric row identity differs')
            for k in ('nrmse_pct', 'nmae_pct'):
                require(abs(float(row[k]) - m[k]) < 1e-8, 'Fault metric numerical mismatch')
            variant_metrics[name] = m
            old = load_npz(Path(ctrl['fault_directory']) / (name + '.npz'))
            check_indices(control_pred, old)
            require(int(old['seed']) == seed and int(old['fault_seed']) == 2026 and json.loads(str(old['scenario_json'])) == specs[name], 'Old control fault identity differs')
            control_metrics[name] = metrics(old['y_true_mw'], old['y_pred_mw'], old['target_valid'], old['capacity_mw'])
        for name in names:
            for model, values in [('VF', control_metrics), ('Without spatial aggregation', variant_metrics)]:
                m, n = values[name], values['normal']
                all_rows.append(dict(model=model, seed=seed, horizon=16, scenario=name, **m,
                                     nrmse_increase_pct=100 * (m['nrmse_pct'] / n['nrmse_pct'] - 1),
                                     nmae_increase_pct=100 * (m['nmae_pct'] / n['nmae_pct'] - 1)))
            row = csv_rows[names.index(name)]
            for k in ('nrmse', 'nmae'):
                increase = 100 * (variant_metrics[name][k + '_pct'] / variant_metrics['normal'][k + '_pct'] - 1)
                require(abs(increase - float(row[k + '_increase_pct'])) < 1e-8, 'Fault degradation CSV differs')
        faults = [n for n in names if n != 'normal']
        p3 = [n for n in faults if specs[n]['kind'] == 'missing' and specs[n]['duration_hours'] == 24]
        require(len(faults) == 39 and len(p3) == 8, 'P2/P3 scenario support differs')
        for model, saved_meta, vals in [('VF', control_meta, control_metrics), ('Without spatial aggregation', meta, variant_metrics)]:
            for family, metric, value in [
                ('P1', 'NRMSE', saved_meta['metrics']['nrmse'] * 100), ('P1', 'NMAE', saved_meta['metrics']['nmae'] * 100),
                ('P2', 'NRMSE', statistics.mean(vals[n]['nrmse_pct'] for n in faults)), ('P2', 'NMAE', statistics.mean(vals[n]['nmae_pct'] for n in faults)),
                ('P3', 'NMAE', statistics.mean(vals[n]['nmae_pct'] for n in p3))]:
                per_seed.append(dict(model=model, seed=seed, family=family, metric=metric, value_pct=value,
                                     scenario_count={'P1': 1, 'P2': 39, 'P3': 8}[family]))

    groups, tests = [], []
    for family, metric in [('P1', 'NRMSE'), ('P1', 'NMAE'), ('P2', 'NRMSE'), ('P2', 'NMAE'), ('P3', 'NMAE')]:
        values = {}
        for model in ('VF', 'Without spatial aggregation'):
            vals = [next(r['value_pct'] for r in per_seed if (r['model'], r['seed'], r['family'], r['metric']) == (model, s, family, metric)) for s in SEEDS]
            values[model] = vals
            groups.append(dict(model=model, family=family, metric=metric, **summary(vals)))
        if family != 'P3':
            tests.append(dict(family=family, metric=metric, **paired(values['Without spatial aggregation'], values['VF'])))
    for family in ('P1', 'P2'):
        bh_two([r for r in tests if r['family'] == family])
    # The plan says P1 "same conditions" as NMAE-based P2. This is frozen as NMAE,
    # with NRMSE fully reported, not selected after observing which test is favorable.
    p1 = next(r for r in tests if r['family'] == 'P1' and r['metric'] == 'NMAE')
    p2 = next(r for r in tests if r['family'] == 'P2' and r['metric'] == 'NMAE')
    A = p2['control_better_seed_count'] >= 4 and p2['q_value'] < .05
    B = p1['control_better_seed_count'] >= 4 and p1['q_value'] < .05
    C = not A and not B and any(r['control_better_seed_count'] >= 4 and r['q_value'] >= .05 for r in (p1, p2))
    decision = dict(status='COMPLETE_VALIDATED', n_trainings=5, n_seeds=5, n_variant_scenarios=200,
                    classification='A' if A else ('B' if B else ('C' if C else 'D')),
                    A_robustness_supported=A, B_normal_NMAE_supported=B,
                    C_consistent_nonsignificant=C, D_no_predeclared_advantage=not (A or B or C),
                    decision_metric_for_A_and_B='NMAE', primary_tests=tests,
                    P3_secondary_no_inferential_test=True, no_new_control_runs=True,
                    legacy_control_realization_hashes_present=False,
                    legacy_same_fault_basis='Exact frozen kernel, sample origin/turbine order/coordinates, preprocessing, seed2026 and all39 scenario specifications; old files do not contain input realization hashes.',
                    inference_scope='Within-example turbine information exchange; parameters remain shared across farm training examples.',
                    accepted_receipt_sha256=sha(acceptance_path), launch_manifest_sha256=sha(deploy_path),
                    control_sources_sha256=CONTROL_SHA, generated_utc=datetime.now(timezone.utc).isoformat())
    decision['pretraining_analysis_addendum_sha256'] = ADDENDUM_SHA
    return dict(decision=decision, groups=groups, tests=tests, per_seed=per_seed, all_rows=all_rows, evidence=evidence)


def publish(root, backup, result):
    # New output directory only. Never alter/overwrite completed or failed scientific artifacts.
    out = root / 'summary'
    out.mkdir(exist_ok=False)
    write_csv(out / 'all_scenarios_400_rows.csv', result['all_rows'])
    write_csv(out / 'endpoint_per_seed.csv', result['per_seed'])
    write_csv(out / 'endpoint_mean_sample_sd.csv', result['groups'])
    write_csv(out / 'paired_tests_BH_two_families.csv', result['tests'])
    (out / 'decision.json').write_text(json.dumps(result['decision'], ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    def mean_sd(family, metric, model):
        r = next(x for x in result['groups'] if (x['family'], x['metric'], x['model']) == (family, metric, model))
        return f'{r["mean"]:.6f} ± {r["sample_sd"]:.6f}'
    for lang, name in [('zh', 'E4_结果报告_中文.md'), ('en', 'E4_results_report_English.md')]:
        zh = lang == 'zh'
        lines = ['# ' + ('E4：跨机组空间聚合消融结果' if zh else 'E4: ablation of inter-turbine spatial aggregation'), '',
                 ('五个种子13/21/42/87/3407全部完成并通过来源、自然退出、完整归档及D/E核验。冻结完整模型直接复用，变体重新训练5次。' if zh else
                  'All five seeds (13/21/42/87/3407) passed source, native-exit, archive and D/E checks. Existing full-model controls were reused; the ablation was trained five times.'), '',
                 ('|终点|指标|完整模型VF（均值±样本SD，%）|Without spatial aggregation（均值±样本SD，%）|' if zh else '|Endpoint|Metric|Full VF (mean ± sample SD, %)|Without spatial aggregation (mean ± sample SD, %)|'),
                 '|---|---|---:|---:|']
        for fam, met in [('P1','NRMSE'),('P1','NMAE'),('P2','NRMSE'),('P2','NMAE'),('P3','NMAE')]:
            lines.append(f'|{fam}|{met}|{mean_sd(fam,met,"VF")}|{mean_sd(fam,met,"Without spatial aggregation")}|')
        lines += ['', ('P1采用保存的正常测试指标。P2在每个seed内先等权平均39故障场景，P3先等权平均8个24h缺失场景，再以五个seed为统计单位。各组SD为样本SD（ddof=1）。P3不作新增显著性检验。' if zh else
                        'P1 uses saved normal-test metrics. P2 averages the 39 fault scenarios within each seed; P3 averages the eight 24-hour missingness scenarios within each seed. Five seeds are the statistical units; SD uses ddof=1. No additional significance test is performed for P3.'), '',
                  ('|族|指标|变体−完整模型（百分点）|均值相对变化（%）|完整模型更好seed数|双侧p|族内BH q|' if zh else '|Family|Metric|Ablation − full (pp)|Relative change of means (%)|Seeds favoring full|Two-sided p|Within-family BH q|'),
                  '|---|---|---:|---:|---:|---:|---:|']
        for t in result['tests']:
            lines.append(f'|{t["family"]}|{t["metric"]}|{t["difference_variant_minus_control_pp"]:.6f}|{t["relative_change_of_means_pct"]:.6f}|{t["control_better_seed_count"]}/5|{t["p_value"]:.8g}|{t["q_value"]:.8g}|')
        d = result['decision']
        lines += ['', ('两族P1/P2各自只有NRMSE与NMAE两项配对t检验，分别进行BH。正的相对变化表示消融误差上升；所有退步和负变化保留。' if zh else
                        'P1 and P2 each contain exactly two paired t-tests (NRMSE and NMAE), with BH adjustment applied separately. Positive relative change means greater error after ablation. All unfavorable and negative changes are retained.'), '',
                  ('预定判定：' if zh else 'Predeclared classification: ') + d['classification'] + f'; A={d["A_robustness_supported"]}; B={d["B_normal_NMAE_supported"]}.', '',
                  ('A/B分别按P2/P1的NMAE判断：完整模型至少4/5seed更低且q<0.05；NRMSE全部独立报告，不按有利指标事后替换。判定不推广到未测试风场、时长或故障。' if zh else
                   'A/B use NMAE in P2/P1, respectively: at least four of five seeds must favor the full model and q must be below 0.05. NRMSE is reported in full and is not substituted post hoc. Conclusions are limited to the tested farm, horizon and fault conditions.'), '',
                  ('self_only只删除单个样本内机组间的信息交换，模型参数仍由全场数据共享学习；这不是分别训练独立机组模型。完整模型39故障来源为旧F1_verified，不是仅含14基线的E1 v2。旧控制没有新式realization哈希，本文不虚构其存在；同故障依据详见decision.json和控制源审计。' if zh else
                   'The self_only policy removes inter-turbine exchange within an example; model parameters remain shared during farm-wide training. This is not a collection of separately trained turbine models. Full-model fault controls come from the existing F1_verified artifacts, not E1 v2, which contains only fourteen baselines. Legacy controls lack the newer input-realization hashes; none are fabricated. The exact common-kernel evidence is documented in decision.json and the source audit.'), '',
                  f'[400 rows]({(out/"all_scenarios_400_rows.csv").as_posix()}) · [Per seed]({(out/"endpoint_per_seed.csv").as_posix()}) · [Tests]({(out/"paired_tests_BH_two_families.csv").as_posix()})',
                  f'[Control audit]({(root/"ops/control_source_audit_20261028.md").as_posix()}) · [Training artifacts]({(root/"project/artifacts").as_posix()}) · [Fault predictions]({(root/"faults").as_posix()})',
                  f'[D report directory]({out.as_posix()}) · [E report directory]({(backup/"summary").as_posix()})']
        (out / name).write_text('\n'.join(lines) + '\n', 'utf-8')
    proof = dict(status='PASS', normal_control_and_variant_seed_count=10, scenario_rows=400,
                 per_seed_endpoint_rows=50, group_rows=10, primary_test_rows=4,
                 script_sha256=sha(__file__), evidence=result['evidence'],
                 outputs=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(out.iterdir())])
    (out / 'summary_validation.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    dest = backup / 'summary'
    dest.mkdir(exist_ok=False)
    import shutil
    for p in out.iterdir():
        shutil.copy2(p, dest / p.name)
        require(sha(p) == sha(dest / p.name), 'Report D/E copy differs')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--acceptance-receipt', type=Path)
    ap.add_argument('--backup-root', type=Path, default=Path('results/paper'))
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    receipt = a.acceptance_receipt or a.root / 'ops/final_acceptance.json'
    result = analyze(a.root, receipt, a.backup_root)
    if a.validate_only:
        print(json.dumps(dict(status='PASS_VALIDATION_ONLY', primary_tests=result['tests']), ensure_ascii=False))
    else:
        out = publish(a.root, a.backup_root, result)
        print(json.dumps(dict(status='PASS', report_directory=str(out), decision=result['decision']['classification']), ensure_ascii=False))


if __name__ == '__main__':
    main()
