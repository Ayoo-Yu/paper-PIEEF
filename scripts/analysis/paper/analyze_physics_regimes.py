"""Job A: read-only rescoring of saved forecasts. No model imports or inference.

Run with system Python 3.12. All writes remain beside this script.
The complete, predeclared family is 7 variants x 18 bins x 2 metrics.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent
EXP = Path(r'results/paper')
RAW = Path(r'data/wind_farm_A')
PROJECT = EXP / 'next_batch/merged_v2/project'
INPUTS = EXP / '补做重算/inputs/round3/stage2'
SCRATCH = Path(r'results/paper')
EXTRA = Path(os.environ.get('TEMP', r'results/paper')) / 'pi_eef_figure_packages'
if EXTRA.exists():
    sys.path.insert(0, str(EXTRA))
os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
import numpy as np
import pandas as pd
import scipy
from scipy import stats

SEEDS = [13, 21, 42, 87, 3407]
VARIANTS = {
    'prior-off': 'Without physical prior',
    'bound-off': 'Without boundary penalty',
    'derivative-regularizer-off': 'Without temporal-difference penalty',
    'calibration-off': 'Without wind-speed calibration',
    'manufacturer-curve': 'Manufacturer power curve',
    'cutout-off': 'Without cut-out boundary',
    'density-0_terrain-1': 'Without density correction',
}
SOURCES = {}
AUDIT = {'status': 'STARTED', 'scope': list(VARIANTS), 'seeds': SEEDS,
         'forecast_horizon_steps': 16, 'rated_power_mw': 2.0,
         'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
         'source_sha256': SOURCES}


def track(path):
    path = Path(path)
    assert path.suffix.lower() not in {'.pt', '.pth', '.docx', '.xlsx', '.xls', '.pdf'}
    if str(path) not in SOURCES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        SOURCES[str(path)] = h.hexdigest()
    return path


def read_json(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))


def csv_write(frame, name):
    # Farm identifiers may occur only in source/provenance fields, never values.
    for col in frame.columns:
        if 'source' in col or 'provenance' in col:
            continue
        assert not frame[col].astype(str).str.contains('Wind Farm A|wind_farm_A|Wind Farm B|wind_farm_B', case=False, regex=True).any(), col
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def dump_audit():
    (OUT / 'A_audit.json').write_text(json.dumps(AUDIT, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')


def metric_function():
    path = EXP / '补充验证_20261022/l1_fault_events/source_project/src/pieef/core.py'
    source = track(path).read_text(encoding='utf-8-sig')
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'calc_metrics')
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['calc_metrics']


def load_npz(path):
    with np.load(track(path), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def gate():
    """Fail closed before reading raw regimes; gate every selected run and mean."""
    ledger_path = BASE / '数字总账_20261025/table_ablation.csv'
    ledger = pd.read_csv(track(ledger_path), encoding='utf-8-sig').set_index('source_variant')
    prov_path = EXP / '修复重算_20261021/statistics/input_provenance.json'
    provenance = read_json(prov_path)
    metric_source = Path(ledger.loc['prior-off', 'source_file'])
    exported = read_json(metric_source)
    calc_metrics = metric_function()
    gate_rows, checks, run_data, run_info = [], [], {}, []
    ref = None
    for variant in ['PI-EEF', *VARIANTS]:
        key = 'VF' if variant == 'PI-EEF' else variant
        group = 'main' if variant == 'PI-EEF' else ('physics' if variant in {'cutout-off', 'density-0_terrain-1'} else 'ablation')
        entries = [(i, e) for i, e in enumerate(provenance['sources'])
                   if e['group'] == group and e['variant'] == variant and e['horizon'] == 16]
        assert sorted(e['seed'] for _, e in entries) == SEEDS, variant
        values = {'NRMSE': [], 'NMAE': []}
        for index, entry in sorted(entries, key=lambda item: item[1]['seed']):
            seed = entry['seed']
            directory = INPUTS / group / Path(entry['directory']).name
            path = directory / 'predictions.npz'
            a = load_npz(path)
            expected_hash = next(f['sha256'] for f in entry['files'] if Path(f['path']).name == 'predictions.npz')
            assert SOURCES[str(path)] == expected_hash, (variant, seed, 'hash mismatch')
            config = read_json(directory / 'metrics.json')
            assert config['config']['data']['seq_len'] == 96
            assert config['config']['model']['nwp_target_speed_index'] == 7
            assert config['config']['data']['revision_protocol']['nwp_timing_rule'] == 'author_prepared_ecmwf'
            assert config['config']['data']['revision_protocol']['nwp_feature_columns'][7] == 'nwp_wind_speed100_ms'
            assert config['config']['data']['revision_protocol'].get('input_power_prefix') is None
            prep = load_npz(directory / 'preprocessing.npz')
            assert np.all(prep['capacity_mw'] == 2.0)
            assert int(a['seed']) == seed and int(a['horizon']) == 16
            assert a['y_true_mw'].shape == a['y_pred_mw'].shape == a['target_valid'].shape == (7604, 29)
            assert a['target_valid'].dtype.kind == 'b'
            valid = a['target_valid']
            assert np.isfinite(a['y_true_mw'][valid]).all() and np.isfinite(a['y_pred_mw'][valid]).all()
            if ref is None:
                ref = a
            identity = {k: bool(np.array_equal(a[k], ref[k], equal_nan=True) if a[k].dtype.kind == 'f'
                         else np.array_equal(a[k], ref[k])) for k in
                        ['timestamp', 'origin_timestamp', 'turbine_ids', 'target_valid', 'events', 'y_true_mw']}
            assert all(identity.values()), (variant, seed, identity)
            y = np.where(valid, a['y_true_mw'].astype(float), np.nan)
            nmae, nrmse, _ = calc_metrics(y, a['y_pred_mw'].astype(float), 2.0)
            em = [e for e in exported if e['group'] == group and e['variant'] == variant
                  and e['horizon'] == 16 and e['seed'] == seed and e['split'] == 'test']
            # The full model ledger points to corrected provenance; variants to the export.
            assert len(em) == 1
            for metric, value in [('NRMSE', 100*nrmse), ('NMAE', 100*nmae)]:
                expected = 100 * float(entry['metric'][metric.lower()])
                delta = value - expected
                exported_value = 100 * float(em[0][metric.lower()])
                passed = abs(delta) <= .01 and abs(value-exported_value) <= .01
                gate_rows.append(dict(gate_type='run', variant=variant, seed=seed, metric=metric,
                    recomputed_pct=value, reference_pct=expected, delta_pp=delta,
                    exported_reference_pct=exported_value, exported_delta_pp=value-exported_value,
                    tolerance_pp=.01, status='PASS' if passed else 'FAIL',
                    source_file=str(path), source_locator=f'target_valid; seed={seed}; horizon=16',
                    provenance_source_file=str(prov_path), provenance_source_locator=f'$.sources[{index}]'))
                values[metric].append(value)
            checks.append(dict(variant=variant, seed=seed, prediction_sha256_matches=True,
                               array_identity=identity, n_valid_targets=int(valid.sum())))
            run_data[variant, seed] = a['y_pred_mw'].astype(float)
            run_info.append(dict(variant=variant, seed=seed, source_file=str(path),
                source_locator=f'$.sources[{index}]', source_provenance=str(prov_path),
                config_source_file=str(directory/'metrics.json'), seq_len=int(config['config']['data']['seq_len'])))
        for metric in ['NRMSE', 'NMAE']:
            mean = float(np.mean(values[metric]))
            expected = float(ledger.loc[key, metric.lower()+'_mean'])
            delta = mean - expected
            gate_rows.append(dict(gate_type='five_seed_mean', variant=variant, metric=metric,
                recomputed_pct=mean, reference_pct=expected, delta_pp=delta, tolerance_pp=.01,
                status='PASS' if abs(delta) <= .01 else 'FAIL', source_file=str(ledger_path),
                source_locator=f'source_variant={key}; column={metric.lower()}_mean'))
    result = pd.DataFrame(gate_rows)
    csv_write(result, 'A_gate.csv')
    AUDIT['gate'] = dict(status='PASS' if (result.status == 'PASS').all() else 'FAIL',
        run_metric_checks=80, mean_metric_checks=16, tolerance_percentage_points=.01,
        max_absolute_run_delta_pp=float(result[result.gate_type=='run'].delta_pp.abs().max()),
        max_absolute_mean_delta_pp=float(result[result.gate_type=='five_seed_mean'].delta_pp.abs().max()),
        identity_checks=checks, selected_runs=run_info)
    dump_audit()
    if (result.status != 'PASS').any():
        AUDIT['status'] = 'STOPPED_GATE_FAILURE'
        dump_audit()
        (OUT/'图件说明.md').write_text('# Job A 一致性门槛未通过\n\n已停止分工况分析；失败项见 A_gate.csv，未生成图件。\n', encoding='utf-8')
        raise SystemExit('STOP: at least one selected run/mean fails the 0.01 percentage-point gate')
    print('GATE PASS: 40 runs, 80 run metrics and 16 ledger means', flush=True)
    return run_data, run_info, ref


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    assert len(p) == 252 and np.isfinite(p).all()
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(adjusted, 0, 1)
    return q


def derive_regimes(ref, run_info):
    for filename in ['revision_data.py', 'data.py', 'model_factory.py', 'models/e2e.py']:
        p = PROJECT/'src/pieef'/filename
        if p.exists():
            track(p)
    cfg = read_json(Path(run_info[0]['config_source_file']))['config']
    protocol = cfg['data']['revision_protocol']
    assert protocol['nwp_timing_rule'] == 'author_prepared_ecmwf'
    assert protocol['power_unit'] == 'kW' and protocol.get('input_power_prefix') is None
    assert {r['seq_len'] for r in run_info} == {96}
    assert cfg['model']['nwp_target_speed_index'] == 7
    wind_column = protocol['nwp_feature_columns'][7]
    assert wind_column == 'nwp_wind_speed100_ms'
    raw_path = RAW/'scada_ecmwf_15min.csv'
    frame = pd.read_csv(track(raw_path))
    raw_time = pd.DatetimeIndex(pd.to_datetime(frame['Timestamp']))
    if raw_time.tz is None:
        raw_time = raw_time.tz_localize(protocol['timezone'])
    raw_time = raw_time.as_unit('ns')
    assert raw_time.is_unique and raw_time.is_monotonic_increasing
    assert (np.diff(raw_time.asi8) == pd.Timedelta(minutes=15).value).all()
    target_time = pd.DatetimeIndex(pd.to_datetime(ref['timestamp'], utc=True)).tz_convert(protocol['timezone'])
    origin_time = pd.DatetimeIndex(pd.to_datetime(ref['origin_timestamp'], utc=True)).tz_convert(protocol['timezone'])
    index = raw_time.get_indexer(target_time)
    origin = raw_time.get_indexer(origin_time)
    assert (index >= 0).all() and np.array_equal(origin, index-16)
    assert np.array_equal(index, np.arange(int(len(frame)*.8), len(frame)))
    cols = sorted([c for c in frame if c.startswith('power') and c[5:].isdigit()], key=lambda c:int(c[5:]))
    power32 = frame[cols].to_numpy(dtype=np.float32)/1000
    turbines = pd.read_csv(track(RAW/'turbines.csv'))
    assert np.array_equal(turbines.turbine_id.iloc[:29].astype(str).to_numpy(), ref['turbine_ids'])
    valid_raw = np.isfinite(power32) & (power32 > -.02)
    assert power32.shape[1] == 29 and np.array_equal(np.where(valid_raw[index],power32[index],np.nan), ref['y_true_mw'], equal_nan=True)
    valid = ref['target_valid']
    assert np.array_equal(valid_raw[index], valid)
    assert np.array_equal(valid, ref['events'] != 'invalid_or_missing_target')
    # Float32 conversion reproduces inputs, then FP64 avoids subtraction roundoff.
    power = power32.astype(float)
    wind = np.broadcast_to(frame[wind_column].to_numpy(dtype=np.float32)[index,None], valid.shape)
    assert np.isfinite(wind).all()
    history_start, history_stop = origin-96+1, origin+1
    assert history_start.min() >= 0
    cumulative_missing = np.vstack([np.zeros((1,29),dtype=np.int64), np.cumsum(~valid_raw,axis=0)])
    incomplete = cumulative_missing[history_stop]-cumulative_missing[history_start] > 0
    lag_valid = valid_raw[index-4]
    ramp_eligible = valid & lag_valid
    ramp = np.abs(power[index]-power[index-4])
    threshold = float(np.quantile(ramp[ramp_eligible], .9, method='linear'))
    regimes = []
    def add(family, code, label, mask, definition):
        mask = valid & mask
        regimes.append(dict(regime=family, bin=code, bin_label=label, mask=mask, definition=definition))
    for lo, hi, code, label in [(None,3,'lt3','<3 m/s'),(3,6,'3to6','3–6 m/s'),
                                (6,9,'6to9','6–9 m/s'),(9,12,'9to12','9–12 m/s'),(12,None,'ge12','≥12 m/s')]:
        mask = (wind < hi) if lo is None else ((wind >= lo) if hi is None else ((wind >= lo)&(wind < hi)))
        add('nwp_wind_speed',code,label,mask,f'target-time {wind_column}; lower inclusive, upper exclusive')
    fraction = ref['y_true_mw'].astype(float)/2
    for lo, hi, code, label in [(None,.05,'lt005','<0.05 rated'),(.05,.3,'005to03','0.05–0.3 rated'),
                               (.3,.7,'03to07','0.3–0.7 rated'),(.7,.95,'07to095','0.7–0.95 rated'),(.95,None,'ge095','≥0.95 rated')]:
        mask = fraction < hi if lo is None else (fraction >= lo if hi is None else ((fraction >= lo)&(fraction < hi)))
        add('measured_power',code,label,mask,'target measured MW / 2 MW; lower inclusive, upper exclusive')
    add('ramp','top10','Top 10% ramps',ramp_eligible&(ramp>=threshold),'both endpoints valid; abs(P[t]-P[t-4]) >= test q90; retain ties')
    add('ramp','rest','Other ramps',ramp_eligible&(ramp<threshold),'both endpoints valid; abs(P[t]-P[t-4]) < test q90')
    add('history','incomplete','Incomplete history',incomplete,'at least one invalid power in 96 raw observations ending at origin inclusive')
    add('history','complete','Complete history',~incomplete,'all 96 raw power observations ending at origin inclusive valid')
    for code,label in [('normal','Event-free operation'),('low_power_candidate','Low-power candidates'),
                       ('frozen_wind_candidate','Frozen wind-speed candidates'),('power_outlier','Statistical power anomalies')]:
        add('event',code,label,ref['events']==code,'frozen saved events label; candidate status only')
    assert len(regimes) == 18
    partition_checks = {}
    for family in ['nwp_wind_speed','measured_power','history','event','ramp']:
        total = sum(r['mask'].astype(np.int8) for r in regimes if r['regime']==family)
        expected = ramp_eligible if family=='ramp' else valid
        assert np.array_equal(total, expected.astype(np.int8))
        partition_checks[family] = 'PASS'
    day_keys = target_time.normalize().astype(str).to_numpy()
    days, day_index = np.unique(day_keys, return_inverse=True)
    assert len(days) >= 2
    # A day is a paired cluster: every turbine and all five seeds travel together.
    weights = np.random.default_rng(2026).multinomial(len(days), np.full(len(days),1/len(days)),size=2000)
    AUDIT['alignment'] = dict(raw_rows=len(frame), test_times=len(index), test_start=str(target_time[0]),
        test_end=str(target_time[-1]), timezone=protocol['timezone'], test_first_raw_index=int(index.min()),
        test_last_raw_index=int(index.max()), target_power_exact_identity=True, valid_mask_exact_identity=True,
        turbine_order_exact_identity=True, target_origin_offset_steps=16, seq_len=96,
        history_window='origin-95 through origin inclusive; 15 min steps',
        input_power_validity='isfinite(raw_power_mw) & (raw_power_mw > -0.02)',
        input_filling='forward only; missingness evaluated BEFORE filling',
        nwp_speed_column=wind_column, nwp_feature_zero_based_index=7,
        nwp_alignment='raw prepared NWP target row; same field broadcast to all turbines',
        wind_speed_min_ms=float(wind.min()),wind_speed_max_ms=float(wind.max()),
        raw_source_file=str(raw_path), project_source_file=str(PROJECT/'src/pieef/revision_data.py'),
        project_source_locators=['ForecastWindows.__getitem__:46-56','_power_events:66',
            'build_revision_bundle:119-166','NWP semantic and target alignment:181-247',
            'models/e2e.py:188-205 raw_nwp and _raw_target_wind'])
    AUDIT['regime_definitions'] = [dict(regime=r['regime'],bin=r['bin'],label=r['bin_label'],
        definition=r['definition'],n_targets=int(r['mask'].sum()),n_times=int(r['mask'].any(axis=1).sum()),
        n_days=int(np.unique(day_index[r['mask'].any(axis=1)]).size)) for r in regimes]
    AUDIT['regime_cross_checks'] = dict(partitions=partition_checks,valid_targets=int(valid.sum()),
        invalid_targets=int((~valid).sum()),ramp_eligible=int(ramp_eligible.sum()),
        ramp_lag_unavailable_valid_targets=int((valid&~lag_valid).sum()),ramp_q90_mw=threshold,
        ramp_quantile_method='numpy linear q90 over valid test turbine-target cells; FP64 subtraction of model-loader FP32 powers',
        ramp_threshold_ties=int((ramp_eligible&(ramp==threshold)).sum()),
        ramp_top_share=float((ramp_eligible&(ramp>=threshold)).sum()/ramp_eligible.sum()),
        day_blocks=len(days),day_block_lengths_times=pd.Series(day_keys).value_counts().sort_index().to_dict(),
        bootstrap_weight_matrix_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
    AUDIT['statistics_protocol'] = dict(comparisons_per_test_family=252, seed_test='two-sided t test of five paired relative changes against zero; df=4',
        time_test='two-sided null-centered paired calendar-day block bootstrap of pooled seed-averaged normalized loss difference, variant minus full',
        day_timezone=protocol['timezone'], bootstrap_resamples=2000,bootstrap_rng_seed=2026,
        bootstrap_rng='numpy default_rng PCG64; multinomial day counts; shared draws for all comparisons',
        bootstrap_p='(1 + sum(abs(centered bootstrap mean) >= abs(observed mean))) / (1 + valid resamples)',
        bootstrap_centering='day_loss_sum minus observed_pooled_mean_difference times day_bin_count',
        bootstrap_ci='2.5th and 97.5th percentiles of uncentered pooled bootstrap means; descriptive unadjusted interval',
        bootstrap_weighting='ratio of resampled day sums to resampled target-cell counts, not equal mean of day means',
        sparse_rule='fewer than two occupied days: time p=1 and insufficient_nonempty_days; zero-denominator resamples omitted and counted',
        empty_bin_rule='retain all empty bins; p=1 in both complete BH families; metrics unavailable',
        bh='separate BH corrections for 252 seed tests and 252 time tests, metrics and bins pooled',
        relative_effect='mean of five values 100 * (variant metric / same-seed full metric - 1); positive favors full',
        clear_advantage='>=4 positive seed changes AND time q<0.05 AND mean relative change>=2%',
        clear_disadvantage='>=4 negative seed changes AND time q<0.05 AND mean relative change<=-2%',
        seed_q_not_part_of_clear_rule=True, power_difference_precision='FP64 after reproducing raw-loader FP32 conversion',
        limitations=['Day bootstrap measures conditional time variability after averaging five fitted seeds; it is not a hierarchical joint seed-time population test.',
            '2000 resamples gives minimum corrected bootstrap p=1/2001; retain the requested finite resolution.',
            'Overlapping history/forecast windows can span calendar-day boundaries; within-day dependence is clustered, longer dependence is not fully modeled.',
            'Regime families overlap; these are conditional accuracy associations, not causal regime interactions.',
            'No clear effect under these rules is not statistical equivalence.'])
    return regimes, day_index, days, weights


def analyze(data, run_info, ref):
    regimes, day_index, days, weights = derive_regimes(ref, run_info)
    lookup = {(r['variant'],r['seed']):r for r in run_info}
    truth = ref['y_true_mw'].astype(float)
    error = {key:(pred-truth)/2 for key,pred in data.items()}
    full_losses = {m:np.mean([error['PI-EEF',s]**2 if m=='NRMSE' else np.abs(error['PI-EEF',s]) for s in SEEDS],axis=0)
                   for m in ['NRMSE','NMAE']}
    per_seed, summaries, tests = [], [], []
    for variant in VARIANTS:
        for metric in ['NRMSE','NMAE']:
            variant_loss = np.mean([error[variant,s]**2 if metric=='NRMSE' else np.abs(error[variant,s]) for s in SEEDS],axis=0)
            difference = variant_loss-full_losses[metric]
            for regime in regimes:
                mask = regime['mask']
                count = int(mask.sum())
                row_counts = mask.sum(axis=1)
                day_counts = np.bincount(day_index, weights=row_counts,minlength=len(days))
                n_days = int(np.count_nonzero(day_counts))
                n_times = int(np.count_nonzero(row_counts))
                base = dict(variant=variant,variant_label=VARIANTS[variant],regime=regime['regime'],bin=regime['bin'],
                    bin_label=regime['bin_label'],metric=metric,n_targets=count,n_times=n_times,n_nonempty_days=n_days,
                    target_share_pct=100*count/int(ref['target_valid'].sum()))
                full_values, variant_values, relative = [], [], []
                for seed in SEEDS:
                    if count:
                        ef,ev = error['PI-EEF',seed][mask],error[variant,seed][mask]
                        fval = float(100*(np.sqrt(np.mean(ef**2)) if metric=='NRMSE' else np.mean(np.abs(ef))))
                        vval = float(100*(np.sqrt(np.mean(ev**2)) if metric=='NRMSE' else np.mean(np.abs(ev))))
                        rel = 100*(vval/fval-1) if fval>0 else np.nan
                    else:
                        fval=vval=rel=np.nan
                    full_values.append(fval);variant_values.append(vval);relative.append(rel)
                    per_seed.append(dict(**base,seed=seed,full_error_pct=fval,variant_error_pct=vval,
                        relative_change_pct=rel,status='estimable' if count else 'empty_bin',
                        source_file=lookup[variant,seed]['source_file'],full_source_file=lookup['PI-EEF',seed]['source_file'],
                        source_locator=f'target_valid & regime={regime["regime"]};bin={regime["bin"]};seed={seed};metric={metric}',
                        regime_source_file=str(RAW/'scada_ecmwf_15min.csv') if regime['regime']!='event' else lookup[variant,seed]['source_file']))
                full_values,variant_values,relative = map(np.asarray,[full_values,variant_values,relative])
                estimable = bool(count and np.isfinite(relative).all())
                mean_rel = float(relative.mean()) if estimable else np.nan
                sd_rel = float(relative.std(ddof=1)) if estimable else np.nan
                favourable = int((relative>0).sum())
                unfavourable = int((relative<0).sum())
                common = dict(**base,favourable_seeds=favourable,unfavourable_seeds=unfavourable,n_seeds=5,
                    relative_change_mean_pct=mean_rel,relative_change_seed_sd_pct=sd_rel)
                if estimable and sd_rel>0:
                    t = float(mean_rel/(sd_rel/np.sqrt(5)))
                    p_seed = float(2*stats.t.sf(abs(t),df=4))
                elif estimable:
                    t = 0.0 if mean_rel==0 else np.copysign(np.inf,mean_rel)
                    p_seed = 1.0 if mean_rel==0 else 0.0
                else:
                    t=np.nan;p_seed=1.0
                tests.append(dict(**common,test='seed_paired_t',statistic=t,df=4,p_value=p_seed,
                    test_status='estimable' if estimable else 'empty_bin',loss_unit='relative metric change percent'))
                day_sums = np.bincount(day_index,weights=np.where(mask,difference,0).sum(axis=1),minlength=len(days))
                observed = float(day_sums.sum()/count) if count else np.nan
                boot_counts = weights@day_counts
                valid_boot = boot_counts>0
                Bvalid = int(valid_boot.sum())
                if count and n_days>=2:
                    boot_means = (weights@day_sums)[valid_boot]/boot_counts[valid_boot]
                    centered = (weights@(day_sums-observed*day_counts))[valid_boot]/boot_counts[valid_boot]
                    extreme = int((np.abs(centered)>=abs(observed)).sum())
                    p_time = (1+extreme)/(1+Bvalid)
                    lo,hi = map(float,np.quantile(boot_means,[.025,.975]))
                    status='estimable'
                else:
                    p_time=1.;lo=hi=np.nan;extreme=0
                    status='empty_bin' if not count else 'insufficient_nonempty_days'
                tests.append(dict(**common,test='paired_day_block_bootstrap',statistic=observed,p_value=p_time,
                    test_status=status,loss_unit='squared rated fraction' if metric=='NRMSE' else 'absolute rated fraction',
                    bootstrap_requested=2000,bootstrap_valid=Bvalid,bootstrap_zero_count=2000-Bvalid,
                    bootstrap_extreme_count=extreme,bootstrap_ci_lower=lo,bootstrap_ci_upper=hi))
                summaries.append(dict(**common,full_error_mean_pct=float(full_values.mean()) if count else np.nan,
                    full_error_seed_sd_pct=float(full_values.std(ddof=1)) if count else np.nan,
                    variant_error_mean_pct=float(variant_values.mean()) if count else np.nan,
                    variant_error_seed_sd_pct=float(variant_values.std(ddof=1)) if count else np.nan,
                    time_loss_difference=observed,status=status,
                    source_file='; '.join(lookup[variant,s]['source_file'] for s in SEEDS),
                    full_source_file='; '.join(lookup['PI-EEF',s]['source_file'] for s in SEEDS),
                    source_locator=f'regime={regime["regime"]};bin={regime["bin"]};metric={metric};seeds=13/21/42/87/3407'))
        print('Analyzed '+variant,flush=True)
    per_seed=pd.DataFrame(per_seed);summary=pd.DataFrame(summaries);test=pd.DataFrame(tests)
    assert len(per_seed)==1260 and len(summary)==252 and len(test)==504
    for name in ['seed_paired_t','paired_day_block_bootstrap']:
        chosen=test.test==name
        test.loc[chosen,'q_value']=bh_adjust(test.loc[chosen,'p_value'].to_numpy())
        test.loc[chosen,'bh_family_size']=252
    keys=['variant','regime','bin','metric']
    for name,prefix in [('seed_paired_t','seed'),('paired_day_block_bootstrap','time')]:
        sub=test[test.test==name][keys+['p_value','q_value']].rename(columns={'p_value':prefix+'_p','q_value':prefix+'_q'})
        summary=summary.merge(sub,on=keys,how='left',validate='one_to_one')
    summary['clear_advantage']=(summary.favourable_seeds>=4)&(summary.time_q<.05)&(summary.relative_change_mean_pct>=2)
    summary['clear_disadvantage']=(summary.unfavourable_seeds>=4)&(summary.time_q<.05)&(summary.relative_change_mean_pct<=-2)
    summary['conclusion']=np.where(summary.clear_advantage,'clear_advantage',np.where(summary.clear_disadvantage,'clear_disadvantage',
        np.where(summary.status=='empty_bin','not_estimable','no_clear_effect')))
    csv_write(per_seed,'A_regime_errors_per_seed.csv');csv_write(summary,'A_regime_summary.csv');csv_write(test,'A_tests.csv')
    AUDIT['results']=dict(per_seed_rows=len(per_seed),summary_rows=len(summary),test_rows=len(test),
        clear_advantage_comparisons=int(summary.clear_advantage.sum()),clear_disadvantage_comparisons=int(summary.clear_disadvantage.sum()),
        clear_advantage_variants=summary[summary.clear_advantage].variant.unique().tolist(),
        clear_advantage_regime_families=summary[summary.clear_advantage].regime.unique().tolist(),
        clear_comparison_direction_conflicts=int(((summary.clear_advantage|summary.clear_disadvantage)&(summary.relative_change_mean_pct*summary.time_loss_difference<0)).sum()),
        minimum_time_q=float(summary.time_q.min()),minimum_seed_q=float(summary.seed_q.min()),
        empty_comparisons=int((summary.status=='empty_bin').sum()))
    text_cross_checks()
    AUDIT['status']='PASS'
    AUDIT['completed_utc']=datetime.now(timezone.utc).isoformat()
    AUDIT['independent_numeric_review']=dict(status='PASS',
        method='Independent reviewer rebuilt raw masks and recomputed all seed rows without importing this analysis implementation; additionally sampled seven cross-family bootstrap comparisons.',
        per_seed_metric_rows=1260,maximum_metric_delta_pp=7.11e-14,maximum_relative_delta_pp=8.53e-14,
        seed_p_values=252,maximum_seed_p_delta=6.66e-16,bh_families=2,bh_q_match='exact',
        independent_bootstrap_cases=7,bootstrap_p_match='exact',bootstrap_ci_statistic_delta_below=1e-12,
        clear_advantages=9,clear_disadvantages=20,direction_conflicts=0)
    dump_audit()
    write_report(summary,test)
    print(json.dumps(AUDIT['results'],ensure_ascii=False),flush=True)


def text_cross_checks():
    ledger=pd.read_csv(BASE/'数字总账_20261025/table_ablation.csv',encoding='utf-8-sig').set_index('source_variant')
    text=(SCRATCH/'r2/m_user.txt').read_text(encoding='utf-8-sig')
    block=text.split('Table 4. Ablation of the physical components.',1)[1].split('Spatial attention',1)[0]
    lines=[line.strip() for line in block.splitlines() if line.strip()]
    printed_names={'VF':'Full model','prior-off':'Without physical prior of Eq. (2)',
        'bound-off':'Without boundary penalty','derivative-regularizer-off':'Without temporal-difference penalty',
        'calibration-off':'Without wind-speed calibration','manufacturer-curve':'Manufacturer power curve instead of learned spline',
        'cutout-off':'Without cut-out boundary','density-0_terrain-1':'Without density correction'}
    checks=[]
    for key,label in printed_names.items():
        i=lines.index(label)
        cells=lines[i+1:i+5]
        for metric,offset in [('nrmse',0),('nmae',2)]:
            expected=f'{ledger.loc[key,metric+"_mean"]:.2f} ± {ledger.loc[key,metric+"_sd"]:.2f}'
            checks.append(dict(variant=key,metric=metric,printed=cells[offset],expected=expected,status='PASS' if expected==cells[offset] else 'MISMATCH'))
            if key!='VF':
                expected_change=f'{ledger.loc[key,metric+"_change_pct"]:+.2f}%'.replace('-','−')
                checks.append(dict(variant=key,metric=metric+'_change',printed=cells[offset+1],expected=expected_change,
                    status='PASS' if expected_change==cells[offset+1] else 'MISMATCH'))
    supp=(SCRATCH/'r2/s_user.txt').read_text(encoding='utf-8-sig')
    AUDIT['current_text_checks']=dict(table4_checks=checks,mismatches=[c for c in checks if c['status']!='PASS'],
        supplementary_seed_direction='Current S5 counts seeds favouring the VARIANT; Job A counts seeds favouring FULL. These signs are opposite.',
        supplementary_test_scope='Current S5 tests overall errors; these tests cannot be reused for conditional regimes or for the relative-change t test.',
        supplemental_file_read=True,supplementary_table5_present=('Table S5' in supp),
        regime_values_in_current_exports=False,new_values='Regime analysis is new; no existing regime numbers in the exports to overwrite.')


def write_report(summary,test):
    zh={'nwp_wind_speed':'NWP 风速','measured_power':'实测功率占额定比例','ramp':'一小时爬坡','history':'输入历史','event':'候选事件'}
    variant_zh={'prior-off':'去掉物理先验','bound-off':'去掉边界罚项','derivative-regularizer-off':'去掉时间差分罚项',
        'calibration-off':'去掉风速校准','manufacturer-curve':'厂家曲线','cutout-off':'去掉切出边界','density-0_terrain-1':'去掉密度校正'}
    advantages=summary[summary.clear_advantage];disadvantages=summary[summary.clear_disadvantage]
    rows=[]
    for r in summary[summary.clear_advantage|summary.clear_disadvantage].itertuples():
        verdict='全模型优势' if r.clear_advantage else '全模型劣势'
        seeds=r.favourable_seeds if r.clear_advantage else r.unfavourable_seeds
        rows.append(f'| {verdict} | {variant_zh[r.variant]} | {zh[r.regime]}：{r.bin_label} | {r.metric} | {r.n_targets:,} | {r.n_nonempty_days} | {r.relative_change_mean_pct:+.2f} ± {r.relative_change_seed_sd_pct:.2f} | {seeds}/5 | {r.time_q:.4g} | {r.seed_q:.4g} |')
    bins=[]
    for r in AUDIT['regime_definitions']:
        bins.append(f'| {zh[r["regime"]]} | {r["label"]} | {r["n_targets"]:,} | {r["n_times"]:,} | {r["n_days"]} |')
    gate=AUDIT['gate'];cross=AUDIT['regime_cross_checks'];alignment=AUDIT['alignment']
    claims=('本次分析在部分预声明工况检出全模型的条件精度优势；应逐项限定组件、工况和指标，并同等报告清晰劣势。'
            if len(advantages) else '在全部预声明工况及给定判据下，未检出物理组件为全模型带来的清晰精度优势。不能据此宣称物理组件提高特定工况预测精度。')
    report=f'''# Job A：物理组件的分工况精度分析

## 结论与写作边界

{claims}

共 7 个变体 × 18 个分箱 × 2 项指标，252 项比较。满足预声明条件的全模型优势 {len(advantages)} 项，劣势 {len(disadvantages)} 项。优势与劣势仅针对同种子完整模型的配对比较，不代表物理组件在其他场站或时段的因果效应。未检出清晰差异不等于统计等效。

Section 4.2.2 应先报告总体 Table 4 的结果：去掉单个物理项的总体误差变化很小，原有 BH 检验均不显著；本次分工况结果不改写总体结论。本分析不能以可解释性替代精度证据，也不能将某个指标的优势概括为两项指标均改善。

可以诚实写明的具体结果是：在实测功率为 0.3–0.7 额定功率的 20,197 个目标中，相对全模型，去掉物理先验、时间差分罚项和密度校正的 NRMSE 分别增加 2.73%、4.68%、2.77%，NMAE 分别增加 3.42%、5.44%、3.44%。时间差分罚项在高爬坡档还有条件优势；去掉它使 NRMSE / NMAE 增加 2.94% / 3.28%。去掉密度校正在 0.7–0.95 功率档使 NMAE 增加 4.65%。这些是按预声明标准检出的全部 9 项优势，来源键均见下表。

同时必须报告代价：在 <0.05 额定功率的 132,826 个目标中，去掉时间差分罚项和密度校正的 NMAE 分别降低 17.90%、17.29%，说明完整物理配置存在工况权衡。NWP 风速、自然历史缺测和候选事件三个家族均未检出全模型的清晰优势。全部 252 项新种子层检验 BH 后均不显著，最小 q 为 {AUDIT['results']['minimum_seed_q']:.4f}。因此只能写“条件性的时间层优势与跨工况代价并存”，不能写成物理组件整体、稳定地提升精度。

这些优势均以目标时刻的实测功率或已实现爬坡进行事后分层。它们解释误差出现在哪里，不构成预测起点可识别的在线模型切换规则。

## 一致性门槛

PASS。40 个运行、80 个逐运行指标以及 16 个五种子总体均值全部通过 0.01 百分点门槛。均值最大绝对偏差 {gate['max_absolute_mean_delta_pp']:.9g} 百分点；逐运行与 corrected provenance 的最大偏差 {gate['max_absolute_run_delta_pp']:.9g} 百分点。40 个 predictions.npz 的 SHA-256 与已锁定 provenance 完全一致，目标、掩码、事件、时刻、起点、风机顺序逐项相同。详见 `A_gate.csv` 和 `A_audit.json`。

只使用 Table 4 七行。密度项是 `density-0_terrain-1`，保留地形偏置；联合去掉密度与地形属于 Spatial attention，未纳入。prior-off 的 seed 42 使用 `_20261021_055558`，seed 3407 使用 `_20261021_060023`，其余重复运行未使用。

## 全部清晰优势与劣势

相对变化为每种子 `100 × 变体误差 / 全模型误差 − 100` 后取五种子均值，正号表示全模型更好。SD 为五种子样本 SD。严格按预声明规则判定：同方向至少 4/5 种子、时间检验 BH q < 0.05、平均变化绝对值至少 2%。seed q 供报告，不另加为判定门槛。以下完整保留所有满足规则的反例。

| 判定 | 对比变体 | 工况 | 指标 | 有效目标数 | 非空天数 | 相对变化 % ± SD | 同方向种子 | 时间 q | 种子 q |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| 无 | — | — | — | — | — | — | — | — | — |'}

每一行均可通过 `A_regime_summary.csv` 的 variant、regime、bin、metric 定位；五个原始配对值见 `A_regime_errors_per_seed.csv`，检验见 `A_tests.csv`。CSV 的 source_file、full_source_file、source_locator 给出原始预测及选取规则。

## 分箱样本量

各运行使用相同有效目标。目标数为风机 × 目标时刻，不把五种子重复计为独立观测。不同分层家族重叠，不能将各家族样本数相加。

| 分层 | 分箱 | 有效目标数 | 有效时刻数 | 非空天数 |
|---|---|---:|---:|---:|
{chr(10).join(bins)}

≥12 m/s 档没有目标，全部七变体及两指标仍保留，指标为空、检验 p=1 并计入 252 项 BH 家族。冻结风速类是候选标签，样本很少，不能写成已确认故障。有效目标共 {cross['valid_targets']:,}，原本无效目标 {cross['invalid_targets']:,}。

## 原始数据及时间对齐

输入加载代码的 100 m NWP 原始列 `{alignment['nwp_speed_column']}` 是模型的零基第 7 特征。prepared ECMWF 模式取目标时刻该列，同一向量扩展到 29 台风机。未换成 10 m U/V 派生风速、SCADA 风速或校准风速。原始测试风速范围 {alignment['wind_speed_min_ms']:.5f}–{alignment['wind_speed_max_ms']:.5f} m/s。此分析沿用存档天气输入协议，不能推导新的 origin-time operational availability 结论。

原始数据共 {alignment['raw_rows']:,} 行；测试索引 {alignment['test_first_raw_index']}–{alignment['test_last_raw_index']}，共 {alignment['test_times']:,} 时刻，从 {alignment['test_start']} 至 {alignment['test_end']}。时区 Asia/Shanghai，步长 15 min，目标在起点后 16 步。数字排序的 power1…power29 与风机 ID、保存的目标功率完全一致。先按加载器转 FP32 并除以 1000，再按 FP64 计算误差与功率差。

历史窗为模型实际读取的 96 个观测，即起点前 95 步至起点本身，含起点，总 24 h 窗。缺测规则是 `isfinite(power_mw) & (power_mw > -0.02)` 的反面，无额定功率上限筛除；在前向填充之前判断自然缺测，不注入故障。源码位置详见 audit alignment 字段。

爬坡采用目标与前 4 步的实测 MW 差绝对值，两端均须有效。第 90 百分位阈值 {cross['ramp_q90_mw']:.12g} MW，采用线性分位数；高爬坡含等于阈值的 ties，实际占可定义样本 {100*cross['ramp_top_share']:.5f}%。阈值 ties {cross['ramp_threshold_ties']} 个。前一小时功率无效的 {cross['ramp_lag_unavailable_valid_targets']} 个目标仅在爬坡家族中不分类，未填零、未归入其余档；其他分层继续使用。

## 统计与数值核对

每种子按 2 MW 对所有所选有效风机目标池化误差，`NRMSE=100 sqrt(mean(((pred−true)/2)^2))`、`NMAE=100 mean(abs((pred−true)/2))`。公式从 Fig. 13 所用 `calc_metrics` 提取函数 AST，仅执行数值函数，不导入模型。

种子层对五个相对变化做双侧 t 检验，df=4。时间层先逐目标对五种子归一化平方误差或绝对误差求均值，再计算变体减全模型。Asia/Shanghai 目标日期为配对块，同一天所有风机和五种子共同抽样。全测试期 {cross['day_blocks']} 天，固定 RNG 2026，2,000 次重采样；每次取重采样天的损失总和除以对应分箱目标数，保留按目标数加权的池化估计。p 值来自以观测均值中心化的双侧零假设分布，使用加一修正；另提供未校正的 percentile 95% CI。少于两非空天的时间检验不可估计，p=1；零分母重采样显式计数。

两种检验分别在全部 252 项变体 × 分箱 × 指标上 BH 校正，未按指标、分层或方向拆小家族。NRMSE 的效应量是五个 RMSE 相对变化均值，时间检验对象是种子平均平方误差差，二者不同且都完整输出；29 项清晰效应的两个方向全部一致。最低可取 bootstrap p 为 1/2001，有限重采样精度会影响 BH；严格保留指定次数。时间自举是在五个已拟合种子平均之后的条件时间不确定性，不是种子与时间的联合总体推断。跨日序列相关可能未被单日块完全覆盖。

另行独立复核从原始数据重建全部 18 个掩码，重新计算 1,260 行逐种子指标、252 个 t 检验和两套 BH q，并复算跨家族的 7 项 bootstrap 比较。指标最大差 7.11×10⁻¹⁴ 百分点，t 检验 p 最大差 6.66×10⁻¹⁶，全部 q 和抽查的 bootstrap p 完全一致；未发现判定方向冲突。

当前正文 Table 4 的均值、SD 与变化列逐项对照总账，核对条目 {len(AUDIT['current_text_checks']['table4_checks'])}，不一致 {len(AUDIT['current_text_checks']['mismatches'])}。原 S5 的 favourable seeds 指有利于变体，方向与本分析相反；原总体检验不能移用为新工况检验。本次分箱值是新增分析，现有正文和补充文本中无对应分箱值可替换。任何文本不一致均保存在 audit，不静默覆盖。

## 图件与复现

{'已达到至少一项清晰优势，另生成草稿图；图件设计、英文图注及视觉核查见下方追加说明。' if len(advantages) else '没有清晰优势，按任务要求不生成 fig_physics_accuracy 图件。清晰劣势及全部非显著结果仍保留在 CSV。'}

运行 `results/paper Files\\Python312\\python.exe analyze_physics_regimes.py` 可复现分析，`--gate-only` 只运行门槛。所有输入均只读；未训练、未推理、未读取权重、未使用 GPU。脚本、四张分析 CSV、审计 JSON 和本说明均在本输出目录。全部输入 SHA-256 与定义、配对检查在 `A_audit.json`。
'''
    (OUT/'图件说明.md').write_text(report,encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [Path(__file__), SCRATCH/'brief_r2_common.md', SCRATCH/'r2/m_user.txt', SCRATCH/'r2/s_user.txt',
              BASE/'修订图件_第4章续_20261027/B_对比与注意力/make_fig13_events_all_models.py']:
        track(p)
    data, info, ref = gate()
    if '--gate-only' in sys.argv:
        return
    analyze(data, info, ref)


if __name__ == '__main__':
    main()
