"""Author-operated fault inference only; never train or change source artifacts."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import e1_common as c

# v2 cross-machine reproduction criterion for the fault-free re-run of each full-run task, fixed by the
# authors on 2026-10-27 before any v2 fault result existed. The saved benchmark predictions came from
# other GPUs and CUDA/cuDNN versions, where recurrent kernels differ in the last digits (GRU seed 13:
# max 2.14e-4 pu). The verification gate keeps the strict c.assert_prediction_equal tolerance.
REPRO_MAX_ABS_PU = 1e-3
REPRO_METRIC_PP = 0.01


def assert_reproduced(saved, pred, capacity):
    import numpy as np
    reference = saved['y_pred_mw']
    if reference.shape != pred.shape or not np.isfinite(pred).all() or not np.isfinite(reference).all():
        raise ValueError('Prediction shape/nonfinite mismatch')
    delta = np.abs((pred - reference) / capacity)
    saved_met = c.metrics(saved['y_true_mw'], reference, saved['target_valid'], capacity)
    rerun_met = c.metrics(saved['y_true_mw'], pred, saved['target_valid'], capacity)
    result = dict(criterion='cross_machine_v2', max_abs_diff_pu_limit=REPRO_MAX_ABS_PU,
                  metric_diff_pp_limit=REPRO_METRIC_PP, bitwise_equal=bool(np.array_equal(pred, reference)),
                  max_abs_diff_mw=float(np.abs(pred - reference).max()), max_abs_diff_pu=float(delta.max()),
                  mean_abs_diff_pu=float(delta.mean()),
                  nrmse_diff_pp=abs(rerun_met['nrmse_pct'] - saved_met['nrmse_pct']),
                  nmae_diff_pp=abs(rerun_met['nmae_pct'] - saved_met['nmae_pct']))
    if (result['max_abs_diff_pu'] > REPRO_MAX_ABS_PU or result['nrmse_diff_pp'] > REPRO_METRIC_PP
            or result['nmae_diff_pp'] > REPRO_METRIC_PP):
        raise ValueError(f'Reproduction FAIL {result}')
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    c.add_paths(ap)
    ap.add_argument('--phase', choices=('verification', 'run'), required=True)
    ap.add_argument('--which', choices=('l1', 'huber'), help='Required for verification')
    ap.add_argument('--device', default='cuda:0', help='cuda:0 or cpu; trees/Persistence stay CPU')
    ap.add_argument('--tasks', type=Path)
    ap.add_argument('--task-id', help='Run one exact task ID')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--cpu-threads', type=int, default=2)
    return ap


def source_for(args, task):
    if task.get('verification_kind') == 'huber':
        selected = c.read_json(args.reference_root / '补做重算/final/selected_retrained_runs.json')
        entry = next(x for x in selected if x['model'] == 'ACFormer' and x['horizon'] == args.horizon and x['seed'] == 13)
        # Portable rebasing of the saved absolute source path.
        tail = str(entry['directory']).replace('\\', '/').split('/补做重算/', 1)[1]
        return args.reference_root / '补做重算' / tail
    return args.main_root / task['run_directory']


def infer_task(args, ev, scenarios, task, destination, identity):
    import numpy as np
    import torch
    source = source_for(args, task)
    meta = c.read_json(source / 'metrics.json')
    # Compatibility is restricted to a separately proven, exact legacy Huber source.
    verified_split = meta.get('evaluation_split')
    if 'evaluation_split' not in meta:
        helper_path = Path(__file__).with_name('e1_legacy_schema.py')
        contract_path = Path(__file__).with_name('e1_huber_legacy_contract.json')
        if (c.sha(helper_path) != '6e56efc6d5d80f03346ea96fffc4044b5a3f5bfa0eb4d4f5e2f94dc190b2955d'
                or c.sha(contract_path) != '80b732c24f3a21e435da5511ac132a91e9770f4f9ad6634588365b9847926b5d'):
            raise ValueError('Legacy schema helper/contract changed; source remains rejected')
        from e1_legacy_schema import verify_test_source
        verified_split = verify_test_source(args, task, source, meta)
    if (meta['status'] != 'completed' or verified_split != 'test'
            or (meta['model'], meta['seed'], meta['horizon']) != (task['model'], task['seed'], args.horizon)):
        raise ValueError(f'Wrong source identity: {source}')
    cfg = deepcopy(meta['config'])
    if cfg['experiment']['horizon'] != args.horizon or cfg['data']['seq_len'] < 96:
        raise ValueError('Wrong horizon/history length')
    cfg['data']['scada_path'] = str(args.data_root.resolve() / 'scada_ecmwf_15min.csv')
    cfg['data']['turbines_path'] = str(args.data_root.resolve() / 'turbines.csv')
    record = dict(source_dir=source, config=cfg, model=task['model'],
                  seed=task['seed'], horizon=args.horizon)
    checkpoint = source / ('model.pkl' if task['model'] in ('LightGBM', 'XGBoost') else 'weights_best.pt')
    checkpoint_hash = c.sha(checkpoint) if task['model'] != 'Persistence' else None
    source_hashes = {n: c.sha(source / n) for n in ('metrics.json', 'preprocessing.npz', 'predictions.npz')}
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'completed.json'
    if marker.exists():
        prior = c.read_json(marker)
        if (prior['identity'] != identity or prior['task'] != task or prior['source_sha256'] != source_hashes
                or prior['checkpoint_sha256'] != checkpoint_hash
                or prior['predictions_sha256'] != c.sha(destination / 'predictions.npz')
                or prior['metrics_sha256'] != c.sha(destination / 'metrics.csv')):
            raise ValueError(f'Existing output differs; use a NEW output root: {destination}')
        print(f'[reuse complete] {task["task_id"]}', flush=True)
        return
    if any(destination.iterdir()):
        raise ValueError(f'Partial output exists; preserve it and use a NEW output root/task directory: {destination}')
    start = time.perf_counter()
    c.dump_json(destination / 'started.json', dict(task=task, identity=identity, source=str(source), pid=os.getpid()))
    bundle = ev._bundle(record)  # Retains original trained batch size and preprocessing checks.
    fields = {k: np.asarray(v).astype(str) if np.asarray(v).dtype.kind == 'O' else np.asarray(v)
              for k, v in bundle.handler.metadata['test'].items()}
    fields['turbine_ids'] = np.asarray(bundle.handler.turbine_ids, dtype=str)
    capacity = np.asarray(bundle.capacity_mw)
    with np.load(source / 'preprocessing.npz', allow_pickle=False) as pre:
        if not np.array_equal(capacity, pre['capacity_mw']):
            raise ValueError('Capacity differs from saved source')
    device = torch.device(args.device)
    model, kind, actual_checkpoint = ev._predictor(record, bundle, device)
    if checkpoint_hash and c.sha(actual_checkpoint) != checkpoint_hash:
        raise ValueError('Predictor loaded a different checkpoint')
    if model is not None and kind == 'dl' and model.training:
        raise ValueError('Predictor not in eval mode')
    if task.get('verification_kind') == 'huber' and (cfg['train'].get('loss_name') != 'huber' or cfg['train'].get('huber_delta') != .1):
        raise ValueError('Expected saved Huber delta=0.1 source')
    if task.get('verification_kind') == 'l1' and cfg.get('baseline_protocol', {}).get('loss') != 'l1':
        raise ValueError('Expected original main ACFormer L1 source')
    rows, all_predictions, fault_hashes = [], [], {}
    specifications = [('normal', None)] + [(s['name'], {k: v for k, v in s.items() if k != 'name'}) for s in scenarios]
    for name, scenario in specifications:
        t0 = time.perf_counter()
        hashes = {k: hashlib.sha256() for k in ('power', 'wind', 'power_mask', 'wind_mask')}
        original_fault = ev.apply_fault
        def audited_fault(batch, spec, std, coords, seed):
            if 'sample_ids' not in spec or seed != c.FAULT_SEED:
                raise ValueError('Origin-indexed fault seed/identities missing')
            corrupted, pm, wm = original_fault(batch, spec, std, coords, seed)
            # Separate streams preserve identical hashes even if batch sizes differ.
            for key, value in zip(hashes, (corrupted[0], corrupted[1], pm, wm)):
                hashes[key].update(value.detach().cpu().contiguous().numpy().tobytes())
            return corrupted, pm, wm
        ev.apply_fault = audited_fault
        try:
            truth, pred = ev._infer(model, kind, bundle, scenario, c.FAULT_SEED, 'test', device)
        finally:
            ev.apply_fault = original_fault
        if name == 'normal':
            with np.load(source / 'predictions.npz', allow_pickle=False) as saved:
                c.assert_indices(saved, dict(**fields, y_true_mw=truth), args.horizon)
                if task.get('verification_kind'):
                    normal_check = c.assert_prediction_equal(saved['y_pred_mw'], pred, capacity)
                else:
                    normal_check = assert_reproduced(saved, pred, capacity)
                print(f'{task["task_id"]} normal reproduction: {normal_check}', flush=True)
            canonical = c.references(args.reference_root, args.horizon)['l1'] / 'normal.npz'
            with np.load(canonical, allow_pickle=False) as saved:
                c.assert_indices(saved, dict(**fields, y_true_mw=truth), args.horizon)
                if not np.array_equal(capacity, saved['capacity_mw']):
                    raise ValueError('Baseline not aligned with canonical ACFormer/PI-EEF targets/capacity')
            target = truth
        elif not np.array_equal(truth, target, equal_nan=True):
            raise ValueError('Fault changed targets')
        met = c.metrics(truth, pred, fields['target_valid'], capacity)
        if name == 'normal':
            normal = met
        row = dict(model=task['model'], seed=task['seed'], horizon=args.horizon, scenario=name, **met,
                   nrmse_increase_pct=100 * (met['nrmse_pct'] / normal['nrmse_pct'] - 1),
                   nmae_increase_pct=100 * (met['nmae_pct'] / normal['nmae_pct'] - 1))
        rows.append(row)
        all_predictions.append(pred)
        if scenario is not None:
            fault_hashes[name] = {k: h.hexdigest() for k, h in hashes.items()}
        print(f'{task["task_id"]} {len(rows):02}/40 {name} NRMSE={met["nrmse_pct"]:.8f}% seconds={time.perf_counter()-t0:.2f}', flush=True)
    archive = destination / 'predictions.npz'
    with archive.open('wb') as stream:
        np.savez_compressed(stream, **fields, y_true_mw=target, y_pred_mw=np.stack(all_predictions),
                            scenario=np.array([n for n, _ in specifications]),
                            scenario_json=np.array([json.dumps(s, sort_keys=True) for _, s in specifications]),
                            seed=task['seed'], horizon=args.horizon, model=task['model'],
                            capacity_mw=capacity, fault_seed=c.FAULT_SEED)
    c.write_csv(destination / 'metrics.csv', rows)
    if source_hashes != {n: c.sha(source / n) for n in source_hashes} or (checkpoint_hash and checkpoint_hash != c.sha(checkpoint)):
        raise ValueError('Source changed during inference')
    c.dump_json(marker, dict(status='completed', task=task, identity=identity,
                            source_directory=str(source), source_sha256=source_hashes,
                            checkpoint_sha256=checkpoint_hash, normal_equivalence=normal_check,
                            realization_sha256=fault_hashes, predictions_sha256=c.sha(archive),
                            metrics_sha256=c.sha(destination / 'metrics.csv'), cases=40,
                            batch_size=cfg['train']['batch_size'], trained=False,
                            elapsed_seconds=time.perf_counter()-start,
                            runtime=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                         device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)))
    del model, bundle
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = parser()
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.cpu_threads < 1:
        ap.error('Invalid shard/thread count')
    if args.phase == 'verification' and not args.which:
        ap.error('--which l1|huber is required for verification')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.cpu_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.cpu_threads))
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = c.protocol_identity(args.project_root, args.data_root)
    tasks = c.load_tasks(args.horizon, args.tasks)
    if args.phase == 'verification':
        task = deepcopy(next(t for t in tasks if t['model'] == 'ACFormer' and t['seed'] == 13))
        task.update(verification_kind=args.which, task_id=f'ACFormer_{args.which}_h{args.horizon}_seed13')
        selected = [(task, c.gate_dir(args.output_root, args.horizon) / args.which)]
    else:
        c.require_gate(args.output_root, args.horizon, identity)
        if args.task_id and args.task_id not in {t['task_id'] for t in tasks}:
            ap.error('Unknown --task-id')
        selected = [(t, args.output_root / f'h{args.horizon}' / 'tasks' / t['task_id'])
                    for i, t in enumerate(tasks)
                    if i % args.num_shards == args.shard_index and (not args.task_id or t['task_id'] == args.task_id)]
        if not selected:
            ap.error('Empty task selection')
    ev, scenarios = c.load_evaluation(args.project_root)
    for task, dest in selected:
        infer_task(args, ev, scenarios, task, dest, identity)


if __name__ == '__main__':
    main()
