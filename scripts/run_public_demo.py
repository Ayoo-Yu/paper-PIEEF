#!/usr/bin/env python3
"""CPU-only synthetic VF smoke run; this is not a paper experiment or benchmark."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[name] = '2'

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pieef.artifacts import save_preprocessing, save_run_predictions, write_json_atomic
from pieef.core import set_seed
from pieef.data import NWP_COLS, build_data_bundle
from pieef.forecast_corruption import ForecastCorruption
from pieef.model_factory import build_main_model
from pieef.train_main import TrainConfig, train_multistep, evaluate_multistep
from pieef.training_timing import save_training_timing


def synthetic_inputs(folder: Path, seed: int):
    """Generate fictitious turbines, observations and explicitly fictitious forecast vintages."""
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    n, nt = 10, 800
    timestamps = pd.date_range('2024-01-01', periods=nt, freq='15min', tz='UTC')
    t = np.arange(nt, dtype=float)
    speed = 7 + 1.5 * np.sin(t / 31) + rng.normal(0, .35, nt)
    wind = np.maximum(.1, speed[:, None] + rng.normal(0, .5, (nt, n)))
    power = 2000 * np.clip((wind - 3) / 7, 0, 1) ** 3
    power *= rng.uniform(.95, 1.05, power.shape)
    # Include missing history, invalid targets and an observable low-power interval.
    power[100:104, 0] = np.nan
    power[600:604, 1] = np.nan
    wind[80:84, 0] = np.nan
    power[220:228, 2] = 2.0
    frame = pd.DataFrame({'Timestamp': timestamps.astype(str)})
    for i in range(n):
        frame[f'power{i+1}'] = power[:, i]
        frame[f'turbine_ws{i+1}'] = wind[:, i]
    frame.to_csv(folder / 'observations.csv', index=False)
    pd.DataFrame({
        'turbine_id': [f'SYNTH-{i+1:02d}' for i in range(n)],
        'x_coord': (np.arange(n) % 5) * 400.,
        'y_coord': (np.arange(n) // 5) * 500.,
        'elevation_m': np.linspace(20, 70, n),
    }).to_csv(folder / 'turbines.csv', index=False)
    v = np.arange(0, 31, dtype=float)
    p = np.where((v >= 3) & (v < 22), np.clip((v - 3) / 7, 0, 1) ** 3, 0)
    pd.DataFrame({'wind_speed_ms': v, 'power_pu': p}).to_csv(folder / 'synthetic_curve.csv', index=False)
    # Deterministic toy forecasts, created independently of SCADA observations.
    refs = pd.date_range(timestamps[0] - pd.Timedelta('12h'), timestamps[-1], freq='6h')
    leads = np.arange(0, 49, dtype=float)
    valid_hours = ((refs.asi8 - timestamps[0].value) / 3.6e12)[:, None] + leads[None, :]
    ws = 7 + 1.5 * np.sin(valid_hours * 4 / 31)
    cube = np.empty((len(refs), len(leads), len(NWP_COLS)), dtype=np.float64)
    cube[..., 0] = np.maximum(0, 500 * np.sin(valid_hours * np.pi / 12))
    cube[..., 1] = cube[..., 0] * .9
    cube[..., 2] = 101325 + 100 * np.cos(valid_hours / 48)
    cube[..., 3] = 288.15 + 3 * np.sin(valid_hours * np.pi / 12)
    cube[..., 4] = ws * .8
    cube[..., 5] = ws * .6
    cube[..., 6], cube[..., 7], cube[..., 8] = ws * 1.15, ws * 1.1, ws
    forecast_path = folder / 'synthetic_forecasts.npz'
    np.savez_compressed(forecast_path, reference_time=refs.astype(str).to_numpy(dtype=str),
        available_time=(refs + pd.Timedelta('1h')).astype(str).to_numpy(dtype=str),
        lead_hours=leads, feature_names=np.asarray(NWP_COLS), values=cube)
    return dict(nwp_timing_rule='forecast_vintages', nwp_alignment_verified=True,
        nwp_vintages_path=str(forecast_path), nwp_feature_columns=NWP_COLS,
        timezone='UTC', step_minutes=15, power_unit='kW', wind_height_m=10,
        input_filling='forward-only', target_policy='raw-measured-valid-power',
        synthetic_demo=True, source='Fictitious deterministic forecasts; not ECMWF/GFS or paper data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(ROOT / 'artifacts/public_demo/vf'))
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    torch.set_num_threads(2)
    set_seed(args.seed)
    run = Path(args.output_root).resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT / 'configs/vf_main_h16.yaml').read_text(encoding='utf-8'))
    protocol = synthetic_inputs(run / 'inputs', args.seed)
    cfg['experiment'].update(seeds=[args.seed], site='SYNTHETIC', finalized=False, selection_scope='validation')
    cfg['data'].update(data_version='synthetic_public_vf_smoke_v1',
        dms_path=str(run/'inputs/observations.csv'), turbines_path=str(run/'inputs/turbines.csv'), revision_protocol=protocol)
    cfg['model'].update(d_model=32, encoder_depth=1, manufacturer_curve_path=str(run/'inputs/synthetic_curve.csv'))
    cfg['train'].update(batch_size=4, epochs_stage1=1, epochs_stage2=1, epochs_stage3=2,
        max_batches_per_epoch=2, record_gradient_conflicts=False)
    cfg['tracking'] = dict(config_id='synthetic_vf_cpu_smoke', implementation_id='energy-r3-vf-F2F3F6', artifact_root=str(run))
    (run / 'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    bundle = build_data_bundle(Path(cfg['data']['dms_path']), Path(cfg['data']['turbines_path']),
        seq_len=96, horizon=16, batch_size=4, train_ratio=.7, val_ratio=.1, capacity_mw=2.,
        revision_protocol=protocol)
    model = build_main_model(bundle, 96, cfg['model'], torch.device('cpu'))
    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    tcfg = TrainConfig(**{k:v for k,v in cfg['train'].items() if k in train_fields})
    tcfg.checkpoint_dir = str(run)
    tcfg.stage3_corruption = ForecastCorruption('f2_block', bundle.handler.coords,
        bundle.handler.scada_w_std, args.seed, bundle.num_turbines, 96)
    save_preprocessing(run, bundle, revin_missing_policy=cfg['model']['revin_missing_policy'])
    model, history = train_multistep(model, bundle.train_loader, bundle.val_loader, torch.device('cpu'), tcfg)
    save_training_timing(run, model, cfg, args.seed, 16, 'cpu')
    metrics, truth, prediction = evaluate_multistep(model, bundle.val_loader, torch.device('cpu'), bundle.capacity_mw)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Synthetic smoke run generated non-finite predictions')
    save_run_predictions(run, bundle, truth, prediction, args.seed, 16, split='val')
    write_json_atomic(run / 'loss_history.json', history)
    write_json_atomic(run / 'metrics.json', dict(status='completed', synthetic_demo=True,
        evaluation_split='val', metrics=metrics, config=cfg,
        note='CPU software smoke test only. Reduced width/depth/epochs/batches are not paper settings.'))
    print(json.dumps({'status':'completed','synthetic_demo':True,'artifact_directory':str(run),
        'validation_samples':len(prediction), 'finite_predictions':True}, indent=2))


if __name__ == '__main__':
    main()
