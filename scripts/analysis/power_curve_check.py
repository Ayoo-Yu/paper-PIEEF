
"""Learned power curve check: is predicted power monotone in wind speed up to cut-out?

This is what R3-Q14 actually asks about. The earlier reply cited the training-loss curve, which is
a different object; this script bins the test-set predictions by measured wind speed and reports
monotonicity, the maximum reversal and the behaviour above and below cut-in / cut-out.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

V2 = Path(r'.')
EX = Path(r'results/paper')
OUT = Path(r'results/paper/source/power_curve_check.json')
sys.path.insert(0, str(V2/'src')); sys.path.insert(0, str(V2/'scripts'))

run_dirs = []
for mp in EX.rglob('metrics.json'):
    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('status')!='completed': continue
    if '_main_' in m['config']['tracking']['config_id'] and m['horizon']==16:
        run_dirs.append((m['seed'], mp.parent))
run_dirs.sort()
print('h16 runs:', [s for s,_ in run_dirs])

import evaluate_revision as evaluation
results = []
for seed, rd in run_dirs[:3]:
    meta = json.loads((rd/'metrics.json').read_text(encoding='utf-8'))
    record = dict(source_dir=rd, config=meta['config'], model='PI-EEF', seed=seed, horizon=16)
    bundle = evaluation._bundle(record)
    handler = bundle.handler
    mean_w = float(getattr(handler, 'scada_w_mean', 0.0)); std_w = float(getattr(handler, 'scada_w_std', 1.0))
    with np.load(rd/'predictions.npz') as z:
        y_pred = z['y_pred_mw'].astype(np.float64)
    winds = []
    with torch.no_grad():
        for batch in bundle.test_loader:
            # x_w is [batch, turbines, seq_len, 1]; the last step is the freshest observed wind
            winds.append(batch[1][:, :, -1, 0].numpy()*std_w + mean_w)
    wind = np.concatenate(winds, axis=0)
    n = min(wind.shape[0], y_pred.shape[0])
    w = wind[:n]; p = y_pred[:n]
    m = np.isfinite(w) & np.isfinite(p)
    w = w[m]; p = p[m]
    edges = np.arange(0, 26.5, 1.0)
    centres, means, counts = [], [], []
    for i in range(len(edges)-1):
        sel = (w >= edges[i]) & (w < edges[i+1])
        if sel.sum() < 200: continue
        centres.append((edges[i]+edges[i+1])/2); means.append(float(p[sel].mean())); counts.append(int(sel.sum()))
    centres = np.array(centres); means = np.array(means); counts = np.array(counts)
    below_cutout = centres <= 21.5
    c2, m2 = centres[below_cutout], means[below_cutout]
    diffs = np.diff(m2)
    reversals = int((diffs < 0).sum())
    results.append(dict(seed=seed, n=int(w.size), bins=int(len(c2)),
                        wind_range=[float(w.min()), float(w.max())],
                        frac_above_22=float((w > 22).mean()),
                        monotone_frac=float((diffs >= 0).mean()),
                        worst_reversal=float(min(diffs.min(), 0.0)),
                        curve=dict(wind=[float(x) for x in c2], power=[float(x) for x in m2], n=[int(x) for x in counts]),
                        above_cutout_mean_power=float(p[w > 22].mean()) if (w > 22).any() else None))
    print(json.dumps({k:v for k,v in results[-1].items() if k!='curve'}, ensure_ascii=False))
OUT.write_text(json.dumps(dict(version=1, note='learned power curve, h16 test set', rows=results), ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)
