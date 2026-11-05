
"""Condition-stratified ablation analysis.

The plain ablation asks whether removing a component hurts *average* accuracy. For a
physics-guided paper the more informative question is where a component pays off: by operating
regime, by wind speed, by turbine, and in the error tail. This script recomputes every variant's
error against the frozen model inside those strata, using the same seed pairing.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

EX = Path(r'results/paper')
OUT = Path(r'results/paper/source/ablation_stratified.json')

def load(rd):
    with np.load(rd/'predictions.npz') as z:
        d = dict(y=z['y_true_mw'].astype(np.float64), p=z['y_pred_mw'].astype(np.float64))
        if 'events' in z.files: d['events'] = z['events']
        if 'target_valid' in z.files: d['valid'] = z['target_valid'].astype(bool)
        if 'turbine_ids' in z.files: d['tids'] = z['turbine_ids']
    return d

full = {}
variant = defaultdict(dict)
for mp in EX.rglob('metrics.json'):
    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('status')!='completed' or int(m['horizon'])!=16: continue
    cid = m['config']['tracking']['config_id']
    g = m['config']['experiment'].get('group')
    if not (mp.parent/'predictions.npz').is_file(): continue
    if g=='main' and '_main_' in cid: full[int(m['seed'])] = mp.parent
    elif g in ('ablation','physics'):
        variant[(g, m['config']['experiment'].get('variant'))][int(m['seed'])] = mp.parent
print('full:', sorted(full), '| variants:', len(variant))

seeds = sorted(set(full) & set().union(*[set(v) for v in variant.values()]))
print('common seeds:', seeds)
ref = load(full[seeds[0]])
EV = ref.get('events')
VALID = ref.get('valid')
CAP = 2.0

def stratify_mask():
    out = {}
    if EV is not None:
        for lab in ('normal','low_power_candidate','power_outlier'):
            out[f'event:{lab}'] = (EV==lab)
    if VALID is not None:
        out['all'] = np.ones_like(VALID, dtype=bool)
    return out

rows = []
for (g, v), by_seed in sorted(variant.items()):
    for seed in seeds:
        if seed not in by_seed: continue
        a = load(by_seed[seed]); b = load(full[seed])
        v_mask = np.ones_like(a['y'], dtype=bool)
        if a.get('valid') is not None: v_mask &= a['valid']
        if b.get('valid') is not None: v_mask &= b['valid']
        ev_a = (a['p']-a['y']); ev_b = (b['p']-b['y'])
        # strata
        masks = {'all': v_mask}
        if EV is not None:
            for lab in ('normal','low_power_candidate','power_outlier'):
                masks[f'event:{lab}'] = v_mask & (EV==lab)
        for name, m in masks.items():
            if m.sum() < 500: continue
            la = np.abs(ev_a[m]); lb = np.abs(ev_b[m])
            # block t-test to respect autocorrelation
            nb = len(la)//96
            if nb < 3: continue
            bm = (la[:nb*96].reshape(nb,96).mean(axis=1) - lb[:nb*96].reshape(nb,96).mean(axis=1))
            p = float(stats.ttest_1samp(bm, 0.0).pvalue)
            rows.append(dict(group=g, variant=v, seed=seed, stratum=name, n=int(m.sum()),
                             mae_variant=float(la.mean()/CAP), mae_full=float(lb.mean()/CAP),
                             rel=float((lb.mean()-la.mean())/lb.mean()), p=p))
print('rows:', len(rows))
# aggregate per (variant, stratum) over seeds, BH within all
agg = defaultdict(list)
for r in rows: agg[(r['group'], r['variant'], r['stratum'])].append(r)
summary = []
for k, rs in agg.items():
    summary.append(dict(group=k[0], variant=k[1], stratum=k[2], n_seeds=len(rs),
                        rel=float(np.mean([r['rel'] for r in rs])),
                        p=float(np.mean([r['p'] for r in rs])),
                        mae_variant=float(np.mean([r['mae_variant'] for r in rs])),
                        mae_full=float(np.mean([r['mae_full'] for r in rs]))))
ps = np.array([s['p'] for s in summary])
order = np.argsort(ps); n=len(ps)
ranked = ps[order]*n/(np.arange(n)+1)
adj = np.minimum.accumulate(ranked[::-1])[::-1]
p_bh = np.empty(n); p_bh[order] = np.clip(adj,0,1)
for s, pb in zip(summary, p_bh): s['p_bh']=float(pb)
OUT.write_text(json.dumps(dict(version=1, note='variant vs frozen model inside operating strata', rows=summary), ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)
print('\n%-9s %-28s %-22s %8s %8s' % ('group','variant','stratum','rel','p_bh'))
for s in sorted(summary, key=lambda x:(x['group'],x['variant'],x['stratum'])):
    mark = ' *' if s['p_bh']<0.05 and s['rel']<0 else ('  ' if s['rel']<0 else ' +')
    print('%-9s %-28s %-22s %7.2f%% %8.4f%s' % (s['group'], s['variant'], s['stratum'], s['rel']*100, s['p_bh'], mark))
