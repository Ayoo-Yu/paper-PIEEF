
"""Rigorous version of the stratified ablation: per-seed signs + block bootstrap CI.

Averaging per-seed p-values overstates precision. Here each stratum is summarised by the
per-seed relative improvement, the sign consistency across seeds, and a block-bootstrap CI on the
pooled block means, so a claim needs both a consistent direction and an interval excluding zero.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

EX = Path(r'results/paper')
OUT = Path(r'results/paper/source/ablation_by_regime.json')
BLOCK = 96

def load(rd):
    with np.load(rd/'predictions.npz') as z:
        d = dict(y=z['y_true_mw'].astype(np.float64), p=z['y_pred_mw'].astype(np.float64))
        for k in ('events','target_valid'):
            if k in z.files: d[k]=z[k]
    return d

full, var = {}, defaultdict(dict)
for mp in EX.rglob('metrics.json'):
    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('status')!='completed' or int(m['horizon'])!=16: continue
    if not (mp.parent/'predictions.npz').is_file(): continue
    g = m['config']['experiment'].get('group'); v = m['config']['experiment'].get('variant')
    cid = m['config']['tracking']['config_id']
    if g=='main' and '_main_' in cid: full[int(m['seed'])] = mp.parent
    elif g in ('ablation','physics'): var[(g,v)][int(m['seed'])] = mp.parent

seeds = sorted(full)
ref = load(full[seeds[0]])
EV = ref['events']; VALID = ref.get('target_valid')
CAP = 2.0

def ci_pooled(diff_blocks, n_boot=5000, alpha=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n = len(diff_blocks)
    if n < 5: return float('nan'), float('nan')
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff_blocks[idx].mean(axis=1)
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))

results = []
for (g,v), by_seed in sorted(var.items()):
    strata = {'all': np.ones_like(EV, dtype=bool)}
    for lab in ('normal','low_power_candidate','power_outlier'):
        strata[lab] = (EV==lab)
    for name, mask in strata.items():
        per_seed_rel = []
        blocks = []
        for s in seeds:
            if s not in by_seed: continue
            a = load(by_seed[s]); b = load(full[s])
            m = mask.copy()
            if a.get('target_valid') is not None: m &= a['target_valid']
            if b.get('target_valid') is not None: m &= b['target_valid']
            if m.sum() < 500: continue
            la = np.abs((a['p']-a['y'])[m])/CAP
            lb = np.abs((b['p']-b['y'])[m])/CAP
            per_seed_rel.append((lb.mean()-la.mean())/lb.mean())
            nb = len(la)//BLOCK
            if nb >= 3:
                bm = la[:nb*BLOCK].reshape(nb,BLOCK).mean(axis=1) - lb[:nb*BLOCK].reshape(nb,BLOCK).mean(axis=1)
                blocks.append(-bm/lb.mean())    # positive = variant better
        if not per_seed_rel: continue
        rel = float(np.mean(per_seed_rel))
        n_seeds = len(per_seed_rel)
        n_removal_worse = int(sum(1 for r in per_seed_rel if r < 0))   # removal worse => component helps
        pooled = np.concatenate(blocks) if blocks else np.array([])
        lo, hi = ci_pooled(pooled) if pooled.size else (float('nan'), float('nan'))
        results.append(dict(group=g, variant=v, stratum=name, n_seeds=n_seeds,
                            rel=rel, seeds_removal_worse=n_removal_worse,
                            ci_lo=lo, ci_hi=hi,
                            per_seed=[round(x,4) for x in per_seed_rel]))
OUT.write_text(json.dumps(dict(version=1, note='negative rel = component helps', rows=results), ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)

print('\n=== 限电工况（low_power_candidate）：负值 = 组件在这里有用 ===')
print('%-9s %-28s %9s %7s %20s' % ('group','variant','rel','seeds','95% CI'))
sel = [r for r in results if r['stratum']=='low_power_candidate']
for r in sorted(sel, key=lambda x: x['rel']):
    sig = ' **' if (r['ci_hi']<0) else (' *' if (r['ci_lo']<0 and r['ci_hi']>0 and r['rel']<-0.02) else '')
    print('%-9s %-28s %8.2f%% %4d/%-2d [%7.2f%%, %7.2f%%]%s' % (r['group'], r['variant'], r['rel']*100,
          r['seeds_removal_worse'], r['n_seeds'], r['ci_lo']*100, r['ci_hi']*100, sig))
print('\n=== 正常工况对比 ===')
sel2 = [r for r in results if r['stratum']=='normal']
for r in sorted(sel2, key=lambda x: x['rel']):
    print('%-9s %-28s %8.2f%% %4d/%-2d [%7.2f%%, %7.2f%%]' % (r['group'], r['variant'], r['rel']*100,
          r['seeds_removal_worse'], r['n_seeds'], r['ci_lo']*100, r['ci_hi']*100))
