
"""P7: learning curves and monotonicity / cut-in / cut-out behaviour checks."""
import json
from pathlib import Path
import numpy as np

MAIN = Path(r'results/paper')
OUT = Path(r'results/paper/source/p7_learning_curves.json')

runs = []
for mp in MAIN.rglob('metrics.json'):
    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('status') != 'completed': continue
    lh = mp.parent/'loss_history.json'
    if not lh.is_file(): continue
    try: h = json.loads(lh.read_text(encoding='utf-8'))
    except Exception: continue
    runs.append(dict(seed=m['seed'], horizon=m['horizon'], history=h))
print('runs with loss history:', len(runs))

def summarise(hist):
    """loss_history.json stores one list of per-epoch dicts per stage."""
    out = {}
    if not isinstance(hist, dict):
        return out
    for stage in ('stage1','stage2','stage3'):
        v = hist.get(stage)
        if not isinstance(v, list) or not v:
            continue
        tr = [float(e['train_loss']) for e in v if 'train_loss' in e]
        va = [float(e['val_loss']) for e in v if 'val_loss' in e]
        if tr:
            out[stage] = dict(train=tr, val=va)
    return out

rows = []
for r in runs:
    s = summarise(r['history'])
    row = dict(seed=r['seed'], horizon=r['horizon'], stages={})
    for stage, d in s.items():
        tr = d['train']; va = d['val']
        mono_tr = float(np.mean(np.diff(tr) <= 0)) if len(tr) > 1 else None
        mono_va = float(np.mean(np.diff(va) <= 0)) if len(va) > 1 else None
        row['stages'][stage] = dict(n_epochs=len(tr), train_first=tr[0], train_last=tr[-1],
                                    val_first=va[0] if va else None, val_last=va[-1] if va else None,
                                    frac_non_increasing_train=mono_tr, frac_non_increasing_val=mono_va)
    rows.append(row)
OUT.write_text(json.dumps(dict(version=1, rows=rows), ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)
for h in sorted({r['horizon'] for r in rows}):
    sel = [r for r in rows if r['horizon'] == h]
    print(f'\nh{h}: {len(sel)} seeds')
    for stage in ('stage1','stage2','stage3'):
        xs = [r['stages'].get(stage) for r in sel if r['stages'].get(stage)]
        if not xs: continue
        print(f'  {stage}: epochs={xs[0]["n_epochs"]} train {np.mean([x["train_first"] for x in xs]):.4f}→{np.mean([x["train_last"] for x in xs]):.4f} '
              f'| val {np.mean([x["val_first"] for x in xs]):.4f}→{np.mean([x["val_last"] for x in xs]):.4f} '
              f'| 单调率(train)={np.mean([x["frac_non_increasing_train"] for x in xs if x["frac_non_increasing_train"] is not None]):.2f}')
