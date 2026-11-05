
"""Definitive post-freeze main-table comparison: test split, per-horizon 5 seeds vs archived old-A."""
import json, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(r'results/paper')
OLD = {  # archived PI-EEF 5-seed test mean (frozen pre-revision run)
    1:  dict(nrmse=8.7754, nmae=4.3143, r2=0.8971),
    4:  dict(nrmse=13.8139, nmae=7.2295, r2=0.7449),
    16: dict(nrmse=20.5249, nmae=12.1355, r2=0.4368),
    48: dict(nrmse=24.4137, nmae=14.6923, r2=0.2029),
    96: dict(nrmse=25.5723, nmae=15.9657, r2=0.1260),
}
rows = []
_seen = {}  # (horizon, seed) -> run dict ; duplicates arise from earlier double publication

for m in ROOT.rglob('metrics.json'):
    try: d = json.loads(m.read_text(encoding='utf-8'))
    except Exception: continue
    if d.get('status') != 'completed': continue
    e = d['config']['experiment']
    if e.get('group') != 'main': continue
    t = d.get('metrics', {})
    key = (e['horizon'], d.get('seed'))
    candidate = dict(h=e['horizon'], s=d.get('seed'), nrmse=t['nrmse']*100, nmae=t['nmae']*100,
                     r2=t['r2'], n=t.get('num_samples'), cid=d['config']['tracking']['config_id'],
                     started=str(d.get('started_utc') or d.get('timestamp') or ''))
    prev = _seen.get(key)
    if prev is None or candidate['started'] < prev['started']:
        _seen[key] = candidate
rows = list(_seen.values())
by = defaultdict(list)
for r in rows: by[r['h']].append(r)

print('=' * 108)
print('FINAL MODEL MAIN TABLE (test split, 7604 samples/seed)')
print('=' * 108)
hdr = f"{'horizon':>8} {'n':>3} {'NRMSE%':>9} {'SD':>7} {'NMAE%':>8} {'SD':>7} {'R2':>8} | {'old NRMSE':>10} {'delta':>8} {'old NMAE':>9} {'delta':>8}"
print(hdr); print('-' * 108)
for h in sorted(by):
    xs = sorted(by[h], key=lambda r: r['s'])
    nr = [x['nrmse'] for x in xs]; na = [x['nmae'] for x in xs]; r2 = [x['r2'] for x in xs]
    sd = lambda a: statistics.stdev(a) if len(a) > 1 else float('nan')
    o = OLD.get(h, {})
    dn = (o['nrmse'] - statistics.mean(nr)) / o['nrmse'] * 100 if len(nr) == 5 else float('nan')
    dm = (o['nmae'] - statistics.mean(na)) / o['nmae'] * 100 if len(na) == 5 else float('nan')
    print(f"{h:>8} {len(xs):>3} {statistics.mean(nr):9.4f} {sd(nr):7.4f} {statistics.mean(na):8.4f} {sd(na):7.4f} {statistics.mean(r2):8.4f} | {o.get('nrmse',0):10.4f} {dn:7.2f}% {o.get('nmae',0):9.4f} {dm:7.2f}%")
    print(f"         seeds={[x['s'] for x in xs]}  per-seed NRMSE={[round(x['nrmse'],2) for x in xs]}")
print()
incomplete = [h for h in sorted(by) if len(by[h]) != 5] + [h for h in (1,4,16,48,96) if h not in by]
print('horizons not yet at 5 seeds:', sorted(set(incomplete)) or 'none - TABLE COMPLETE')
