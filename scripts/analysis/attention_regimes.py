
"""P4: attention statistics split by operating regime (normal / curtailment / outlier)."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-batches', type=int, default=0)
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    PROJECT = Path(a.project)
    sys.path.insert(0, str(PROJECT/'src')); sys.path.insert(0, str(PROJECT/'scripts'))
    import evaluate_revision as evaluation
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
    record = dict(source_dir=run_dir, config=meta['config'], model='PI-EEF', seed=meta['seed'], horizon=meta['horizon'])
    device = torch.device(a.device)
    bundle = evaluation._bundle(record)
    model, kind, _ = evaluation._predictor(record, bundle, device)
    loader = getattr(bundle, f'{a.split}_loader')
    meta_val = bundle.handler.metadata[a.split]
    events = np.asarray(meta_val['events']) if 'events' in meta_val else None
    per_regime = {}
    offset = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if a.max_batches and i >= a.max_batches: break
            count = len(batch[0])
            pm = torch.zeros_like(batch[0]); wm = torch.zeros_like(batch[1])
            evaluation._predict(model, kind, batch, pm, wm, device)
            att = getattr(model.graph, 'last_attn_weights', None)
            if att is None: raise RuntimeError('no attention')
            p = att.detach().float().cpu().numpy()
            n = p.shape[-1]; eps = 1e-12
            ent = -(p*np.log(p+eps)).sum(axis=-1)/np.log(n)   # [B, T]
            mx = p.max(axis=-1)                                # [B, T]
            if events is not None:
                labs = events[offset:offset+count]
                regime_of_sample = [Counter(row).most_common(1)[0][0] for row in labs]
            else:
                regime_of_sample = ['unknown']*count
            for b in range(count):
                r = str(regime_of_sample[b])
                per_regime.setdefault(r, []).append((float(ent[b].mean()), float(mx[b].mean())))
            offset += count
    payload = dict(version=1, config_id=meta['config']['tracking']['config_id'], seed=meta['seed'],
                   split=a.split, gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                   regimes={k: dict(n=len(v), mean_entropy=float(np.mean([x[0] for x in v])),
                                    mean_max_weight=float(np.mean([x[1] for x in v])))
                            for k, v in per_regime.items()})
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(payload['regimes'], ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
