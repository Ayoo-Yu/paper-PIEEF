"""E4 inference adapter: reuse frozen E1 v2 metrics/index checks and production 39-fault kernel."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('project','source','data','output','e1-scripts'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(a.e1_scripts.resolve()))
    import e1_common as c
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ev,scenarios=c.load_evaluation(a.project.resolve())
    source=a.source.resolve();meta=c.read_json(source/'metrics.json');cfg=copy.deepcopy(meta['config'])
    assert meta['status']=='completed' and meta['seed']==a.seed and meta['horizon']==16
    assert meta['evaluation_split']=='test' and cfg['model']['graph_neighbour_policy']=='self_only'
    assert cfg['train']['batch_size']==16 and len(scenarios)==39
    cfg['data']['scada_path']=str(a.data.resolve()/'scada_ecmwf_15min.csv')
    cfg['data']['turbines_path']=str(a.data.resolve()/'turbines.csv')
    cfg['model']['manufacturer_curve_path']=str(a.data.resolve()/'manufacturer_curve.csv')
    names=('metrics.json','preprocessing.npz','predictions.npz','weights_best.pt','weights_stage1.pt','weights_stage2.pt','weights_last.pt')
    source_hash={n:sha(source/n) for n in names}
    identity=c.protocol_identity(a.project,a.data)
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Existing/partial inference output retained; no duplicate attempt')
    c.dump_json(a.output/'started.json',dict(pid=os.getpid(),seed=a.seed,source=str(source),identity=identity))
    start=time.perf_counter();record=dict(source_dir=source,config=cfg,model='PI-EEF',seed=a.seed,horizon=16)
    bundle=ev._bundle(record);device=torch.device(a.device)
    fields={k:np.asarray(v).astype(str) if np.asarray(v).dtype.kind=='O' else np.asarray(v) for k,v in bundle.handler.metadata['test'].items()}
    fields['turbine_ids']=np.asarray(bundle.handler.turbine_ids,dtype=str)
    capacity=np.asarray(bundle.capacity_mw)
    model,kind,checkpoint=ev._predictor(record,bundle,device)
    assert kind=='main' and not model.training and sha(checkpoint)==source_hash['weights_best.pt']
    layers=model.graph.layers if hasattr(model.graph,'layers') else [model.graph]
    specifications=[('normal',None)]+[(s['name'],{k:v for k,v in s.items() if k!='name'}) for s in scenarios]
    all_predictions=[];rows=[];realizations={};attn_checks=[]
    for name,scenario in specifications:
        t0=time.perf_counter();hashes={k:hashlib.sha256() for k in ('power','wind','power_mask','wind_mask')}
        original=ev.apply_fault
        def audited(batch,spec,std,coords,seed):
            assert 'sample_ids' in spec and seed==2026
            damaged,pm,wm=original(batch,spec,std,coords,seed)
            for k,v in zip(hashes,(damaged[0],damaged[1],pm,wm)):hashes[k].update(v.detach().cpu().contiguous().numpy().tobytes())
            return damaged,pm,wm
        ev.apply_fault=audited
        try:truth,pred=ev._infer(model,kind,bundle,scenario,2026,'test',device)
        finally:ev.apply_fault=original
        for layer in layers:
            w=layer.last_attn_weights;eye=torch.eye(w.shape[-1],device=w.device,dtype=w.dtype).expand_as(w)
            if not torch.equal(w,eye):raise ValueError('E4 self-only attention is not the identity')
        attn_checks.append(name)
        if name=='normal':
            with np.load(source/'predictions.npz',allow_pickle=False) as saved:
                c.assert_indices(saved,dict(**fields,y_true_mw=truth),16)
                normal_check=c.assert_prediction_equal(saved['y_pred_mw'],pred,capacity)
            target=truth
        elif not np.array_equal(truth,target,equal_nan=True):raise ValueError('Fault modified targets')
        metrics=c.metrics(truth,pred,fields['target_valid'],capacity)
        if name=='normal':normal=metrics
        rows.append(dict(model='PI-EEF-self-only',seed=a.seed,horizon=16,scenario=name,**metrics,
            nrmse_increase_pct=100*(metrics['nrmse_pct']/normal['nrmse_pct']-1),
            nmae_increase_pct=100*(metrics['nmae_pct']/normal['nmae_pct']-1)))
        all_predictions.append(pred)
        if scenario is not None:realizations[name]={k:h.hexdigest() for k,h in hashes.items()}
        print(f'[E4 {a.seed}] {len(rows):02}/40 {name} NRMSE={metrics["nrmse_pct"]:.8f}% seconds={time.perf_counter()-t0:.2f}',flush=True)
    archive=a.output/'predictions.npz'
    with archive.open('wb') as stream:
        np.savez_compressed(stream,**fields,y_true_mw=target,y_pred_mw=np.stack(all_predictions),
            scenario=np.array([n for n,_ in specifications]),scenario_json=np.array([json.dumps(s,sort_keys=True) for _,s in specifications]),
            seed=a.seed,horizon=16,model='PI-EEF-self-only',capacity_mw=capacity,fault_seed=2026)
    c.write_csv(a.output/'metrics.csv',rows)
    assert source_hash=={n:sha(source/n) for n in names}
    c.dump_json(a.output/'completed.json',dict(status='completed',seed=a.seed,horizon=16,cases=40,
        source=str(source),source_sha256=source_hash,identity=identity,e4_adapter_sha256=sha(__file__),normal_equivalence=normal_check,
        identity_attention_checked_cases=attn_checks,realization_sha256=realizations,
        predictions_sha256=sha(archive),metrics_sha256=sha(a.output/'metrics.csv'),elapsed_seconds=time.perf_counter()-start,
        runtime=dict(python=sys.version,torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(device))))

if __name__=='__main__':main()
