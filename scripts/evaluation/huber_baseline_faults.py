"""Task F only: immutable checkpoint inference, original revision fault operator."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util,traceback
from copy import deepcopy
import numpy as np
import torch
ROOT=Path('results/paper')
OUT=Path(__file__).resolve().parent
PROJECT=ROOT/'next_batch/merged_v2/project'
DATA=Path('data/wind_farm_A')
RUNS=ROOT/'修复重算_20261021/baselines/candidate_project/artifacts/energy_baseline_repair_20261021'
sys.path.insert(0,str(PROJECT/'src'))
import rerun_h16_faults as ref
spec=importlib.util.spec_from_file_location('evaluation',PROJECT/'scripts/evaluate_revision.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
from pieef.revision_fault_jobs import scada_scenarios
torch.set_num_threads(2)
def dump(p,x): ref.atomic_json(p,x)
def progress(s):
    with (OUT/'进度.md').open('a',encoding='utf-8') as f: f.write('\n- '+s+'\n')
def grid(fields):
    import pandas as pd
    t=pd.to_datetime(fields['timestamp'],utc=True).asi8
    o=pd.to_datetime(fields['origin_timestamp'],utc=True).asi8
    assert np.all(t-o==ref.HORIZON*15*60*10**9),'target-origin mismatch'
    assert np.all(np.diff(t)>0)
ref.check_h16_grid=grid
def main():
    selected=[]
    for entry in json.loads((OUT/'selected_retrained_runs.json').read_text(encoding='utf-8')):
        p=Path(entry['directory'])/'metrics.json';m=json.loads(p.read_text(encoding='utf-8'))
        if (m['model'],m['horizon']) in [('ACFormer',16),('ACFormer',96),('TCN',96),('TimesNet',96)]: selected.append((p,m))
    assert len(selected)==12,len(selected)
    sources=[dict(path=str(p.parent),config_id=m['config']['tracking']['config_id'],seed=m['seed'],horizon=m['horizon'],model=m['model'],files=[dict(path=str(q),sha256=ref.sha256(q)) for q in sorted(p.parent.iterdir()) if q.is_file()]) for p,m in selected]
    projectfiles=[dict(path=str(p),sha256=ref.sha256(p)) for p in sorted(PROJECT.rglob('*.py'))]
    manifest=dict(sources=sources,project=projectfiles,data=[dict(path=str(DATA/n),sha256=ref.sha256(DATA/n)) for n in ('scada_ecmwf_15min.csv','turbines.csv')],device=torch.cuda.get_device_name(0),seeds=[13,42,3407],runner_sha256=ref.sha256(Path(__file__)))
    if (OUT/'baseline_manifest.json').exists():
        assert json.loads((OUT/'baseline_manifest.json').read_text(encoding='utf-8'))==manifest
    else: dump(OUT/'baseline_manifest.json',manifest)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for (p,m),src in zip(selected,sources):
        start=time.perf_counter();h=m['horizon'];seed=m['seed'];name=m['model'];ref.HORIZON=h
        dest=OUT/'F2'/f'{name}_h{h}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'completed.json').exists(): continue
        saved=ref.load_npz(p.parent/'predictions.npz')
        assert int(saved['horizon'])==h==m['config']['experiment']['horizon']
        assert int(saved['seed'])==seed and m['status']=='completed' and (OUT/'G1_runs.json').is_file()
        cfg=deepcopy(m['config']);cfg['data']['scada_path']=str(DATA/'scada_ecmwf_15min.csv');cfg['data']['turbines_path']=str(DATA/'turbines.csv')
        assert cfg['train']['loss_name']=='huber' and cfg['train']['huber_delta']==.1
        record=dict(source_dir=p.parent,config=cfg,model=name,seed=seed,horizon=h)
        bundle=ev._bundle(record);fields=ref.index_fields(bundle);ref.same_indices(saved,fields);grid(fields)
        cap=np.asarray(bundle.capacity_mw)
        assert np.array_equal(cap,np.load(p.parent/'preprocessing.npz')['capacity_mw'])
        model,kind,_=ev._predictor(record,bundle,device)
        assert not model.training
        batch=next(iter(bundle.test_loader));torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): pred=ev._predict(model,kind,batch,torch.zeros_like(batch[0]),torch.zeros_like(batch[1]),device)
        actual={**ref.index_fields(bundle,len(batch[0])),'y_true_mw':batch[-1].numpy()*cap,'y_pred_mw':np.maximum(pred*cap,0)}
        gate=ref.normal_equivalence(saved,actual,cap,count=len(batch[0]))
        scenarios=scada_scenarios('structured_39');assert len(scenarios)==39
        for row in scenarios:
            scenario=dict(row);scenario.pop('name');scenario['sample_ids']=fields['origin_timestamp'][:len(batch[0])]
            corrupt,pm,wm=ev.apply_fault(batch,scenario,bundle.handler.scada_w_std,bundle.handler.coords,2026)
            for i in (2,3,4): assert torch.allclose(corrupt[i],batch[i],atol=0,rtol=0,equal_nan=True)
        gate.update(device=str(device),gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),all39_operators_checked=True)
        dump(dest/'cuda_gate.json',gate)
        truth,pred=ev._infer(model,kind,bundle,None,2026,'test',device)
        gate=ref.normal_equivalence(saved,{**fields,'y_true_mw':truth,'y_pred_mw':pred},cap)
        delta=np.abs((pred-saved['y_pred_mw'])/cap)
        allowed=1e-5+1e-5*np.abs(saved['y_pred_mw']/cap)
        assert np.isfinite(delta).all() and np.all(delta<=allowed)
        gate.update(all_prediction_cells=int(delta.size),all_cells_pass=True,all_cells_max_delta_pu=float(delta.max()))
        dump(dest/'normal_equivalence.json',gate)
        source=dict(config_id=src['config_id'],run_id=p.parent.name,checkpoint_sha256=ref.sha256(p.parent/'weights_best.pt'),source_manifest_sha256=ref.sha256(OUT/'baseline_manifest.json'))
        ref.save_prediction(dest/'normal.npz',fields,truth,pred,seed=seed,scenario='normal',capacity=cap,source=source)
        normal=ref.measured_metrics(truth,pred,fields['target_valid'],cap);rows=[dict(scenario='normal',**normal)]
        for i,original in enumerate(scenarios):
            scenario=dict(original);s=scenario.pop('name')
            ft,fp=ev._infer(model,kind,bundle,scenario,2026,'test',device)
            assert np.array_equal(ft,truth,equal_nan=True)
            met=ref.measured_metrics(ft,fp,fields['target_valid'],cap)
            ref.save_prediction(dest/f'{s}.npz',fields,ft,fp,seed=seed,scenario=s,capacity=cap,source=source,scenario_spec=scenario)
            rows.append(dict(scenario=s,scenario_json=scenario,**met,nrmse_increase=met['nrmse']/normal['nrmse']-1,nmae_increase=met['nmae']/normal['nmae']-1))
            print(f'{name} h{h} seed{seed}: {i+1}/39 {s}',flush=True)
        dump(dest/'completed.json',dict(status='completed',source=source,source_directory=str(p.parent),horizon=h,seed=seed,model=name,device=str(device),gpu=torch.cuda.get_device_name(0),runner_sha256=ref.sha256(Path(__file__)),rows=rows,elapsed_seconds=time.perf_counter()-start,trained=False))
        progress(f'F2 {name} h{h} seed{seed} 正常门禁及 39 场景完成。')
        del model,bundle;torch.cuda.empty_cache()
if __name__=='__main__':
    try: main()
    except Exception as e:
        dump(OUT/'F2_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))
        progress('F2 停止：'+repr(e));raise
