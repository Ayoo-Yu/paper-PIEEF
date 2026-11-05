from pathlib import Path
import os, sys, json, hashlib, time, importlib.util, copy
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LOKY_MAX_CPU_COUNT='2')
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent/'attention'
OUT.mkdir(exist_ok=True)
PLAN=Path(__file__).resolve().parents[2]
P=PLAN/'补做重算/physics'
PROJECT=PLAN/'next_batch/merged_v2/project'
DATA=Path('data/wind_farm_A')
sys.path.insert(0,str(PROJECT/'src'))
spec=importlib.util.spec_from_file_location('attention_eval',PROJECT/'scripts/evaluate_revision.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 started=time.time()
 protocol=dict(version=1,purpose='post-hoc explanatory attention diagnostics; no model selection or training',seeds=[13,21,42,87,3407],horizon_steps=16,horizon_minutes=240,split='test',batch_size=16,precision='FP32',proxy='sample standard deviation / mean of five raw nonnegative finite wind-speed observations ending at forecast origin, spanning 60 min',minimum_mean_wind_ms=3.0,threshold='90th percentile of eligible turbine-origin coefficients on chronological training rows only',strata=['event_free_low_variability','event_free_high_variability','candidate_low_power'],row_unit='query turbine and forecast origin; valid measured target required; candidate class from saved target event label',selection='all eligible rows; no downsampling',interpretation='15-min historical variability proxy, not IEC turbulence intensity; not confirmed curtailment',statistics='descriptive per-seed means and sample SD, no tests or CI',direction='Pearson correlation across 28 off-diagonal keys between attention and signed alignment with the actual model NWP vector, each query row')
 js('design_frozen.json',protocol)
 rows=[r for r in json.loads((P/'run_inventory.json').read_text(encoding='utf-8')) if r['horizon']==16]
 assert sorted(r['seed'] for r in rows)==protocol['seeds']
 raw=pd.read_csv(DATA/'scada_ecmwf_15min.csv')
 ts=pd.DatetimeIndex(pd.to_datetime(raw.Timestamp)).tz_localize('Asia/Shanghai').tz_convert('UTC')
 assert np.all(np.diff(ts.asi8)==pd.Timedelta(minutes=15).value)
 winds=raw[[f'turbine_ws{i+1}' for i in range(29)]].where(lambda x: x>=0)
 rollmean=winds.rolling(5,min_periods=5).mean().to_numpy()
 rolls=winds.rolling(5,min_periods=5).std(ddof=1).to_numpy()
 with np.errstate(invalid='ignore',divide='ignore'):cv=rolls/rollmean
 eligible=np.isfinite(cv)&(rollmean>=3.)
 train_end=int(len(raw)*.7)
 threshold=float(np.quantile(cv[:train_end][eligible[:train_end]],.9))
 js('proxy_definition.json',dict(threshold=threshold,training_end_row_exclusive=train_end,training_eligible_turbine_times=int(eligible[:train_end].sum()),raw_rows=len(raw),raw_step_minutes=15,has_raw_subminute_standard_deviation=False,available_columns=list(raw.columns),raw_sha256=sha(DATA/'scada_ecmwf_15min.csv')))
 sources=[];summ=[];gates=[];matrices={};bundle=None
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.55)
 for r in sorted(rows,key=lambda x:x['seed']):
  d=Path(r['path']);m=json.loads((d/'metrics.json').read_text(encoding='utf-8'));cfg=copy.deepcopy(m['config']);seed=r['seed']
  assert m['status']=='completed' and m['evaluation_split']=='test'
  assert cfg['tracking']['implementation_id']=='energy-r3-vf-F2F3F6'
  cfg['data'].update(scada_path=str(DATA/'scada_ecmwf_15min.csv'),turbines_path=str(DATA/'turbines.csv'))
  cfg['model']['manufacturer_curve_path']=str(DATA/'manufacturer_curve.csv')
  rec=dict(source_dir=d,config=cfg,seed=seed,horizon=16,model='PI-EEF')
  if bundle is None:bundle=ev._bundle(rec)
  z=dict(np.load(d/'predictions.npz',allow_pickle=False));meta=bundle.handler.metadata['test']
  for k in ('timestamp','origin_timestamp','target_valid','events'):
   a=np.asarray(meta[k]);b=z[k]
   assert np.array_equal(a.astype(str),b.astype(str)) if a.dtype.kind in 'OU' else np.array_equal(a,b),k
  assert np.array_equal(np.asarray(bundle.handler.turbine_ids).astype(str),z['turbine_ids'].astype(str))
  ix=ts.get_indexer(pd.to_datetime(z['origin_timestamp'],utc=True));assert (ix>=0).all()
  assert np.array_equal(ix+16,bundle.test_loader.dataset.indices)
  ok=eligible[ix]&z['target_valid'].astype(bool)
  high=cv[ix]>=threshold
  masks=dict(event_free_low_variability=ok&(z['events']=='normal')&~high,event_free_high_variability=ok&(z['events']=='normal')&high,candidate_low_power=ok&(z['events']=='low_power_candidate'))
  model,kind,_=ev._predictor(rec,bundle,device)
  before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  pred=[];truth=[];ent=[];mx=[];corr=[];offset=0
  ms={k:np.zeros((29,29),dtype=np.float64) for k in masks};mc={k:np.zeros(29,dtype=np.int64) for k in masks}
  offdiag=~np.eye(29,dtype=bool)
  with torch.no_grad():
   for bi,b in enumerate(bundle.test_loader):
    pp=ev._predict(model,kind,b,torch.zeros_like(b[0]),torch.zeros_like(b[1]),device)
    pred.append(pp);truth.append(b[-1].numpy())
    att=model.graph.last_attn_weights.cpu().double().numpy();bs=len(att)
    assert np.allclose(att.sum(-1),1,rtol=1e-6,atol=1e-6)
    e=-(att*np.log(np.maximum(att,1e-300))).sum(-1)/np.log(29);ent.append(e);mx.append(att.max(-1))
    target=b[3].to(device);wind=model.raw_nwp(target)[...,-2:].mean(1)
    w=torch.nn.functional.normalize(wind,dim=-1).cpu().double().numpy()
    rel=torch.nn.functional.normalize(model.rel_coords,dim=-1).cpu().double().numpy()
    align=np.einsum('bd,ijd->bij',w,rel)
    a=att[:,offdiag].reshape(bs,29,28);v=align[:,offdiag].reshape(bs,29,28)
    ac=a-a.mean(-1,keepdims=True);vc=v-v.mean(-1,keepdims=True)
    den=np.sqrt((ac*ac).sum(-1)*(vc*vc).sum(-1))
    co=np.divide((ac*vc).sum(-1),den,out=np.full_like(den,np.nan),where=den>1e-15);corr.append(co)
    for name,mask in masks.items():
     mm=mask[offset:offset+bs];ms[name]+=(att*mm[:,:,None]).sum(0);mc[name]+=mm.sum(0)
    offset+=bs
    if bi%150==0:print(f'seed={seed} batch={bi}/{len(bundle.test_loader)}',flush=True)
  preds=np.maximum(np.concatenate(pred)*bundle.capacity_mw,0);tr=np.concatenate(truth)*bundle.capacity_mw
  assert np.array_equal(tr,z['y_true_mw'],equal_nan=True)
  delta=np.abs(preds-z['y_pred_mw'])/bundle.capacity_mw;allowed=1e-5+1e-5*np.abs(z['y_pred_mw']/bundle.capacity_mw)
  assert np.all(delta<=allowed) and np.isfinite(preds).all()
  assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
  gates.append(dict(seed=seed,status='PASS',max_delta_pu=float(delta.max()),points=int(preds.size),indices_targets_capacity_exact=True,weights_unchanged=True))
  ee,xx,cc=np.concatenate(ent),np.concatenate(mx),np.concatenate(corr)
  for name,mask in masks.items():
   summ.append(dict(seed=seed,stratum=name,n_turbine_times=int(mask.sum()),n_origins=int(mask.any(1).sum()),entropy=float(ee[mask].mean()),max_weight=float(xx[mask].mean()),alignment_correlation=float(np.nanmean(cc[mask])),absolute_alignment_correlation=float(np.nanmean(np.abs(cc[mask])))))
   matrices[f'{name}_seed{seed}']=np.divide(ms[name],mc[name][:,None],out=np.full_like(ms[name],np.nan),where=mc[name][:,None]>0)
  np.savez_compressed(OUT/f'row_diagnostics_seed{seed}.npz',timestamp=z['timestamp'],origin_timestamp=z['origin_timestamp'],turbine_ids=z['turbine_ids'],events=z['events'],target_valid=z['target_valid'],history_cv=cv[ix],eligible=ok,high_variability=high,entropy=ee,max_weight=xx,alignment_correlation=cc)
  sources.append(dict(seed=seed,config_id=cfg['tracking']['config_id'],files=[dict(path=str(d/n),sha256=sha(d/n)) for n in ['metrics.json','predictions.npz','preprocessing.npz','weights_best.pt']]))
  pd.DataFrame(summ).to_csv(OUT/'attention_per_seed.csv',index=False,encoding='utf-8-sig');js('reproduction_gates.json',gates)
  del model,before
  if device.type=='cuda':torch.cuda.empty_cache()
 frame=pd.DataFrame(summ);group=[]
 for name,g in frame.groupby('stratum',sort=False):
  rr=dict(stratum=name,seeds=len(g),n_turbine_times_per_seed=int(g.iloc[0].n_turbine_times),n_origins_per_seed=int(g.iloc[0].n_origins))
  for metric in ['entropy','max_weight','alignment_correlation','absolute_alignment_correlation']:
   rr[metric+'_mean']=float(g[metric].mean());rr[metric+'_sd']=float(g[metric].std(ddof=1))
  group.append(rr)
 pd.DataFrame(group).to_csv(OUT/'attention_summary.csv',index=False,encoding='utf-8-sig')
 np.savez_compressed(OUT/'mean_attention_matrices.npz',**matrices)
 source_files=list((PROJECT/'src/pieef').rglob('*.py'))+[PROJECT/'scripts/evaluate_revision.py',Path(__file__),DATA/'scada_ecmwf_15min.csv',DATA/'turbines.csv']
 js('source_manifest.json',dict(runs=sources,files=[dict(path=str(p),sha256=sha(p)) for p in source_files]))
 js('completed.json',dict(status='completed',pid=os.getpid(),device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,seconds=time.time()-started,protocol_sha256=sha(OUT/'design_frozen.json'),results=group,gates=gates,training=False))
 print(json.dumps(group,indent=2),flush=True)
if __name__=='__main__': main()
