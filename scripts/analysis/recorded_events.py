from pathlib import Path
import os,sys,json,time,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[k]='2'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
L=Path(__file__).resolve().parent
import reference_helpers as ref
sys.path.insert(0,str(L/'source_project/src'))
from pieef.core import calc_metrics
from pieef.revision_data import _power_events
from pieef.data import _extract_power_cols
START=time.time();SOURCES={}
def js(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
def dump(n,x):ref.atomic_json(L/n,clean(x))
def record(p,expected=None):
 p=Path(p);h=ref.sha256(p)
 if expected:assert h==expected,str(p)
 SOURCES[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=h)
 return h
def records(d,c):
 for f in c['outputs']:
  q=d/f['path'];assert q.stat().st_size==f['bytes'];record(q,f['sha256'])
def event_metrics(a,mask):
 if not mask.any():return dict(nrmse_pct=None,nmae_pct=None,r2=None)
 ma,nr,r2=calc_metrics(np.where(mask,a['y_true_mw'],np.nan),a['y_pred_mw'],a['capacity_mw'])
 return dict(nrmse_pct=100*float(nr),nmae_pct=100*float(ma),r2=float(r2))
def save_csv(n,rows):pd.DataFrame(rows).to_csv(L/n,index=False,encoding='utf-8-sig')
def metric_fields(m):return dict(nrmse_pct=100*m['nrmse'],nmae_pct=100*m['nmae'],r2=m['r2'])
def aggregate(rows,keys,metrics):
 frame=pd.DataFrame(rows);result=[]
 for key,g in frame.groupby(keys,dropna=False,sort=True):
  key=key if isinstance(key,tuple) else (key,);r=dict(zip(keys,key))
  assert len(g)==3 and sorted(g.seed.tolist())==[13,42,3407]
  r.update(seeds='13/42/3407',n_seeds=3)
  for m in metrics:
   values=g[m]
   r[m+'_mean']=float(values.mean()) if values.notna().all() else None
   r[m+'_sd']=float(values.std(ddof=1)) if values.notna().all() else None
  result.append(r)
 return result
protocol=js(L/'protocol.json');execution=js(L/'execution_manifest.json')
record(L/'protocol.json',execution['protocol_sha256']);record(L/'execution_manifest.json');record(__file__)
operator_audit=js(L/'fault_operator_identity_audit.json');assert len(operator_audit)==6 and all(x['equal'] for x in operator_audit)
record(L/'fault_operator_identity_audit.json')
for row in operator_audit:
 record(row['manifest'],row['manifest_sha256']);assert ref.sha256(L/'source_project'/row['file'])==row['L1_sha256']==row['VF_sha256']
all_done=js(L/'all_inference_completed.json');assert all_done['status']=='completed' and all_done['fault_cells']==234 and all_done['gpu_released']
record(L/'all_inference_completed.json')
for f in protocol['source_files']+protocol['data_files']:record(f['path'],f['sha256'])
fault_rows=[];event_rows=[];identities=[];pair_gates=[];run_gates=[];event_baseline=None
scenario_specs={'normal':None,**{s['name']:{k:v for k,v in s.items() if k!='name'} for s in protocol['scenarios']}}
assert len(scenario_specs)==40
for task in protocol['sources']:
 h,s=task['horizon'],task['seed'];d=L/'runs'/task['task_id'];vf=Path(task['vf_directory'])
 c=js(d/'completed.json');v=js(vf/'completed.json')
 record(d/'completed.json');record(task['vf_completed']['path'],task['vf_completed']['sha256'])
 assert c['status']==v['status']=='completed' and c['process_exit']['returncode']==0
 assert c['protocol_sha256']==execution['protocol_sha256'] and c['execution_manifest_sha256']==ref.sha256(L/'execution_manifest.json')
 assert c['source']['config_id']==task['config_id'] and c['loss']=='l1' and c['trained']==v['trained']==False
 assert v['horizon']==h and v['seed']==s and v['evaluation_split']=='test' and v['fault_seed']==2026 and v['scenario_scope']=='structured_39'
 assert v['source']['config_id']==f'energy_r3s2_main_PI-EEF_h{h}_seed{s}' and len(c['rows'])==len(v['rows'])==40
 assert v['job_manifest_sha256']==next(x['manifest_sha256'] for x in operator_audit if x['VF_horizon']==f'h{h}')
 assert v['full_normal_equivalence']['status']=='PASS'
 records(d,c);records(vf,v)
 for f in task['files']:record(f['path'],f['sha256'])
 source=Path(task['directory']);vm=Path(v['source_directory_original'])
 assert vm.is_dir(),str(vm)
 cm=js(source/'metrics.json');vmeta=js(vm/'metrics.json')
 assert cm['status']==vmeta['status']=='completed'
 assert vmeta['config']['tracking']['config_id']==v['source']['config_id']
 for f in ('metrics.json','preprocessing.npz','data_protocol.json','predictions.npz','weights_best.pt'):
  record(vm/f,v['source']['checkpoint_sha256'] if f=='weights_best.pt' else None)
 with np.load(source/'preprocessing.npz',allow_pickle=False) as z:ccap=z['capacity_mw'].copy()
 with np.load(vm/'preprocessing.npz',allow_pickle=False) as z:vcap=z['capacity_mw'].copy()
 assert np.array_equal(ccap,vcap)
 assert js(source/'data_protocol.json')==cm['config']['data']['revision_protocol']
 assert js(vm/'data_protocol.json')==vmeta['config']['data']['revision_protocol']
 for key in ('revision_protocol','seq_len','split','capacity_mw'):
  assert cm['config']['data'][key]==vmeta['config']['data'][key],key
 assert cm['config']['train']['batch_size']==vmeta['config']['train']['batch_size']==16
 identities.append(dict(horizon=h,horizon_minutes=h*15,seed=s,ACFormer_L1_directory=str(source),ACFormer_L1_config_id=task['config_id'],ACFormer_L1_checkpoint_sha256=c['source']['checkpoint_sha256'],VF_directory=str(vm),VF_config_id=v['source']['config_id'],VF_checkpoint_sha256=v['source']['checkpoint_sha256'],source_protocol_same=True,ACFormer_L1_producer=task['producer'],fault_outputs_L1=str(d),fault_outputs_VF=str(vf)))
 gate=js(d/'normal_equivalence.json');assert gate['status']=='PASS' and gate['all_cells_pass'] and gate['vf_pair_identity']=='PASS'
 run_gates.append(dict(task_id=task['task_id'],status='PASS',normal_gate=gate,process_exit=c['process_exit'],gpu=c['gpu'],peak_allocated_bytes=c['peak_allocated_bytes'],elapsed_seconds=c['process_exit']['elapsed_seconds']))
 normal_metrics={}
 for name,spec in scenario_specs.items():
  a=ref.load_npz(d/(name+'.npz'));b=ref.load_npz(vf/(name+'.npz'))
  checks={k:bool(np.array_equal(a[k],b[k])) for k in ('timestamp','origin_timestamp','turbine_ids','target_valid','events','capacity_mw')}
  checks['truth']=bool(np.array_equal(a['y_true_mw'],b['y_true_mw'],equal_nan=True));assert all(checks.values())
  assert np.array_equal(a['capacity_mw'],ccap)
  for arr,completed,model in ((a,c,'ACFormer_L1'),(b,v,'VF')):
   assert int(arr['seed'])==s and int(arr['horizon'])==h and int(arr['fault_seed'])==2026
   assert str(arr['source_config_id'])==completed['source']['config_id'] and str(arr['checkpoint_sha256'])==completed['source']['checkpoint_sha256']
   assert json.loads(str(arr['scenario_json']))==spec and str(arr['scenario'])==name
   assert np.array_equal(np.isfinite(arr['y_true_mw']),arr['target_valid']) and np.isfinite(arr['y_pred_mw']).all()
   ts=pd.to_datetime(arr['timestamp'],utc=True);ot=pd.to_datetime(arr['origin_timestamp'],utc=True)
   assert np.all(ts-ot==pd.Timedelta(minutes=h*15)) and ts.is_unique and np.all(np.diff(ts.asi8)==15*60*10**9)
   m=ref.measured_metrics(arr['y_true_mw'],arr['y_pred_mw'],arr['target_valid'],arr['capacity_mw'])
   saved=next(x for x in completed['rows'] if x['scenario']==name)
   for key in ('nrmse','nmae','r2'):assert np.isclose(m[key],saved[key],atol=1e-10,rtol=1e-10),(model,h,s,name,key)
   if name=='normal':normal_metrics[model]=m
   fault_rows.append(dict(model=model,site='wind_farm_A',horizon_steps=h,horizon_minutes=h*15,seed=s,evaluation_split='test',scenario=name,n_targets=m['n'],**metric_fields(m),nrmse_relative_degradation_pct=100*(m['nrmse']/normal_metrics[model]['nrmse']-1),nmae_relative_degradation_pct=100*(m['nmae']/normal_metrics[model]['nmae']-1),prediction_file=str((d if model=='ACFormer_L1' else vf)/(name+'.npz'))))
   if h==16 and name=='normal':
    for label in ('all_valid',)+tuple(sorted(np.unique(arr['events']).tolist())):
     mask=arr['target_valid'] if label=='all_valid' else arr['target_valid']&(arr['events']==label)
     event_rows.append(dict(model=model,site='wind_farm_A',horizon_minutes=240,seed=s,evaluation_split='test',event=label,n_targets=int(mask.sum()),n_times=int(mask.any(axis=1).sum()),n_labelled_cells=int(mask.size) if label=='all_valid' else int((arr['events']==label).sum()),**event_metrics(arr,mask),prediction_file=str((d if model=='ACFormer_L1' else vf)/'normal.npz')))
    if event_baseline is None:event_baseline=a
  pair_gates.append(dict(horizon=h,seed=s,scenario=name,status='PASS',checks=checks,finite_all_prediction_cells=True,scenario_spec_exact=True))
# Independently reproduce saved event labels from the original observed series.
raw=Path(protocol['data_files'][0]['path']);frame=pd.read_csv(raw)
power=frame[_extract_power_cols(frame)].to_numpy(dtype=np.float32)/1000
wind=frame[[f'turbine_ws{i+1}' for i in range(29)]].to_numpy(dtype=np.float32)
valid,labels=_power_events(power,wind,int(len(frame)*0.7),2.)
ts=pd.DatetimeIndex(pd.to_datetime(frame.Timestamp));ts=ts.tz_localize('Asia/Shanghai') if ts.tz is None else ts
ix=ts.tz_convert('UTC').get_indexer(pd.to_datetime(event_baseline['timestamp'],utc=True));assert (ix>=0).all()
assert np.array_equal(valid[ix],event_baseline['target_valid']) and np.array_equal(labels[ix],event_baseline['events'])
assert np.allclose(np.where(valid[ix],power[ix],np.nan),event_baseline['y_true_mw'],atol=1e-7,rtol=0,equal_nan=True)
assert (ix>=int(len(frame)*0.8)).all()
metrics=['nrmse_pct','nmae_pct','r2','nrmse_relative_degradation_pct','nmae_relative_degradation_pct']
fault_groups=aggregate(fault_rows,['model','horizon_steps','horizon_minutes','scenario'],metrics)
perseed_overall=[]
for (model,h,s),g in pd.DataFrame(fault_rows).query("scenario != 'normal'").groupby(['model','horizon_steps','seed']):
 assert len(g)==39
 perseed_overall.append(dict(model=model,horizon_steps=h,horizon_minutes=h*15,seed=s,n_scenarios=39,**{m:float(g[m].mean()) for m in metrics}))
overall_groups=aggregate(perseed_overall,['model','horizon_steps','horizon_minutes'],metrics)
def pair_rows(rows,keys):
 f=pd.DataFrame(rows);pairs=[]
 for key,g in f.groupby(keys,sort=True,dropna=False):
  key=key if isinstance(key,tuple) else (key,);v=g[g.model=='VF'].iloc[0];b=g[g.model=='ACFormer_L1'].iloc[0];r=dict(zip(keys,key))
  assert len(g)==2
  for m in ['nrmse_pct','nmae_pct','r2']:
   r[m+'_VF_minus_L1']=float(v[m]-b[m]) if pd.notna(v[m]) and pd.notna(b[m]) else None
   if m!='r2':r[m+'_VF_relative_improvement_pct']=float(100*(1-v[m]/b[m])) if pd.notna(v[m]) and pd.notna(b[m]) and b[m]!=0 else None
  pairs.append(r)
 return pairs
fault_pairs=pair_rows(fault_rows,['horizon_steps','horizon_minutes','seed','scenario'])
pair_metrics=[k for k in fault_pairs[0] if '_VF_' in k]
fault_pair_groups=aggregate(fault_pairs,['horizon_steps','horizon_minutes','scenario'],pair_metrics)
event_groups=aggregate(event_rows,['model','event'],['nrmse_pct','nmae_pct','r2'])
for g in event_groups:
 r=[r for r in event_rows if r['model']==g['model'] and r['event']==g['event']]
 assert len({r['n_targets'] for r in r})==1
 g.update(n_targets_per_seed=r[0]['n_targets'],n_times_per_seed=r[0]['n_times'],n_labelled_cells_per_seed=r[0]['n_labelled_cells'])
event_pairs=pair_rows(event_rows,['seed','event'])
event_pair_groups=aggregate(event_pairs,['event'],pair_metrics)
comparisons=[]
for h in (16,96):
 v=[g for g in fault_groups if g['model']=='VF' and g['horizon_steps']==h];b={g['scenario']:g for g in fault_groups if g['model']=='ACFormer_L1' and g['horizon_steps']==h}
 dct=dict(horizon_steps=h,horizon_minutes=h*15,n_scenarios=39)
 for m in ('nrmse_pct','nmae_pct'):
  wins=[g['scenario'] for g in v if g['scenario']!='normal' and g[m+'_mean']<b[g['scenario']][m+'_mean']]
  losses=[g['scenario'] for g in v if g['scenario']!='normal' and g[m+'_mean']>b[g['scenario']][m+'_mean']]
  dct[m+'_VF_better_scenarios']=len(wins);dct[m+'_VF_worse_scenarios']=len(losses);dct[m+'_VF_worse_scenario_names']=losses
  for scope,gg in [('normal',fault_groups),('fault39_mean',overall_groups)]:
   vg=next(g for g in gg if g['model']=='VF' and g['horizon_steps']==h and (scope!='normal' or g['scenario']=='normal'))
   bg=next(g for g in gg if g['model']=='ACFormer_L1' and g['horizon_steps']==h and (scope!='normal' or g['scenario']=='normal'))
   dct[scope+'_'+m+'_relative_improvement_of_group_means_pct']=100*(1-vg[m+'_mean']/bg[m+'_mean'])
 comparisons.append(dct)
tables={'fault_per_run.csv':fault_rows,'fault_group_results.csv':fault_groups,'fault39_mean_per_seed.csv':perseed_overall,'fault39_mean_group.csv':overall_groups,'fault_matched_seed_differences.csv':fault_pairs,'fault_matched_group_differences.csv':fault_pair_groups,'event_per_run.csv':event_rows,'event_group_results.csv':event_groups,'event_matched_seed_differences.csv':event_pairs,'event_matched_group_differences.csv':event_pair_groups}
for name,rows in tables.items():save_csv(name,rows)
dump('source_identity.json',identities);dump('comparisons.json',comparisons)
dump('results.json',dict(fault_per_run=fault_rows,fault_groups=fault_groups,fault39_per_seed=perseed_overall,fault39_groups=overall_groups,fault_pairs=fault_pairs,fault_pair_groups=fault_pair_groups,event_per_run=event_rows,event_groups=event_groups,event_pairs=event_pairs,event_pair_groups=event_pair_groups,comparisons=comparisons))
for f in SOURCES.values():assert ref.sha256(f['path'])==f['sha256']
dump('source_files_sha256.json',list(SOURCES.values()));save_csv('source_files_sha256.csv',list(SOURCES.values()))
dump('completion_gate.json',dict(status='PASS',source_runs=6,fault_cells=234,normal_runs=6,VF_reused_fault_cells=234,VF_reused_normal_runs=6,run_gates=run_gates,full_pair_gates=pair_gates,event_raw_reconstruction='PASS',all_source_hashes_unchanged=True,statistics='3 common seeds; descriptive mean and sample SD only, no new tests or CI/BH',fault39_mean_definition='Mean of 39 scenario metrics within each seed, followed by the mean and sample SD across the three seeds; not pooled independent replicates',event_counts='Counts per seed; identical targets are not multiplied by three',fault_metric=str(L/'reference_helpers.py'),event_metric=str(L/'source_project/src/pieef/core.py'),cpu_threads=2,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],elapsed_seconds=time.time()-START))
print(json.dumps(dict(status='PASS',comparisons=comparisons,event_groups=event_groups,output=str(L)),ensure_ascii=False))
