from pathlib import Path
import os,sys,json,time,hashlib,importlib.util,copy,platform,argparse
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(Path(__file__).parent/'mplconfig'))
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PROJECT=ROOT/'next_batch/merged_v2/project'
DATA=Path('data/wind_farm_A')
sys.path.insert(0,str(PROJECT/'src'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
ev=module('p_evaluation',PROJECT/'scripts/evaluate_revision.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 return x
def js(name,x): (OUT/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def progress(s):
 print(s,flush=True)
 with (OUT/'进度.md').open('a',encoding='utf-8') as f:f.write('\n- '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+s+'\n')
def quant(x):
 a=np.asarray(x,dtype=float).reshape(-1);a=a[np.isfinite(a)]
 return dict(n=len(a),**dict(zip(['min','p05','p50','p95','max'],np.quantile(a,[0,.05,.5,.95,1]) if len(a) else [np.nan]*5)))
def records(): return json.loads((OUT/'run_inventory.json').read_text(encoding='utf-8'))
def record(r):
 d=Path(r['path']);m=json.loads((d/'metrics.json').read_text(encoding='utf-8'));c=copy.deepcopy(m['config'])
 assert m['status']=='completed' and m['evaluation_split']=='test'
 assert c['tracking']['implementation_id']=='energy-r3-vf-F2F3F6'
 assert c['experiment']['horizon']==r['horizon'] and c['experiment']['seeds']==[r['seed']]
 c['data']['scada_path']=str(DATA/'scada_ecmwf_15min.csv');c['data']['turbines_path']=str(DATA/'turbines.csv')
 c['model']['manufacturer_curve_path']=str(DATA/'manufacturer_curve.csv')
 return dict(source_dir=d,config=c,seed=r['seed'],horizon=r['horizon'],model='PI-EEF'),m
def identity(r,bundle,saved):
 h=r['horizon'];s=r['seed'];assert int(saved['horizon'])==h and int(saved['seed'])==s
 fields={k:np.asarray(v).astype(str) if np.asarray(v).dtype.kind=='O' else np.asarray(v) for k,v in bundle.handler.metadata['test'].items()}
 fields['turbine_ids']=np.asarray(bundle.handler.turbine_ids).astype(str)
 for k in ('timestamp','origin_timestamp','turbine_ids','target_valid','events'):assert np.array_equal(fields[k],saved[k]),k
 ts=pd.to_datetime(fields['timestamp'],utc=True);orig=pd.to_datetime(fields['origin_timestamp'],utc=True)
 assert np.all(ts-orig==pd.Timedelta(minutes=15*h)) and ts.is_unique and ts.is_monotonic_increasing
 cap=np.asarray(bundle.capacity_mw)
 with np.load(Path(r['path'])/'preprocessing.npz') as z:assert np.array_equal(cap,z['capacity_mw'])
 assert np.array_equal(cap,np.asarray(r.get('capacity',2.0)))
 return fields
def provenance():
 files=list((PROJECT/'src/pieef').rglob('*.py'))+[PROJECT/'scripts/evaluate_revision.py',ROOT/'修复重算_20261021/statistics/corrected_statistics.py',ROOT/'修复重算_20261021/statistics/revision_statistics_frozen.py']
 js('source_manifest.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files])
 js('audit_metadata.json',dict(status='running',device='CPU',cpu=platform.processor(),torch=torch.__version__,python=platform.python_version(),source_root=str(PROJECT),source_manifest_sha256=sha(OUT/'source_manifest.json'),seeds=[13,21,42,87,3407],started=time.strftime('%Y-%m-%d %H:%M:%S'),design_sha256=sha(OUT/'design_fixed.json')))
 js('input_manifest.json',[dict(run=r,files=[dict(path=str(Path(r['path'])/n),sha256=sha(Path(r['path'])/n)) for n in ['metrics.json','weights_best.pt','predictions.npz','preprocessing.npz','loss_history.json']]) for r in records()])
def curves(model,r,rho,ids):
 seed,h=r['seed'],r['horizon'];n=len(ids);v=torch.linspace(0,30,301).reshape(-1,1).expand(-1,n).clone().requires_grad_(True)
 base=model.dcn.power_curve(v);der=torch.autograd.grad(base.sum(),v)[0];cp=model.dcn.power_curve.get_control_points().detach().numpy();cap=model.dcn.power_curve.rated_power.clamp(.8,1).detach().numpy().ravel()
 bv=base.detach().numpy();dv=der.detach().numpy();vv=v.detach().numpy()[:,0]
 levels=np.quantile(rho,[.05,.5,.95]);full=[]
 with torch.no_grad():
  for lev in levels:full.append(model.dcn.power_at(v,torch.full_like(v,float(lev))).squeeze(-1).numpy())
 rows=[];summary=[]
 for j,tid in enumerate(ids):
  at100=np.flatnonzero(bv[:,j]>=1);atcap=np.flatnonzero(bv[:,j]>=cap[j]-1e-6)
  summary.append(dict(horizon=h,seed=seed,turbine=tid,negative_derivative_pct=100*np.mean(dv[:,j]<0),min_derivative=float(dv[:,j].min()),negative_forward_difference_pct=100*np.mean(np.diff(bv[:,j])<0),rated100_wind_ms=float(vv[at100[0]]) if len(at100) else None,cap_wind_ms=float(vv[atcap[0]]) if len(atcap) else None,cap_pct=100*cap[j],curve30_pct=100*bv[-1,j],cp_max_pct=100*cp[j].max()))
  for i,x in enumerate(vv):rows.append(dict(horizon=h,seed=seed,turbine=tid,wind_ms=x,base_pct=100*bv[i,j],derivative_pct_per_ms=100*dv[i,j],rho_p05=levels[0],rho_p50=levels[1],rho_p95=levels[2],full_p05_pct=100*full[0][i,j],full_p50_pct=100*full[1][i,j],full_p95_pct=100*full[2][i,j]))
 csv(f'curves_h{h}_s{seed}.csv',rows);csv(f'curve_summary_h{h}_s{seed}.csv',summary)
 return summary
def p14():
 t=time.perf_counter();provenance()
 rows=[r for r in records() if r['horizon']==16];assert sorted(r['seed'] for r in rows)==[13,21,42,87,3407]
 wind_farm_A=pd.read_csv(DATA/'scada_ecmwf_15min.csv');alltimes=pd.DatetimeIndex(pd.to_datetime(wind_farm_A.Timestamp)).tz_localize('Asia/Shanghai').tz_convert('UTC')
 turbine_meta=pd.read_csv(DATA/'turbines.csv');allwind=wind_farm_A[[f'turbine_ws{i+1}' for i in range(29)]].to_numpy(np.float32)
 bundle=None;ref=None; summaries=[];winds=[];p2=[];p3=[];curveall=[]
 for r in sorted(rows,key=lambda x:x['seed']):
  seed=r['seed'];rec,meta=record(r)
  if bundle is None:bundle=ev._bundle(rec)
  else:
   with np.load(Path(r['path'])/'preprocessing.npz') as pre:
    for k,val in [('nwp_mean',bundle.handler.nwp_mean),('nwp_std',bundle.handler.nwp_std),('capacity_mw',bundle.capacity_mw),('scada_wind_mean',bundle.handler.scada_w_mean),('scada_wind_std',bundle.handler.scada_w_std)]:assert np.array_equal(pre[k],np.asarray(val)),k
  saved=dict(np.load(Path(r['path'])/'predictions.npz',allow_pickle=False));fields=identity(r,bundle,saved)
  if ref is not None:
   for k in ['timestamp','origin_timestamp','turbine_ids','target_valid','events','y_true_mw']:assert np.array_equal(ref[k],saved[k],equal_nan=True) if saved[k].dtype.kind=='f' else np.array_equal(ref[k],saved[k]),k
  ref=saved
  model,kind,_=ev._predictor(rec,bundle,torch.device('cpu'));pred=[];truth=[];vc=[];raw=[];rho=[];delta=[];pp=[];p0=[];t1=time.perf_counter()
  with torch.no_grad():
   for bi,b in enumerate(bundle.test_loader):
    pred.append(ev._predict(model,kind,b,torch.zeros_like(b[0]),torch.zeros_like(b[1]),torch.device('cpu')));truth.append(b[-1].numpy())
    v,p,d=model.forward_stage2_training(b[3]);vc.append(v.squeeze(-1).numpy());pp.append(p.squeeze(-1).numpy());delta.append(d.squeeze(-1).numpy());raw.append(model._raw_target_wind(b[3]).numpy());rho.append(model.density_ratio(b[3]).numpy());p0.append(model.dcn.power_at(v.squeeze(-1),torch.ones_like(v.squeeze(-1))).squeeze(-1).numpy())
    if bi%100==0:print(f'seed {seed} normal batch {bi}/{len(bundle.test_loader)} seconds {time.perf_counter()-t1:.1f}',flush=True)
  pred=np.maximum(np.concatenate(pred)*bundle.capacity_mw,0);truth=np.concatenate(truth)*bundle.capacity_mw
  assert np.array_equal(truth,saved['y_true_mw'],equal_nan=True)
  old=saved['y_pred_mw'];error=np.abs(pred-old)/bundle.capacity_mw;allowed=1e-5+1e-5*np.abs(old/bundle.capacity_mw)
  assert np.isfinite(pred).all() and np.all(error<=allowed),'normal reproduction failed'
  gate=dict(seed=seed,status='PASS',all_points=pred.size,valid_points=int(fields['target_valid'].sum()),max_abs_delta_pu=error.max(),max_tolerance_fraction=(error/allowed).max(),indices_truth_capacity_exact=True,horizon_minutes=240,seconds=time.perf_counter()-t1,batch_size=16,device='CPU')
  js(f'normal_equivalence_s{seed}.json',gate);summaries.append(gate)
  vc,raw,rho,delta,pp,p0=[np.concatenate(x) for x in [vc,raw,rho,delta,pp,p0]]
  ti=alltimes.get_indexer(pd.to_datetime(fields['timestamp'],utc=True));assert (ti>=0).all()
  assert np.array_equal(ti,bundle.test_loader.dataset.indices)
  ids=fields['turbine_ids'];assert np.array_equal(turbine_meta['turbine_id'].astype(str).to_numpy(),ids)
  obs=allwind[ti];valid=np.isfinite(obs)&(obs>=0)
  np.savez_compressed(OUT/f'physical_samples_s{seed}.npz',**fields,y_true_mw=truth,y_pred_mw=pred,capacity_mw=bundle.capacity_mw,scada_wind=obs,wind_valid=valid,v_cal=vc,raw_nwp=raw,rho_ratio=rho,delta_v=delta,p_phys=pp,p_phys_rho1=p0,seed=seed,horizon=16)
  for scope,mask in [('all',np.ones_like(valid)),('valid_power',fields['target_valid'])]:
   for name,a in [('rho_ratio',rho),('equivalent_wind_factor',rho**(1/3)),('power_effect_capacity_pp',100*(pp-p0))]:p2.append(dict(seed=seed,scope=scope,quantity=name,**quant(a[mask])))
   p2.append(dict(seed=seed,scope=scope,quantity='power_relative_change_pct_positive_denominator',zero_denominator=int((mask&(p0==0)).sum()),**quant(100*(pp[mask&(p0>0)]/p0[mask&(p0>0)]-1))))
   p3.append(dict(seed=seed,scope=scope,**quant(vc[mask]),above22_pct=100*np.mean(vc[mask]>22),above25_pct=100*np.mean(vc[mask]>25),gate_lt099_pct=100*np.mean(vc[mask]>=22),gate_lt05_pct=100*np.mean(vc[mask]>=22),below_cut_in_pct=100*np.mean(vc[mask]<3)))
  scopes=[('all','all',valid)]+[('turbine',tid,valid & (np.arange(len(ids))[None,:]==j)) for j,tid in enumerate(ids)]+[('wind_bin',label,valid&m) for label,m in [('<3',obs<3),('[3,10)',(obs>=3)&(obs<10)),('[10,15]',(obs>=10)&(obs<=15)),('>15',obs>15)]]
  for st,label,mask in scopes:
   for method,a in [('raw_nwp',raw),('v_cal',vc)]:
    e=(a-obs)[mask].astype(float);winds.append(dict(seed=seed,scope=st,label=label,method=method,n=len(e),rmse_ms=np.sqrt(np.mean(e**2)) if len(e) else np.nan,mae_ms=np.mean(np.abs(e)) if len(e) else np.nan,bias_ms=np.mean(e) if len(e) else np.nan))
  curveall.extend(curves(model,r,rho,ids))
  js(f'runtime_s{seed}.json',dict(reference_density=model.reference_density,pressure_column=bundle.handler.feature_columns[model.nwp_pressure_index],temperature_column=bundle.handler.feature_columns[model.nwp_temperature_index],wind_column=bundle.handler.feature_columns[model.nwp_target_speed_index],cut_in=model.cut_in_speed,cut_out=model.cut_out_speed,config=meta['config'],source_dir=r['path']))
  progress(f'P1–P4：h16 seed {seed} 完整正常复现 PASS；曲线、密度、切出和风速误差已计算。')
 csv('P1_curve_summary.csv',curveall);csv('P2_density_summary.csv',p2);csv('P3_cutout_summary.csv',p3);csv('P4_wind_metrics.csv',winds);js('P14_completed.json',dict(seconds=time.perf_counter()-t,gates=summaries))
 # Cheap direct curves for all other main horizons, using their own actual target NWP indices.
 for r in records():
  if r['horizon']==16:continue
  rec,_=record(r);model,_,_=ev._predictor(rec,bundle,torch.device('cpu'))
  saved=dict(np.load(Path(r['path'])/'predictions.npz',allow_pickle=False))
  assert int(saved['horizon'])==r['horizon'] and int(saved['seed'])==r['seed']
  ti=alltimes.get_indexer(pd.to_datetime(saved['timestamp'],utc=True));assert (ti>=0).all()
  target=bundle.test_loader.dataset.future_nwp[ti].unsqueeze(1).expand(-1,len(ids),-1)
  with torch.no_grad():rr=model.density_ratio(target).numpy()
  curveall.extend(curves(model,r,rr,ids))
 csv('P1_curve_summary.csv',curveall);progress('P1 完成：五个提前量 × 五个种子的直接曲线已计算；P2–P4 五种子全测试样本完成。')
if __name__=='__main__':p14()
