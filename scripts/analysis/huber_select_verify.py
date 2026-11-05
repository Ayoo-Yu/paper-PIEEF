from pathlib import Path
import sys, json, time, hashlib, shutil
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent; R=O.parents[1]; S=R/'修复重算_20261021/statistics'
sys.path.insert(0,str(S))
import corrected_statistics as cs
SEEDS=(13,42,3407)
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(n,x):(O/n).write_text(json.dumps(cs.clean(x),ensure_ascii=False,indent=2),encoding='utf-8')
def csv(n,x):pd.DataFrame(x).to_csv(O/n,index=False,encoding='utf-8-sig')
def progress(s):
 with (O/'进度.md').open('a',encoding='utf-8') as f:f.write('\n- '+cs.utc()+' '+s+'\n')
def run(p):
 m=js(p/'metrics.json'); c=m['config'];e=c['experiment']
 return cs.Run(p,m,c,int(m['seed']),int(m['horizon']),m.get('model',e.get('model')),e.get('group',''),e.get('variant',''))
def pairing(a,b):
 cs.assert_paired(a,b)
 with np.load(a.run.directory/'predictions.npz') as x,np.load(b.run.directory/'predictions.npz') as y:assert np.array_equal(x['events'],y['events'])
def main():
 start=time.time();progress('开始G1：仅使用completed.json登记的attempt；G2各损失8项BH已预定义。')
 vf={}
 for p in (R/'补做重算/inputs/round3/stage2').rglob('metrics.json'):
  m=js(p);cid=m['config']['tracking']['config_id']
  if cid.startswith('energy_r3s2_main_PI-EEF_'):
   key=(m['horizon'],m['seed']);assert key not in vf;vf[key]=cs.load_saved(run(p.parent))
 assert len(vf)==25
 roots=[R/'补做重算/s3_retrain'/p for p in ('job_A_workstation_results','job_A_local_results','job_B_local/results')]
 new={};audit=[];files=[];excluded=[]
 for root in roots:
  for cp in sorted(root.rglob('completed.json')):
   c=js(cp);parents={str(Path(e['path']).parent) for e in c['files']};assert len(parents)==1
   d=cp.parent/parents.pop();assert d.resolve().is_relative_to(root.resolve())
   for x in c['files']:
    p=cp.parent/x['path'];assert p.stat().st_size==x['bytes'] and cs.sha(p)==x['sha256'],str(p)
    files.append(dict(path=str(p),**{k:x[k] for k in ('bytes','sha256')}))
   excluded.extend(str(p) for p in cp.parent.glob('attempt_*') if p!=d)
   rr=run(d);m=rr.metadata;t=rr.config['train'];summ=m['training_summary'];hist=pd.read_csv(d/'epoch_losses.csv');ex=js(d/'execution.json');dev=js(d/'device.json')
   assert t['epochs']==50 and t['batch_size']==16 and t['patience']==5 and t['min_delta']==0
   assert t['loss_name']==summ['loss_name']=='huber' and t['huber_delta']==summ['huber_delta']==.1
   assert ex['status']=='completed' and ex['returncode']==0 and c['config_id']==rr.config['tracking']['config_id']
   n=len(hist);best=int(hist.val_loss.idxmin()+1)
   assert list(hist.epoch)==list(range(1,n+1)) and n==summ['epochs_run']<50
   assert best==summ['best_epoch']==int(hist.iloc[-1].best_epoch) and n-best==5
   assert np.isclose(hist.val_loss.min(),summ['best_validation_loss'],rtol=0,atol=1e-15)
   prov=js(d/'provenance.json');assert prov['source_manifest_sha256']==cs.sha(d/'source_manifest.json')==m['source_manifest_sha256']
   assert prov['config_sha256']==cs.sha(d/'config.yaml') and prov['package_manifest_sha256']==c['package_manifest_sha256']
   sv=cs.load_saved(rr);pairing(vf[(rr.horizon,rr.seed)],sv);key=(rr.model,rr.horizon,rr.seed);assert key not in new;new[key]=sv
   row=dict(model=rr.model,horizon=rr.horizon,seed=rr.seed,directory=str(d),completed=str(cp),completed_sha256=cs.sha(cp),source_sha256=m['source_manifest_sha256'],device=dev,epochs=n,best_epoch=best,last_val=float(hist.iloc[-1].val_loss),previous_val=float(hist.iloc[-2].val_loss),last_change_percent=100*(hist.iloc[-1].val_loss/hist.iloc[-2].val_loss-1),early_stop=True,hashes_pass=True,identity_pass=True,n_time=len(sv.timestamp),n_valid=int(sv.valid.sum()),elapsed_training_seconds=ex['elapsed_s'])
   audit.append(row)
 assert len(new)==21
 dump('G1_runs.json',audit);csv('G1_runs.csv',audit);csv('G1_verified_files.csv',files);dump('excluded_attempts.json',excluded)
 dump('selected_retrained_runs.json',[dict(model=k[0],horizon=k[1],seed=k[2],directory=str(v.run.directory)) for k,v in new.items()])
 progress('G1完成：21/21 SHA、预算、Huber参数、逐项身份通过；全部早停且未满50轮。')
 oldroot=R/'修复重算_20261021/baselines/candidate_project/artifacts/energy_baseline_repair_20261021'
 old={}
 for p in oldroot.rglob('metrics.json'):
  rr=run(p.parent);key=(rr.model,rr.horizon,rr.seed)
  if (rr.model,rr.horizon) in cs.S3_CELLS and rr.seed in SEEDS:
   assert key not in old;old[key]=cs.load_saved(rr)
 assert len(old)==24
 for s in SEEDS:new[('LightGBM',48,s)]=old[('LightGBM',48,s)]
 rows=[];tests=[];per=[];evidence=[]
 for model,h in cs.S3_CELLS:
  row=dict(model=model,horizon=h,seeds=list(SEEDS),analysis_device='CPU')
  for s in SEEDS:
   v,b,o=vf[(h,s)],new[(model,h,s)],old[(model,h,s)];pairing(v,b);pairing(v,o)
   for side,x in [('VF',v),('new',b),('old',o)]:
    evidence.append(x.evidence);per.append(dict(model=model,horizon=h,seed=s,side=side,directory=str(x.run.directory),**x.metric))
  for metric in ('nrmse','nmae','r2'):
   scale=1 if metric=='r2' else 100
   for side,vals in [('VF',[vf[(h,s)].metric[metric] for s in SEEDS]),('new',[new[(model,h,s)].metric[metric] for s in SEEDS]),('old',[old[(model,h,s)].metric[metric] for s in SEEDS])]:
    row[side+'_'+metric+'_mean']=scale*np.mean(vals);row[side+'_'+metric+'_sd']=scale*np.std(vals,ddof=1)
   row[metric+'_new_minus_old']=row['new_'+metric+'_mean']-row['old_'+metric+'_mean']
   if metric!='r2':
    row[metric+'_VF_improvement_percent']=100*(1-row['VF_'+metric+'_mean']/row['new_'+metric+'_mean'])
    row[metric+'_old_VF_improvement_percent']=100*(1-row['VF_'+metric+'_mean']/row['old_'+metric+'_mean'])
    row[metric+'_baseline_change_percent']=100*(row['new_'+metric+'_mean']/row['old_'+metric+'_mean']-1)
  rows.append(row)
  for fam,attr in [('mse','mse_time'),('mae','mae_time')]:
   a=np.array([getattr(vf[(h,s)],attr) for s in SEEDS]);b=np.array([getattr(new[(model,h,s)],attr) for s in SEEDS])
   t=cs.inference(b.mean(axis=0),a.mean(axis=0),h);t.update(model=model,horizon=h,family=fam,seeds=list(SEEDS),direction='VF_minus_baseline')
   t.update({'seed_'+k:v for k,v in cs.seed_supplement((a.mean(axis=1)-b.mean(axis=1)).tolist()).items()});tests.append(t)
 for fam in ('mse','mae'):
  tt=[t for t in tests if t['family']==fam];assert len(tt)==8
  for p,q in [('p_value','time_q_BH8'),('seed_paired_t_p_raw','seed_t_q_BH8'),('seed_exact_sign_p_two_sided_raw','seed_sign_q_BH8')]:
   for t,v in zip(tt,cs.established.bh_adjust([t[p] for t in tt])):t[q]=v
 csv('G2_summary.csv',rows);dump('G2_summary.json',rows);csv('G2_tests.csv',tests);dump('G2_tests.json',tests);csv('G2_per_seed.csv',per);dump('G12_identity_and_sources.json',evidence)
 dump('G12_receipt.json',dict(elapsed_seconds=time.time()-start,finished_utc=cs.utc(),source_sha256={str(p):cs.sha(p) for p in (Path(__file__),S/'corrected_statistics.py',S/'revision_statistics_frozen.py')},device='CPU',seeds=SEEDS))
 progress('G2完成：8组新旧描述统计、两层检验、移动块CI及MSE/MAE各8项BH；全部结果保留。')
if __name__=='__main__':main()
