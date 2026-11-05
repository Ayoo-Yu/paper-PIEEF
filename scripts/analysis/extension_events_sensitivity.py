from __future__ import annotations
import os,sys,time,json,csv,hashlib,itertools,re,platform,importlib.util,collections,math,functools
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;P=O.parents[1];ROOT=Path('results/paper');START=time.time()
STAT=P/'修复重算_20261021/statistics';sys.path.insert(0,str(STAT))
import corrected_statistics as cs
SRC=P/'next_batch/merged_v2/project'
SEEDS=(13,42,3407)
@functools.lru_cache(maxsize=None)
def sha(p):return cs.sha(p)
def jread(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jout(name,data): (O/name).write_text(json.dumps(cs.clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(cs.clean(v),ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else cs.clean(v) for k,v in r.items()})
def progress(text):
 with (O/'进度.md').open('a',encoding='utf-8') as f:f.write('\n- '+text+'\n')
 print(text,flush=True)
AR=jread(O/'archive_runs.json');INV=jread(O/'inventory.json')
for a in AR:
 a['path']=a['directory'];a['cid']=a['metadata']['config']['tracking']['config_id']
def ar(group,variant=None,h=None,seed=None):
 return [a for a in AR if 'r3s2' in a['cid'] and a['metadata']['config']['experiment']['group']==group and (variant is None or a['metadata']['config']['experiment']['variant']==variant) and (h is None or a['metadata']['horizon']==h) and (seed is None or a['metadata']['seed']==seed)]
def unique(xs):
 assert len(xs)==1,[(x.get('cid'),x['path']) for x in xs]
 return xs[0]
def convert(r):return dict(path=r['path'],metadata=jread(Path(r['path'])/'metrics.json'),source=r['source'])
AUD=[];EVID={};CACHE={}
def load(r):
 key=r['path']
 if key in CACHE:return CACHE[key]
 d=Path(key);m=r.get('metadata') or jread(d/'metrics.json');cfg=m['config'];h=int(m['horizon']);s=int(m['seed'])
 assert m['status']=='completed' and m['evaluation_split']=='test'
 assert cfg['experiment']['horizon']==h and cfg['experiment']['seeds']==[s]
 with np.load(d/'predictions.npz',allow_pickle=False) as z:a={k:np.array(z[k],copy=True) for k in z.files}
 assert int(a['horizon'])==h and int(a['seed'])==s
 with np.load(d/'preprocessing.npz',allow_pickle=False) as z:pre={k:np.array(z[k],copy=True) for k in z.files}
 cap=pre['capacity_mw'];assert np.isfinite(cap).all() and (cap>0).all()
 if 'capacity_mw' in cfg['data']:assert np.array_equal(cap,np.asarray(cfg['data']['capacity_mw']))
 assert np.array_equal(a['turbine_ids'],pre['turbine_ids'])
 assert jread(d/'data_protocol.json')==cfg['data']['revision_protocol']
 ts=pd.to_datetime(a['timestamp'],utc=True).tz_localize(None).to_numpy();orig=pd.to_datetime(a['origin_timestamp'],utc=True).tz_localize(None).to_numpy()
 assert not np.isnat(ts).any() and not np.isnat(orig).any() and np.all(np.diff(ts)>np.timedelta64(0,'m'))
 assert np.all(ts-orig==np.timedelta64(h*15,'m'))
 y=a['y_true_mw'];pred=a['y_pred_mw'];v=a['target_valid']
 assert v.dtype.kind=='b' and y.shape==pred.shape==v.shape and y.shape==(len(ts),len(a['turbine_ids']))
 assert np.array_equal(np.isfinite(y),v) and v.any(axis=1).all() and np.isfinite(pred[v]).all()
 err=np.where(v,(pred.astype(float)-y.astype(float))/cap,0);cnt=v.sum(axis=1)
 metrics={'nrmse':np.sqrt(np.mean(err[v]**2)),'nmae':np.mean(abs(err[v]))}
 for k,x in metrics.items():assert np.isclose(x,m['metrics'][k],rtol=1e-5,atol=1e-7),(d,k,x,m['metrics'][k])
 EVID[key]=dict(directory=key,archive=r.get('archive'),archive_member=r.get('member'),config_id=cfg['tracking']['config_id'],seed=s,horizon=h,source_tracking=cfg['tracking'],metrics=metrics,n_time=len(ts),n_valid=int(v.sum()),n_turbines=len(cap) if cap.ndim else len(a['turbine_ids']),first_target=str(ts[0]),last_target=str(ts[-1]),target_intervals_minutes={str(k):int(n) for k,n in zip(*np.unique(np.diff(ts).astype('timedelta64[m]').astype(int),return_counts=True))},files=[dict(path=str(d/f),sha256=sha(d/f)) for f in ('metrics.json','predictions.npz','preprocessing.npz','data_protocol.json')])
 result=dict(a=a,m=m,cfg=cfg,cap=cap,pre=pre,ts=ts,orig=orig,metric=metrics,mse_time=(err**2).sum(axis=1)/cnt,mae_time=abs(err).sum(axis=1)/cnt,path=key)
 CACHE[key]=result
 return result
def pair(x,y,task):
 fields={}
 for k in ('timestamp','origin_timestamp','turbine_ids','target_valid','events'):
  fields[k]=bool(np.array_equal(x['a'][k],y['a'][k]))
 fields['truth']=bool(np.array_equal(x['a']['y_true_mw'],y['a']['y_true_mw'],equal_nan=True));fields['capacity']=bool(np.array_equal(x['cap'],y['cap']))
 fields['horizon_seed']=x['m']['seed']==y['m']['seed'] and x['m']['horizon']==y['m']['horizon']
 fields['data_config']=x['cfg']['data']==y['cfg']['data']
 fields['protocol']=x['cfg']['data']['revision_protocol']==y['cfg']['data']['revision_protocol']
 AUD.append(dict(task=task,reference=x['path'],candidate=y['path'],status='PASS' if all(fields.values()) else 'FAIL',checks=fields))
 assert all(fields.values()),AUD[-1]
def seedstats(v):return cs.seed_supplement(v)
def bhrows(rows,key,out):
 p=[np.nan if r.get(key) is None else r[key] for r in rows]
 for r,q in zip(rows,cs.established.bh_adjust(p)):r[out]=float(q) if np.isfinite(q) else None
def mean_sd(v):return float(np.mean(v)),float(np.std(v,ddof=1))
def main():
 # T1: all five original cross-farm baselines, no model selection before identity checks.
 rows=[];tests=[];perseed=[];best=[]
 for site in ('kelmarsh','wind_farm_B'):
  for h in (1,4,16):
   vf={s:load(unique(ar('multifarm',site,h,s))) for s in SEEDS}
   models=sorted({r['model'] for r in INV if r['source']=='multifarm' and r['model'] is not None and site in r['path']})
   assert models==['DLinear','Jensen-Net','Persistence','Wake-GNN','XGBoost']
   group=[]
   for model in models:
    pairs={};ok=True
    for s in SEEDS:
     b=unique([r for r in INV if r['source']=='multifarm' and r['model']==model and r['h']==h and r['seed']==s and site in r['path']])
     try:
      y=load(convert(b));pair(vf[s],y,'T1');pairs[s]=(vf[s],y)
     except Exception as exc:
      AUD.append(dict(task='T1',status='FAIL',site=site,h=h,model=model,seed=s,error=str(exc)));ok=False
    if not ok:continue
    row=dict(site=site,horizon=h,baseline=model,seeds=list(SEEDS),n_time=len(vf[13]['ts']),n_valid=int(vf[13]['a']['target_valid'].sum()),n_turbines=len(vf[13]['a']['turbine_ids']))
    for metric in ('nrmse','nmae'):
     vm,vs=mean_sd([100*pairs[s][0]['metric'][metric] for s in SEEDS]);bm,bs=mean_sd([100*pairs[s][1]['metric'][metric] for s in SEEDS])
     row.update({f'VF_{metric}_mean_pct':vm,f'VF_{metric}_sd_pp':vs,f'baseline_{metric}_mean_pct':bm,f'baseline_{metric}_sd_pp':bs,f'VF_{metric}_improvement_pct':100*(1-vm/bm)})
    for s,(v,b) in pairs.items():perseed.append(dict(site=site,horizon=h,baseline=model,seed=s,VF_dir=v['path'],baseline_dir=b['path'],**{f'{side}_{metric}_pct':100*x['metric'][metric] for side,x in [('VF',v),('baseline',b)] for metric in ('nrmse','nmae')}))
    for family,attr in (('mse','mse_time'),('mae','mae_time')):
     t=dict(site=site,horizon=h,baseline=model,family=family,n_time=len(vf[13]['ts']),n_valid=row['n_valid'],seeds=list(SEEDS),difference_direction='VF_minus_baseline',declared_family_size=30)
     diffs=[np.mean(pairs[s][0][attr])-np.mean(pairs[s][1][attr]) for s in SEEDS]
     t.update({'seed_'+k:v for k,v in seedstats(diffs).items()})
     t['seed_multiplicity']='T1_predeclared_30_comparisons_per_loss_and_test_type_BH'
     if site=='kelmarsh':
      steps=int(np.ceil(h*15/30));assert np.all(np.diff(vf[13]['ts'])==np.timedelta64(30,'m'))
      result=cs.inference(np.mean([pairs[s][1][attr] for s in SEEDS],axis=0),np.mean([pairs[s][0][attr] for s in SEEDS],axis=0),steps)
      t.update(result);t.update(horizon=h,effective_observation_horizon=steps,target_step_minutes=30,time_test_status='PASS_regular_30min_actual_target_grid')
     else:t.update(p_value=None,time_test_status='NOT_COMPUTED_irregular_30_60min_targets_no_time_compression')
     tests.append(t)
    group.append(row);rows.append(row)
   for metric in ('nrmse','nmae'):
    learned=[r for r in group if r['baseline']!='Persistence'];winner=min(learned,key=lambda r:r[f'baseline_{metric}_mean_pct'])
    best.append(dict(site=site,horizon=h,metric=metric,baseline=winner['baseline'],VF_pct=winner[f'VF_{metric}_mean_pct'],baseline_pct=winner[f'baseline_{metric}_mean_pct'],improvement_pct=winner[f'VF_{metric}_improvement_pct'],n_time=winner['n_time'],n_valid=winner['n_valid']))
 for fam in ('mse','mae'):
  f=[t for t in tests if t['family']==fam];assert len(f)==30
  for key,out in [('p_value','time_BH_q'),('seed_paired_t_p_raw','seed_t_BH_q'),('seed_exact_sign_p_two_sided_raw','seed_sign_BH_q')]:bhrows(f,key,out)
 csvout('T1_all_baselines.csv',rows);csvout('T1_best_learning_baselines.csv',best);csvout('T1_tests.csv',tests);csvout('T1_per_seed.csv',perseed)
 jout('T1_results.json',dict(rows=rows,best=best,tests=tests))
 progress('T1 已完成全部30组的描述统计与种子检验；Kelmarsh 15组时间DM已计算，Wind Farm B15组因非等间隔时间网格保留未定义槽。')
 # T2: repeated runs are not silently collapsed. All seed-13 x seed-42 branches.
 vf={s:load(unique(ar('main','PI-EEF',16,s))) for s in SEEDS}
 runs=[unique(ar('main','PI-EEF',16,s)) for s in SEEDS]+ar('loss')
 strata=[];diffs=[]
 for r in runs:
  s=r['metadata']['seed'];v=vf[s];a=load(r);pair(v,a,'T2')
  loss='huber' if r['metadata']['config']['experiment']['group']=='main' else r['metadata']['config']['experiment']['variant']
  for label in sorted(np.unique(a['a']['events']).tolist()):
   valid=a['a']['target_valid']&(a['a']['events']==label);n=int(valid.sum())
   if n:
    err=((a['a']['y_pred_mw'].astype(float)-a['a']['y_true_mw'].astype(float))/a['cap'])[valid];ve=((v['a']['y_pred_mw'].astype(float)-v['a']['y_true_mw'].astype(float))/v['cap'])[valid]
    nr=100*np.sqrt(np.mean(err**2));ma=100*np.mean(abs(err));dr=nr-100*np.sqrt(np.mean(ve**2));dm=ma-100*np.mean(abs(ve))
   else:nr=ma=dr=dm=None
   rr=dict(loss=loss,seed=s,run_id=a['m']['run_id'],archive=r['archive'],directory=r['path'],label=label,n_valid=n,n_time_with_valid=int(valid.any(axis=1).sum()),nrmse_pct=nr,nmae_pct=ma,delta_nrmse_vs_Huber_pp=dr,delta_nmae_vs_Huber_pp=dm)
   strata.append(rr);diffs.append(rr.copy())
 summary=[]
 for loss in ('huber','mse','quantile'):
  candidates={s:[r for r in runs if r['metadata']['seed']==s and ('huber' if r['metadata']['config']['experiment']['group']=='main' else r['metadata']['config']['experiment']['variant'])==loss] for s in SEEDS}
  for choice in itertools.product(*(candidates[s] for s in SEEDS)):
   chosen=[r['metadata']['run_id'] for r in choice];branch=' + '.join(chosen)
   for label in sorted({r['label'] for r in strata}):
    selected=[r for r in strata if r['run_id'] in chosen and r['label']==label];assert len(selected)==3
    row=dict(loss=loss,branch=branch,seeds=list(SEEDS),label=label,n_valid_per_seed=selected[0]['n_valid'],n_seeds=3)
    for k in ('nrmse_pct','nmae_pct','delta_nrmse_vs_Huber_pp','delta_nmae_vs_Huber_pp'):
     values=[r[k] for r in selected];row[k+'_mean'],row[k+'_sample_sd']=mean_sd(values) if all(v is not None for v in values) else (None,None)
    summary.append(row)
 csvout('T2_event_per_run.csv',strata);csvout('T2_event_summary_all_branches.csv',summary)
 jout('T2_results.json',dict(per_run=strata,summary=summary))
 progress('T2 已完成：全部事件标签、无效目标剔除、逐种子配对差；MSE两次seed13和两次seed42均独立列出，全部4种组合均报告。')
 # T3: Original validation metrics exist; exact shared validation definition rebuilt separately.
 sensitive=ar('sensitivity');sruns=[unique(ar('main','PI-EEF',16,s)) for s in SEEDS]+sensitive
 valrows=[]
 for r in sruns:
  a=load(r);v=vf[a['m']['seed']];pair(v,a,'T3_test')
  assert a['cfg']['data']==v['cfg']['data']
  assert set(a['pre'])==set(v['pre'])
  # Production _bundle uses this exact preprocessing comparison tolerance;
  # capacity/targets/index identity remain exact, not relaxed.
  for k in a['pre']:
   if a['pre'][k].dtype.kind=='f':assert np.allclose(a['pre'][k],v['pre'][k],rtol=1e-6,atol=1e-7,equal_nan=True)
   else:assert np.array_equal(a['pre'][k],v['pre'][k])
  val=a['m']['validation_metrics'];h=jread(Path(r['path'])/'loss_history.json');be=h['best_epoch'];hv=next(x for x in h['stage3'] if x['epoch']==be)['val_nrmse']
  # Epoch history is per-unit normalized power; final metric is same checkpoint.
  assert abs(hv-val['nrmse'])<2e-6,(r['path'],hv,val['nrmse'])
  rr=dict(variant='VF' if a['cfg']['experiment']['group']=='main' else a['cfg']['experiment']['variant'],seed=a['m']['seed'],run_id=a['m']['run_id'],archive=r['archive'],directory=r['path'],validation_n_time=val['num_samples'],test_n_time=len(a['ts']),best_epoch=be,history_validation_nrmse=hv,validation_metric_source='original_saved_metrics_at_weights_best',validation_index_evidence='deterministic_shared_config_preprocessing_and_producer; reconstructed separately')
  for metric in ('nrmse','nmae'):
   rr['validation_'+metric+'_pct']=100*val[metric];rr['test_'+metric+'_pct']=100*a['metric'][metric]
   rr['validation_delta_'+metric+'_pp']=100*(val[metric]-v['m']['validation_metrics'][metric]);rr['test_delta_'+metric+'_pp']=100*(a['metric'][metric]-v['metric'][metric])
  valrows.append(rr)
 csvout('T3_per_run_validation_test.csv',valrows)
 valsummary=[]
 for variant in ('VF','encoder-2','encoder-4','graph-2','hidden-256','hidden-64'):
  candidates={s:[r for r in valrows if r['variant']==variant and r['seed']==s] for s in SEEDS}
  for choice in itertools.product(*(candidates[s] for s in SEEDS)):
   row=dict(variant=variant,run_ids=[r['run_id'] for r in choice],seeds=list(SEEDS))
   for split in ('validation','test'):
    for metric in ('nrmse','nmae'):
     k=split+'_'+metric+'_pct';row[k+'_mean'],row[k+'_sample_sd']=mean_sd([r[k] for r in choice]);base=np.mean([r[k] for r in valrows if r['variant']=='VF'])
     row[split+'_'+metric+'_change_pct']=100*(row[k+'_mean']/base-1)
     row[split+'_'+metric+'_VF_improvement_pct']=100*(1-base/row[k+'_mean'])
   valsummary.append(row)
 csvout('T3_summary_all_branches.csv',valsummary);jout('T3_results.json',dict(per_run=valrows,summary=valsummary))
 progress('T3 原验证指标与测试指标已并列，全部19个运行的原best_epoch验证NRMSE与保存指标一致；hidden-64两次seed13分别报告，验证样本重建门禁另行执行。')
 # T4: Huber logs and all five named strongest neural original baseline model families.
 logrows=[];epochs=[];orig_models=('ACFormer','TCN','TimesNet','DLinear','Wake-GNN','GRU','iTransformer','Jensen-Net','PatchTST','STAN','TimeMixer')
 recovered=jread(O/'T4_original_epoch_observations.json')['rows']
 selected=[r for r in INV if r['source']=='huber' and r['model']!='LightGBM' or r['source']=='original' and r['model'] in orig_models]
 assert sum(r['source']=='huber' for r in selected)==21
 for r in selected:
  d=Path(r['path']);m=jread(d/'metrics.json');tr=m['config']['train'];s=m.get('training_summary') or {};cid=m['config']['tracking']['config_id']
  row=dict(source=r['source'],model=r['model'],horizon=r['h'],seed=r['seed'],config_id=cid,directory=str(d),configured_epochs=tr.get('epochs'),batch_size=tr.get('batch_size'),patience=tr.get('patience'),loss=(m['config'].get('baseline_protocol',{}).get('loss') or tr.get('loss_name')),**s)
  log=P/'修复重算_20261021/ops/baseline_process_receipts'/cid/'training.log'
  history=[]
  if log.is_file():
   text=log.read_text(encoding='utf-8-sig',errors='replace');pattern=re.compile(r'\[train\]\s+'+re.escape(str(r['model']))+r'\|s'+str(r['seed'])+r'\|h'+str(r['h'])+r'\s+epoch=(\d+)/(\d+).*?val_loss=([0-9.eE+\-]+)')
   history=[(int(e),float(v)) for e,_,v in pattern.findall(text)];row['log']=str(log);row['log_sha256']=sha(log)
  obs=[]
  if r['source']=='original':
   obs=[x for x in recovered if x['config_id']==cid and x['run_id']==m['run_id']]
   history=sorted((x['epoch'],x['val_loss']) for x in obs)
   if obs:row['log']=obs[0]['source'];row['log_sha256']=sha(Path(obs[0]['source']))
  lookup=dict(history);n=s.get('epochs_run',0);complete=set(lookup)==set(range(1,n+1))
  if complete and history:assert np.isclose(min(lookup.values()),s['best_validation_loss'])
  if s.get('best_epoch') in lookup:assert np.isclose(lookup[s['best_epoch']],s['best_validation_loss'])
  row.update(observed_epochs=sorted(lookup),missing_epochs=[e for e in range(1,n+1) if e not in lookup],n_observed_epochs=len(lookup),minimum_at_last=(s.get('best_epoch')==n) if s.get('best_epoch') is not None else None)
  if n in lookup and n-1 in lookup:
   prev,last=lookup[n-1],lookup[n];row.update(last_val_loss=last,previous_val_loss=prev,last_improvement_pct=100*(1-last/prev),last_improvement_gt1pct=bool(100*(1-last/prev)>1),history_status='COMPLETE_LOG' if complete else 'PARTIAL_SNAPSHOT_HAS_FINAL_TWO')
  else:row.update(last_improvement_pct=None,last_improvement_gt1pct=None,history_status='PARTIAL_SNAPSHOT_MISSING_FINAL_TWO' if history else 'MISSING_EPOCH_LOSSES_summary_only')
  if obs:epochs.extend(obs)
  else:epochs.extend(dict(config_id=cid,epoch=e,val_loss=v,source=str(log)) for e,v in history)
  logrows.append(row)
 csvout('T4_training_convergence.csv',logrows);csvout('T4_validation_epoch_losses.csv',epochs);jout('T4_results.json',dict(rows=logrows,epochs=epochs))
 progress('T4 新Huber 21个神经运行完整逐轮日志已提取；扩展清点原正式11类神经基线275个运行，从同步快照补回可验证轮次，缺失不填造。')
 # T5: every duplicate completed config, plus every summary record.
 groups=collections.defaultdict(list)
 for r in AR:
  if r['metadata'].get('status')=='completed':groups[(r['cid'],r['metadata']['seed'])].append(r)
 stagepath=P/'交付_20261021/02_实验结果/round3_stage2_metrics.json';stage=jread(stagepath);stagegroups=collections.defaultdict(list)
 for i,r in enumerate(stage):stagegroups[(r['config_id'],r['seed'])].append(dict(row_number=i+1,**r))
 repeats=[];summary_repeats=[];registry=jread(O/'archive_registry_rows.json')
 for key,rs in sorted(groups.items()):
  if len(rs)<2:continue
  for r in rs:
   m=r['metadata'];track=m['config']['tracking'];d=Path(r['path']);pred=d/'predictions.npz'
   reg=[x for x in registry if x['archive']==r['archive'] and x['run_id']==m['run_id'] and x['config_id']==key[0] and x['status']=='success']
   assert len(reg)<=1
   reg=reg[0] if reg else {}
   repeats.append(dict(config_id=key[0],seed=key[1],run_id=m['run_id'],node=r['archive'],run_timestamp_from_id=m['run_id'][-15:],registry_start_time=reg.get('start_time'),registry_end_time=reg.get('end_time'),registry_time_timezone='not encoded; do not rank cross-node times as UTC',registry_code_commit=reg.get('code_commit'),registry_source=reg.get('source'),archive=r['archive'],member=r['member'],directory=r['path'],code_git_commit=track.get('code_git_commit'),source_manifest_sha256=track.get('source_manifest_sha256'),metrics_sha256=sha(d/'metrics.json'),predictions_sha256=sha(pred) if pred.is_file() else None,nrmse_pct=100*m['metrics']['nrmse'],nmae_pct=100*m['metrics']['nmae'],device=m.get('device'),n_archive_records=len(rs),n_distinct_run_ids=len({x['metadata']['run_id'] for x in rs})))
 for key,rs in sorted(stagegroups.items()):
  if len(rs)>1:summary_repeats.extend(rs)
 csvout('T5_duplicate_runs.csv',repeats);csvout('T5_duplicate_stage2_rows.csv',summary_repeats);jout('T5_results.json',dict(archive_duplicates=repeats,summary_duplicates=summary_repeats,total_archive_run_records=len(AR),stage2_rows=len(stage),summary_source=str(stagepath),summary_sha256=sha(stagepath)))
 progress(f'T5 清点完成：归档{len(AR)}条运行记录；重复身份{sum(len(v)>1 for v in groups.values())}组；所有重复均保留，未选择运行。')
 # T6: Post-hoc MAE family of exactly 70 original conditional comparisons.
 table=pd.read_csv(STAT/'original_revision_main_70_corrected.csv');mae=table[table['family']=='mae'].copy();assert len(mae)==70 and mae['hypothesis_id'].nunique()==70
 q=cs.established.bh_adjust(mae['p_value'].to_numpy());mae['posthoc_MAE70_BH_q']=q;mae['posthoc_MAE70_reject_005']=q<=.05;mae['family_definition']='post_hoc_added_MAE_14_baselines_x_5_horizons_70';mae['input_sha256']=sha(STAT/'original_revision_main_70_corrected.csv')
 mae.to_csv(O/'T6_MAE70_posthoc_BH.csv',index=False,encoding='utf-8-sig');jout('T6_results.json',dict(n=70,reject=int((q<=.05).sum()),by_horizon=mae.groupby('horizon')['posthoc_MAE70_reject_005'].sum().to_dict(),rows=mae.to_dict('records')))
 progress(f'T6 完成：事后补加MAE70族，BH后{int((q<=.05).sum())}/70达到q≤0.05。')
 jout('identity_audit.json',AUD);jout('input_provenance.json',list(EVID.values()))
 source_files=[STAT/'corrected_statistics.py',STAT/'revision_statistics_frozen.py',SRC/'scripts/evaluate_revision.py',SRC/'src/pieef/revision_data.py',SRC/'src/pieef/data.py',SRC/'src/pieef/train_main.py',SRC/'scripts/train.py',SRC/'src/pieef/artifacts.py',SRC/'provenance/source_manifest.json']
 jout('execution.json',dict(device='CPU',cpu=platform.processor(),seeds=list(SEEDS),python=sys.version,elapsed_analysis_seconds=time.time()-START,start_epoch=START,end_epoch=time.time(),training_performed=False,inference_performed=False,sources=[dict(path=str(p),sha256=sha(p)) for p in source_files],reproduction_gate='Not applicable: analysis of saved predictions/metrics; no normal re-inference',identity_pass=sum(r.get('status')=='PASS' for r in AUD),identity_fail=[r for r in AUD if r.get('status')=='FAIL']))
 print('DONE',time.time()-START,flush=True)
if __name__=='__main__':main()
