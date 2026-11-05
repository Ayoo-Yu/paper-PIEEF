from analyze_g12 import *
import importlib.util,re
from datetime import datetime,timezone,timedelta
T=R/'补做重算/tables';N=R/'next_batch'
AR=js(T/'archive_runs.json')
mask_inputs=js(O/'G4_mask_archive_runs.json')
for item in mask_inputs:
 for i,a in enumerate(AR):
  if a['metadata']['run_id']==item['metadata']['run_id'] and a['archive']==item['archive']:AR[i]=item
def cid(a):return a['metadata']['config']['tracking']['config_id']
def unique(xs):assert len(xs)==1,len(xs);return xs[0]
def get(c):return unique([a for a in AR if cid(a)==c])
def metrics(a):return cs.load_saved(run(Path(a['directory'])))
def main():
 start=time.time();dups=pd.read_csv(T/'T5_duplicate_runs.csv').fillna('').to_dict('records');dispatch=js(N/'_stage2_dispatch.json');reserved=js(N/'_local_reserved.json')['reserved']
 owner={c:n for n,cc in dispatch.items() for c in cc}
 decisions=[];selected={};backups=[];allrows=[];tzproof=[]
 for c in sorted(set(r['config_id'] for r in dups)):
  rows=[r.copy() for r in dups if r['config_id']==c];assert len(rows)==2
  orig='045' if c.startswith('energy_r3_stage1_F3_') else owner[c]
  assigned='local' if c in reserved else orig
  for row in rows:
   row['node_short']='local' if row['node'].startswith('local') else row['node'][3:6]
   off=-7 if row['node_short']=='local' else 8
   row['completion_utc']=datetime.fromisoformat(row['registry_end_time']).replace(tzinfo=timezone(timedelta(hours=off))).astimezone(timezone.utc).isoformat()
   a=unique([a for a in AR if a['metadata']['run_id']==row['run_id'] and a['archive']==row['archive']]);m=a['metadata'];d=Path(a['directory'])
   assert cs.sha(d/'metrics.json')==row['metrics_sha256']
   row.update(directory=str(d),validation_metrics=m.get('validation_metrics'),stored_metrics=m['metrics'],evaluation_split=m.get('evaluation_split'),source_sha=m['config']['tracking'].get('source_manifest_sha256'),files=[dict(path=str(p),sha256=cs.sha(p),bytes=p.stat().st_size) for p in sorted(d.iterdir()) if p.is_file()])
   if row['predictions_sha256']:assert cs.sha(d/'predictions.npz')==row['predictions_sha256']
   if c.startswith('energy_r3s2_'):
    x=metrics(a);v=metrics(get(f"energy_r3s2_main_PI-EEF_h16_seed{row['seed']}"));pairing(v,x)
    row['recomputed_metrics']=x.metric
   allrows.append(row)
  matching=[r for r in rows if r['node_short']==assigned]
  if len(matching)==1:chosen=matching[0];rule='指定位置'
  elif len(matching)==2:chosen=min(matching,key=lambda r:r['completion_utc']);rule='同一指定位置重复，取UTC先完成'
  else:chosen=min(rows,key=lambda r:r['completion_utc']);rule='初始指定节点无对应完成运行，保留改派歧义，回退UTC先完成'
  backup=next(r for r in rows if r['run_id']!=chosen['run_id']);selected[c]=chosen;backups.append(backup)
  basis=('stage1 _launch_stage1.py按variant排序整组分派：F3→045；_place_remaining.py另给562补跑，不覆盖原指定位置。' if c.startswith('energy_r3_stage1') else '_stage2_dispatch.json初始节点='+orig+'；'+('stage2_monitor.py第21–23行与_local_reserved.json明确移交local，优先于旧节点队列。' if c in reserved else '_local_stage2.ps1也包含该配置，且后续work-stealing改变节点；保留冲突，不把后来启动节点反推为原指定位置。'))
  decisions.append(dict(config_id=c,original_owner=orig,effective_explicit_owner=assigned,selected_run_id=chosen['run_id'],selected_node=chosen['node_short'],selected_directory=chosen['directory'],backup_run_id=backup['run_id'],backup_node=backup['node_short'],rule=rule,basis=basis,selected_completion_utc=chosen['completion_utc'],backup_completion_utc=backup['completion_utc']))
 assert len(decisions)==9 and len(backups)==9
 # Freeze selection before computing group means; no metric appears in the selection predicate.
 dump('G4_selection.json',decisions);csv('G4_selection.csv',decisions);dump('G4_backup_full_values.json',backups);csv('G4_backup_values.csv',backups);dump('G4_all_duplicate_records.json',allrows)
 source_names=['_stage2_dispatch.json','_local_reserved.json','stage2_monitor.py','collect_stage2.py','_local_stage2.ps1','_local_stage2b.ps1','_local_stage2.log','_local_stage2b.log','_stage2_monitor.log','_launch_stage1.py','_place_remaining.py','round3_adoption_decisions.json','round3_stage1_configs/_stage1_manifest.json']
 evidence=[dict(path=str(N/n),sha256=cs.sha(N/n)) for n in source_names]
 # UTC offsets are proved by the same-run launch/end logs, not machine locale or file mtimes.
 for row in allrows:
  c=row['config_id'];node=row['node_short']
  if node=='local':
   candidates=[]
   for name in ('_local_stage2.log','_local_stage2b.log'):
    for i,line in enumerate((N/name).read_text(encoding='utf-8-sig').splitlines(),1):
     if 'END '+c+' rc=0' in line:
      t=pd.Timestamp(re.search(r'\[(.*?)\]',line)[1]).to_pydatetime(warn=False);delta=abs((t.astimezone(timezone.utc)-datetime.fromisoformat(row['completion_utc'])).total_seconds())
      if delta<10:candidates.append(dict(path=str(N/name),line=i,text=line,offset_hours=-7,end_delta_seconds=delta))
   assert len(candidates)==1;tzproof.append(dict(run_id=row['run_id'],proof=candidates[0]))
  elif c.startswith('energy_r3s2_') and 'hidden-64' not in c:
   hits=[]
   for i,line in enumerate((N/'_stage2_monitor.log').read_text(encoding='utf-8-sig').splitlines(),1):
    if node+': START '+c+' ' in line:
     t=pd.Timestamp(re.search(r'\[(.*?)\]',line)[1]).to_pydatetime(warn=False);rt=datetime.fromisoformat(row['registry_start_time']).replace(tzinfo=timezone(timedelta(hours=8)))
     delta=(rt.astimezone(timezone.utc)-t).total_seconds();assert -10<delta<90;hits.append(dict(path=str(N/'_stage2_monitor.log'),line=i,text=line,offset_hours=8,start_delta_seconds=delta))
   assert hits;tzproof.append(dict(run_id=row['run_id'],proof=hits))
 dump('G4_selection_evidence.json',dict(sources=evidence,timezone_proofs=tzproof,policy='原始分派优先；显式local保留名单覆盖旧分派；原指定位置无完成者或同位置两次者以已核实时区的首次完成为准。未把后续work-stealing视为初始分派。',time_zone_limitation='hidden-64云端及F3不依赖时间排序；其+08显示沿用同节点云端日志时区，采用判定仅依赖指定位置。'))
 def chosen(c):
  if c in selected:return unique([a for a in AR if a['metadata']['run_id']==selected[c]['run_id'] and a['archive']==selected[c]['archive']])
  return get(c)
 vfs={s:metrics(get(f'energy_r3s2_main_PI-EEF_h16_seed{s}')) for s in (13,21,42,87,3407)}
 group=[];events=[];eventsummary=[];used=[]
 for loss in ('huber','mse','quantile'):
  runs={s:(get(f'energy_r3s2_main_PI-EEF_h16_seed{s}') if loss=='huber' else chosen(f'energy_r3s2_loss_{loss}_h16_seed{s}')) for s in SEEDS}
  sv={s:metrics(a) for s,a in runs.items()}
  for s,a in sv.items():pairing(vfs[s],a);used.append(a.evidence)
  row=dict(loss=loss,seeds=SEEDS,run_ids=[runs[s]['metadata']['run_id'] for s in SEEDS])
  for k in ('nrmse','nmae','r2'):
   z=np.array([sv[s].metric[k] for s in SEEDS])*(1 if k=='r2' else 100);row[k+'_mean']=z.mean();row[k+'_sd']=z.std(ddof=1)
  group.append(row)
  for s,a in sv.items():
   with np.load(a.run.directory/'predictions.npz') as z:labels=z['events']
   for label in sorted(np.unique(labels)):
    valid=a.valid&(labels==label);count=int(valid.sum());r=dict(loss=loss,seed=s,run_id=a.run.metadata['run_id'],label=label,n_valid=count)
    for k in ('nrmse','nmae'):
     if count:
      e=((a.prediction-a.truth)/a.capacity)[valid];ve=((vfs[s].prediction-vfs[s].truth)/a.capacity)[valid]
      f=(lambda x:np.sqrt(np.mean(x*x))) if k=='nrmse' else (lambda x:np.mean(abs(x)))
      r[k]=100*f(e);r['delta_'+k+'_vs_Huber_pp']=100*(f(e)-f(ve))
     else:r[k]=r['delta_'+k+'_vs_Huber_pp']=None
    events.append(r)
  for label in sorted({r['label'] for r in events if r['loss']==loss}):
   rr=[r for r in events if r['loss']==loss and r['label']==label];assert len(rr)==3
   out=dict(loss=loss,label=label,n_valid_per_seed=rr[0]['n_valid'],seeds=SEEDS)
   for k in ('nrmse','nmae','delta_nrmse_vs_Huber_pp','delta_nmae_vs_Huber_pp'):
    vals=[r[k] for r in rr];out[k+'_mean']=np.mean(vals) if vals[0] is not None else None;out[k+'_sd']=np.std(vals,ddof=1) if vals[0] is not None else None
   eventsummary.append(out)
 csv('G4_loss_summary.csv',group);csv('G4_events_per_seed.csv',events);csv('G4_event_summary.csv',eventsummary)
 hidden=[];validation_runs=[]
 for variant in ('VF','hidden-64'):
  for s in SEEDS:
   a=chosen(f'energy_r3s2_sensitivity_hidden-64_h16_seed{s}') if variant=='hidden-64' else get(f'energy_r3s2_main_PI-EEF_h16_seed{s}')
   sv=metrics(a);pairing(vfs[s],sv);used.append(sv.evidence);validation_runs.append(a)
   hist=js(Path(a['directory'])/'loss_history.json');be=hist['best_epoch'];val=a['metadata']['validation_metrics'];assert abs(next(r for r in hist['stage3'] if r['epoch']==be)['val_nrmse']-val['nrmse'])<2e-6
   row=dict(variant=variant,seed=s,run_id=a['metadata']['run_id'],directory=a['directory'],best_epoch=be)
   for split,m in [('validation',val),('test',sv.metric)]:
    for k in ('nrmse','nmae','r2'):row[split+'_'+k]=(1 if k=='r2' else 100)*m[k]
   hidden.append(row)
 hs=[]
 for variant in ('VF','hidden-64'):
  row=dict(variant=variant,seeds=SEEDS)
  for split in ('validation','test'):
   for k in ('nrmse','nmae','r2'):
    key=split+'_'+k;vv=[r[key] for r in hidden if r['variant']==variant];row[key+'_mean']=np.mean(vv);row[key+'_sd']=np.std(vv,ddof=1)
  hs.append(row)
 csv('G4_hidden64_per_seed.csv',hidden);csv('G4_hidden64_summary.csv',hs);dump('G4_validation_runs.json',validation_runs)
 pairs={}
 for s in vfs:
  a=metrics(chosen(f'energy_r3s2_masking_power-and-wind_h16_seed{s}'));pairing(vfs[s],a);pairs[s]=(vfs[s],a);used.append(a.evidence)
 mask=cs.summarize_group('masking:power-and-wind',pairs)
 dump('G4_P3_joint_mask.json',mask)
 # Update the original 30-item supplementary t/sign BH display, retaining other rows.
 old=js(S/'p3_training_seed_supplement.json');dump('G4_P3_original_schema.json',dict(type=type(old).__name__,keys=list(old) if isinstance(old,dict) else None))
 # F3 is a historical decision audit, not a new validation/test comparison.
 c='energy_r3_stage1_F3_h96_seed3407';decision=js(N/'round3_adoption_decisions.json');rr=[a for a in AR if cid(a).startswith('energy_r3_stage1_F3_h96_seed')]
 f3=[]
 for a in rr:
  m=a['metadata'];assert m['status']=='completed' and m['evaluation_split']=='val' and m['horizon']==96
  f3.append(dict(run_id=m['run_id'],seed=m['seed'],archive=a['archive'],selected=(m['seed']!=3407 or m['run_id']==selected[c]['run_id']),validation_metrics=m['validation_metrics']))
 dump('G4_F3_adoption_audit.json',dict(original_path=str(N/'round3_adoption_decisions.json'),original_sha256=cs.sha(N/'round3_adoption_decisions.json'),recorded_utc=decision['recorded_utc'],original_F3=decision['decisions']['F3'],selected_run=selected[c]['run_id'],runs=f3,decision_unchanged=None,historical_record_unchanged=True,conclusion='指定文件F3.adopted原值是pending，而非true/false。已核查并选定运行，但该文件没有可重验的最终采用判定，因此不能证明“最终采用结论不变”。h96证据缺失这一历史描述现已过时；没有据此补写作者决定。',limitation='后续verify_f3_archive.py补读并核验阶段1验证预测；本项缺少实际作出最终F3采用判定的冻结记录，不能用仅有h1/h16的历史PASS代替最终采用。'))
 dump('G4_input_run_evidence.json',used);dump('G4_receipt.json',dict(elapsed_seconds=time.time()-start,source_sha256=cs.sha(Path(__file__)),device='CPU',finished_utc=cs.utc()))
 progress('G4数值重算完成：9组按位置/核实UTC选定，9份备份完整保留；MSE事件、hidden64、联合掩码已重算。F3原文件为pending，历史记录未改；验证定义重建另行核验。')
if __name__=='__main__':main()
