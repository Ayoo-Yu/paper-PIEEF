import os,sys,json,time,importlib.util,traceback,argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
import torch
import reference_helpers as ref
from prepare import js,sha,dump,info
L=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def verify(f):assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f['path']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);a=ap.parse_args()
 protocol=js(L/'protocol.json');task=next(t for t in protocol['sources'] if t['task_id']==a.task)
 dest=L/'runs'/a.task;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completed.json').exists():
  done=js(dest/'completed.json')
  for f in done['outputs']:verify(f)
  print('VERIFIED COMPLETED',a.task,flush=True);return
 selected=protocol
 fallback=False
 if task['model_or_mode']=='V1' and (dest/'normal_equivalence.json').exists() and js(dest/'normal_equivalence.json')['status']=='FAIL':
  selected=js(L/'v1_fallback_manifest.json');fallback=True
  archive=L/'fallback_attempts'/a.task;archive.mkdir(parents=True,exist_ok=True)
  import shutil
  shutil.copyfile(dest/'normal_equivalence.json',archive/'frozen_source_normal_gate_failed.json')
 started=utc();t0=time.monotonic();attempt=f'{time.time_ns()}_{os.getpid()}'
 ident=dict(pid=os.getpid(),command=[sys.executable,*sys.argv],start_utc=started,task_id=a.task,trained=False,runner_sha256=sha(Path(__file__)))
 dump(dest/f'process_{attempt}.json',ident)
 try:
  for f in selected['source_files']+protocol['data_files']+task['files']+[protocol['helper']]:verify(f)
  torch.set_num_threads(2);device=torch.device('cuda:0');assert torch.cuda.is_available()
  project=Path(selected['source_project']);sys.path.insert(0,str(project/'src'))
  spec=importlib.util.spec_from_file_location('frozen_eval',project/'scripts/evaluate_revision.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
  from pieef.revision_fault_jobs import scada_scenarios
  assert scada_scenarios('structured_39')==protocol['scenarios']
  h,s=task['horizon'],task['seed'];ref.HORIZON=h
  def grid(fields):
   t=pd.to_datetime(fields['timestamp'],utc=True).asi8;o=pd.to_datetime(fields['origin_timestamp'],utc=True).asi8
   assert np.all(t-o==h*15*60*10**9) and np.all(np.diff(t)==15*60*10**9)
  ref.check_h16_grid=grid
  d=Path(task['directory']);meta=js(d/'metrics.json');cfg=deepcopy(meta['config'])
  data={Path(f['path']).name:f['path'] for f in protocol['data_files']}
  cfg['data']['scada_path']=data['scada_ecmwf_15min.csv'];cfg['data']['turbines_path']=data['turbines.csv'];cfg['model']['manufacturer_curve_path']=data['manufacturer_curve.csv']
  record=dict(source_dir=d,config=cfg,model='PI-EEF',seed=s,horizon=h)
  bundle=ev._bundle(record);fields=ref.index_fields(bundle);grid(fields);saved=ref.load_npz(d/'predictions.npz');ref.same_indices(saved,fields)
  cap=np.asarray(bundle.capacity_mw);assert int((~fields['target_valid']).sum())==115
  model,kind,weight=ev._predictor(record,bundle,device);assert kind=='main' and not model.training
  if fallback:
   original_forward=model.forward_stage3_forecast
   def forward_compat(features,v,physics,target,hist_power=None,hist_power_mask=None):
    return original_forward(features,v,physics,target)
   model.forward_stage3_forecast=forward_compat
  assert all(torch.isfinite(v).all() for v in model.state_dict().values())
  source=dict(config_id=task['config_id'],run_id=d.name,checkpoint_sha256=sha(weight),source_manifest_sha256=sha(L/'protocol.json'),source_snapshot_sha256=selected['source_snapshot_sha256'])
  def state(status,completed=0,scenario=None):
   payload=dict(ident);payload.update(status=status,completed_faults=completed,scenario=scenario,updated_utc=utc(),elapsed_seconds=time.monotonic()-t0)
   ref.atomic_json(dest/'progress.json',payload)
  state('normal_reproduction');truth,pred=ev._infer(model,kind,bundle,None,2026,'test',device)
  delta=np.abs((pred-saved['y_pred_mw'])/cap);allowed=1e-5+1e-5*np.abs(saved['y_pred_mw']/cap)
  gate=dict(status='PASS' if np.isfinite(pred).all() and np.all(delta<=allowed) else 'FAIL',all_prediction_cells=int(delta.size),failed_cells=int((~np.isfinite(pred)|(delta>allowed)).sum()),original_v1_model_fallback=fallback,source_project=str(project),max_abs_delta_pu=float(delta.max()),atol_pu=1e-5,rtol=1e-5,source_snapshot_sha256=selected['source_snapshot_sha256'],source=task['task_id'],utc=utc())
  dump(dest/'normal_equivalence.json',gate)
  if gate['status']!='PASS':
   tag=selected['source_snapshot_sha256'][:12]
   ref.save_prediction(dest/f'normal_failed_{tag}.npz',fields,truth,pred,seed=s,scenario='normal',capacity=cap,source=source)
   ii,jj=np.where(~np.isfinite(pred)|(delta>allowed));caps=np.broadcast_to(cap,pred.shape)
   pd.DataFrame(dict(timestamp=fields['timestamp'][ii],origin_timestamp=fields['origin_timestamp'][ii],turbine_id=fields['turbine_ids'][jj],target_valid=fields['target_valid'][ii,jj],prediction_mw=pred[ii,jj],saved_mw=saved['y_pred_mw'][ii,jj],capacity_mw=caps[ii,jj],abs_delta_pu=delta[ii,jj],allowed_pu=allowed[ii,jj],source_snapshot_sha256=selected['source_snapshot_sha256'])).to_csv(dest/f'normal_failed_cells_{tag}.csv',index=False)
   raise ValueError('Normal reproduction all-cell gate FAILED; no faults permitted')
  gate.update(ref.normal_equivalence(saved,{**fields,'y_true_mw':truth,'y_pred_mw':pred},cap))
  vf=next(v for v in protocol['vf_sources'] if v['horizon']==h and v['seed']==s);vdir=Path(vf['fault_directory']);v=ref.load_npz(vdir/'normal.npz');ref.same_indices(v,fields)
  assert np.array_equal(v['y_true_mw'],truth,equal_nan=True) and np.array_equal(v['capacity_mw'],cap)
  gate.update(vf_pair_identity='PASS',invalid_targets=115,original_v1_model_fallback=fallback,source_project=str(project));dump(dest/'normal_equivalence.json',gate)
  ref.save_prediction(dest/'normal.npz',fields,truth,pred,seed=s,scenario='normal',capacity=cap,source=source)
  normal=ref.measured_metrics(truth,pred,fields['target_valid'],cap)
  for k in ['nrmse','nmae','r2']:assert abs(normal[k]-meta['metrics'][k])<2e-6
  rows=[dict(scenario='normal',**normal)]
  print('NORMAL PASS',a.task,gate['max_abs_delta_pu'],flush=True)
  for i,original in enumerate(protocol['scenarios']):
   scene=dict(original);name=scene.pop('name');state('fault_inference',i,name)
   va=ref.load_npz(vdir/(name+'.npz'));ref.same_indices(va,fields)
   assert int(va['horizon'])==h and int(va['seed'])==s and int(va['fault_seed'])==2026 and str(va['scenario'])==name
   assert json.loads(str(va['scenario_json']))==scene and np.array_equal(va['y_true_mw'],truth,equal_nan=True)
   path=dest/(name+'.npz');receipt=dest/(name+'.receipt.json')
   if path.exists() and receipt.exists() and sha(path)==js(receipt)['sha256']:
    fpdata=ref.load_npz(path);ref.same_indices(fpdata,fields)
    assert str(fpdata['source_manifest_sha256'])==source['source_manifest_sha256'] and str(fpdata['checkpoint_sha256'])==source['checkpoint_sha256']
    assert np.array_equal(fpdata['y_true_mw'],truth,equal_nan=True) and json.loads(str(fpdata['scenario_json']))==scene
    ft,fp=fpdata['y_true_mw'],fpdata['y_pred_mw']
   else:
    ft,fp=ev._infer(model,kind,bundle,scene,2026,'test',device)
    assert np.array_equal(ft,truth,equal_nan=True) and np.isfinite(fp).all()
    ref.save_prediction(path,fields,ft,fp,seed=s,scenario=name,capacity=cap,source=source,scenario_spec=scene)
    dump(receipt,info(path))
   met=ref.measured_metrics(ft,fp,fields['target_valid'],cap)
   rows.append(dict(scenario=name,scenario_json=scene,**met,nrmse_increase=met['nrmse']/normal['nrmse']-1,nmae_increase=met['nmae']/normal['nmae']-1))
   print(a.task,f'{i+1}/39',name,flush=True)
  for f in selected['source_files']+protocol['data_files']+task['files']:verify(f)
  end=utc();ident.update(end_utc=end,elapsed_seconds=time.monotonic()-t0,status='inference_complete');dump(dest/f'process_{attempt}.json',ident)
  done=dict(status='awaiting_process_exit',task_id=a.task,trained=False,model_or_mode=task['model_or_mode'],horizon=h,seed=s,fault_seed=2026,source_run=str(d),source=source,rows=rows,normal_gate=gate,process=ident,outputs=[info(p) for p in sorted(dest.iterdir()) if p.suffix=='.npz' or p.name=='normal_equivalence.json'])
  dump(dest/'candidate_completed.json',done);state('awaiting_process_exit',39)
 except Exception as e:
  ident.update(end_utc=utc(),status='failed',error=repr(e),traceback=traceback.format_exc())
  dump(dest/f'failure_{attempt}.json',ident);print(traceback.format_exc(),flush=True);raise
if __name__=='__main__':main()
