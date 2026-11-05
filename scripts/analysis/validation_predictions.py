from pathlib import Path
import sys,os,json,time,importlib.util,gc,hashlib
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parents[1];SRC=P/'next_batch/merged_v2/project';DATA=Path('data/wind_farm_A')
sys.path.insert(0,str(SRC/'src'));torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('table_evaluation',SRC/'scripts/evaluate_revision.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
runs=json.loads((O/'G4_validation_runs.json').read_text(encoding='utf-8'));seeds=(13,42,3407)
runs=[r for r in runs if r['metadata']['config']['tracking']['config_id'].startswith('energy_r3s2_') and r['metadata']['seed'] in seeds and r['metadata']['horizon']==16 and r['metadata']['config']['experiment']['group'] in ('main','sensitivity')]
assert len(runs)==6
base=None;audit=[];start=time.time()
for r in runs:
 m=r['metadata'];cfg=json.loads(json.dumps(m['config']));cfg['data']['scada_path']=str(DATA/'scada_ecmwf_15min.csv');cfg['data']['turbines_path']=str(DATA/'turbines.csv')
 bundle=ev._bundle(dict(config=cfg,source_dir=Path(r['directory']),horizon=16))
 def fields(split):
  ds=getattr(bundle,split+'_loader').dataset
  y=(ds.targets[ds.indices].numpy()*np.asarray(bundle.capacity_mw)).astype('float32')
  return dict(**{k:np.asarray(v,dtype=str) if np.asarray(v).dtype.kind=='O' else np.asarray(v) for k,v in bundle.handler.metadata[split].items()},turbine_ids=np.asarray(bundle.handler.turbine_ids,dtype=str),capacity_mw=np.asarray(bundle.capacity_mw),y_true_mw=y)
 val=fields('val');test=fields('test')
 with np.load(Path(r['directory'])/'predictions.npz') as saved:
  for k in ('timestamp','origin_timestamp','turbine_ids','target_valid','events'):assert np.array_equal(test[k],saved[k]),(m['run_id'],k)
  assert np.array_equal(test['y_true_mw'],saved['y_true_mw'],equal_nan=True)
 if base is None:
  base={k:v.copy() for k,v in val.items()};np.savez_compressed(O/'G4_reconstructed_validation_identity.npz',**base)
 checks={k:(bool(np.array_equal(val[k],base[k],equal_nan=True)) if val[k].dtype.kind=='f' else bool(np.array_equal(val[k],base[k]))) for k in val}
 assert all(checks.values()),(m['run_id'],checks)
 assert len(val['timestamp'])==m['validation_metrics']['num_samples']
 audit.append(dict(run_id=m['run_id'],source_dir=r['directory'],checks=checks,status='PASS_reconstructed_definition',n_time=len(val['timestamp']),n_valid=int(val['target_valid'].sum()),limitation='Original validation per-point predictions absent; shared deterministic validation identity reconstructed from actual saved preprocessing/config/producer; original validation_metrics retained unchanged.'))
 print(m['run_id'],'PASS',flush=True)
 del bundle;gc.collect()
result=dict(status='PASS',run_count=len(audit),n_time=len(base['timestamp']),n_valid=int(base['target_valid'].sum()),first_target=str(base['timestamp'][0]),last_target=str(base['timestamp'][-1]),audits=audit,elapsed_seconds=time.time()-start,data=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (DATA/'scada_ecmwf_15min.csv',DATA/'turbines.csv')],inference_performed=False)
(O/'G4_validation_identity_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'进度.md').open('a',encoding='utf-8') as f:f.write('\n- T3 验证身份重建通过：6/6，全部目标时间、起点、机组、真值、有效掩码、容量完全一致；原验证指标原样复用，未重新推理。\n')
print('DONE',time.time()-start,flush=True)
