"""Finite, read-only CPU inference of the 19 frozen neural comparison controls.

No fitting, checkpoint selection, or modification of project/old artifacts.
Only a genuine child exit0 plus full prediction/attention/source/D-E checks
accepts a control. Non-complete atomic claims stop rather than silently retry.
"""
from pathlib import Path
import argparse, contextlib, datetime, hashlib, json, os, shutil, socket, subprocess, sys, time
import numpy as np
import psutil

WORK=Path(__file__).resolve().parents[1];PROJECT=WORK/'project';OUT=WORK.parent/'补充验证_20261023'
EBASE=Path('results/paper')
INDEX=WORK/'audit/baseline/control_index.json';STATE=WORK/'postprocess/control_state'
DEST=OUT/'M_controls';EDEST=EBASE/'results/M_controls'
FAMILIES={'VF_wind_farm_A':5,'direction_off_wind_farm_A':5,'VF_Kelmarsh':9}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(t,p)
def identity(pid):
    p=psutil.Process(pid);return dict(pid=pid,create_time=p.create_time(),command=p.cmdline(),cwd=p.cwd())
def control_id(row):return f"{row['family']}_h{row['horizon']}_seed{row['seed']}"
def selected():
    import collections
    records=[x for x in read(INDEX)['controls'] if x['family'] in FAMILIES]
    require(dict(collections.Counter(r['family'] for r in records))==FAMILIES,'Exact19 neural controls required')
    require(len({control_id(x) for x in records})==19,'Duplicate control ID')
    return sorted(records,key=lambda x:(list(FAMILIES).index(x['family']),x['horizon'],x['seed']))
def resolved(value):
    p=Path(value);return p if p.is_absolute() else PROJECT/p
@contextlib.contextmanager
def lock(p):
    import msvcrt
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def infer(row):
    import copy
    import torch
    sys.path.insert(0,str(PROJECT/'src'))
    from pieef.core import calc_metrics,set_seed
    from pieef.data import build_data_bundle
    from pieef.model_factory import build_main_model
    from pieef.cfd_features import attach_local_forecast_inputs
    from pieef.r4_attention import evaluate_and_record
    from pieef.artifacts import save_preprocessing,save_run_predictions
    torch.set_num_threads(2)
    require(not torch.cuda.is_available(),'CPU control lane must actually hide CUDA')
    rid=control_id(row);directory=DEST/rid;directory.mkdir(parents=True,exist_ok=False)
    source=Path(row['artifact_directory']);oldmetrics=read(source/'metrics.json')
    cfg=copy.deepcopy(row['config']);require(oldmetrics['config']==cfg,'Old control config changed')
    require(oldmetrics['seed']==row['seed'] and oldmetrics['horizon']==row['horizon'],'Control seed/horizon differs')
    d=cfg['data'];m=cfg['model'];protocol=copy.deepcopy(d['revision_protocol'])
    if protocol.get('nwp_vintages_path'):protocol['nwp_vintages_path']=str(resolved(protocol['nwp_vintages_path']))
    if m.get('manufacturer_curve_path'):m['manufacturer_curve_path']=str(resolved(m['manufacturer_curve_path']))
    # Existing config semantics stay fixed; the path-only resolution above is
    # recorded and is not a model/scientific-identity modification.
    set_seed(int(row['seed']));t0=time.perf_counter()
    bundle=build_data_bundle(resolved(d['scada_path']),resolved(d['turbines_path']),int(d['seq_len']),int(row['horizon']),
        int(cfg['train']['batch_size']),float(d['split'][0]),float(d['split'][1]),float(d.get('capacity_mw',2.)),
        revision_protocol=protocol)
    require(attach_local_forecast_inputs(bundle,d,m) is bundle,'Disabled control wrapper must be no-op')
    device=torch.device('cpu');model=build_main_model(bundle,d['seq_len'],m,device)
    weights=source/'weights_best.pt';model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.eval();save_preprocessing(directory,bundle,revin_missing_policy=m.get('revin_missing_policy','legacy'))
    for name in ('metrics.json','preprocessing.npz','predictions.npz'):
        shutil.copy2(source/name,directory/('source_'+name))
    # Reference hashes plus an exact, independently usable copy retain the real
    # selected original checkpoint. It is never fine-tuned or saved anew.
    shutil.copy2(weights,directory/'source_weights_best.pt')
    atomic(directory/'source_control.json',dict(row,source_index_sha256=sha(INDEX),
        original_checkpoint_sha256=sha(weights),project_source_manifest_sha256=sha(WORK/'source_manifest.json'),
        original_project_unchanged=True,control_inference_only=True,original_config=oldmetrics['config']))
    outputs={}
    for split in ('val','test'):
        metrics,y,p=evaluate_and_record(model,getattr(bundle,split+'_loader'),device,bundle,split,directory)
        save_run_predictions(directory,bundle,y,p,row['seed'],row['horizon'],split)
        outputs[split]=metrics
    # Original source archives are locally trusted author artifacts. Convert
    # only their index arrays in memory; never change the original NPZ.
    with np.load(source/'predictions.npz',allow_pickle=True) as z:
        original={k:z[k] for k in z.files}
    with np.load(directory/'predictions.npz',allow_pickle=False) as z:
        repeated={k:z[k] for k in z.files}
    require(original['y_pred_mw'].shape==repeated['y_pred_mw'].shape,'Original/repeated prediction support differs')
    for name in ('timestamp','origin_timestamp','turbine_ids'):
        require(np.array_equal(np.asarray(original[name],dtype=str),np.asarray(repeated[name],dtype=str)),'Original/repeated index differs: '+name)
    require(np.array_equal(original['y_true_mw'],repeated['y_true_mw'],equal_nan=True),'Measured target differs')
    capacity=np.asarray(bundle.capacity_mw);delta=np.abs(original['y_pred_mw']-repeated['y_pred_mw'])/capacity
    tolerance=1e-5+1e-5*np.abs(original['y_pred_mw']/capacity)
    require(np.isfinite(repeated['y_pred_mw']).all() and (delta<=tolerance).all(),'CPU original-control reproduction tolerance failed')
    primary=dict(zip(('nmae','nrmse','r2'),calc_metrics(original['y_true_mw'],original['y_pred_mw'],capacity)))
    result=dict(status='completed',control_id=rid,family=row['family'],seed=row['seed'],horizon=row['horizon'],
        inference_only=True,training_performed=False,selected_checkpoint='original weights_best.pt, no re-selection',
        device='cpu',threads=2,source_control_directory=str(source),validation_metrics=outputs['val'],
        metrics=primary,repeated_test_metrics=outputs['test'],primary_test_origin='original verified stored test predictions',
        default_reproduction_gate=dict(status='PASS',max_abs_error_pu=float(delta.max()),
            tolerance='1e-5 pu + 1e-5 relative',full_indices_truth_equal=True),
        child_identity=identity(os.getpid()),control_script_sha256=sha(__file__),elapsed_inference_and_save_seconds=time.perf_counter()-t0,
        source_config=oldmetrics['config'],config=cfg)
    atomic(directory/'metrics.json',result)
    print(json.dumps({'event':'control_inference_outputs_saved','control_id':rid,'elapsed_seconds':result['elapsed_inference_and_save_seconds']},ensure_ascii=False),flush=True)

def complete(row,child,code,wall,peak):
    rid=control_id(row);directory=DEST/rid
    require(code==0,'Control inference child failed; preserve files and stop')
    m=read(directory/'metrics.json');require(m['control_id']==rid and m['default_reproduction_gate']['status']=='PASS','Control output identity/gate differs')
    needed=['metrics.json','preprocessing.npz','predictions.npz','val_predictions.npz','source_control.json','source_weights_best.pt',
            'source_predictions.npz','source_metrics.json','source_preprocessing.npz',
            'val_attention_diagnostics.npz','test_attention_diagnostics.npz','val_attention_summary.json','test_attention_summary.json']
    for n in needed:require((directory/n).is_file(),'Missing actual control output '+n)
    log=STATE/'claims'/rid/'inference.log'
    require(log.is_file(),'Actual finite inference log is missing')
    shutil.copy2(log,directory/'inference.log')
    members={p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in directory.rglob('*') if p.is_file()}
    target=EDEST/rid
    require(not target.exists(),'Existing second control directory requires review, not overwrite')
    target.mkdir(parents=True)
    for rel,desc in members.items():
        q=target/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/rel,q)
        require(q.stat().st_size==desc['bytes'] and sha(q)==desc['sha256'],'Control D/E member differs')
    receipt=dict(status='completed',control_id=rid,family=row['family'],seed=row['seed'],horizon=row['horizon'],
        original_control_directory=row['artifact_directory'],actual_child=child,natural_returncode=code,
        full_child_wall_seconds=wall,peak_child_RSS_bytes=peak,completed_utc=now(),
        inference_only=True,training_performed=False,D_path=str(directory),E_path=str(target),files=members,
        all_full_index_predictions_attention_D_E_verified=True)
    for p in (directory/'completed.json',target/'completed.json'):atomic(p,receipt)
    return receipt

def supervise(limit):
    STATE.mkdir(parents=True,exist_ok=True)
    with lock(STATE/'supervisor.lock'):
        me=identity(os.getpid());atomic(STATE/'state.json',dict(status='running',parent=me,started_utc=now(),controls=19))
        completed=[];new=0
        for row in selected():
            rid=control_id(row);claim=STATE/'claims'/rid;receipt=claim/'receipt.json'
            if receipt.exists():
                prior=read(receipt);require(prior.get('status')=='completed','Prior control failure/unfinished task; inspect before retry')
                d,e=Path(prior['D_path']),Path(prior['E_path'])
                for rel,desc in prior['files'].items():
                    require(sha(d/rel)==desc['sha256'] and sha(e/rel)==desc['sha256'],'Previously completed control changed')
                completed.append(rid);continue
            if limit is not None and new>=limit:break
            require(not claim.exists(),'Atomic control claim already exists')
            claim.mkdir(parents=True);command=[sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'--single',rid]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LOKY_MAX_CPU_COUNT='2',PYTHONUTF8='1')
            record=dict(status='starting',control_id=rid,source_index_sha256=sha(INDEX),control_script_sha256=sha(__file__),parent=me,command=command,started_utc=now(),inference_only=True)
            atomic(receipt,record);start=time.perf_counter();peak=0
            with (claim/'inference.log').open('x',encoding='utf8') as log:
                p=subprocess.Popen(command,cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                actual=identity(p.pid);record.update(status='running',child=actual);atomic(receipt,record)
                atomic(STATE/'state.json',dict(status='running',parent=me,current_control=rid,child=actual,completed=completed,observed_utc=now()))
                print(json.dumps({'event':'control_started','control_id':rid,'child':actual},ensure_ascii=False),flush=True)
                while p.poll() is None:
                    try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
                    except psutil.NoSuchProcess:pass
                    time.sleep(1)
                code=p.wait()
            if code:
                record.update(status='failed',natural_returncode=code,ended_utc=now(),full_child_wall_seconds=time.perf_counter()-start)
                atomic(receipt,record);atomic(STATE/'state.json',dict(status='failed',parent=me,control_id=rid,actual_natural_returncode=code));raise RuntimeError('Control failed: '+rid)
            result=complete(row,actual,code,time.perf_counter()-start,peak);atomic(receipt,result);completed.append(rid);new+=1
            atomic(STATE/'state.json',dict(status='running',parent=me,completed=completed,observed_utc=now()))
            print(json.dumps({'event':'control_completed','control_id':rid,'wall_seconds':result['full_child_wall_seconds'],'peak_RSS_bytes':peak},ensure_ascii=False),flush=True)
        atomic(STATE/'state.json',dict(status='all19_completed' if len(completed)==19 else 'finite_limit_completed',parent=me,completed=completed,new_completed_this_invocation=new,ended_utc=now()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--single');ap.add_argument('--supervise',action='store_true');ap.add_argument('--max-controls',type=int)
    args=ap.parse_args()
    if args.single:
        matches=[r for r in selected() if control_id(r)==args.single];require(len(matches)==1,'Unknown frozen control');infer(matches[0])
    else:
        require(args.supervise,'Explicit finite supervision mode required');supervise(args.max_controls)
