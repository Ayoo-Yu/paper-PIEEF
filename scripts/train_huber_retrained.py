"""Portable sequential S3R queue. No training unless --run is explicitly supplied."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, importlib.util, json, os, platform, re
import shutil, subprocess, sys, time, traceback, uuid
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode=True

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_json(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def now():return datetime.now(timezone.utc).isoformat()

def inside(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes package')
    return p

def verify(root):
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf8'))
    for x in manifest['files']:
        p=inside(root,x['path'])
        if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('SHA mismatch: '+x['path'])
    return sha(root/'package_manifest.json')

class EpochLog:
    def __init__(self,path,terminal):
        self.file=path.open('w',encoding='utf8',newline='');self.writer=csv.DictWriter(self.file,fieldnames=['epoch','batches','train_loss','val_loss','best_epoch','elapsed_s']);self.writer.writeheader()
        self.terminal=terminal;self.pending='';self.count=0
    def write(self,s):
        self.terminal.write(s);self.pending+=s
        while '\n' in self.pending:
            line,self.pending=self.pending.split('\n',1)
            m=re.search(r'epoch=(\d+)/\d+ batches=(\d+) avg_loss=([^ ]+) val_loss=([^ ]+) best_epoch=(\d+) elapsed=([\d.]+)s',line)
            if m:
                self.writer.writerow(dict(zip(self.writer.fieldnames,m.groups())));self.file.flush();self.count+=1
    def flush(self):self.terminal.flush();self.file.flush()
    def close(self):self.file.close()

def worker(args,entry,run_dir):
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(args.cpu_threads)
    root=args.package_root;project=root/'candidate_project'
    sys.path.insert(0,str(project/'src'))
    spec=importlib.util.spec_from_file_location('s3r_factory',project/'scripts/train_baselines.py')
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    from pieef import baselines as bl
    from pieef.core import set_seed
    from pieef.data import build_data_bundle
    from pieef.artifacts import save_preprocessing,save_run_predictions
    from pieef.revision_data import require_verified_forecast_input
    from pieef.training_timing import save_training_timing
    config_path=inside(root,entry['config']);cfg=yaml.safe_load(config_path.read_text(encoding='utf8'))
    t=cfg['train'];e=cfg['experiment'];d=cfg['data'];cid=cfg['tracking']['config_id']
    assert cid==entry['config_id'] and cid.startswith('s3r_')
    assert e['seeds']==[entry['seed']] and e['horizon']==entry['horizon']
    assert e['model']==entry['model'] and cfg['baselines']['models']==[entry['model']]
    assert (t['epochs'],t['batch_size'],t['patience'],t['min_delta'])==(50,16,5,0.)
    assert t['loss_name']=='huber' and t['huber_delta']==.1 and t['lr']==.001
    assert t.get('max_batches_per_epoch') is None and t.get('max_test_batches') is None
    require_verified_forecast_input(d['revision_protocol'])
    dl=bl.DLTrainConfig(**{k:v for k,v in t.items() if k!='batch_size'})
    device=torch.device(args.device)
    if args.run and device.type!='cuda':raise ValueError('Formal runs require CUDA')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device_info=dict(device=str(device),hostname=platform.node(),platform=platform.platform(),processor=platform.processor(),python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,cpu_threads=torch.get_num_threads(),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    write_json(run_dir/'device.json',device_info)
    shutil.copy2(config_path,run_dir/'config.yaml')
    shutil.copy2(root/'source_manifest.json',run_dir/'source_manifest.json')
    write_json(run_dir/'provenance.json',dict(source_manifest_sha256=sha(root/'source_manifest.json'),package_manifest_sha256=sha(root/'package_manifest.json'),config_sha256=sha(config_path),data_manifest_sha256=sha(root/'data_manifest.json'),command=sys.argv,output_directory=str(run_dir),started_at=now()))
    bundle=build_data_bundle(scada_path=args.data_root/'scada_ecmwf_15min.csv',turbines_path=args.data_root/'turbines.csv',seq_len=d['seq_len'],horizon=e['horizon'],batch_size=t['batch_size'],train_ratio=d['split'][0],val_ratio=d['split'][1],capacity_mw=d['capacity_mw'],revision_protocol=d['revision_protocol'])
    checks={}
    for split in ('val','test'):
        loader=getattr(bundle,split+'_loader')
        # Dataset access only: no model inference, no train-loader RNG consumption.
        truth=np.stack([loader.dataset[i][-1].numpy() for i in range(len(loader.dataset))])*bundle.capacity_mw
        actual=dict(bundle.handler.metadata[split],turbine_ids=np.asarray(bundle.handler.turbine_ids,dtype=str),y_true_mw=truth)
        paths=[root/'references'/f's3_h{e["horizon"]}_{split}.npz']
        if split=='test':paths.append(inside(root,entry['reference']))
        for rp in paths:
            with np.load(rp,allow_pickle=False) as ref:
                for k in ref.files:
                    a=np.asarray(actual[k]);b=ref[k]
                    if k in ('timestamp','origin_timestamp'):
                        # Compare physical instants, tolerating equivalent timezone string formatting.
                        import pandas as pd
                        a=pd.to_datetime(a,utc=True).asi8;b=pd.to_datetime(b,utc=True).asi8
                    np.testing.assert_array_equal(a,b,err_msg=f'{split}: {rp.name}: {k}')
            checks[rp.name]=dict(status='PASS',samples=len(truth),turbines=bundle.num_turbines,valid=int(np.isfinite(truth).sum()),sha256=sha(rp))
    write_json(run_dir/'identity_checks.json',checks)
    save_preprocessing(run_dir,bundle)
    old_shapes={}
    if args.dry_run:
        old_path=project/'configs/baseline_repair'/f'{cid.removeprefix("s3r_")}.yaml'
        old=yaml.safe_load(old_path.read_text(encoding='utf8'))
        assert old['data']==d and old['experiment']==e and old['train']['batch_size']==64
        prior=build_data_bundle(scada_path=args.data_root/'scada_ecmwf_15min.csv',turbines_path=args.data_root/'turbines.csv',seq_len=d['seq_len'],horizon=e['horizon'],batch_size=64,train_ratio=d['split'][0],val_ratio=d['split'][1],capacity_mw=d['capacity_mw'],revision_protocol=d['revision_protocol'])
        for split in ('train','val','test'):
            new_ds=getattr(bundle,split+'_loader').dataset;old_ds=getattr(prior,split+'_loader').dataset
            np.testing.assert_array_equal(new_ds.indices,old_ds.indices)
            for field in ('power','wind','nwp','future_nwp','targets'):
                torch.testing.assert_close(getattr(new_ds,field),getattr(old_ds,field),rtol=0,atol=0,equal_nan=True)
            old_shapes[split]=dict(samples=len(old_ds),sample_shapes=[list(x.shape) for x in old_ds[0]],old_batches=len(getattr(prior,split+'_loader')),new_batches=len(getattr(bundle,split+'_loader')),all_input_arrays_and_indices_exact=True)
        del prior
    # Match native producer ordering: bundle -> seed -> one test batch -> model.
    set_seed(entry['seed']);sample=next(iter(bundle.test_loader))
    model=factory.build_model(entry['model'],d['seq_len'],bundle=bundle,turbines_path=args.data_root/'turbines.csv')
    assert bl.baseline_loss_spec(model,dl)==('huber',.1)
    if args.dry_run:
        model=model.to(device)
        if isinstance(model,bl.ACFormerModel):model._decoder(bundle.num_turbines,device)
        batch=tuple(x.to(device) for x in next(iter(bundle.train_loader)))
        model.train();pred=bl._predict_dl(model,*batch[:4]);loss,count=bl._observed_loss(model,pred,batch[4],dl)
        expected=torch.nn.HuberLoss(delta=.1)(pred[torch.isfinite(batch[4])],batch[4][torch.isfinite(batch[4])])
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        probe=torch.tensor([0.,.05,.2]);target=torch.zeros(3)
        pl,_=bl._observed_loss(model,probe,target,dl)
        torch.testing.assert_close(pl,torch.tensor((0.+.5*.05**2+.1*(.2-.05))/3))
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(x).all() for x in grads)
        result=dict(status='PASS',mode='dry-run',config_id=cid,loss_function='torch.nn.functional.huber_loss',equivalence_check='torch.nn.HuberLoss(delta=0.1)',delta=.1,loss=float(loss),observed_targets=count,backward_finite=True,optimizer_steps=0,epochs_executed=0,patience=dl.patience,min_delta=dl.min_delta,epochs_cap=dl.epochs,selection='full validation observed-target-weighted Huber loss; strictly smaller than best - min_delta',batch_shapes=[list(x.shape) for x in batch],train_samples=len(bundle.train_loader.dataset),val_samples=len(bundle.val_loader.dataset),test_samples=len(bundle.test_loader.dataset),train_batches=len(bundle.train_loader),val_batches=len(bundle.val_loader),test_batches=len(bundle.test_loader),identity_checks=checks,device=device_info,finished_at=now())
        write_json(run_dir/'dry_run.json',result)
        write_json(run_dir/'s3_shape_comparison.json',dict(old_batch_size=64,new_batch_size=16,splits=old_shapes,status='PASS'))
        print('[dry-run] PASS '+cid,flush=True);return
    log=EpochLog(run_dir/'epoch_losses.csv',sys.stdout)
    try:
        with contextlib.redirect_stdout(log):
            model=bl.train_dl_model(model,bundle.train_loader,device,dl,progress_label=cid,val_loader=bundle.val_loader,checkpoint_dir=run_dir)
    finally:log.close()
    assert log.count==model.training_summary['epochs_run']
    save_training_timing(run_dir,model,cfg,entry['seed'],entry['horizon'],device)
    metrics={}
    for split in ('val','test'):
        met,truth,pred=bl.eval_dl_model(model,getattr(bundle,split+'_loader'),device,bundle.capacity_mw)
        save_run_predictions(run_dir,bundle,truth,pred,entry['seed'],entry['horizon'],split=split)
        metrics[split]=met
    write_json(run_dir/'metrics.json',dict(status='completed',run_id=cid,model=entry['model'],seed=entry['seed'],horizon=entry['horizon'],config=cfg,metrics=metrics['test'],validation_metrics=metrics['val'],training_summary=model.training_summary,device=str(device),source_manifest_sha256=sha(root/'source_manifest.json')))
    for fn in ('weights_best.pt','weights_last.pt'):
        state=torch.load(run_dir/fn,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values())
        model.load_state_dict(state,strict=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package-root','data-root','queue','output-root'):p.add_argument('--'+name,type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--device',required=True);p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--config-id');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--attempt-dir',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    for k in ('package_root','data_root','queue','output_root'):setattr(args,k,getattr(args,k).resolve())
    manifest_sha=verify(args.package_root)
    # Allow data relocation, but require exact packaged bytes.
    for x in json.loads((args.package_root/'data_manifest.json').read_text(encoding='utf8'))['files']:
        assert sha(args.data_root/Path(x['path']).name)==x['sha256']
    queue=json.loads(args.queue.read_text(encoding='utf8'))
    frozen=json.loads((args.package_root/'queue.json').read_text(encoding='utf8'))
    assert queue==frozen,'Queue differs from frozen package'
    if args.config_id:queue=[x for x in queue if x['config_id']==args.config_id]
    assert queue and len({x['config_id'] for x in queue})==len(queue)
    if args.worker:
        assert len(queue)==1 and args.attempt_dir
        worker(args,queue[0],args.attempt_dir);return
    args.output_root.mkdir(parents=True,exist_ok=True)
    lock=args.output_root/'runner.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Output root is locked; inspect runner.lock before restarting')
    os.write(fd,json.dumps(dict(pid=os.getpid(),host=platform.node(),started_at=now())).encode());os.close(fd)
    summary=[]
    try:
        for entry in queue:
            cid=entry['config_id'];home=args.output_root/cid;home.mkdir(exist_ok=True)
            marker=home/'completed.json'
            try:
                if marker.exists():
                    done=json.loads(marker.read_text(encoding='utf8'))
                    assert done['config_id']==cid and done['package_manifest_sha256']==manifest_sha
                    for f in done['files']:assert sha(inside(home,f['path']))==f['sha256']
                    print('[skip] '+cid+' verified completed marker',flush=True);summary.append(dict(config_id=cid,status='skipped'));continue
                run_dir=home/('dry_' if args.dry_run else 'attempt_')
                run_dir=home/(run_dir.name+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);run_dir.mkdir()
                started=now();t0=time.perf_counter()
                print('[start] '+cid+(' dry-run' if args.dry_run else ' run'),flush=True)
                cmd=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--package-root',str(args.package_root),'--data-root',str(args.data_root),'--queue',str(args.queue),'--output-root',str(args.output_root),'--device',args.device,'--cpu-threads',str(args.cpu_threads),'--config-id',cid,'--worker','--attempt-dir',str(run_dir),'--dry-run' if args.dry_run else '--run']
                env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
                if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']='-1'
                with (run_dir/'run.log').open('w',encoding='utf8') as log:
                    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',env=env,cwd=args.package_root)
                    for line in proc.stdout:
                        log.write(line);log.flush()
                        if line.startswith('[train]'):print(line.rstrip(),flush=True)
                    rc=proc.wait()
                result=dict(config_id=cid,status='passed' if args.dry_run and rc==0 else 'completed' if rc==0 else 'failed',returncode=rc,pid=proc.pid,command=cmd,started_at=started,finished_at=now(),elapsed_s=time.perf_counter()-t0,directory=str(run_dir))
                write_json(run_dir/'execution.json',result)
                if rc==0 and args.run:
                    needed=['metrics.json','weights_best.pt','weights_last.pt','predictions.npz','val_predictions.npz','epoch_losses.csv','config.yaml','source_manifest.json','device.json','identity_checks.json','training_timing.json']
                    assert all((run_dir/f).is_file() for f in needed)
                    files=[dict(path=f.relative_to(home).as_posix(),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(run_dir.rglob('*')) if f.is_file()]
                    write_json(marker,dict(config_id=cid,package_manifest_sha256=manifest_sha,completed_at=now(),files=files))
                summary.append(result);print(f'[end] {cid} {result["status"]} {result["elapsed_s"]:.1f}s',flush=True)
            except Exception:
                error=traceback.format_exc();print('[end] '+cid+' FAILED '+error.splitlines()[-1],flush=True)
                error_path=home/('runner_error_'+uuid.uuid4().hex+'.log');error_path.write_text(error,encoding='utf8')
                summary.append(dict(config_id=cid,status='failed',error=error))
        sp=args.output_root/('summary_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json');write_json(sp,summary)
        print('[summary] '+str(sp),flush=True)
    finally:lock.unlink()
    if any(x['status']=='failed' for x in summary):raise SystemExit(1)

if __name__=='__main__':main()
