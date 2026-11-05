"""Finite R4 F1 LightGBM+CFD runner; control mode only reuses existing trees."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[key]='4'
from pathlib import Path
import argparse,datetime,hashlib,json,platform,shutil,sys,time,traceback,warnings
import joblib,numpy as np,torch,yaml,psutil,lightgbm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pieef.data import build_data_bundle
from pieef.baselines import _iter_flat_xy,_training_flat_cache,_baseline_metrics
from pieef.core import set_seed
from pieef.artifacts import save_preprocessing,save_run_predictions,write_json_atomic
from pieef.r4_lightgbm_features import use_stored_wind_farm_A_preprocessing,append_shared_cfd_features
torch.set_num_threads(4)
warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMRegressor was fitted with feature names',category=UserWarning)

def path(x):
    p=Path(x);return p if p.is_absolute() else ROOT/p
def desc(p):
    p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def j(p):return json.loads(Path(p).read_bytes())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def need(v,m):
    if not v:raise ValueError(m)

def validate_config(cfg):
    ref=path(cfg['round4_reference']['artifact_directory']);m=j(ref/'metrics.json');base=m['config'];seed=int(cfg['experiment']['seeds'][0])
    for field,name in [('metrics_sha256','metrics.json'),('model_sha256','model.pkl'),('preprocessing_sha256','preprocessing.npz')]:
        need(cfg['round4_reference'].get(field)==desc(ref/name)['sha256'],'Published control SHA differs: '+name)
    need(len(cfg['experiment']['seeds'])==1 and seed in (13,21,42,87,3407),'Exactly one frozen F1 seed required')
    need(m['status']=='completed' and m['model']=='LightGBM' and int(m['horizon'])==16 and int(m['seed'])==seed,'Exact matching published LightGBM control required')
    need(cfg['experiment']['horizon']==16 and cfg['train']==base['train'],'F1 horizon or original tree training protocol changed')
    for key in ('seq_len','split','capacity_mw','revision_protocol','data_version'):
        need(cfg['data'].get(key)==base['data'].get(key),'F1 original data protocol changed: '+key)
    a=base['model']['architecture'];expected={'objective':'regression','n_estimators':300,'learning_rate':.05,'num_leaves':64,'subsample':.8,'colsample_bytree':.8}
    need(all(a.get(k)==v for k,v in expected.items()),'Original fixed LightGBM architecture differs')
    need(cfg['model'].get('architecture')==a and cfg.get('baselines',{}).get('models')==['LightGBM'],'Candidate must retain original LightGBM model settings')
    need(cfg['model'].get('direction_source')=='per_turbine' and cfg['model'].get('nwp_speed_correction')=='cfd_speedup','F1 must add exactly M2 information')
    need(cfg['data']['local_wind']['mode']=='cfd' and cfg['data']['local_wind']['site']=='wind_farm_A','F1 scope is wind_farm_A CFD')
    return ref,m,seed

def build_bundle(cfg,ref):
    d=cfg['data'];bundle=build_data_bundle(path(d['scada_path']),path(d['turbines_path']),int(d['seq_len']),16,int(cfg['train']['batch_size']),float(d['split'][0]),float(d['split'][1]),float(d.get('capacity_mw',2)),revision_protocol=d['revision_protocol'])
    pp=use_stored_wind_farm_A_preprocessing(bundle,path(d['scada_path']),ref/'preprocessing.npz')
    return bundle,pp

def predict(model,bundle,split):
    truth=[];pred=[]
    for x,y,n in _iter_flat_xy(getattr(bundle,split+'_loader')):
        p=np.asarray(model.predict(x,num_threads=4));need(p.shape==y.shape,'Wrong flattened prediction shape')
        truth.append(y.reshape(-1,n));pred.append(p.reshape(-1,n))
    y=np.concatenate(truth)*bundle.capacity_mw;p=np.maximum(np.concatenate(pred)*bundle.capacity_mw,0)
    need(np.isfinite(p).all(),'Nonfinite LightGBM predictions');return _baseline_metrics(y,p,bundle.capacity_mw),y,p

def compare_old(ref,bundle,split,y,p):
    with np.load(ref/('predictions.npz' if split=='test' else 'val_predictions.npz'),allow_pickle=False) as z:
        need(np.array_equal(y,z['y_true_mw'],equal_nan=True),'Original measured truths differ')
        need(np.array_equal(bundle.handler.turbine_ids,z['turbine_ids']),'Turbine identities differ')
        for k,v in bundle.handler.metadata[split].items():need(np.array_equal(v,z[k]),'Control index/protocol differs: '+k)
        error=np.abs(p-z['y_pred_mw']);tol=1e-5*bundle.capacity_mw+1e-5*np.abs(z['y_pred_mw'])
        old_metrics=_baseline_metrics(z['y_true_mw'],z['y_pred_mw'],bundle.capacity_mw);new_metrics=_baseline_metrics(y,p,bundle.capacity_mw)
        return {'raw_truth_origin_timestamp_turbine_mask_exact':True,'max_abs_prediction_difference_mw':float(error.max()),'cells_outside_VF_reference_tolerance':int((error>tol).sum()),'cells':int(error.size),'diagnostic_not_the_VF_adoption_gate':True,'original_published_metrics':old_metrics,'same_tree_same_platform_metrics':new_metrics,'metric_difference':{k:new_metrics[k]-old_metrics[k] for k in old_metrics}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output');ap.add_argument('--mode',choices=['gate','control','fit'],default='gate');args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'));ref,original,seed=validate_config(cfg)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and torch.get_num_threads()==4,'CPU-only4thread guard')
    output=path(args.output) if args.output else path(cfg['tracking']['artifact_root'])/cfg['tracking']['config_id']
    if args.mode!='gate':
        need(not output.exists(),'Refuse duplicate/partial output; retain previous attempt and choose a reviewed unique output')
        output.mkdir(parents=True)
    started=utc();clock=time.perf_counter();me=psutil.Process();bundle,pp=build_bundle(cfg,ref)
    provenance={'published_control':str(ref),'control_metrics':desc(ref/'metrics.json'),'control_model':desc(ref/'model.pkl'),'original_preprocessing':pp,'mode':args.mode,'effective_CFD_features_applied':args.mode!='control','control_trees_refitted':False,'original_base_feature_count':96*(2+bundle.nwp_dim)+bundle.nwp_dim,'appended_feature_count':3 if args.mode!='control' else 0}
    if args.mode!='control':
        # No random draws occur while constructing the fixed target features.
        provenance['local_wind']=append_shared_cfd_features(bundle,cfg['data'],cfg['model'])
    if args.mode=='gate':
        sample=next(_iter_flat_xy(bundle.test_loader));need(sample[0].shape[1]==1359 and np.isfinite(sample[0]).all(),'F1 fixed original1356+3 features required')
        print(json.dumps({'status':'F1_DATA_FEATURE_GATE_PASS','features':1359,'train_windows':len(bundle.train_loader.dataset),'val_windows':len(bundle.val_loader.dataset),'test_windows':len(bundle.test_loader.dataset),'provenance':provenance},ensure_ascii=False));return
    source_files=[Path(__file__),ROOT/'src/pieef/r4_lightgbm_features.py',ROOT/'src/pieef/cfd_features.py',ROOT/'src/pieef/revision_data.py',ROOT/'src/pieef/baselines.py']
    write_json_atomic(output/'config.json',cfg);write_json_atomic(output/'source_manifest.json',{'files':[desc(p) for p in source_files]});write_json_atomic(output/'provenance.json',provenance)
    write_json_atomic(output/'execution_started.json',{'pid':os.getpid(),'create_time':me.create_time(),'argv':sys.argv,'cwd':str(Path.cwd()),'started_utc':started,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'cpu_threads':4})
    try:
        save_preprocessing(output,bundle)
        set_seed(seed)
        # Preserve the original baseline's pre-fit sequential sample draw.
        next(iter(bundle.test_loader))
        if args.mode=='control':
            model=joblib.load(ref/'model.pkl');shutil.copy2(ref/'model.pkl',output/'model.pkl')
        else:
            model=lightgbm.LGBMRegressor(objective='regression',n_estimators=300,learning_rate=.05,num_leaves=64,subsample=.8,colsample_bytree=.8,random_state=seed,verbosity=-1,force_col_wise=True,histogram_pool_size=64,n_jobs=4)
            with _training_flat_cache(bundle.train_loader,cache_dir=output/'_tabular_cache') as (x,y,_):
                need(np.isfinite(y).all(),'Only finite original measured training targets may enter F1')
                training_rows=int(len(y));feature_count=int(x.shape[1]);need(feature_count==1359,'Original+3 feature layout differs')
                print(json.dumps({'state':'fit_starting','training_rows':training_rows,'features':feature_count,'seed':seed}),flush=True)
                model.fit(x,y)
            joblib.dump(model,output/'model.pkl')
        # Native LightGBM fopen cannot write this Windows Unicode project path.
        # The full native model string is identical tree content, written by Python.
        (output/'booster.txt').write_text(model.booster_.model_to_string(),encoding='utf8')
        test,y,p=predict(model,bundle,'test');val,vy,vp=predict(model,bundle,'val')
        save_run_predictions(output,bundle,y,p,seed,16);save_run_predictions(output,bundle,vy,vp,seed,16,split='val')
        if args.mode=='control':write_json_atomic(output/'platform_numerics_comparison.json',{'test':compare_old(ref,bundle,'test',y,p),'val':compare_old(ref,bundle,'val',vy,vp),'not_a_refit':True,'same_serialized_trees_preserved':desc(output/'model.pkl')['sha256']==desc(ref/'model.pkl')['sha256']})
        timing={'started_utc':started,'completed_utc':utc(),'complete_wall_seconds':time.perf_counter()-clock,'includes':'bundle,preprocessing,sharedfeatures,fullfit(ifcandidate),val/testprediction,modelandartifactsave','mode':args.mode}
        write_json_atomic(output/'training_timing.json',timing)
        write_json_atomic(output/'metrics.json',{'run_id':cfg['tracking']['config_id']+('_control_reinference' if args.mode=='control' else ''),'status':'completed','model':'LightGBM','variant':'LightGBM_same_platform_control' if args.mode=='control' else 'LightGBM+CFD','seed':seed,'horizon':16,'evaluation_split':'test','metrics':test,'validation_metrics':val,'config':cfg,'mode':args.mode,'new_training':args.mode=='fit','features':1356 if args.mode=='control' else 1359,'training_summary':None if args.mode=='control' else {'rows':training_rows,'features':feature_count,'n_estimators_budget':300,'actual_tree_count':model.booster_.num_trees()},'device':'cpu','lightgbm_version':lightgbm.__version__})
        members={p.name:desc(p) for p in output.iterdir() if p.is_file()}
        write_json_atomic(output/'completed.json',{'status':'COMPLETED_OUTPUTS_SAVED','mode':args.mode,'config_sha256':desc(output/'config.json')['sha256'],'files':members,'completion_utc':utc(),'requires_parent_natural_exit_and_D_E_verification':True})
        print(json.dumps({'status':'completed','output':str(output),'metrics':test,'validation_metrics':val,'wall_seconds':timing['complete_wall_seconds']},ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json_atomic(output/'failed.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'failed_utc':utc(),'all_partial_outputs_preserved':True});raise

if __name__=='__main__':main()
