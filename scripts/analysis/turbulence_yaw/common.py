"""Read-only inputs; all outputs are confined to this analysis directory."""
import os, sys, json, hashlib, datetime, time
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
import pandas as pd
O = Path(__file__).resolve().parents[1]
V1 = O.parent
W = O.parents[2]
PROJECT = W/'round4_prediction_20261024/project'
DATA = Path('data/kelmarsh')
ZIP = Path('data/kelmarsh/Kelmarsh_SCADA_2024_5962.zip')
SEEDS = (13,42,3407)
HORIZONS = (1,4,16)
TI_LAYERS = ('low','medium','high','very_high','undefined')
YAW_LAYERS = ('lt5','5to10','10to20','ge20')
INPUTS = {}
INCIDENTS = []
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def register(p):
    p=Path(p).resolve(); key=str(p)
    if key not in INPUTS:INPUTS[key]=dict(path=key,sha256=sha(p),bytes=p.stat().st_size)
    return p
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    return x
def js(name,x):
    p=O/name;assert p.resolve().is_relative_to(O)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def readjson(p):return json.loads(register(p).read_text(encoding='utf-8-sig'))
def npz(p):
    with np.load(register(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csv(name,rows):
    p=O/name;assert p.resolve().is_relative_to(O)
    pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
def incident(code,reason,action,kind='documented_detail'):
    row=dict(utc=utc(),code=code,kind=kind,reason=reason,action=action)
    if not any(x['code']==code for x in INCIDENTS):INCIDENTS.append(row)
    js('operational_incidents.json',INCIDENTS)
def log(message):
    print(message,flush=True)
    with (O/'run_log.md').open('a',encoding='utf-8') as f:f.write(f'\n- {utc()} {message}\n')
def begin(command):
    t=time.perf_counter();log('START command: `'+command+'`');return t
def end(t,command,status='completed'):log(f'END {command}; status={status}; duration_seconds={time.perf_counter()-t:.6f}')
def masks_ti(ws,std):
    ti=np.divide(std,ws,out=np.full_like(ws,np.nan,dtype=float),where=(ws>=4)&np.isfinite(ws)&np.isfinite(std))
    return ti,dict(low=ti<.124,medium=(ti>=.124)&(ti<.166),high=ti>=.166,very_high=ti>=.216,undefined=np.isfinite(ws)&(ws<4))
def masks_yaw(phi):
    a=np.abs(phi)
    return dict(lt5=a<5,**{'5to10':(a>=5)&(a<10),'10to20':(a>=10)&(a<20)},ge20=a>=20)
def metrics(y,p,v,capacity):
    e=((p.astype(float)-y.astype(float))/capacity)[v]
    return dict(n_valid=int(v.sum()),nrmse_pct=100*np.sqrt(np.mean(e*e)) if len(e) else np.nan,nmae_pct=100*np.mean(abs(e)) if len(e) else np.nan)
def control(h,s):return W/f'补充验证_20261023/M_controls/VF_Kelmarsh_h{h}_seed{s}'
def checkpoint():
    if (O/'operational_incidents.json').exists():INCIDENTS[:]=json.loads((O/'operational_incidents.json').read_text(encoding='utf-8'))
    if (O/'k0_identity_checks.json').exists():
        k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf-8'))
        INPUTS.update({x['path']:x for x in k.get('input_files',[])})
def save_k0(k):k['input_files']=list(INPUTS.values());js('k0_identity_checks.json',k)
