"""Read-only reproduction of saved TI and yaw summaries; never imports torch.

Only named, inspected numerical functions are extracted from source ASTs.
All writers are redirected to raw_reproduction in this output folder.
"""
from pathlib import Path
import os, sys, ast, json, hashlib, zipfile, csv as csvlib
sys.dont_write_bytecode = True
EXTRA = Path(os.environ.get("TEMP", r"results/paper")) / "pi_eef_figure_packages"
if EXTRA.exists():
    sys.path.insert(0, str(EXTRA))
os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent
R = OUT/'raw_reproduction'
EXP = Path(r'results/paper')
SRC = EXP/'补充验证_20261024/K_turbulence_yaw/v2_勘误后'
DATA = Path(r'data/kelmarsh')
ZIP = Path(r'data/kelmarsh/Kelmarsh_SCADA_2024_5962.zip')
PROJECT = EXP/'round4_prediction_20261024/project'
INPUTS = {}
def register(path):
    path=Path(path)
    if str(path) not in INPUTS:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):h.update(block)
        INPUTS[str(path)]=h.hexdigest()
    return path
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return str(value) if isinstance(value,(Path,pd.Timestamp)) else value
def js(name,value):
    R.mkdir(parents=True,exist_ok=True)
    (R/name).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):
    R.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(R/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def incident(*args):pass
def log(message):print(message,flush=True)
def save_k0(k):pass
TI_LAYERS=('low','medium','high','very_high','undefined')
YAW_LAYERS=('lt5','5to10','10to20','ge20')

def load_functions(path,names,transform=None):
    tree=ast.parse(register(path).read_text(encoding='utf-8-sig'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    if transform:selected=[transform.visit(n) for n in selected]
    unit=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    exec(compile(unit,str(path),'exec'),globals())

class AvoidModuleImport(ast.NodeTransformer):
    """Replace only the producer's revision_data module import with its pure function."""
    def visit_Assign(self,node):
        # The producer places path/spec/module assignments and exec_module on one line.
        if any(isinstance(t,ast.Name) and t.id in {'path','spec','module'} for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='exec_module':return None
        return self.generic_visit(node)
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='module' and node.attr=='_power_events':
            return ast.copy_location(ast.Name(id='_power_events',ctx=ast.Load()),node)
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='pd' and node.func.attr in {'DatetimeIndex','date_range'}:
            return ast.copy_location(ast.Call(func=ast.Attribute(value=node,attr='as_unit',ctx=ast.Load()),args=[ast.Constant(value='ns')],keywords=[]),node)
        return node

def main():
    R.mkdir(parents=True,exist_ok=True)
    load_functions(SRC/'scripts/common.py',{'masks_ti','masks_yaw'})
    load_functions(PROJECT/'src/pieef/revision_data.py',{'_power_events'})
    load_functions(SRC/'scripts/raw_data.py',{'wrap','header_info','run'},AvoidModuleImport())
    raw_run=globals()['run']
    load_functions(SRC/'scripts/ti_analysis.py',{'power_curves'})
    load_functions(SRC/'scripts/yaw_analysis.py',{'fit_p','power_ratio'})
    k=json.loads(register(SRC/'k0_identity_checks.json').read_text(encoding='utf-8-sig'))
    d=raw_run(k)
    power_curves(d)
    power_ratio(k,d)
    checks=[]
    for name,keys,cols in [
        ('k2_ti_power_curves.csv',['turbine','stratum','wind_lower_ms'],['wind_upper_ms','wind_center_ms','n_records','mean_power_mw']),
        ('k4_yaw_power_ratio.csv',['reference_speed','bin_type','stratum'],['abs_phi_lower_deg','abs_phi_upper_deg','abs_phi_center_deg','n_records','mean_power_ratio','ci95_lower','ci95_upper','n_valid_bootstrap'])]:
        old=pd.read_csv(register(SRC/name),float_precision='round_trip').set_index(keys).sort_index()
        new=pd.read_csv(R/name,float_precision='round_trip').set_index(keys).sort_index()
        assert old.index.equals(new.index)
        for col in cols:
            a=old[col].to_numpy(float);b=new[col].to_numpy(float)
            finite=np.isfinite(a)&np.isfinite(b)
            err=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0
            ok=bool(np.array_equal(np.isnan(a),np.isnan(b)) and np.array_equal(np.isinf(a),np.isinf(b)) and np.allclose(a[finite],b[finite],rtol=0,atol=1e-12))
            checks.append(dict(file=name,column=col,rows=len(a),max_abs_difference=err,status='PASS' if ok else 'FAIL'))
    old=json.loads(register(SRC/'k4_cos_fit.json').read_text(encoding='utf-8'))
    new=json.loads((R/'k4_cos_fit.json').read_text(encoding='utf-8'))
    for a,b in zip(old['fits'],new['fits']):
        for col in ['p','ci95_lower','ci95_upper']:
            err=abs(a[col]-b[col]);checks.append(dict(file='k4_cos_fit.json',reference=a['reference_speed'],column=col,max_abs_difference=err,status='PASS' if err<1e-9 else 'FAIL'))
    js('reproduction_audit.json',dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'MISMATCH',checks=checks,source_sha256=INPUTS,operations='Pure saved-data filtering, bin aggregation, least-squares fit and day bootstrap only. No model imports, weights, inference, training, or GPU.',source_function_extraction='AST selects named functions only; revision_data module import is removed and _power_events pure function is extracted directly. Output functions are confined to this raw_reproduction directory. DatetimeIndex and date_range explicitly use ns resolution because installed pandas defaults to us whereas original pandas used ns; timestamp values are unchanged.',ti_sampling='15-minute grid; original power events include 3-sigma anomaly exclusions; stop/derating excluded',yaw_sampling='10-minute raw grid; no 3-sigma anomaly exclusion'))
    print('Reproduction finished',len(checks),'checks',flush=True)
if __name__=='__main__':main()
