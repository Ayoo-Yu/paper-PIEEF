"""Read-only verification and F1/F2(4-5) analysis; no model training or inference."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from analyze_g12 import R as ROOT, O as OUT, dump as dump_named, cs
PROJECT=ROOT/'next_batch/merged_v2/project'
RUNS=ROOT/'补做重算/inputs/round3'
SEEDS=(13,21,42,87,3407)
sha=cs.sha
def dump(p,obj):dump_named(str(p.relative_to(OUT)),obj)
import sys, json, csv, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import rerun_h16_faults as ref
STAT=ROOT/'修复重算_20261021/statistics'
sys.path.insert(0,str(STAT))
import corrected_statistics as cs
sys.path.insert(0,str(PROJECT/'src'))
from pieef.revision_fault_jobs import scada_scenarios
COMMON=(13,42,3407)
CELLS=(('ACFormer',16),('ACFormer',96),('TCN',96),('TimesNet',96))
SCENARIOS=scada_scenarios('structured_39')
NAMES=['normal']+[r['name'] for r in SCENARIOS]
SPECS={r['name']:{k:v for k,v in r.items() if k!='name'} for r in SCENARIOS}
inputs={}; checks=[]; cache={}
def inp(p):
    p=p.resolve();k=str(p)
    if k not in inputs:inputs[k]=dict(path=k,bytes=p.stat().st_size,sha256=sha(p))
    return p
def js(p):return json.loads(inp(p).read_text(encoding='utf-8'))
def npz(p):return ref.load_npz(inp(p))
def writejson(name,obj):dump(OUT/name,cs.clean(obj))
def writecsv(name,rows):
    df=pd.DataFrame(rows);df.to_csv(OUT/name,index=False,encoding='utf-8-sig');return df
def same(a,b):
    for key in ('timestamp','origin_timestamp','turbine_ids','target_valid','events','capacity_mw'):
        assert np.array_equal(a[key],b[key]),f'Identity mismatch: {key}'
    assert np.array_equal(a['y_true_mw'],b['y_true_mw'],equal_nan=True),'truth mismatch'
    assert int(a['horizon'])==int(b['horizon'])
def grid(a,h,s):
    assert int(a['horizon'])==h and int(a['seed'])==s
    t=pd.to_datetime(a['timestamp'],utc=True).asi8;o=pd.to_datetime(a['origin_timestamp'],utc=True).asi8
    assert np.all(t-o==h*15*60*10**9) and np.all(np.diff(t)==15*60*10**9)
    assert len(np.unique(a['turbine_ids']))==len(a['turbine_ids'])
    v=a['target_valid'];assert v.dtype.kind=='b' and np.array_equal(np.isfinite(a['y_true_mw']),v)
    assert v.any(axis=1).all() and np.isfinite(a['y_pred_mw']).all()
    assert np.all(np.asarray(a['capacity_mw'])>0)
def load_result(directory,h,s,model):
    key=str(directory)
    if key in cache:return cache[key]
    c=js(directory/'completed.json');assert c['status']=='completed' and c['seed']==s and c['horizon']==h
    assert c['trained'] is False and len(c['rows'])==40
    assert [r['scenario'] for r in c['rows']]==NAMES
    for item in c.get('outputs',[]):
        p=inp(directory/item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    src=c['source'];cid=src['config_id']
    if model=='VF':
        matches=[]
        for p in RUNS.rglob('metrics.json'):
            m=json.loads(p.read_text(encoding='utf-8'))
            if m.get('config',{}).get('tracking',{}).get('config_id')==cid:matches.append(p.parent)
        assert len(matches)==1,'Duplicate or absent VF source';source=matches[0]
    else:
        source=__import__('pathlib').Path(c['source_directory'])
        bm_path=OUT/'baseline_manifest.json';bm=js(bm_path)
        assert sha(bm_path)==src['source_manifest_sha256']
        assert bm['runner_sha256']==c['runner_sha256']
        matching=[r for r in bm['sources'] if __import__('pathlib').Path(r['path'])==source]
        assert len(matching)==1
        for entry in matching[0]['files']+bm['project']+bm['data']:
            p=inp(__import__('pathlib').Path(entry['path']))
            assert inputs[str(p)]['sha256']==entry['sha256'],'Baseline source manifest mismatch'
    meta=js(source/'metrics.json');cfg=meta['config']
    assert meta['status']=='completed'
    if model=='VF':assert meta['evaluation_split']=='test'
    else: assert (OUT/'G1_runs.json').is_file()
    assert meta['horizon']==cfg['experiment']['horizon']==h and meta['seed']==s
    assert cfg['tracking']['config_id']==cid
    assert sha(inp(source/'weights_best.pt'))==src['checkpoint_sha256']
    if model!='VF':assert cfg['train']['loss_name']=='huber' and cfg['train']['huber_delta']==.1
    saved=npz(source/'predictions.npz');pre=npz(source/'preprocessing.npz')
    normal=npz(directory/'normal.npz');grid(normal,h,s)
    assert np.array_equal(normal['capacity_mw'],pre['capacity_mw'])
    for field in ('timestamp','origin_timestamp','turbine_ids','target_valid','events'):
        assert np.array_equal(normal[field],saved[field]),field
    assert np.array_equal(normal['y_true_mw'],saved['y_true_mw'],equal_nan=True)
    delta=np.abs((normal['y_pred_mw']-saved['y_pred_mw'])/normal['capacity_mw'])
    allowed=1e-5+1e-5*np.abs(saved['y_pred_mw']/normal['capacity_mw'])
    assert np.isfinite(saved['y_pred_mw']).all() and np.all(delta<=allowed),'Saved normal reproduction threshold failed'
    gate=js(directory/'normal_equivalence.json');assert gate['status']=='PASS'
    assert gate['atol_per_unit_capacity']==1e-5 and gate['rtol']==1e-5
    metrics=[];losses=[];specs=[]
    for name,row in zip(NAMES,c['rows']):
        a=normal if name=='normal' else npz(directory/(name+'.npz'))
        same(normal,a);grid(a,h,s)
        assert str(a['scenario'])==name and int(a['fault_seed'])==2026
        assert str(a['source_config_id'])==cid and str(a['checkpoint_sha256'])==src['checkpoint_sha256']
        spec=json.loads(str(a['scenario_json']));specs.append(spec)
        assert spec==(None if name=='normal' else SPECS[name])
        met=ref.measured_metrics(a['y_true_mw'],a['y_pred_mw'],a['target_valid'],a['capacity_mw'])
        for k in ('nrmse','nmae','r2'):assert abs(met[k]-row[k])<1e-10,(directory,name,k)
        assert met['n']==row['n']
        metrics.append(dict(seed=s,scenario=name,**met))
        err=np.where(a['target_valid'],(a['y_pred_mw'].astype(float)-a['y_true_mw'])/a['capacity_mw'],0.)
        counts=a['target_valid'].sum(axis=1)
        losses.append(np.stack(((err**2).sum(axis=1)/counts,np.abs(err).sum(axis=1)/counts)))
    for m in metrics:
        for key in ('nrmse','nmae'):m[key+'_increase']=m[key]/metrics[0][key]-1
    protocol={k:cfg['data'].get(k) for k in ('revision_protocol','seq_len','split','capacity_mw')}
    checks.append(dict(directory=str(directory),model=model,horizon=h,seed=s,completed_sha256=sha(directory/'completed.json'),source=str(source),checkpoint_sha256=src['checkpoint_sha256'],n_time=len(normal['timestamp']),n_turbines=len(normal['turbine_ids']),n_valid=int(normal['target_valid'].sum()),all_prediction_cells=int(normal['y_pred_mw'].size),normal_max_abs_delta_pu=float(delta.max()),index_truth_mask_capacity_exact=True,scenario_count=39,all_metrics_recomputed=True,source_protocol=protocol,device=c.get('runtime',{}).get('gpu',c.get('gpu',c.get('device'))),elapsed_seconds=c.get('elapsed_full_inference_seconds',c.get('elapsed_seconds'))))
    result=dict(completed=c,normal=normal,metrics=metrics,losses=np.stack(losses),protocol=protocol,directory=directory)
    cache[key]=result;return result
def group(runs,seeds):
    rows=[]
    for j,name in enumerate(NAMES):
        vals=[runs[s]['metrics'][j] for s in seeds]
        row=dict(scenario=name,common_seeds=len(seeds),seeds='/'.join(map(str,seeds)))
        for metric in ('nrmse','nmae','r2','nrmse_increase','nmae_increase'):
            arr=np.array([x[metric] for x in vals]);suffix=metric+('_percent' if metric!='r2' else '')
            scale=1 if metric=='r2' else 100
            row[suffix+'_mean']=arr.mean()*scale;row[suffix+'_sample_sd']=arr.std(ddof=1)*scale
        for metric in ('nrmse','nmae'):
            row[metric+'_degraded_seeds']=sum(v[metric+'_increase']>0 for v in vals)
            row[metric+'_increase_ratio_of_matched_means_percent']=100*(np.mean([v[metric] for v in vals])/np.mean([runs[s]['metrics'][0][metric] for s in seeds])-1)
        rows.append(row)
    return rows
def summary(runs,seeds):
    arr=np.array([[r['nrmse'] for r in runs[s]['metrics']] for s in seeds])
    mae=np.array([[r['nmae'] for r in runs[s]['metrics']] for s in seeds]);deg=100*(arr[:,1:]/arr[:,:1]-1)
    worst=1+np.argmax(arr[:,1:].mean(axis=0))
    return dict(seeds='/'.join(map(str,seeds)),normal_nrmse=100*arr[:,0].mean(),normal_sd=100*arr[:,0].std(ddof=1),normal_nmae=100*mae[:,0].mean(),fault_mean_nrmse=100*arr[:,1:].mean(),fault_mean_nrmse_sd=100*arr[:,1:].mean(axis=1).std(ddof=1),fault_mean_nmae=100*mae[:,1:].mean(),mean_degradation=deg.mean(),mean_degradation_sd=deg.mean(axis=1).std(ddof=1),worst_scenario=NAMES[worst],worst_nrmse=100*arr[:,worst].mean(),worst_sd=100*arr[:,worst].std(ddof=1),worst_degradation=100*(arr[:,worst]/arr[:,0]-1).mean())
def main():
    started=time.time()
    snapshot=[dict(model=m,horizon=h,eligible=True) for m,h in CELLS]
    vf={16:{},96:{}}
    for h in vf:
        root=ROOT/'补做重算/faults_vf/F1_verified' if h==16 else ROOT/'修复重算_20261021/h96_faults/workstation_capture/h96_faults/results'
        for s in SEEDS:vf[h][s]=load_result(root/f'seed{s}',h,s,'VF')
        for s in SEEDS:same(vf[h][13]['normal'],vf[h][s]['normal'])
    comparisons=[];scenario_comparisons=[];seedtests=[];timetests=[];perseed=[]
    for snap in snapshot:
        model,h=snap['model'],snap['horizon'];label=f'{model}_h{h}'
        if not snap['eligible']:
            for metric in ('fault_mean_nrmse','mean_degradation'):seedtests.append(dict(comparison=label,metric=metric,status='not_completed',paired_t_p_raw=None,exact_sign_p_two_sided_raw=None))
            for family in ('mse','mae'):
                for scope in ('normal','fault_mean'):timetests.append(dict(comparison=label,family=family,scope=scope,status='not_completed',p_value=None))
            continue
        base={s:load_result(OUT/f'F2/{model}_h{h}_seed{s}',h,s,model) for s in COMMON}
        for s in COMMON:
            same(vf[h][s]['normal'],base[s]['normal']);assert vf[h][s]['protocol']==base[s]['protocol'],'Training/data protocol mismatch'
            checks.append(dict(comparison=label,seed=s,paired_identity='PASS',fields='target_time/origin/turbines/truth/valid/capacity/events/protocol; all40 via transitivity'))
        p=summary(vf[h],COMMON);b=summary(base,COMMON);p5=summary(vf[h],SEEDS)
        comp=dict(comparison=label,horizon=h,model=model,**{'vf_'+k:v for k,v in p.items()},**{'baseline_'+k:v for k,v in b.items()},**{'vf5_'+k:v for k,v in p5.items()})
        for metric in ('normal_nrmse','fault_mean_nrmse','normal_nmae','fault_mean_nmae'):comp[metric+'_improvement_percent']=100*(1-p[metric]/b[metric])
        scen=[]
        for j,name in enumerate(NAMES):
            r=dict(comparison=label,scenario=name,seeds='13/42/3407')
            for metric in ('nrmse','nmae','nrmse_increase','nmae_increase'):
                r['vf_'+metric+'_percent']=100*np.mean([vf[h][s]['metrics'][j][metric] for s in COMMON]);r['baseline_'+metric+'_percent']=100*np.mean([base[s]['metrics'][j][metric] for s in COMMON])
            for metric in ('nrmse','nmae'):r[metric+'_improvement_percent']=100*(1-r['vf_'+metric+'_percent']/r['baseline_'+metric+'_percent'])
            scen.append(r)
        comp['vf_lower_nrmse_scenarios']=sum(r['vf_nrmse_percent']<r['baseline_nrmse_percent'] for r in scen[1:]);comp['vf_lower_nmae_scenarios']=sum(r['vf_nmae_percent']<r['baseline_nmae_percent'] for r in scen[1:])
        comparisons.append(comp);scenario_comparisons+=scen
        values={side:[] for side in ('VF','baseline')}
        for s in COMMON:
            for side,runs in (('VF',vf[h]),('baseline',base)):
                a=np.array([r['nrmse'] for r in runs[s]['metrics']]);row=dict(comparison=label,seed=s,side=side,normal_nrmse=100*a[0],fault_mean_nrmse=100*a[1:].mean(),mean_degradation=100*(a[1:]/a[0]-1).mean())
                perseed.append(row);values[side].append(row)
        for metric in ('fault_mean_nrmse','mean_degradation'):
            diff=[a[metric]-b[metric] for a,b in zip(values['VF'],values['baseline'])]
            test=cs.seed_supplement(diff);test.update(comparison=label,metric=metric,status='ok',difference_direction='VF_minus_baseline_percentage_points',multiplicity='declared_F2_8_per_test_method');seedtests.append(test)
        for family,k in (('mse',0),('mae',1)):
            for scope,sl in (('normal',slice(0,1)),('fault_mean',slice(1,40))):
                pl=np.mean([vf[h][s]['losses'][sl,k].mean(axis=0) for s in COMMON],axis=0);bl=np.mean([base[s]['losses'][sl,k].mean(axis=0) for s in COMMON],axis=0)
                test=cs.inference(bl,pl,h);test.update(comparison=label,family=family,scope=scope,seeds='13/42/3407');timetests.append(test)
    for pkey,qkey in (('paired_t_p_raw','paired_t_q_BH8'),('exact_sign_p_two_sided_raw','sign_q_BH8')):
        vals=cs.established.bh_adjust([r.get(pkey) if r.get(pkey) is not None else np.nan for r in seedtests]);assert len(vals)==8
        for r,q in zip(seedtests,vals):r[qkey]=q
    for family in ('mse','mae'):
        rows=[r for r in timetests if r['family']==family];assert len(rows)==8
        qs=cs.established.bh_adjust([r['p_value'] if r['p_value'] is not None else np.nan for r in rows])
        for r,q in zip(rows,qs):r['q_BH8']=q
    writecsv('G3_comparisons.csv',comparisons);writecsv('G3_scenario_comparisons.csv',scenario_comparisons);writecsv('G3_per_seed.csv',perseed)
    writecsv('G3_seed_tests.csv',seedtests);writejson('G3_seed_tests.json',seedtests)
    writecsv('G3_time_tests.csv',timetests);writejson('G3_time_tests.json',timetests)
    writejson('G3_identity_and_completion_verification.json',checks)

    writecsv('G3_input_files_sha256.csv',list(inputs.values()))
    writejson('G3_receipt.json',dict(elapsed_seconds=time.time()-started,finished_utc=datetime.now(timezone.utc).isoformat(),analysis_device='CPU',source_sha256={str(p):sha(p) for p in (Path(__file__),STAT/'corrected_statistics.py',STAT/'revision_statistics_frozen.py',OUT/'baseline_faults.py',OUT/'g3_worker.py')}))
    with (OUT/'进度.md').open('a',encoding='utf-8') as f:f.write('\n- G3完成：12个基线正常复现与468个故障推理、VF配对、种子与时间检验及各8项BH完成。\n')
if __name__=='__main__':main()
