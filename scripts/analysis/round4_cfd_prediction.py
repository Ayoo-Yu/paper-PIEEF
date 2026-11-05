"""R4 v2 with author-authorized M0/D1 addendum; no fitting or queue changes.

Before all 33 neural candidates, five F1 fits and 19 neural controls have actual
wait0, complete indexed predictions and D/E receipts, this writes progress only.
F1 statistics are merged from their sole owner, never recomputed here.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, io, json, math, os, shutil
import numpy as np
from scipy.stats import binomtest,ttest_rel

WORK=Path(__file__).resolve().parents[1];PROJECT=WORK/'project';OUT=WORK.parent/'补充验证_20261023'
EBASE=Path('results/paper');DEST=OUT/'M_summary_v2';EDEST=EBASE/'results/M_summary_v2'
M0=WORK/'scope_addendum_20261024/M0'
SEEDS_wind_farm_A=(13,21,42,87,3407);SEEDS_KEL=(13,42,3407)
FAMILIES={'VF_wind_farm_A':5,'direction_off_wind_farm_A':5,'VF_Kelmarsh':9}
VARIANTS={'M0':'V-nwp100-global','M1':'V-dir','M2':'V-dir+spd','M3a':'K-cfd','M3b':'K-meas',
          'VF_wind_farm_A':'VF','direction_off_wind_farm_A':'direction-off','VF_Kelmarsh':'VF'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def need(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
def atomic(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_bytes(b);os.replace(t,p)
def pair(name,b,final=False):
    for root in (DEST,EDEST):
        p=root/name
        if final and p.exists():need(p.read_bytes()==b,'Existing final report differs; preserve before review: '+str(p))
        else:atomic(p,b)
    need((DEST/name).read_bytes()==(EDEST/name).read_bytes(),'Report D/E bytes differ')
def csvraw(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode('utf-8-sig')
def safe(p,root):
    p=Path(p);root=Path(root).resolve();need(p.resolve().is_relative_to(root),'Unexpected artifact root: '+str(p));return p

def bh(values,family_size):
    need(len(values)==family_size,'The complete fixed family is required')
    if any(p is None for p in values):return [None]*family_size
    need(all(math.isfinite(p) and 0<=p<=1 for p in values),'Invalid actual p-values')
    order=np.argsort(values);out=[None]*family_size;minimum=1.
    for rank in range(family_size-1,-1,-1):
        i=int(order[rank]);minimum=min(minimum,float(values[i])*family_size/(rank+1));out[i]=minimum
    return out

def paired_row(candidate,control,*,family,comparison,endpoint,seeds,horizon,site):
    a=np.asarray(candidate,dtype=float);b=np.asarray(control,dtype=float)
    need(a.shape==b.shape==(len(seeds),) and len(seeds) in (3,5) and np.isfinite(a).all() and np.isfinite(b).all(),'Incomplete/nonfinite matched seeds')
    diff=a-b;t=ttest_rel(a,b);estimable=bool(np.isfinite(t.statistic) and np.isfinite(t.pvalue))
    wins=int((diff<0).sum());losses=int((diff>0).sum());n=wins+losses
    return dict(family=family,comparison=comparison,site=site,horizon_steps=horizon,horizon_minutes=15*horizon,
        endpoint=endpoint,split='test',common_seeds=len(seeds),seeds='/'.join(map(str,seeds)),
        candidate_mean=float(a.mean()),control_mean=float(b.mean()),mean_difference=float(diff.mean()),
        sample_sd_difference=float(diff.std(ddof=1)),difference_definition='candidate_minus_control',
        relative_error_improvement_percent=float(100*(1-a.mean()/b.mean())) if b.mean()!=0 else None,
        candidate_better_seeds=wins,candidate_worse_seeds=losses,tied_seeds=len(seeds)-n,
        paired_t_statistic=float(t.statistic) if estimable else None,paired_t_p=float(t.pvalue) if estimable else None,
        paired_t_test_estimable=estimable,exact_two_sided_sign_p=float(binomtest(wins,n,.5).pvalue) if n else 1.)

def correct_family(rows,name):
    need(len(rows)==6 and all(r['family']==name for r in rows),'Exactly six predefined family endpoints required')
    for field,qfield in [('paired_t_p','paired_t_BH_q'),('exact_two_sided_sign_p','sign_BH_q')]:
        for r,q in zip(rows,bh([r[field] for r in rows],6)):r[qfield]=q
    for r in rows:
        n=r['common_seeds'];wins=r['candidate_better_seeds'];q=r['paired_t_BH_q']
        r['planned_result_category']='支持' if wins==n and q is not None and q<.05 else '趋势' if (n==5 and wins>=4 or n==3 and wins==3) else '无差异或不利'
    return rows

def planned_tests(values):
    result=[]
    for candidate,control in [('M1','VF_wind_farm_A'),('M2','VF_wind_farm_A'),('M1','direction_off_wind_farm_A')]:
        for metric in ('nrmse','nmae'):
            result.append(paired_row([values[(candidate,16,s,'test')][metric] for s in SEEDS_wind_farm_A],
                [values[(control,16,s,'test')][metric] for s in SEEDS_wind_farm_A],family='M-wind_farm_A',comparison=VARIANTS[candidate]+' vs '+VARIANTS[control],
                endpoint=metric,seeds=SEEDS_wind_farm_A,horizon=16,site='wind_farm_A'))
    result=correct_family(result,'M-wind_farm_A');kel=[]
    for candidate in ('M3a','M3b'):
        for horizon in (1,4,16):
            kel.append(paired_row([values[(candidate,horizon,s,'test')]['nmae'] for s in SEEDS_KEL],
                [values[('VF_Kelmarsh',horizon,s,'test')]['nmae'] for s in SEEDS_KEL],family='M-KEL',
                comparison=VARIANTS[candidate]+' vs VF',endpoint='nmae',seeds=SEEDS_KEL,horizon=horizon,site='Kelmarsh'))
    height=[]
    for candidate,control in [('M0','VF_wind_farm_A'),('M1','M0'),('M2','M0')]:
        for metric in ('nrmse','nmae'):
            height.append(paired_row([values[(candidate,16,s,'test')][metric] for s in SEEDS_wind_farm_A],
                [values[(control,16,s,'test')][metric] for s in SEEDS_wind_farm_A],family='M-HEIGHT',comparison=VARIANTS[candidate]+' vs '+VARIANTS[control],
                endpoint=metric,seeds=SEEDS_wind_farm_A,horizon=16,site='wind_farm_A'))
    return result+correct_family(kel,'M-KEL')+correct_family(height,'M-HEIGHT')

def controls():
    selected=[r for r in read(WORK/'audit/baseline/control_index.json')['controls'] if r['family'] in FAMILIES]
    need(dict(collections.Counter(r['family'] for r in selected))==FAMILIES,'Original19 control identities differ')
    return selected
def cid(r):return f"{r['family']}_h{r['horizon']}_seed{r['seed']}"

def admission():
    protocol=read(WORK/'protocol.json');tasks=protocol['neural_tasks'];need(len(tasks)==28,'Frozen28 neural scope differs')
    need(dict(collections.Counter(t['group'] for t in tasks))=={'M1':5,'M2':5,'M3a':9,'M3b':9},'Candidate groups differ')
    supplement=read(M0/'scope_protocol.json');extra=supplement['tasks']
    need(len(extra)==5 and {t['seed'] for t in extra}==set(SEEDS_wind_farm_A)
         and all(t['group']=='M0' and t['horizon']==16 for t in extra),'Exact authorized M0 five-seed scope differs')
    tasks=tasks+extra
    receipts={t['task_id']:OUT/t['group']/'runs'/t['task_id']/'verification_receipt.json' for t in tasks}
    cpaths={cid(r):OUT/'M_controls'/cid(r)/'completed.json' for r in controls()}
    fpaths={t['task_id']:OUT/'F1/runs'/t['task_id']/'verification_receipt.json' for t in protocol['F1_tasks']}
    need(len(fpaths)==5,'Frozen F1 scope differs')
    missing=[dict(kind=kind,task_id=rid,required_path=str(p)) for kind,mapping in [('neural',receipts),('control_inference',cpaths),('F1',fpaths)] for rid,p in mapping.items() if not p.is_file()]
    for name in ('f1_results.json','f1_runs.csv','f1_groups.csv','f1_paired_by_seed.csv','f1_tests.csv','summary_D_E_SHA.json'):
        p=OUT/'F1/summary'/name
        if not p.is_file():missing.append(dict(kind='F1_summary',task_id=name,required_path=str(p)))
    for name in ('verification_receipt.json','d1_results.json','d1_summary.md'):
        p=OUT/'D1'/name
        if not p.is_file():missing.append(dict(kind='D1',task_id=name,required_path=str(p)))
    return protocol,tasks,receipts,cpaths,fpaths,missing

def source_wait(receipt,neural):
    need(receipt.get('actual_natural_returncode')==0,'Actual accepted child wait0 missing')
    execution=receipt['source_execution_receipt']
    if neural:
        need(isinstance(execution,dict) and execution.get('natural_returncode')==0 and execution.get('status')=='completed','Original NN natural wait0/completion missing')
        need(execution.get('child',{}).get('pid',0)>0 and execution.get('child',{}).get('create_time',0)>0,'Actual NN child identity missing')
    else:
        need(execution.get('actual_natural_exit_code')==0 and execution.get('actual_natural_returncode')==0,'F1 true wait0 attestation missing')
        p,q=Path(execution['D_path']),Path(execution['E_path']);need(p.read_bytes()==q.read_bytes() and sha(p)==execution['sha256'],'F1 original wait D/E/SHA differs')

def received(path,task,neural=True):
    receipt=read(path);need(receipt['status']=='VERIFIED_COMPLETE_D_E' and receipt['task_id']==task['task_id'],'Received task identity/status differs')
    need(receipt['config_sha256']==task['config_sha256'],'Received frozen config SHA differs')
    if neural:
        manifest=M0/'source_manifest.json' if task['group']=='M0' else WORK/'source_manifest.json'
        need(receipt.get('source_manifest_sha256')==sha(manifest),'Received neural frozen source manifest differs')
    source_wait(receipt,neural)
    directory=Path(receipt.get('D_path') or receipt.get('artifact_directory') or receipt['D_artifact_directory'])
    backup=Path(receipt.get('E_path') or receipt.get('backup_artifact_directory') or receipt['E_artifact_directory'])
    safe(directory,OUT);safe(backup,EBASE/'results')
    need(path.parent.resolve()==directory.resolve(),'Receipt is outside its accepted artifact directory')
    need((backup/'verification_receipt.json').read_bytes()==path.read_bytes(),'Received acceptance receipt D/E differs')
    for rel,entry in receipt['files'].items():
        p,q=directory/rel,backup/rel;safe(p,directory);safe(q,backup)
        need(p.is_file() and q.is_file() and p.stat().st_size==entry['bytes'] and q.stat().st_size==entry['bytes']
             and sha(p)==entry['sha256'] and sha(q)==entry['sha256'],'Received output full D/E SHA differs: '+rel)
    return directory,backup,receipt

def check_controls(row,path):
    receipt=read(path);need(receipt['status']=='completed' and receipt['control_id']==cid(row)
        and receipt['natural_returncode']==0 and receipt['inference_only'] is True and receipt['training_performed'] is False,'Control inference true completion missing')
    d,e=Path(receipt['D_path']),Path(receipt['E_path']);safe(d,OUT/'M_controls');safe(e,EBASE/'results/M_controls')
    need((e/'completed.json').read_bytes()==path.read_bytes(),'Control receipt D/E differs')
    for rel,entry in receipt['files'].items():
        need((d/rel).stat().st_size==entry['bytes'] and sha(d/rel)==entry['sha256'] and sha(e/rel)==entry['sha256'],'Control D/E full member SHA differs')
    need(read(d/'metrics.json')['default_reproduction_gate']['status']=='PASS','Control off reproduction failed')
    return d,e,receipt

def prediction(directory,split,*,original_control=False):
    name='source_predictions.npz' if original_control and split=='test' else 'predictions.npz' if split=='test' else 'val_predictions.npz'
    with np.load(directory/name,allow_pickle=original_control and split=='test') as z:
        values={k:(np.asarray(z[k],dtype=str) if k in ('timestamp','origin_timestamp','turbine_ids') else z[k]) for k in z.files}
    y,p=values['y_true_mw'],values['y_pred_mw'];need(y.ndim==2 and y.shape==p.shape and np.isfinite(p).all(),'Incomplete/nonfinite predictions')
    need(len(values['timestamp'])==len(y) and len(set(values['timestamp']))==len(y)
         and len(values['origin_timestamp'])==len(y) and len(values['turbine_ids'])==y.shape[1],'Full prediction indices differ')
    need(np.array_equal(np.isfinite(y),values['target_valid']),'Measured-target mask differs')
    with np.load(directory/'preprocessing.npz',allow_pickle=False) as z:capacity=z['capacity_mw']
    from pieef.core import calc_metrics
    metrics=dict(zip(('nmae','nrmse','r2'),calc_metrics(y,p,capacity)))
    return values,metrics,capacity

def matching(a,b):
    for name in ('timestamp','origin_timestamp','turbine_ids','target_valid'):need(np.array_equal(a[name],b[name]),'Paired prediction index/support differs: '+name)
    need(np.array_equal(a['y_true_mw'],b['y_true_mw'],equal_nan=True),'Paired measured truth differs')

def attention(directory,split,pred):
    s=read(directory/(split+'_attention_summary.json'))
    with np.load(directory/(split+'_attention_diagnostics.npz'),allow_pickle=False) as z:
        for k in ('timestamp','origin_timestamp','turbine_ids'):need(np.array_equal(z[k].astype(str),pred[k]),'Actual attention indices differ')
        need(z['dsi'].shape==pred['y_pred_mw'].shape,'Actual attention is incomplete')
        valid=np.isfinite(z['dsi']);need(int(valid.sum())==s['dsi_supported_query_count'] and int((~valid).sum())==s['dsi_undefined_query_count'],'DSI support counts differ')
        if valid.any():need(abs(float(z['dsi'][valid].mean())-s['dsi_mean'])<1e-12,'DSI actual mean differs')
        for key,outkey in [('normalized_entropy','normalized_entropy_mean'),('maximum_weight','maximum_weight_mean')]:
            need(np.isfinite(z[key]).all() and abs(float(z[key].mean())-s[outkey])<1e-12,'Actual attention diagnostic differs')
    return s

def accepted_d1():
    d,e=OUT/'D1',EBASE/'results/D1';p=d/'verification_receipt.json';r=read(p)
    need(r.get('status')=='VERIFIED_COMPLETE_D_E' and r.get('actual_natural_returncode')==0
         and r.get('no_new_model_training') is True and r.get('only_training_prefix_analyzed') is True,'D1 completed training-only source gate missing')
    need((e/p.name).read_bytes()==p.read_bytes(),'D1 acceptance D/E differs')
    execution=r['source_execution_receipt'];a,b=Path(execution['D_path']),Path(execution['E_path'])
    need(a.read_bytes()==b.read_bytes() and sha(a)==execution['sha256'] and a.stat().st_size==execution['bytes'],'D1 actual wait D/E differs')
    wait=read(a);need(wait.get('actual_natural_returncode')==0 and wait.get('protocol_sha256')==r['protocol_sha256'],'D1 actual native wait0/protocol differs')
    for name,entry in r['files'].items():
        a,b=d/name,e/name;safe(a,d);safe(b,e)
        need(a.stat().st_size==entry['bytes'] and b.stat().st_size==entry['bytes']
             and sha(a)==entry['sha256'] and sha(b)==entry['sha256'],'D1 full member D/E differs: '+name)
    result=read(d/'d1_results.json')
    need(result['status']=='D1_PRIMARY_TRAINING_SPEED_PATTERN_ANALYSIS_COMPLETE' and result['protocol_sha256']==r['protocol_sha256']
         and result['paired_cells']==464 and result['missing_cells']==0 and result['covered_sectors']==16
         and result['p_value_or_confidence_interval_computed'] is False and result['direction_height_m']==100,'D1 scope or result identity differs')
    need(math.isfinite(result['pearson_r']) and -1<=result['pearson_r']<=1 and math.isfinite(result['mean_absolute_relative_ratio_difference']),'D1 actual descriptive values invalid')
    return result,r

def summarize_all(protocol,tasks,npaths,cpaths,fpaths):
    import sys
    sys.path.insert(0,str(PROJECT/'src'))
    runs=[];attn=[];sources=[];values={};preds={};grouped=[]
    allrecords=[]
    for r in controls():
        d,e,receipt=check_controls(r,cpaths[cid(r)]);allrecords.append((r['family'],r['horizon'],r['seed'],d,e,receipt,False,cid(r)))
    for task in tasks:
        d,e,receipt=received(npaths[task['task_id']],task);allrecords.append((task['group'],task['horizon'],task['seed'],d,e,receipt,True,task['task_id']))
    for group,horizon,seed,d,e,receipt,new,task_id in allrecords:
        site='Kelmarsh' if group in ('VF_Kelmarsh','M3a','M3b') else 'wind_farm_A';m=read(d/'metrics.json')
        for split in ('val','test'):
            pred,metrics,capacity=prediction(d,split,original_control=not new);key=(group,horizon,seed,split);values[key]=metrics;preds[key]=(pred,capacity)
            expected=m['validation_metrics'] if split=='val' else m['metrics']
            need(all(abs(metrics[k]-expected[k])<1e-9 for k in metrics),'Full primary metrics differ: '+task_id+' '+split)
            runs.append(dict(model='PI-EEF',variant=VARIANTS[group],group=group,site=site,horizon_steps=horizon,horizon_minutes=15*horizon,seed=seed,split=split,
                nrmse=metrics['nrmse'],nmae=metrics['nmae'],r2=metrics['r2'],nrmse_percent=metrics['nrmse']*100,nmae_percent=metrics['nmae']*100,
                new_training=new,primary_test_source='current complete predictions' if new else 'original frozen stored test predictions',artifact_directory=str(d),E_artifact_directory=str(e)))
            summary=attention(d,split,pred)
            attn.append(dict(variant=VARIANTS[group],group=group,site=site,horizon_steps=horizon,horizon_minutes=15*horizon,seed=seed,split=split,
                dsi_mean=summary['dsi_mean'],dsi_supported_query_count=summary['dsi_supported_query_count'],dsi_undefined_query_count=summary['dsi_undefined_query_count'],
                normalized_entropy_mean=summary['normalized_entropy_mean'],maximum_weight_mean=summary['maximum_weight_mean'],
                secondary_descriptive_only=True,artifact_directory=str(d)))
        sources.append(dict(task_id=task_id,group=group,seed=seed,horizon_steps=horizon,new_training=new,D_path=str(d),E_path=str(e),
            actual_natural_returncode=0,receipt_path=str((npaths[task_id] if new else cpaths[task_id])),
            receipt_sha256=sha(npaths[task_id] if new else cpaths[task_id]),complete_process_wall_seconds=receipt.get('full_child_wall_seconds') or receipt.get('source_execution_receipt',{}).get('complete_child_wall_seconds')))
    # Check exact index/true/capacity equality for every candidate-control pair,
    # including secondary validation, before seed-level arithmetic.
    for task in tasks:
        group,h,s=task['group'],task['horizon'],task['seed'];control='VF_Kelmarsh' if group.startswith('M3') else 'VF_wind_farm_A'
        for split in ('val','test'):
            a,ca=preds[(group,h,s,split)];b,cb=preds[(control,h,s,split)];matching(a,b);need(np.array_equal(ca,cb),'Matched capacity differs')
            if group=='M1':matching(a,preds[('direction_off_wind_farm_A',h,s,split)][0])
            if group in ('M1','M2'):
                c,cc=preds[('M0',h,s,split)];matching(a,c);need(np.array_equal(ca,cc),'Height-control matched capacity differs')
    for (variant,site,h,split),members in __import__('itertools').groupby(sorted(runs,key=lambda r:(r['variant'],r['site'],r['horizon_steps'],r['split'])),key=lambda r:(r['variant'],r['site'],r['horizon_steps'],r['split'])):
        members=list(members);seeds=[r['seed'] for r in members];expected=set(SEEDS_wind_farm_A if site=='wind_farm_A' else SEEDS_KEL);need(set(seeds)==expected and len(seeds)==len(expected),'Group has incomplete/duplicate seeds')
        row=dict(variant=variant,site=site,horizon_steps=h,horizon_minutes=15*h,split=split,common_seeds=len(seeds),seeds='/'.join(map(str,sorted(seeds))))
        for metric in ('nrmse','nmae','r2'):
            a=np.asarray([r[metric] for r in members]);row[metric+'_mean']=float(a.mean());row[metric+'_sample_sd']=float(a.std(ddof=1))
            worst=int(np.argmin(a) if metric=='r2' else np.argmax(a));row[metric+'_worst_seed']=members[worst]['seed'];row[metric+'_worst_value']=float(a[worst])
        grouped.append(row)
    tests=planned_tests(values);paired=[]
    for candidate,control,h,seeds in [('M1','VF_wind_farm_A',16,SEEDS_wind_farm_A),('M2','VF_wind_farm_A',16,SEEDS_wind_farm_A),('M1','direction_off_wind_farm_A',16,SEEDS_wind_farm_A),
        ('M0','VF_wind_farm_A',16,SEEDS_wind_farm_A),('M1','M0',16,SEEDS_wind_farm_A),('M2','M0',16,SEEDS_wind_farm_A),
        *[(m,'VF_Kelmarsh',h,SEEDS_KEL) for m in ('M3a','M3b') for h in (1,4,16)]]:
        for split in ('val','test'):
            for seed in seeds:
                for metric in ('nrmse','nmae','r2'):
                    a=values[(candidate,h,seed,split)][metric];b=values[(control,h,seed,split)][metric]
                    paired.append(dict(comparison=VARIANTS[candidate]+' vs '+VARIANTS[control],site='wind_farm_A' if h==16 and candidate in ('M0','M1','M2') else 'Kelmarsh',
                        horizon_steps=h,horizon_minutes=15*h,seed=seed,split=split,metric=metric,candidate=a,control=b,candidate_minus_control=a-b,
                        relative_error_improvement_percent=100*(1-a/b) if metric!='r2' and b!=0 else None,candidate_better=a<b if metric!='r2' else a>b))
    f1dir=OUT/'F1/summary';f1=read(f1dir/'f1_results.json');seal=read(f1dir/'summary_D_E_SHA.json')
    need(f1['status']=='F1_ALL_FIVE_REAL_FITS_COMPLETE' and f1['new_training_runs']==5 and f1['existing_control_tree_refits']==0,'F1 complete actual five-fit summary missing')
    need(seal['status']=='F1_SUMMARY_ALL_D_E_BYTE_SHA_PASS' and len(f1['planned_tests'])==2,'F1 summary seal/fixed family differs')
    for name,entry in seal['files'].items():
        p,q=f1dir/name,EBASE/'results/F1/summary'/name
        need(p.read_bytes()==q.read_bytes() and sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'F1 summary D/E SHA differs')
    for task in protocol['F1_tasks']:received(fpaths[task['task_id']],task,neural=False)
    d1,d1receipt=accepted_d1()
    return dict(runs=runs,groups=grouped,attention=attn,paired=paired,tests=tests,sources=sources,F1=f1,F1_summary_files=seal['files'],
                D1=d1,D1_receipt_sha256=sha(OUT/'D1/verification_receipt.json'),M0_source_manifest_sha256=sha(M0/'source_manifest.json'))

def report(value):
    lines=['# 第四轮 CFD 接入预测与 LightGBM 公平性实验完整汇总 v2（含M0与D1）','',
        '本报告覆盖原28项神经预测训练、作者追加的5项M0高度对照、5项LightGBM公平性训练，以及19项原神经控制只读CPU再次推理和D1训练期物理模式验证。38项新增训练及其控制/辅助来源均有真实自然退出0、完整预测/来源与D/E逐文件SHA门禁。原VF、去方向偏置控制及主表均未重训或替换。','',
        '## 逐运行与组结果','',
        '完整逐运行指标见m_runs.csv，先列模型/变体、风场、实际提前量、种子、评价集、NRMSE、NMAE、R²；下表按完整共同种子汇总均值与样本标准差。误差单位为容量百分比，R²不乘100。验证与测试分别列出，不用测试表现回改候选或检查点。','',
        '| 评价集 | 风场 | 变体 | 提前量(min) | 共同种子 | NRMSE均值±样本SD (%) | NMAE均值±样本SD (%) | R²均值±样本SD |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(value['groups'],key=lambda r:(r['split'],r['site'],r['horizon_steps'],r['variant'])):
        lines.append(f"| {r['split']} | {r['site']} | {r['variant']} | {r['horizon_minutes']} | {r['common_seeds']} | {100*r['nrmse_mean']:.3f} ± {100*r['nrmse_sample_sd']:.3f} | {100*r['nmae_mean']:.3f} ± {100*r['nmae_sample_sd']:.3f} | {r['r2_mean']:.4f} ± {r['r2_sample_sd']:.4f} |")
    lines+=['','## 预先冻结的配对比较','',
        '主对照测试指标来自原完整存储预测；本轮CPU再次推理仅补验证和注意力并核兼容，全部再次预测同样保存。正的相对改善表示候选误差更低；逐种子改善、退步和最坏种子均保存在m_paired_by_seed.csv与m_runs.csv，不删不利结果。','',
        '| 族 | 比较 | 提前量(min) | 测试指标 | 相对改善(%) | 改善/退步/相同 | 配对t p/q | 双侧符号p/q | 预定档次 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    def fmt(v):return '不可估' if v is None else f'{v:.5g}'
    for r in value['tests']:
        lines.append(f"| {r['family']} | {r['comparison']} | {r['horizon_minutes']} | {r['endpoint']} | {fmt(r['relative_error_improvement_percent'])} | {r['candidate_better_seeds']}/{r['candidate_worse_seeds']}/{r['tied_seeds']} | {fmt(r['paired_t_p'])}/{fmt(r['paired_t_BH_q'])} | {fmt(r['exact_two_sided_sign_p'])}/{fmt(r['sign_BH_q'])} | {r['planned_result_category']} |")
    lines+=['','原M-wind_farm_A六项、M-KEL六项及F1两项统计族保持不变；作者授权新增的M-HEIGHT六项（M0对VF、M1对M0、M2对M0各NRMSE/NMAE）独立列出。每族内配对t与双侧精确符号检验分别BH，不把新族并入旧族或反向改原q值；主档次看本族配对t q和共同种子方向。零方差导致t不可估时保持空值，固定族t-BH不删项重算。Kelmarsh三种子双侧精确符号p最小0.25、五种子最小0.0625。没有添加时间DM、额外CI或未授权统计族。','',
        '## 注意力、来源与方法边界','',
        'm_attention_dsi.csv列出每运行/划分的实际DSI、支持/未定义查询数、归一化注意力熵和最大权重；均为次要描述，不升级为主结论。DSI由真实attention与同批方向基准计算，rel[i,j]=r_i-r_j，正流向余弦表示j位于i上风向。方向偏置关闭的控制仍按相同物理基准读取实际attention。','',
        '原VFWind Farm A方向来自10m U/V，原M1/M2按第四轮方案用100m U/V查询局地CFD。作者在原第四轮开始后追加M0，只把目标方向换为100m全场方向、保留farm_mean图且不用CFD，不改风速。M0对VF检验高度输入差异；M1/M2对M0描述局地CFD修正及相应方向图接入的增量，不把旧M1对VF的全部差异直接归因为地形修正。历史、实测目标、划分、原训练统计、batch16、FP32、5/5/20预算与验证NRMSE检查点均冻结。','',
        '补充时点透明记录：根协调器2026-10-24 11:08:36 UTC快照显示Wind Farm A尚无本轮完整神经metrics，但Kelmarsh M3a h16 seed13已于11:03:42完成，F1独立结果亦已产生。M0理由仅为已知的10m→100m混杂，不是根据这些得分选择参数；这是结果已部分产生后的明确方法补充，不称全部结果产生前的原始预注册。原R4计划及新增说明均随报告保留。Wind Farm Aauthor_prepared_ecmwf确认语义未改变，Kelmarsh保持真实起点可用GFS vintage。','',
        'K-meas仅使用预测起点t0的机组实测风向相对于圆周场均风向的偏差；不读目标时刻实测。K-cfd使用冻结静态CFD查找表，预测时按目标NWP风向作圆周插值。M2仅额外修正选定目标NWP风速字段，不改变历史输入或标签。','',
        'CFD查找表的原始False标志及MPI收尾/采样恢复、加载适用计数修正记录均保留，见先前CFD交付报告；这些来源验收不等于现场流场验证或网格独立性结论。预测收益由本次真实完整配对另行检验。原控制真实权重、原预测、原预处理和CPU再次推理证据在M_controls完整保留。','',
        '## LightGBM公平性（F1）','',
        '以下F1结果直接合并独立F1所有者的五种子完整统计，不重新拟合、重算其统计族或选取有利项。主输入、目标及树预算固定，只追加与M2完全同源的局地流向sin/cos和修正目标NWP风速三列。','',
        '| 评价集 | 模型 | 共同种子 | NRMSE均值±样本SD (%) | NMAE均值±样本SD (%) | R²均值±样本SD |','|---|---|---:|---:|---:|---:|']
    for r in value['F1']['group_results']:
        lines.append(f"| {r['split']} | {r['model']} | {r['common_seeds']} | {100*r['nrmse_mean']:.3f} ± {100*r['nrmse_sample_sd']:.3f} | {100*r['nmae_mean']:.3f} ± {100*r['nmae_sample_sd']:.3f} | {r['r2_mean']:.4f} ± {r['r2_sample_sd']:.4f} |")
    fg={(r['model'],r['split']):r for r in value['F1']['group_results']};a,b=fg[('LightGBM+CFD','val')],fg[('LightGBM','val')]
    bad=sorted({r['seed'] for r in value['F1']['paired_per_seed'] if r['split']=='test' and r['metric'] in ('nrmse','nmae') and r['candidate_minus_control']>0})
    lines+=['',f"F1验证集NRMSE相对变化（候选−对照，正值为恶化）为{100*(a['nrmse_mean']/b['nrmse_mean']-1):.3f}%，NMAE为{100*(a['nmae_mean']/b['nmae_mean']-1):.3f}%；测试存在误差退步的种子为{('/'.join(map(str,bad)) if bad else '无')}。验证与逐种子不利结果均保留，不以测试均值的有利变化替代它们。"]
    lines+=['','F1逐种子变化、两项固定测试终点及各自t/sign BH详见合并的F1/f1_tests.csv和F1完整报告。种子13原producer已自然退出0，独立消费者R²显式float转换修复、原消费者FAIL和原wait均保留；未重新fit或重跑预测，未放宽验算阈值。','',
        '## D1训练期物理模式验证','',
        f"D1使用原训练期{value['D1']['training_time_rows']}个15分钟时间格，按冻结的29机组共同normal/原始风速4–12m/s规则保留{value['D1']['selected_common_time_rows']}个共同时间点。16扇区×29机组共{value['D1']['paired_cells']}个完整配对cell，Pearson r={value['D1']['pearson_r']:.6f}，相对风速比值平均绝对差={value['D1']['mean_absolute_relative_ratio_difference']:.6f}。未计算p或CI，cell不作为独立重复。完整筛选漏斗、每格数值与掩码见D1原目录及复制的D1报告。100m分箱和原CFD主导工况选频10m来源差异已事先记录，没有另算10m相关择优。该分析仅量化此筛选下相对机组风速模式的一致程度，不等于山地风向现场验证或所有流场条件的验证。",'',
        '## 原始文件与复核','',
        f'神经/控制完整来源索引见m_source_index.csv，任务目录分别位于 {OUT.as_posix()}/M1、M2、M3a、M3b、M_controls；F1位于 {OUT.as_posix()}/F1。完整第二份为 {EBASE.as_posix()}/results。每个目录的验收receipt包含真实子进程身份/自然退出、配置身份、全部文件SHA及D/E路径。','',
        '本汇总只读取已验收产物，没有重新训练、选择检查点或改写冻结主表。完整运行墙钟只引用实际子进程启动至全部训练、预测和保存结束的记录；CPU控制墙钟属于再次推理，不混作新增训练耗时。作者取消的定时任务未恢复。','']
    return '\n'.join(lines).encode('utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');args=ap.parse_args()
    protocol,tasks,npaths,cpaths,fpaths,missing=admission()
    if missing:
        v=dict(status='PENDING_NO_COMPLETE_RESULT_REPORT_V2_WITH_M0_D1',observed_utc=now(),neural_required=33,neural_receipts_present=sum(p.is_file() for p in npaths.values()),
            F1_required=5,F1_receipts_present=sum(p.is_file() for p in fpaths.values()),control_required=19,control_receipts_present=sum(p.is_file() for p in cpaths.values()),
            missing=missing,no_complete_statistics_generated=True)
        pair('progress.json',dump(v));pair('progress.md',(f"# 第四轮预测实验进度v2（含M0/D1）\n\n已出现验收回执：神经（含M0）{v['neural_receipts_present']}/33、F1 {v['F1_receipts_present']}/5、神经控制{v['control_receipts_present']}/19。另核D1。尚有缺项，仅记录进度，不生成完整结果或统计结论。\n").encode('utf8'))
        print(json.dumps({k:v[k] for k in ('status','neural_receipts_present','neural_required','F1_receipts_present','F1_required','control_receipts_present','control_required','no_complete_statistics_generated')},ensure_ascii=False));return
    if not args.build:print(json.dumps(dict(status='ALL_RECEIPTS_PRESENT_USE_BUILD_FOR_FULL_VERIFICATION',formal_completion_not_claimed=True)));return
    existing=DEST/'final_summary_D_E_SHA.json'
    if existing.exists():
        v=read(existing)
        for name,entry in v['files'].items():need((DEST/name).read_bytes()==(EDEST/name).read_bytes() and sha(DEST/name)==entry['sha256'],'Existing final report changed')
        print(json.dumps(dict(status='EXISTING_IDENTICAL_COMPLETE_SUMMARY_VERIFIED',D_path=str(DEST),E_path=str(EDEST))));return
    value=summarize_all(protocol,tasks,npaths,cpaths,fpaths)
    for name,rows in [('m_runs.csv',value['runs']),('m_groups.csv',value['groups']),('m_attention_dsi.csv',value['attention']),
        ('m_paired_by_seed.csv',value['paired']),('m_tests.csv',value['tests']),('m_source_index.csv',value['sources'])]:pair(name,csvraw(rows),True)
    summary=dict(status='ALL38_NEW_TASKS_AND19_CONTROLS_D1_FULLY_VERIFIED',created_utc=now(),new_neural_trainings=33,new_tree_trainings=5,
        original_neural_controls_reinferred=19,original_tree_control_refits=0,protocol_sha256=sha(WORK/'protocol.json'),summary_source_sha256=sha(__file__),
        methods_document_sha256=sha(Path(__file__).with_name('统计与控制推理执行补充说明.md')),**value)
    pair('m_results.json',dump(summary),True);pair('第四轮CFD接入预测与公平性实验报告_v2_含M0与D1.md',report(value),True)
    pair('m_summary.md',report(value),True)
    for name in value['F1_summary_files']:pair('F1/'+name,(OUT/'F1/summary'/name).read_bytes(),True)
    pair('F1/summary_D_E_SHA.json',(OUT/'F1/summary/summary_D_E_SHA.json').read_bytes(),True)
    pair('统计与控制推理执行补充说明.md',Path(__file__).with_name('统计与控制推理执行补充说明.md').read_bytes(),True)
    pair('scripts/summarize_round4_v2.py',Path(__file__).read_bytes(),True)
    pair('M0/方法与统计补充说明.md',(M0/'方法与统计补充说明.md').read_bytes(),True)
    pair('M0/scope_protocol.json',(M0/'scope_protocol.json').read_bytes(),True)
    pair('M0/source_manifest.json',(M0/'source_manifest.json').read_bytes(),True)
    for name in ('d1_results.json','d1_summary.md','verification_receipt.json'):pair('D1/'+name,(OUT/'D1'/name).read_bytes(),True)
    pair('progress.json',dump(dict(status='ALL38_NEW_TASKS_AND19_CONTROLS_D1_FULLY_VERIFIED',completed_utc=summary['created_utc'],neural_verified=33,F1_verified=5,controls_verified=19,D1_verified=True,complete_summary_generated=True)))
    pair('progress.md',('# 第四轮预测实验进度v2\n\n33项神经训练（含5项M0）、5项LightGBM公平性训练、19项原神经控制只读推理及D1均已真实完成并通过完整D/E验收。完整结果见本目录第四轮CFD接入预测与公平性实验报告_v2_含M0与D1.md。\n').encode('utf8'))
    files={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='final_summary_D_E_SHA.json'}
    for name,entry in files.items():need((DEST/name).read_bytes()==(EDEST/name).read_bytes() and sha(EDEST/name)==entry['sha256'],'Final summary D/E SHA differs')
    pair('final_summary_D_E_SHA.json',dump(dict(status='ALL38_AND19_D1_COMPLETE_SUMMARY_D_E_SHA_PASS',created_utc=summary['created_utc'],files=files,
        new_training_tasks=38,neural_control_inferences=19,D1_verified=True,no_unapproved_models_or_statistics_families=True)),True)
    print(json.dumps(dict(status=summary['status'],D_path=str(DEST),E_path=str(EDEST),files=len(files)),ensure_ascii=False))

if __name__=='__main__':main()
