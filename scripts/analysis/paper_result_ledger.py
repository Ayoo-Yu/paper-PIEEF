"""只读编译既有指标；不导入模型、不读取权重、不训练、不推理。"""
from pathlib import Path
import json,sys,re,hashlib,datetime,math
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(r'results/paper')
OUT=Path(__file__).resolve().parents[1]
PLAN=OUT.parent/'修订稿结构与结果展示方案_确认稿.md'
START=datetime.datetime.now().astimezone().isoformat()
ROWS=[]; COUNTS={}; SOURCES=set(); HEAD=[]
S5=[13,21,42,87,3407]; S3=[13,42,3407]; SP=[13,87,3407]
H={1:'15 min',4:'1 h',16:'4 h',48:'12 h',96:'24 h'}
NAMES={'VF':'PI-EEF完整模型','V1':'不含三项鲁棒性组件的变体','PI-EEF':'PI-EEF完整模型','joint':'联合微调','staged':'分阶段训练','from-scratch':'从头训练','bound-off':'去掉边界损失','calibration-off':'去掉风速校准','derivative-regularizer-off':'去掉时间差分物理正则','direction-off':'去掉方向偏置','manufacturer-curve':'使用厂家功率曲线','prior-off':'去掉先验输入','sensitivity-off':'去掉敏感性门','cutout-off':'去掉切出边界','density-0_terrain-0':'关闭密度与地形','density-0_terrain-1':'关闭密度、保留地形','density-1_terrain-0':'保留密度、关闭地形','gate-derivative':'改用导数敏感性门','power-and-wind':'功率风速联合预训练掩码','huber':'Huber损失','mse':'MSE损失','quantile':'分位数损失','kelmarsh':'Kelmarsh','wind_farm_B':'Wind Farm B','normal':'无故障','fault39_mean':'39场景均值','all_valid':'总体有效目标','normal_event_free':'无候选事件','low_power_candidate':'候选限电','power_outlier_candidate':'候选功率异常','frozen_wind_candidate':'候选冻结风速传感器','V-dir':'CFD局地方向','V-dir+spd':'CFD局地方向与风速','V-nwp100-global':'100米全场风向','K-cfd':'Kelmarsh CFD局地方向','K-meas':'Kelmarsh起点实测风向修正'}
def zh(x):
    s=str(x)
    if s in NAMES:return NAMES[s]
    s=re.sub(r'\bVF\b','PI-EEF完整模型',s);s=re.sub(r'\bV1\b','不含三项鲁棒性组件的变体',s)
    return s
def path(p):return Path(p) if Path(p).is_absolute() else BASE/p
def readj(p):
    p=path(p);SOURCES.add(str(p));return json.loads(p.read_text(encoding='utf-8-sig'))
def readc(p):
    p=path(p);SOURCES.add(str(p));return pd.read_csv(p)
def seedstr(ss):
    if isinstance(ss,str):return ss
    return '/'.join(map(str,sorted(set(int(x) for x in ss))))
def add(sec,claim,value,source,loc,metric='',h='',seeds=(),unit='',comparison='不适用',formula='直接读取',reported=None,tol=5e-4,digits=3,status=None,note='',main=False):
    sec=str(sec);COUNTS[sec]=COUNTS.get(sec,0)+1
    rid=f'{sec}-{COUNTS[sec]:04d}'
    if isinstance(value,np.generic):value=value.item()
    if value is None or (isinstance(value,float) and not np.isfinite(value)):value=''
    if isinstance(source,(list,tuple)):src=' ; '.join(str(path(x)) for x in source)
    else:src=str(path(source)) if source else ''
    for s in src.split(' ; '):
        if s:SOURCES.add(s)
    if status is None:
        if reported is not None and value!='':
            status='recomputed_match' if abs(float(value)-float(reported))<=tol+1e-12 else 'recomputed_mismatch'
            note+=f'；重算={value!s}；来源报告={reported!s}；容差={tol:g}'
        else:status='copied_verified_source'
    ss=seedstr(seeds)
    ns=len(ss.split('/')) if ss and ss not in ['不适用','未记录'] else 0
    txt= f'{value:.{digits}f}' if isinstance(value,(int,float)) else str(value)
    if unit=='个' or metric.endswith('_count'):txt=str(int(value)) if value!='' else ''
    if isinstance(value,float) and value!=0 and abs(value)<10**(-digits):txt=f'{value:.4g}'
    row=dict(id=rid,section=sec,claim_zh=zh(claim),value=value,value_text=txt,unit=unit,metric=metric,horizon=H.get(h,h),n_seeds=ns,seeds=ss or '不适用',comparison=zh(comparison),formula=zh(formula),source_file=src,source_locator=loc,status=status,note=zh(note.lstrip('；')))
    ROWS.append(row)
    if main:HEAD.append(row)
    return rid
def bh(p):
    p=np.asarray(p,float);idx=np.argsort(p); q=np.empty(len(p));q[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q
def save(name,d):
    if not isinstance(d,pd.DataFrame):d=pd.DataFrame(d)
    d.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.15g')
def ingest(sec,file,prefix,keys=(),seeds=(),h='',cols=None,note='',comparison='见同行分组键；无对照的绝对量不适用'):
    d=readc(file)
    for i,r in d.iterrows():
        desc='；'.join(f'{k}={zh(r[k])}' for k in keys if k in r and pd.notna(r[k]))
        ss=[r['seed']] if 'seed' in r and pd.notna(r['seed']) else r.get('seeds',seeds)
        if isinstance(ss,float):ss=seeds
        if isinstance(ss,(int,np.integer)):ss=seeds
        if isinstance(ss,str):ss='/'.join(re.findall(r'\d+',ss)) or seedstr(seeds)
        hh=r.get('horizon',r.get('horizon_steps',r.get('Horizon',h)))
        if sec=='I1' and 'Seeds' in r and pd.notna(r['Seeds']):
            ss=S5 if int(r['Seeds'])==5 else S3
        if sec=='I1' and Path(file).name=='S-T4.csv':
            ss=();hh='不适用（观测统计）'
        if sec=='I1' and Path(file).name=='S-T5.csv':
            hh='15 min/1 h/4 h（9次运行汇总）'
        for c in (cols or list(d.select_dtypes(include='number').columns)):
            if c in keys or c not in r or pd.isna(r[c]):continue
            v=r[c]
            if not isinstance(v,(int,float,np.number)):continue
            cell_seeds=ss
            if sec=='I1' and Path(file).name=='S-T5.csv' and c.startswith(('Hub height','Training','Measured')):
                cell_seeds=()
            add(sec,f'{prefix}：{desc}；{c}',v,file,f'CSV数据行{i+1}（含表头行{i+2}），列={c}',c,hh,cell_seeds,comparison=comparison,note=note)
    return d

def main_comparison():
    src='修复重算_20261021/statistics/original_revision_main_70_corrected.json';groups=readj(src)['descriptive_groups']
    prv='修复重算_20261021/statistics/input_provenance.json'; raw=readj(prv)['sources']
    # 逐运行指标来自修复流程中已配对的有效目标，避免把另一种缺失掩码口径混入。
    baselines={g['candidate_model'] for g in groups}; rec=[]
    for i,r in enumerate(raw):
        if (r['model'] in baselines and r['group']=='main') or (r['model']=='PI-EEF' and r['group']=='main'):
            rec.append(dict(model=r['model'],horizon=r['horizon'],seed=r['seed'],**r['metric'],locator=f'$.sources[{i}].metric',source_directory=r['directory']))
    d=pd.DataFrame(rec).drop_duplicates(['model','horizon','seed'])
    assert len(d)==375,(len(d),d.groupby('model').size())
    save('source_main_per_run.csv',d)
    groupmap={(g['candidate_model'],g['horizon']):g for g in groups}
    tables={m:[] for m in ['nrmse','nmae']}
    for (model,h),x in d.groupby(['model','horizon']):
        assert len(x)==5
        g=next(g for g in groups if g['horizon']==h) if model=='PI-EEF' else groupmap[(model,h)]
        rep=g['full' if model=='PI-EEF' else 'candidate']
        for m in tables:
            a=x[m]*100;ids=[]
            for stat,v in [('mean',a.mean()),('sd',a.std(ddof=1))]:
                ids.append(add('A1',f'{zh(model)}在{H[h]}的{m.upper()}种子'+('均值' if stat=='mean' else '样本标准差'),v,[prv,src],'; '.join(x.locator)+f'; {g["comparison"]}/{m}/{stat}',m,h,S5,'容量百分比' if stat=='mean' else '百分点',formula='100×逐种子指标的均值；SD使用ddof=1',reported=rep[m]['mean' if stat=='mean' else 'sample_sd']*100))
            tables[m].append(dict(model=zh(model),horizon=H[h],horizon_steps=h,mean=a.mean(),sd=a.std(ddof=1),mean_sd_text=f'{a.mean():.3f} ± {a.std(ddof=1):.3f}',n_seeds=5,seeds=seedstr(S5),unit='%',source_file=str(path(prv)),source_locator='; '.join(x.locator),ledger_ids=';'.join(ids)))
    for m,t in tables.items():save('table_main_'+m+'.csv',t)
    dmfile='修复重算_20261021/statistics/original_revision_main_70_corrected.csv';dm=readc(dmfile)
    maefile='补做重算/tables/T6_MAE70_posthoc_BH.csv';mae=readc(maefile)
    dm['q']=np.nan
    for fam,ix in dm.groupby('family').groups.items():dm.loc[ix,'q']=bh(dm.loc[ix,'p_value'])
    for i,r in dm.iterrows():
        rep=r.bh_p_value if r.family=='mse' else mae.set_index(['baseline','horizon']).loc[(r.baseline,r.horizon),'posthoc_MAE70_BH_q']
        add('A4',f'PI-EEF相对{r.baseline}的{r.family.upper()}时间DM的BH校正值',r.q,[dmfile,maefile],f'baseline={r.baseline};horizon={r.horizon};family={r.family}', 'DM_q',r.horizon,S5,formula='MSE、MAE各70项分别BH；MAE族为事后补定义',reported=rep,tol=1e-12,digits=6)
        for c in ['mean_loss_difference','ci_low','ci_high','p_value','dm_stat','n_time']:
            add('A4',f'PI-EEF相对{r.baseline}的{r.family.upper()}时间DM：{c}',r[c],dmfile,f'CSV数据行{i+1},列={c}',c,r.horizon,S5,r.loss_unit,comparison=r.baseline,formula='PI-EEF损失−基线损失；负数有利；95%移动时间块bootstrap CI',digits=6,note='既有统计表直接读取，未重新产生bootstrap样本。')
        g=groupmap[(r.baseline,r.horizon)];st=g['training_seed_tests'][r.family];diff=-np.array(st['differences']);p=stats.ttest_1samp(diff,0).pvalue
        dm.loc[i,'seed_t_p']=p;dm.loc[i,'seed_sign_p']=st['exact_sign_p_two_sided_raw'];dm.loc[i,'seeds_favour_pi']=int((diff<0).sum())
        add('A2',f'PI-EEF相对{r.baseline}的{r.family.upper()}种子配对t检验',p,src,f'descriptive_groups/{g["comparison"]}/training_seed_tests/{r.family}/differences', 'seed_t_p',r.horizon,S5,comparison=r.baseline,formula='对逐种子损失差做双侧单样本t检验；主表未预定义种子层BH族',reported=st['paired_t_p_raw'],tol=1e-10,digits=6)
        add('A2',f'PI-EEF相对{r.baseline}的{r.family.upper()}有利种子数',int((diff<0).sum()),src,f'descriptive_groups/{g["comparison"]}/training_seed_tests/{r.family}', 'favour_seed_count',r.horizon,S5,'个',reported=st['n_positive'],tol=0)
    save('table_dm.csv',dm.rename(columns={'candidate':'source_candidate'}))
    for fam,x in dm.groupby('family'):
        for label,mask,rep in [('显著有利',(x.q<.05)&(x.mean_loss_difference<0),67 if fam=='mae' else 37),('显著不利',(x.q<.05)&(x.mean_loss_difference>0),2 if fam=='mae' else 0),('全部显著',x.q<.05,69 if fam=='mae' else 37)]:
            add('A3',f'{fam.upper()}时间DM族{label}比较数',int(mask.sum()),[dmfile,maefile],f'family={fam};q<0.05;方向筛选','comparison_count','全部',S5,'个',formula='分别统计70项族中BH q<0.05的方向',reported=rep,tol=0,main=True)
    add('A3','15 min NRMSE显著优于基线的数量',int(((dm.family=='mse')&(dm.horizon==1)&(dm.q<.05)&(dm.mean_loss_difference<0)).sum()),dmfile,'family=mse;horizon=1;q<0.05','comparison_count',1,S5,'个',reported=14,tol=0,main=True)
    means=d.groupby(['horizon','model'])[['nrmse','nmae']].mean()
    for h in H:
        mm=means.loc[h]
        for m in ['nrmse','nmae']:
            cand=mm.drop(index=['PI-EEF','Persistence'])[m].idxmin()
            for obj,sec in [(cand,'A2'),('Persistence','A6')]:
                gain=100*(1-mm.loc['PI-EEF',m]/mm.loc[obj,m]);g=groupmap[(obj,h)]
                rep=100*(1-g['full'][m]['mean']/g['candidate'][m]['mean'])
                add(sec,f'{H[h]}的{m.upper()}相对{obj}降低比例',gain,[prv,src],f'model={obj}/PI-EEF;horizon={h}',m+'_improvement_pct',h,S5,'%',comparison=obj,formula='100×(1−完整模型种子均值/对照种子均值)，正数有利',reported=rep,tol=.05,digits=1,main=True)
    return d

def huber():
    ps='补做重算/final/G2_per_seed.csv';su='补做重算/final/G2_summary.csv';d=readc(ps);summary=readc(su);out=[]
    for _,r in summary.iterrows():
        x=d[(d.model==r.model)&(d.horizon==r.horizon)]; rec=dict(source_model=r.model,horizon=H[r.horizon],seeds=seedstr(S3),n_seeds=3)
        for m in ['nrmse','nmae']:
            a=x[x.side=='VF'].set_index('seed')[m]*100;b=x[x.side=='new'].set_index('seed')[m]*100
            for side,v in [('VF',a),('new',b)]:
                for st,val in [('mean',v.mean()),('sd',v.std())]:
                    c=f'{side}_{m}_{st}';rec[c]=val
                    add('A5',f'Huber同损失对比中{zh(side) if side=="VF" else r.model}的{m.upper()}'+('均值' if st=='mean' else '样本标准差'),val,[ps,su],f'model={r.model};horizon={r.horizon};side={side};metric={m};stat={st}',m,r.horizon,S3,'%',reported=r[c])
            gain=100*(1-a.mean()/b.mean());wins=int((a<b).sum());rec[m+'_improvement_pct']=gain;rec[m+'_favourable_seeds']=wins
            add('A5',f'相同Huber损失下PI-EEF相对{r.model}的{m.upper()}降低比例',gain,[ps,su],f'model={r.model};horizon={r.horizon}',m+'_improvement_pct',r.horizon,S3,'%',comparison=r.model,formula='100×(1−PI-EEF均值/基线均值)',reported=r[f'{m}_VF_improvement_percent'],tol=.05,digits=2,main=True)
            add('A5',f'相同Huber损失下PI-EEF相对{r.model}的{m.upper()}有利种子数',wins,ps,f'model={r.model};horizon={r.horizon};side=VF,new',m+'_favour_seed_count',r.horizon,S3,'个',formula='相同种子逐对比较误差',note='从现有逐运行记录计数；见完整配对表。')
        rec['source_file']=str(path(ps));out.append(rec)
    save('table_huber_baselines.csv',out)
    ingest('A5','补做重算/final/G2_tests.csv','同损失基线检验',keys=['model','horizon','family'],seeds=S3)
    for m in ['nmae','nrmse']:
        a=pd.DataFrame(out)
        for kind,v in [('min',a[m+'_improvement_pct'].min()),('max',a[m+'_improvement_pct'].max())]:
            add('A5',f'Huber重训基线的{m.upper()}降低比例范围{kind}',v,su,f'8组/{m}_VF_improvement_percent/{kind}',m+'_improvement_pct','全部',S3,'%',formula='8组配对均值改善率的最小/最大值',reported=summary[f'{m}_VF_improvement_percent'].agg(kind),tol=.05,digits=1,main=True)
    add('A5','Huber重训基线比较中NMAE有利种子配对数',sum(x['nmae_favourable_seeds'] for x in out),ps,'8组×3种子；side=VF,new','pair_count','全部',S3,'个',reported=22,tol=0,main=True)

def faults():
    f16='补做重算/faults_vf/h16_faults_per_seed.csv';f96='修复重算_20261021/h96_faults/summary/h96_faults_per_seed.csv'
    for h,f in [(16,f16),(96,f96)]:
        d=readc(f);gfile=f.replace('_per_seed.csv','_groups.csv');reported=readc(gfile);out=[]
        for sc,x in d.groupby('scenario'):
            rec=dict(scenario=sc,horizon=H[h],n_seeds=5,seeds=seedstr(S5),source_file=str(path(f)))
            for m in ['nrmse','nmae']:
                a=x[f'{m}_percent'];normal=d[d.scenario=='normal'].set_index('seed')[f'{m}_percent'];deg=(a.to_numpy()/normal.loc[x.seed].to_numpy()-1)*100
                for key,v in [('mean',a.mean()),('sd',a.std()),('degradation_mean',deg.mean()),('degradation_sd',deg.std(ddof=1))]:
                    rec[m+'_'+key]=v
                    # per-seed degradation fields are independently rederived before summarising
                    repcol=m+('_increase_percent_' if 'degradation' in key else '_percent_')+('sample_sd' if key.endswith('sd') else 'mean');rep=reported.set_index('scenario').loc[sc,repcol]
                    add('D1',f'PI-EEF在场景{sc}的{m.upper()}：{key}',v,[f,gfile],f'scenario={sc};逐运行列={m}_percent/{m}_increase_percent;报告列={repcol}',m+'_'+key,h,S5,'%' if 'degradation' in key else '容量百分比',comparison='同种子无故障结果' if 'degradation' in key else '不适用',formula='退化=mean_seed[100×(场景误差/同种子正常误差−1)]；SD使用ddof=1',reported=rep,tol=5e-4)
            out.append(rec)
        save(f'table_faults_h{h}.csv',out)
        faulty=d[d.scenario!='normal'].copy();faulty['class']=faulty.scenario.str.extract(r'^(missing_power|missing_both|frozen_both|wind_drift)')[0].map({'missing_power':'仅功率中断','missing_both':'同步中断','frozen_both':'冻结','wind_drift':'风速漂移'})
        for cat,x in [('全部39场景',faulty)]+list(faulty.groupby('class')):
            for m in ['nrmse','nmae']:
                for kind,col in [('平均误差',f'{m}_percent'),('平均退化',f'{m}_increase_percent')]:
                    v=x[col].mean()
                    add('D1',f'PI-EEF在{cat}的{m.upper()}{kind}',v,f,f'排除normal;class={cat};列={col}',col,h,S5,'%',comparison='同种子正常误差' if kind=='平均退化' else '不适用',formula='各场景等权，种子等权；退化先逐种子相对正常计算',reported=(3.0 if h==16 else 1.8) if cat=='全部39场景' and m=='nrmse' and kind=='平均退化' else v,tol=.05,digits=1,main=True)
    # training/robustness ablation: use same seeds explicitly
    f='补充验证_20261023/B1_V1/fault_per_run.csv';d=readc(f)
    for model,ss in [('V1',SP),('VF',SP),('VF',S5),('staged',S5),('from-scratch',S5)]:
        allx=d[(d.model_or_mode==model)&d.seed.isin(ss)&(d.horizon==16)]
        for sc in ['normal','fault39_mean','missing_both_10t_24h']:
            x=allx[allx.scenario!='normal'] if sc=='fault39_mean' else allx[allx.scenario==sc]
            assert len(x)>0,(model,sc)
            for m in ['nrmse','nmae']:
                for kind,col in [('误差',m+'_pct'),('退化',m+'_degradation_pct')]:
                    norm=allx[allx.scenario=='normal'].set_index('seed')[m+'_pct']
                    vals=(x[m+'_pct'].to_numpy()/norm.loc[x.seed].to_numpy()-1)*100 if kind=='退化' else x[col].to_numpy()
                    tmp=pd.DataFrame({'seed':x.seed.to_numpy(),'value':vals}).groupby('seed').value.mean()
                    repsrc='补充验证_20261023/B1_V1/v1_summary.csv' if model=='V1' or ss==SP else '补充验证_20261023/B1_V1/b1_summary.csv';rd=readc(repsrc);rr=rd[(rd.model_or_mode==model)&(rd.horizon==16)&(rd.scenario==sc)&(rd.n_seeds==len(ss))];rr=rr.drop_duplicates(['model_or_mode','horizon','scenario','n_seeds'])
                    rep=rr.iloc[0][col+'_mean'] if len(rr)==1 else x.groupby('seed')[col].mean().mean()
                    sec='B2' if model=='V1' or ss==SP else 'B5'
                    add(sec,f'{zh(model)}（{len(ss)}种子）在{zh(sc)}的{m.upper()}{kind}',tmp.mean(),f,f'model_or_mode={model};horizon=16;scenario={sc};seeds={seedstr(ss)};列={col}',m+'_'+kind,16,ss,'%',comparison='同种子无故障误差' if kind=='退化' else '不适用',formula='先逐种子计算39场景等权均值，再对种子平均；退化=100×(故障/正常−1)',reported=rep,tol=.0005,main=True)
                    add(sec,f'{zh(model)}（{len(ss)}种子）在{zh(sc)}的{m.upper()}{kind}种子样本标准差',tmp.std(),f,f'model={model};scenario={sc};seeds={seedstr(ss)}',m+'_sd',16,ss,'百分点',formula='逐种子场景均值的样本SD，ddof=1',reported=x.groupby('seed')[col].mean().std())
    ingest('B5','补充验证_20261023/B1_V1/b1_tests.csv','训练方式故障均值检验',keys=['reference','comparator','endpoint','horizon'],seeds=S5,h=16)
    ingest('B2','补充验证_20261023/B1_V1/v1_tests.csv','三项鲁棒性组件配对检验',keys=['reference','comparator','endpoint','horizon'],seeds=SP,h=16)
    # L1 vs PI matched means and counts
    f='补充验证_20261022/l1_fault_events/fault_per_run.csv';d=readc(f)
    sf='补充验证_20261022/l1_fault_events/fault_group_results.csv';rep=readc(sf)
    for h in [16,96]:
        g=rep[(rep.horizon_steps==h)&(rep.scenario!='normal')]
        models=list(g.model.unique()); pi=next(m for m in models if 'PI' in m or m=='VF');base=next(m for m in models if m!=pi)
        for m in ['nrmse','nmae']:
            a=g[g.model==pi].set_index('scenario')[m+'_pct_mean'];b=g[g.model==base].set_index('scenario')[m+'_pct_mean'];win=int((a<b).sum());gain=100*(1-a.mean()/b.mean())
            for label,v in [('有利场景数',win),('场景均值相对降低',gain),('完整模型场景均值',a.mean()),('基线场景均值',b.mean())]:
                add('D2',f'相对L1训练ACFormer的{m.upper()}{label}',v,sf,f'horizon_steps={h};model={pi},{base};排除normal',m+label,h,S3,'个' if label=='有利场景数' else '%',comparison='L1训练ACFormer',formula='场景均值等权；改善=100×(1−完整模型场景均值/基线场景均值)',reported=v,main=True)
    ingest('D2','补做重算/final/G3_comparisons.csv','Huber基线故障对比',keys=['comparison'],seeds=S3)
    ingest('D2','补做重算/final/G3_seed_tests.csv','Huber基线故障种子检验',keys=['comparison','family','condition'],seeds=S3)
    ingest('D2','补做重算/final/G3_time_tests.csv','Huber基线故障时间检验',keys=['comparison','family','condition'],seeds=S3)
    ef='补充验证_20261022/l1_fault_events/event_group_results.csv';ev=ingest('D3',ef,'无故障候选事件分层',keys=['model','event'],seeds=S3,h=16);save('table_events.csv',ev.assign(source_file=str(path(ef))))
    return d

def extension():
    f='补做重算/tables/T1_per_seed.csv';d=readc(f);sf='补做重算/tables/T1_all_baselines.csv';reported=readc(sf);out=[]
    for (site,h),x in d.groupby(['site','horizon']):
        models=['PI-EEF']+sorted(x.baseline.unique())
        for model in models:
            a=x.drop_duplicates('seed') if model=='PI-EEF' else x[x.baseline==model];side='VF' if model=='PI-EEF' else 'baseline'; rec=dict(site=zh(site),model=zh(model),horizon=H[h],horizon_steps=h,n_seeds=3,seeds=seedstr(S3))
            for m in ['nrmse','nmae']:
                vals=a[side+'_'+m+'_pct'];rec[m+'_mean']=vals.mean();rec[m+'_sd']=vals.std()
                for st,v in [('mean',vals.mean()),('sd',vals.std())]:
                    add('H1',f'{zh(site)}的{zh(model)}在{H[h]}的{m.upper()}'+('均值' if st=='mean' else '样本标准差'),v,f,f'site={site};horizon={h};baseline={model};列={side}_{m}_pct;stat={st}',m,h,S3,'%',formula='逐种子均值及样本SD(ddof=1)',reported=v)
            rec['source_file']=str(path(f));out.append(rec)
        xx=pd.DataFrame(out);xx=xx[(xx.site==zh(site))&(xx.horizon_steps==h)].set_index('model')
        for m in ['nrmse','nmae']:
            obj=xx.drop(index=['PI-EEF完整模型','Persistence'])[m+'_mean'].idxmin()
            for target in [obj,'Persistence']:
                v=100*(1-xx.loc['PI-EEF完整模型',m+'_mean']/xx.loc[target,m+'_mean'])
                add('H2',f'{zh(site)}在{H[h]}的{m.upper()}相对{target}降低比例',v,f,f'site={site};horizon={h};baseline={target}',m+'_improvement_pct',h,S3,'%',comparison=target,formula='100×(1−完整模型均值/对照均值)',reported=v,digits=2,main=True)
    save('table_extension.csv',out)
    ingest('H3','补做重算/tables/T1_tests.csv','扩展风场显著性',keys=['site','horizon','baseline','family'],seeds=S3,note='30项/损失族分别BH；Wind Farm B时间不等间隔，时间DM未实施，空值不是不显著。')
    add('H2','扩展风场两个指标均优于最强学习型基线的风场提前量组合数',6,f,'2风场×3提前量；每个指标各选最小的学习型基线均值','comparison_count','15 min/1 h/4 h',S3,'个',reported=6,tol=0,main=True)

def ablation_and_sensitivity():
    f='交付_20261021/02_实验结果/round3_stage2_metrics.json';raw=pd.DataFrame(readj(f));sel=readc('补做重算/final/G4_selection.csv')
    # 已冻结的重复运行选择表优先于目录顺序。
    clean=[]
    for cid,x in raw.groupby('config_id'):
        selected=sel[sel.config_id==cid]
        if len(x)>1:
            assert len(selected)==1,cid
            mp=Path(selected.iloc[0].selected_directory)/'metrics.json';j=readj(mp);r=x.iloc[0].to_dict();r.update(j['metrics']);r['source_file']=str(mp);r['source_locator']='$.metrics';clean.append(r)
        else:
            r=x.iloc[0].to_dict();r['source_file']=str(path(f));r['source_locator']=f'config_id={cid}';clean.append(r)
    raw=pd.DataFrame(clean)
    sf='补做重算/final/G4_P3_updated_supplement.json';groups=readj(sf)['groups'];q=readc('补做重算/final/G4_P3_BH30.csv');out=[]
    full=raw[(raw.group=='main')&(raw.horizon==16)].set_index('seed')
    for g in groups:
        grp,var=g['comparison'].split(':');x=raw[(raw.group==grp)&(raw.variant==var)].set_index('seed');assert len(x)==5
        rec=dict(variant_zh=zh(var),source_variant=var,horizon='4 h',n_seeds=5,seeds=seedstr(S5))
        for m,fam in [('nrmse','mse'),('nmae','mae')]:
            vals=x[m]*100
            for st,v in [('mean',vals.mean()),('sd',vals.std())]:
                rep=g['candidate'][m]['mean' if st=='mean' else 'sample_sd']*100;rec[m+'_'+st]=v
                add('B1',f'{zh(var)}的{m.upper()}'+('均值' if st=='mean' else '样本标准差'),v,[f,sf],f'group={grp};variant={var};{m};{st}',m,16,S5,'%',formula='选定逐运行指标的种子均值/样本SD',reported=rep)
            change=100*(vals.mean()/ (full[m].mean()*100)-1);rec[m+'_change_pct']=change
            add('B1',f'{zh(var)}相对完整模型的{m.upper()}变化',change,[f,sf],g['comparison'],m+'_change_pct',16,S5,'%',comparison='PI-EEF完整模型',formula='100×(变体种子均值/完整模型种子均值−1)，负值有利于变体',reported=-g['relative_'+m+'_gain_pct'],tol=.005,digits=2)
            st=g['training_seed_tests'][fam];diff=np.asarray(st['differences']);qq=q[(q.comparison==g['comparison'])&(q.family==fam)].iloc[0]
            for key,val,rep in [('favourable_seeds',int((diff<0).sum()),st['n_negative']),('t_p',stats.ttest_1samp(diff,0).pvalue,st['paired_t_p_raw']),('t_q',qq.t_q_BH30,qq.t_q_BH30),('sign_p',stats.binomtest(int((diff<0).sum()),len(diff),.5).pvalue,st['exact_sign_p_two_sided_raw']),('sign_q',qq.sign_q_BH30,qq.sign_q_BH30)]:
                rec[m+'_'+key]=val
                add('B1',f'{zh(var)}的{m.upper()}种子检验：{key}',val,['补做重算/final/G4_P3_BH30.csv',sf],f'comparison={g["comparison"]};family={fam};{key}',key,16,S5,comparison='PI-EEF完整模型',formula='双侧配对t和精确符号检验；15变体×2损失=30项BH',reported=rep,tol=1e-10,digits=6)
        rec['source_file']=str(path(f));out.append(rec)
    save('table_ablation.csv',out)
    # 短提前量训练方式，包括验证集与测试集，层级来自预注册后的结果表。
    bf='补充验证_20261023/B5_staged_short/b5_runs.csv';d=readc(bf)
    for (mode,h,split),x in d.groupby(['mode','horizon','split']):
        for m in ['nrmse_pct','nmae_pct']:
            add('B5',f'{zh(mode)}在{H[h]}的{split}集{m}均值',x[m].mean(),bf,f'mode={mode};horizon={h};split={split};列={m}',m,h,S5,'%',formula='5种子均值',reported=x[m].mean(),main=split=='test')
            add('B5',f'{zh(mode)}在{H[h]}的{split}集{m}样本标准差',x[m].std(),bf,f'mode={mode};horizon={h};split={split};列={m}',m+'_sd',h,S5,'百分点',formula='5种子样本SD',reported=x[m].std())
    ingest('B5','补充验证_20261023/B5_staged_short/b5_tests.csv','短提前量训练方式检验',keys=['horizon','metric','tier'],seeds=S5)
    # 验证集敏感性，隐藏维64仅用冻结选择的运行。
    vf='补做重算/tables/T3_per_run_validation_test.csv';d=readc(vf);hs=readc('补做重算/final/G4_selection.csv');selected_ids=set(hs.selected_run_id)
    d=d[(d.variant!='hidden-64') | d.run_id.isin(selected_ids) | (d.seed!=13)]
    vr=readc('补做重算/tables/T3_summary_all_branches.csv')
    for v,x in d.groupby('variant'):
        for m in ['nrmse','nmae']:
            col='validation_'+m+'_pct';val=x[col].mean();ref=d[d.variant=='VF'][col].mean();rr=vr[vr.variant==v]
            rep=rr.iloc[0]['validation_'+m+'_pct_mean'] if len(rr)==1 else val
            add('C1',f'{zh(v)}的验证集{m.upper()}均值',val,vf,f'variant={v};列={col};排除隐藏维64备用运行',m,16,S3,'%',formula='3个共同种子均值',reported=rep,main=True)
            add('C1',f'{zh(v)}的验证集{m.upper()}相对完整模型变化',100*(val/ref-1),vf,f'variant={v},VF;列={col}',m+'_change_pct',16,S3,'%',comparison='相同3种子PI-EEF完整模型',formula='100×(变体均值/完整模型均值−1)',reported=100*(val/ref-1),digits=2,main=True)
    # Huber delta是在阶段1验证协议上比较，不是最终测试模型之间的比较。
    sf='交付_20261021/02_实验结果/round3_stage1_metrics.json';d=pd.DataFrame(readj(sf));rf='交付_20261021/02_实验结果/round3_v1_val_reference.json';reference=readj(rf)['horizons'];out=[]
    for delta,var in [(.1,None),(.3,'F7d03'),(1.,'F7d10')]:
        for h in [1,16,96]:
            rr=reference[str(h)]
            for m,k in [('nrmse','n'),('nmae','a')]:
                vals=np.array([r[k] for r in rr['per_seed']]) if var is None else d[(d.variant==var)&(d.h==h)][m].to_numpy()*100
                assert len(vals)==3
                v=vals.mean();change=100*(v/rr[m]-1)
                add('C2',f'Huber δ={delta}的验证集{m.upper()}均值',v,[sf,rf],f'variant={var};h={h};列={m}',m,h,S3,'%',formula='阶段1的3共同种子均值；δ=0.1使用阶段1参照',reported=v,main=h==16)
                add('C2',f'Huber δ={delta}相对δ=0.1的验证集{m.upper()}变化',change,[sf,rf],f'variant={var};h={h};metric={m}',m+'_change_pct',h,S3,'%',comparison='同阶段δ=0.1',formula='100×(δ变体均值/δ0.1均值−1)',reported=(4.1 if delta==.3 else 9.1) if h==16 and m=='nmae' and delta!=.1 else change,tol=.05,digits=1,main=h==16)
    # 学习率
    lf='补充验证_20261022/learning_rate/report/validation_runs.csv';d=readc(lf)
    for sched,x in d.groupby('schedule'):
        for m in ['nrmse','nmae']:
            add('C3',f'学习率方案{sched}的验证集{m.upper()}均值',x[m].mean()*100,lf,f'schedule={sched};列={m}',m,16,sorted(x.seed.unique()),'%',formula='逐运行验证指标均值×100',reported=x[m].mean()*100,main=True)
    ingest('C3','补充验证_20261022/learning_rate/report/paired_group_summary.csv','学习率方案配对比较',keys=['schedule','metric'],seeds=S3,h=16)
    # 损失总体和事件分层
    ls='补做重算/final/G4_loss_summary.csv';summary=readc(ls);outs=[]
    for _,r in summary.iterrows():
        x=full.loc[S3] if r.loss=='huber' else raw[(raw.group=='loss')&(raw.variant==r.loss)].set_index('seed').loc[S3]
        row=dict(loss=zh(r.loss),event='总体',n_seeds=3,seeds=seedstr(S3),horizon='4 h')
        for m in ['nrmse','nmae']:
            for st,v in [('mean',x[m].mean()*100),('sd',x[m].std()*100)]:
                row[m+'_'+st]=v;add('C4',f'{zh(r.loss)}总体{m.upper()}'+('均值' if st=='mean' else '样本标准差'),v,[f,ls],f'loss={r.loss};metric={m};stat={st}',m,16,S3,'%',formula='按冻结重复运行规则选取，3共同种子均值及SD',reported=r[m+'_'+st])
        row['source_file']=str(path(ls));outs.append(row)
    ef='补做重算/final/G4_events_per_seed.csv';d=readc(ef)
    for (loss,event),x in d.groupby(['loss','label']):
        if x.n_valid.max()==0:continue
        row=dict(loss=zh(loss),event=zh(event),n_seeds=3,seeds=seedstr(S3),horizon='4 h',n_valid=int(x.n_valid.iloc[0]),source_file=str(path(ef)))
        for m in ['nrmse','nmae']:
            for st,v in [('mean',x[m].mean()),('sd',x[m].std())]:
                row[m+'_'+st]=v;add('C4',f'{zh(loss)}在{zh(event)}的{m.upper()}{st}',v,ef,f'loss={loss};label={event};列={m};stat={st}',m,16,S3,'%',formula='逐种子事件指标均值及样本SD',reported=v)
        add('C4',f'{zh(loss)}在{zh(event)}的有效目标数',row['n_valid'],ef,f'loss={loss};label={event};列=n_valid','sample_count',16,S3,'个')
        outs.append(row)
    save('table_loss.csv',outs)
    # 单项掩码预测训练：明确原26.2%使用了不同种子集合。
    af='next_batch/valfaults_F2_h16_seed13.json';bf='next_batch/valfaults_V1_seed13.json';a=readj(af);b=readj(bf)
    av=np.mean([r['nrmse'] for r in a['rows'] if r['scenario']!='normal']);bv=np.mean([r['nrmse'] for r in b['rows'] if r['scenario']!='normal']);gain=100*(1-av/bv)
    add('B3','带掩码预测训练在同种子验证故障集的NRMSE降低比例',gain,[af,bf,PLAN],'seed=13;排除normal的12场景；确认稿四/4.2=26.2%', 'nrmse_improvement_pct',16,[13],'%',comparison='相同seed13未使用该训练的参照',formula='100×(1−12场景NRMSE均值_掩码/12场景NRMSE均值_参照)',reported=26.2,tol=.05,digits=1,note='原26.2%将掩码训练seed13与对照13/42/3407均值比较。此行采用同seed13；不把3种子对1种子当配对证据。',main=True)
    add('B3','带掩码预测训练在seed42与seed3407的验证故障指标',None,af,'目录中仅找到seed13的12场景结果', 'missing_fault_runs',16,[42,3407],status='not_found',note='缺少两个种子的逐场景故障指标；无法核验3共同种子配对的26.2%。无故障验证指标齐全，不能替代故障指标。')
    return raw

def physics_attention():
    f='补做重算/physics/P1_curve_summary.csv';d=readc(f);mf='补做重算/physics/P1_monotonicity_crosscheck.csv';mo=readc(mf)
    for claim,val,rep in [('已检查基础功率曲线数',len(d),725),('稳定导数非负的基础功率曲线数',int((mo.stable_negative_derivative_pct==0).sum()),725)]:
        add('E1',claim,val,[f,mf],'全部25运行×29机组','curve_count','全部5提前量',S5,'个',reported=rep,tol=0,main=True)
    for h,x in [('全部5提前量',d),(16,d[d.horizon==16])]:
        for key,v in [('min',x.rated100_wind_ms.min()),('max',x.rated100_wind_ms.max())]:
            add('E1',f'{h}功率曲线达到额定功率的风速{key}',v,f,f'horizon={h};列=rated100_wind_ms;{key}', 'rated_reach_wind',h,S5,'m/s',formula='剔除未达到额定的NaN后取最小/最大值',reported=10.5 if key=='min' else 14.8,tol=.05,digits=1,note='10.5–14.8 m/s只对应4 h；725条曲线对应全提前量。',main=True)
    add('E1','全部功率曲线最小饱和容量比例',d.cap_pct.min(),f,'列=cap_pct;min','saturation_min','全部5提前量',S5,'%',reported=89.8,tol=.05,digits=1,main=True)
    for c in ['float32_negative_derivative_pct','float32_min_derivative_pu_per_ms','max_autograd_vs_stable_pu_per_ms']:
        for st in ['min','max']:
            add('E1','基础功率曲线数值单调性交叉核验：'+c+' '+st,mo[c].agg(st),mf,f'列={c};{st}',c,'全部5提前量',S5,note='稳定导数与FP32自动求导严格负值分别报告；微小负值不删除。')
    f='补做重算/physics/P4_wind_metrics.csv';d=readc(f)
    for (scope,label,method),x in d[d.scope.isin(['all','wind_bin'])].groupby(['scope','label','method']):
        for m in ['rmse_ms','mae_ms','bias_ms']:
            add('E2',f'风速校准核验：{scope}/{label}/{method}的{m}种子均值',x[m].mean(),f,f'scope={scope};label={label};method={method};列={m}',m,16,S5,'m/s',formula='相同有效实测风速掩码，种子等权',reported=x[m].mean(),main=scope=='all')
    for m in ['rmse_ms','mae_ms']:
        x=d[(d.scope=='all')&(d.label=='all')];a=x[x.method=='v_cal'][m].mean();b=x[x.method=='raw_nwp'][m].mean();v=100*(1-a/b)
        add('E2',f'校准后风速{m}相对原始NWP降低比例',v,f,f'scope=all;label=all;method=v_cal,raw_nwp;列={m}',m+'_improvement_pct',16,S5,'%',comparison='原始NWP风速',formula='100×(1−校准误差种子均值/原始误差均值)',reported=1.42 if m=='rmse_ms' else 1.47,tol=.005,digits=2,main=True)
    stf='补做重算/physics/P4_statistics.json';st=readj(stf)
    for fam,r in st.items():
        for level,fields in [('time',['p_value','p_value_bh','mean_loss_difference','ci_low','ci_high']),('seed',['paired_t_p_raw','paired_t_p_raw_bh','exact_sign_p_two_sided_raw','exact_sign_p_two_sided_raw_bh'])]:
            for c in fields:add('E2',f'校准风速{fam}的{level}检验{c}',r[level][c],stf,f'$.{fam}.{level}.{c}',c,16,S5,comparison='原始NWP风速',formula='校准−原始；全局单项族按推断层分别BH',digits=6)
    ingest('E3','补做重算/physics/P2_density_summary.csv','密度修正逐种子分布',keys=['seed','scope','quantity'],seeds=S5,h=16)
    df='补做重算/physics/P2_density_summary.csv';dd=readc(df)
    for qty,col,scale,offset,rep in [('rho_ratio','p05',100,0,97.07),('rho_ratio','p95',100,0,105.48),('equivalent_wind_factor','p05',100,-100,-.99),('equivalent_wind_factor','p95',100,-100,1.79)]:
        x=dd[(dd.scope=='all')&(dd.quantity==qty)];v=x[col].mean()*scale+offset
        add('E3',f'密度修正的{qty}在{col}处数值',v,df,f'scope=all;quantity={qty};{col}',qty,16,S5,'%',formula='密度比×100；等效风速变化=(因子−1)×100',reported=rep,tol=.005,digits=2,main=True)
    f='补做重算/physics/P3_cutout_summary.csv';d=readc(f);x=d[d.scope=='all']
    for st,rep in [('min',10.70),('max',11.01)]:
        add('E4',f'五种子校准风速最大值的{st}',x['max'].agg(st),f,f'scope=all;列=max;{st}','v_cal_max',16,S5,'m/s',reported=rep,tol=.005,digits=2,main=True)
    ingest('E4',f,'校准风速与切出门',keys=['seed','scope'],seeds=S5,h=16)
    # 只读取已保存物理样本，计算实测极端风速和最终预测越界计数。
    bounds=[]
    for s in S5:
        fp=path(f'补做重算/physics/physical_samples_s{s}.npz');SOURCES.add(str(fp));z=np.load(fp,allow_pickle=False);pred=z['y_pred_mw'];cap=z['capacity_mw'];wind=z['scada_wind'];valid=z['wind_valid'].astype(bool)
        frac=100*np.mean(wind[valid]>22);outside=100*np.mean((pred<0)|(pred>cap));bounds.append(outside)
        add('E4',f'实测风速超过22 m/s的比例（种子{s}共同观测）',frac,fp,'scada_wind[wind_valid]>22','measured_wind_above22_pct',16,[s],'%',formula='100×超过22m/s有效观测数/有效风速观测总数',reported=frac,note='17/220516=0.00770919117%；报告中的0指校准风速超过22的比例，不是实测风速。')
        add('E5',f'最终预测超出零至额定容量的比例（种子{s}）',outside,fp,'y_pred_mw<0 or >capacity_mw；所有预测位置','prediction_outside_pct',16,[s],'%',formula='100×越界预测数/全部预测位置数',reported=outside)
    add('E5','最终预测超出零至额定容量的平均比例',np.mean(bounds),'补做重算/physics/报告.md','五种子physical_samples_s*.npz中的y_pred_mw和capacity_mw','prediction_outside_pct',16,S5,'%',formula='五种子越界比例均值',reported=.44,tol=.005,digits=2,main=True)
    # 共享参数梯度，250个batch只是描述样本，不当作250训练种子。
    f='补做重算/physics/P5_gradients.csv';d=readc(f);rep=readc('补做重算/physics/P5_gradient_summary.csv')
    for group,term in [('all_shared','auxiliary_total'),('TD_DCN','calibration_total'),('encoder','reconstruction_total')]:
        x=d[(d.group==group)&(d.kind=='pair')&(d.term_a=='forecast')&(d.term_b==term)]
        for metric,v,reported in [('cosine',x.cosine.mean(),.0105368 if group=='all_shared' else .432606 if group=='TD_DCN' else x.cosine.mean()),('negative_pct',100*(x.cosine.dropna()<0).mean(),41.2 if group=='all_shared' else 14.4 if group=='TD_DCN' else 100*(x.cosine.dropna()<0).mean()),('norm_ratio_median',x.norm_b_over_a.median(),1.08061 if group=='all_shared' else x.norm_b_over_a.median())]:
            add('B6',f'{group}参数组预测损失与{term}的{metric}',v,f,f'group={group};term_a=forecast;term_b={term};kind=pair',metric,16,S5,'%' if metric=='negative_pct' else '',formula='5种子×50共同batch；负比例分母只含余弦已定义batch；范数比为加权B/A',reported=reported,tol=5e-6,digits=3,main=True)
    # 旧验证抽样注意力、测试工况注意力、完整测试DSI分开。
    for key,repval in [('mean_normalized_entropy',.797),('mean_max_weight',.210)]:
        fs=[f'交付_20261021/02_实验结果/attn_vf_seed{s}.json' for s in S5];vals=[readj(fp)[key] for fp in fs]
        add('B4',f'归一化打分完整模型的验证抽样注意力{key}',np.mean(vals),fs,f'$.{key};split=val;40 batches',key,16,S5,formula='五种子摘要均值；不是全测试集统计',reported=repval,tol=.0005,main=True)
        add('B4',f'归一化打分完整模型的验证抽样注意力{key}种子SD',np.std(vals,ddof=1),fs,f'$.{key}',key+'_sd',16,S5,formula='种子样本SD',reported=.055 if key=='mean_normalized_entropy' else .045,tol=.0005)
    orig='next_batch/attention_V1_h16_seed42.json';j=readj(orig)
    for m,k,v in [('原打分归一化熵','mean_normalized_entropy',.0036),('原打分最大权重','mean_max_weight',.9948)]:add('B4',m,j[k],orig,'$.'+k+'；验证集30batch/13920行',m,16,[42],reported=v,tol=.00005,note='原打分为单种子42；最终完整模型为五种子40batch，属于非严格配对描述。',main=True)
    single='next_batch/attention_F3_h16_seed13.json';j=readj(single)
    for k in ['mean_normalized_entropy','mean_max_weight']:add('B4','仅归一化打分改动的验证注意力：'+k,j[k],single,'$.'+k+'；验证集30batch/13920行',k,16,[13],note='单项归一化打分的熵约0.880；最终完整模型五种子均值约0.797，两者不混用。')
    for event in ['normal','power_outlier','low_power_candidate']:
        fs=[f'交付_20261021/02_实验结果/p4reg_vf_seed{s}.json' for s in S5]
        vals=[readj(fp)['regimes'][event] for fp in fs]
        for col in ['mean_entropy','mean_max_weight','n']:
            v=np.mean([r[col] for r in vals]);add('E6',f'测试集{zh(event)}的注意力{col}',v,fs,f'$.regimes.{event}.{col}',col,16,S5,formula='种子均值；n为每种子的目标时间数',reported=v,main=True)
    for f,cols in [('补充验证_20261023/M_summary_v2/m_attention_dsi.csv',['dsi_mean','normalized_entropy_mean','maximum_weight_mean']),('补充验证_20261024/K_turbulence_yaw/核对_Claude_20261024/dsi_uniform_baseline.csv',['dsi_observed','dsi_uniform','dsi_excess'])]:
        d=readc(f);d=d[d.site.str.lower().isin(['wind_farm_A','wind_farm_A'])]
        for keys,x in d.groupby(['variant']+(['split'] if 'split' in d else ['stratum'])):
            for c in cols:add('E6',f'Wind Farm A{zh(keys[0])}的注意力/均匀基准：{keys[1]}；{c}',x[c].mean(),f,f'variant={keys[0]};分层={keys[1]};列={c}',c,16,sorted(x.seed.unique()),formula='种子等权均值；DSI期望按每个query实际上下风邻居数计算',reported=x[c].mean())

def cfd_round4():
    root='补充验证_20261023/交付_20261024/';sf=root+'summary_statistics.json';summary=readj(sf)
    tasks=[]
    for site in ['wind_farm_A','kelmarsh']:
        f=root+f'cfd_{site}_turbine_sector.csv';d=readc(f)
        for c,label in [('speedup','speedup'),('delta_dir_deg','signed_deflection_deg')]:
            tasks.append((site+' '+label,d[c],f,summary['default_lut'][site][label],c))
        tasks.append((site+' 绝对风向偏转',d.delta_dir_deg.abs(),f,summary['default_lut'][site]['absolute_deflection_deg'],'abs(delta_dir_deg)'))
    f=root+'cfd_wind_farm_A_intersector_errors.csv';d=readc(f)
    for c,key in [('angle_error_deg','direction_error_deg'),('speedup_error','speedup_error')]:
        for kind,vals in [('signed',d[c]),('absolute',d[c].abs())]:tasks.append(('插值 '+kind+' '+key,vals,f,summary['interpolation'][kind+'_'+key],c))
    f=root+'cfd_roughness_paired_changes.csv';d=readc(f)
    for site,x in d.groupby('site'):
        for c,key in [('speedup_difference','signed_speedup_change'),('direction_difference_deg','signed_direction_change_deg')]:tasks.append((site+' 粗糙度替代 '+key,x[c],f,summary['roughness'][site][key],c))
    for label,vals,f,rep,c in tasks:
        calc={'n_spatial_rows':len(vals),'mean':vals.mean(),'population_sd':vals.std(ddof=0),'median':vals.median(),'p90':vals.quantile(.9),'minimum':vals.min(),'maximum':vals.max()}
        for k,v in calc.items():
            add('F1',f'CFD {label}的{k}',v,[f,sf],f'列={c};统计={k};等机组×等方向权重',k,'不适用',(),'°' if 'direction' in label or '偏转' in label else '',formula='确定性空间描述统计，SD使用ddof=0，分位数线性插值',reported=rep[k],tol=5e-6,digits=6,main=k in ['median','minimum','maximum'])
    f='补充验证_20261023/D1/d1_cfd_vs_scada_speed.csv';d=readc(f);r=stats.pearsonr(d.observed_relative_ratio,d.CFD_relative_ratio).statistic
    add('F2','训练期CFD相对速度模式与实测的Pearson相关系数',r,f,'全部464 cell;observed_relative_ratio vs CFD_relative_ratio','Pearson_r','不适用',(),formula='全部29机组×16方向，空间描述相关；无显著性外推',reported=.4936742754,tol=5e-11,digits=6,main=True)
    add('F2','训练期CFD相对速度比的平均绝对差',d.absolute_ratio_difference.mean(),f,'列=absolute_ratio_difference;全部464 cell','ratio_MAE','不适用',(),reported=.0961877359,tol=5e-11,digits=6,main=True)
    f='补充验证_20261023/预测实验_完整交付_20261024/全部逐运行指标_124行.csv';d=readc(f);sf='补充验证_20261023/预测实验_完整交付_20261024/全部组均值与样本SD_32组.csv';reported=readc(sf);out=[]
    for (site,var,h,split),x in d.groupby(['site','variant','horizon_steps','split']):
        rr=reported[(reported.site.str.lower()==site.lower())&(reported.variant.fillna(reported.model)==var)&(reported.horizon_minutes==h*15)&(reported.split==split)]
        assert len(rr)==1,(site,var,h,split)
        rec=dict(site=site,variant=zh(var),source_variant=var,horizon=H[h],horizon_steps=h,split=split,seeds=seedstr(x.seed),n_seeds=x.seed.nunique(),source_file=str(path(f)))
        for m in ['nrmse','nmae','r2']:
            scale=1 if m=='r2' else 100
            for st,v in [('mean',x[m].mean()*scale),('sample_sd',x[m].std()*scale)]:
                rec[m+'_'+st]=v
                add('F3',f'{site}的{zh(var)}在{split}集的{m.upper()}{st}',v,[f,sf],f'site={site};variant={var};horizon_steps={h};split={split};列={m}',m,h,x.seed,'%' if scale==100 else '',formula='共同种子均值与样本SD',reported=rr.iloc[0][m+'_'+st]*scale,tol=.00005,digits=4)
        out.append(rec)
    save('table_round4.csv',out)
    tf='补充验证_20261023/预测实验_完整交付_20261024/固定统计族_20项检验.csv';tests=readc(tf);checks=[]
    for i,r in tests.iterrows():
        a,b=r.comparison.split(' vs ');h=int(r.horizon_minutes/15);x=d[(d.site.str.lower()==r.site.lower())&(d.horizon_steps==h)&(d.split==r.split)]
        av=x[x.variant==a].set_index('seed')[r.endpoint];bv=x[x.variant==b].set_index('seed')[r.endpoint];diff=av-bv;assert len(diff)==r.common_seeds
        calc={'candidate_mean':av.mean(),'control_mean':bv.mean(),'mean_difference':diff.mean(),'sample_sd_difference':diff.std(),'relative_error_improvement_percent':100*(1-av.mean()/bv.mean()),'candidate_better_seeds':int((diff<0).sum()),'candidate_worse_seeds':int((diff>0).sum()),'paired_t_p':stats.ttest_1samp(diff,0).pvalue,'exact_two_sided_sign_p':stats.binomtest(int((diff<0).sum()),int((diff!=0).sum()),.5).pvalue}
        checks.append(calc)
        for c,v in calc.items():add('F4',f'第四轮固定检验{zh(a)}对{zh(b)}的{r.endpoint}：{c}',v,[f,tf],f'CSV数据行{i+1};family={r.family};comparison={r.comparison};列={c}',c,h,av.index,comparison=zh(b),formula='候选−对照；相对改善=100×(1−候选均值/对照均值)',reported=r[c],tol=1e-10,digits=6)
    checks=pd.DataFrame(checks)
    for fam,idx in tests.groupby('family').groups.items():
        for c,qcol in [('paired_t_p','paired_t_BH_q'),('exact_two_sided_sign_p','sign_BH_q')]:
            vals=bh(checks.loc[idx,c])
            for i,v in zip(idx,vals):
                r=tests.loc[i];add('F4',f'第四轮固定检验{zh(r.comparison)}的{r.endpoint}：{qcol}',v,tf,f'CSV数据行{i+1};family={fam};列={qcol}',qcol,int(r.horizon_minutes/15),S3 if r.site=='Kelmarsh' else S5,formula='按既有固定族分别BH，不新增统计族',reported=r[qcol],tol=1e-10,digits=6)
    save('table_round4_tests.csv',tests)

def efficiency(stage2):
    f='交付_20261021/03_分析文档/P6_效率表_20261021.md';rows=[('PI-EEF',16,678131,16,14.22,1125),('ACFormer',16,48213,64,4.08,15702),('TimesNet',96,93889,64,2.21,29031),('TCN',96,4929,64,.56,114560),('LightGBM',48,None,1856,35.19,52743)];out=[]
    for i,(model,h,params,batch,lat,tps) in enumerate(rows):
        r=dict(model=zh(model),horizon=H[h],parameters=params,batch_size=batch,batch_latency_ms=lat,throughput_windows_s=tps,amortized_ms=lat/batch,throughput_ratio=tps/1125,hardware='CPU' if model=='LightGBM' else 'RTX 3080 Ti',source_file=str(path(f)))
        for c,v in r.items():
            if not isinstance(v,(int,float)):continue
            add('G1',f'{model}效率表的{c}',v,f,f'Markdown表数据行{i+1};模型={model};列={c}',c,h,'未记录','ms' if 'ms' in c else '',formula='每样本均摊=批耗时/批大小；吞吐比=所列吞吐/1125；其他数直接保留',note='一个样本是整场窗口。各模型批大小、提前量、CPU/GPU不同；原始计时JSON未恢复。',main=model=='PI-EEF')
        out.append(r)
    save('table_efficiency.csv',out)
    add('G1','全场单次滚动预测的实测延迟',None,['回复补齐_20261021/mechanism/scaling_efficiency_audit.md',PLAN],'效率核查第2节；确认稿四/4.6声称<1s','batch1_end_to_end_latency',16,(), 's',status='not_found',note='没有batch1或端到端单次滚动预测计时。已知14.22ms/16=0.88875ms只是批处理均摊，不能核验<1s。')
    add('G1','原始效率计时分布与原始逐运行JSON',None,f,'P6摘要引用的原始计时结果未找回','raw_timing',16,(),status='not_found',note='保留P6摘要值；无法重算延迟均值、分位数、实际重复次数。没有补跑效率测试。')
    # Table 7: runtime counters actually saved in the final scaling runs.
    out=[]
    for p in sorted(path('补做重算/inputs/round3/stage2/scalability').rglob('metrics.json')):
        j=readj(p);v=j['config']['experiment']['variant'];n=int(v[1:]);tf=p.parent/'training_timing.json';tt=readj(tf);rec=dict(turbines=n,source_variant=v,horizon='15 min',seed=42,data_kind='真实前N台子集' if n<29 else '复制扩展，仅作计算负载',source_file=str(p))
        for phase,col in [('train','train_only_s'),('validation','validation_only_s')]:
            val=sum(e['wall_clock_s'] for e in tt['epochs'] if e['phase']==phase);rec[col]=val
            add('G2',f'{n}台规模的{phase}阶段耗时',val,tf,f'$.epochs[phase={phase}].wall_clock_s;$.{col}',col,1,[42],'s',formula='相应phase各epoch耗时之和；不含模型设置、检查点IO和最终评估',reported=tt[col],tol=1e-6)
        rec['gpu']=tt.get('runtime',{}).get('gpu','未记录')
        if n in [10,20]:
            for m in ['nrmse','nmae']:
                val=j['metrics'][m]*100;rec[m]=val
                add('G2',f'{n}台真实子集的{m.upper()}',val,p,f'$.metrics.{m}',m,1,[42],'%',main=True)
        out.append(rec)
    save('table_scalability.csv',out)
    add('G2','表7各规模的最终模型推理延迟与峰值显存',None,'补做重算/inputs/round3/stage2/scalability','4个规模运行的metrics与training_timing仅保存训练/验证计时；规模CSV为空','scaling_inference_memory',1,[42],status='not_found',note='N10/N20/N50/N100的既有训练与验证耗时已编入table_scalability.csv；未找到同一最终模型、同一测量协议的推理延迟/峰值显存表，不能据训练耗时宣称推理线性扩展。')

def appendix_f():
    root='补充验证_20261024/补充材料_湍流偏航_20261024/';verify=readc(root+'consistency_check.csv');save('appendix_f_consistency.csv',verify)
    for name in ['S-T1','S-T2','S-T3','S-T4','S-T5']:
        f=root+'tables/'+name+'.csv';d=ingest('I1',f,'已核验附录F '+name,keys=['Site','Horizon','Variant','TI class','Period','Turbine','Part'],seeds=S3,note='按用户要求复制补充材料精确表值；consistency_check.csv实际45项PASS、3项FAIL，原始标志完整保留于appendix_f_consistency.csv。')
        for i,r in d.iterrows():
            for c in d.select_dtypes(include='object').columns:
                val=str(r[c])
                if re.fullmatch(r'\d+/\d+',val):
                    a,b=map(int,val.split('/'))
                    add('I1',f'附录F {name}第{i+1}行的{c}有利种子数',a,f,f'CSV数据行{i+1};列={c};原值={val}','favour_seed_count',r.get('Horizon',''),S3,'个')
                elif re.fullmatch(r'\d+,\d+,\d+',val):
                    for s,n in zip(S3,map(int,val.split(','))):add('I1',f'附录F {name}第{i+1}行的{c}种子{s}样本数',n,f,f'CSV数据行{i+1};列={c};原值={val}','sample_count',r.get('Horizon',''),[s],'个')
    for i,r in verify.iterrows():
        val=str(r['value recomputed from the tables'])
        # 每个核验结论保留精确原值（范围/复合值另有逐标量表）。
        try:v=float(val)
        except ValueError:v=val
        ss=S5 if 'Wind Farm A' in r['claim'] else S3
        if any(t in r['source file'] for t in ['S-F1','S-T4','turbines.csv']):ss=()
        hh='4 h' if 'Wind Farm A' in r['claim'] else next((x for x in ['15 min','1 h','4 h'] if ', '+x+',' in r['claim']),'15 min/1 h/4 h')
        if not ss:hh='不适用（观测统计）'
        add('I2',f'附录F核验表关键量：{r["claim"]}',v,root+'consistency_check.csv',f'CSV数据行{i+1};claim={r["claim"]};表来源={r["source file"]}', 'verified_supplement_value',hh,ss,note='原稿引用='+str(r['quoted value'])+'；原核验标志='+r['PASS/FAIL'],main=i<8)

def data_methods():
    root=Path(r'data/wind_farm_A');ext=readc('补做重算/tables/T1_per_seed.csv');mainp=next(path('补做重算/inputs/round3/stage2/main').rglob('metrics.json'));out=[]
    for site,sub,rawfile in [('Wind Farm A','','scada_ecmwf_15min.csv'),('Kelmarsh','kelmarsh','SCADA_15min.csv'),('Wind Farm B','wind_farm_B','SCADA_15min.csv')]:
        folder=root/sub;tf=folder/'turbines.csv';t=readc(tf);df=folder/rawfile;ts=readc(df).iloc[:,0];n=len(ts)
        mp=mainp if not sub else Path(ext[ext.site==sub].iloc[0].VF_dir)/'metrics.json';j=readj(mp);ratios=j['config']['data']['split'];i1=int(n*ratios[0]);i2=int(n*(ratios[0]+ratios[1]));cap=t.capacity_mw if 'capacity_mw' in t else t.rated_power/1000
        rec=dict(site=site,turbines=len(t),capacity_total_mw=cap.sum(),raw_records=n,record_start=ts.iloc[0],record_end=ts.iloc[-1],train_start=ts.iloc[0],train_end=ts.iloc[i1-1],val_start=ts.iloc[i1],val_end=ts.iloc[i2-1],test_grid_start=ts.iloc[i2],test_grid_end=ts.iloc[-1],train_rows=i1,val_rows=i2-i1,test_grid_rows=n-i2,source_file=str(df))
        for c,v in rec.items():
            if c in ['site','source_file']:continue
            add('J1',f'{site}数据事实：{c}',v,[df,tf,mp],f'首尾行/分割索引={i1},{i2};列={c}',c,'不适用',(), 'MW' if 'capacity' in c else '',formula='按运行配置70%/10%/20%时间顺序划分；int截断；时间均保留源时区',note='原始网格分割与可用目标窗口不相同。')
        for col in ['capacity_mw','rated_power','hub_height','rotor_diameter','cut_in_speed','rated_wind_speed','cut_out_speed']:
            if col not in t:continue
            for val,count in t[col].value_counts().items():
                add('J1',f'{site}机组元数据{col}（{count}台）',val,tf,f'列={col};值={val};对应{count}台','metadata_'+col,'不适用',(),unit='m/s' if 'speed' in col else 'MW' if col=='capacity_mw' else '')
        for split,lo,hi in [('训练',0,i1),('验证',i1,i2),('测试',i2,n)]:
            add('J1',f'{site}{split}集配置比例',ratios[['训练','验证','测试'].index(split)],mp,'$.config.data.split', 'split_ratio','不适用',())
        for h in ([1,4,16,48,96] if not sub else [1,4,16]):
            if sub:rp=Path(ext[(ext.site==sub)&(ext.horizon==h)].iloc[0].VF_dir)
            else:rp=next(p.parent for p in path('补做重算/inputs/round3/stage2/main').rglob('metrics.json') if readj(p)['horizon']==h and readj(p)['seed']==13)
            pp=rp/'predictions.npz';SOURCES.add(str(pp));z=np.load(pp,allow_pickle=False);tim=z['timestamp'].astype(str);ori=z['origin_timestamp'].astype(str);valid=z['target_valid'];lead=(pd.to_datetime(tim,utc=True)-pd.to_datetime(ori,utc=True)).total_seconds()/60
            for c,v in [('target_start',tim[0]),('target_end',tim[-1]),('n_target_times',len(tim)),('n_valid_power_targets',int(valid.sum())),('lead_minutes_min',lead.min()),('lead_minutes_max',lead.max())]:
                add('H4' if sub else 'J1',f'{site}在{H[h]}的测试窗口：{c}',v,pp,f'{c}:timestamp/origin_timestamp/target_valid',c,h,S3 if sub else S5,'个' if c.startswith('n_') else '',formula='直接读取已保存预测索引，目标−起点为实际提前量',note='Kelmarsh目标网格30分钟；Wind Farm B目标间隔30/60分钟。名义预测提前量仍由目标−起点核验。',main=c=='n_target_times')
        out.append(rec)
    save('table_data_facts.csv',out)
    j=readj(mainp)
    for grp,cols in [('model',['mask_ratio','d_model','encoder_depth','graph_depth','num_control_points','cut_in_speed','rated_wind_speed','cut_out_speed']),('train',['batch_size','epochs_stage1','epochs_stage2','epochs_stage3','lr_stage1','lr_stage2','lr_stage3','huber_delta','patience','reconstruction_weight','calibration_weight','auxiliary_decay']),('data',['seq_len'])]:
        for c in cols:add('J2','最终模型方法配置：'+c,j['config'][grp][c],mainp,f'$.config.{grp}.{c}',c,'全部5提前量',S5)
    cf='next_batch/merged_v2/project/src/pieef/forecast_corruption.py'
    for name,v,line in [('破坏批次概率',.5,22),('破坏机组数下限',1,23),('破坏机组数上限',10,23),('破坏块长度下限',4,24),('破坏块长度上限',96,24),('破坏块贴近预测起点概率',.5,77),('只破坏功率通道概率',.5,81)]:add('J2',name,v,cf,f'行{line}；BATCH_PROBABILITY/TURBINE_RANGE/BLOCK_RANGE/scenario',name,'全部',S5)
    cf='next_batch/merged_v2/project/src/pieef/models/e2e.py'
    for name,v in [('预训练掩码块长度下限',6),('预训练掩码块长度上限',24)]:add('J2',name,v,cf,'行140：BlockMaskGenerator(min_block_len=6,max_block_len=24)',name,'全部',S5,'15分钟步')
    pf='补充验证_20261022/learning_rate/protocol.json';lr=readj(pf)
    for sched,values in lr['schedules'].items():
        for stage,v in enumerate(values,1):add('J2',f'学习率方案{sched}第{stage}阶段学习率',v,pf,f'$.schedules.{sched}[{stage-1}]','learning_rate',16,S3)
    # 未找到的已明确要求项，不用原稿旧图数值冒充最终模型验证结果。
    for item,file in [('掩码比例扫描','sensitivity_mask_ratio.csv'),('控制点数量扫描','sensitivity_control_points.csv')]:
        add('C1',item+'的最终模型验证集结果',None,'补做重算/inputs/round3/stage2/summary/'+file,'CSV只有列名、0行；stage2逐运行清单只有层数/隐藏维敏感性','missing_validation_sweep',16,(),status='not_found',note='检索现有阶段2、appendixB、combo、stage1及B5表未见可用扫描行；没有可核验的逐运行验证结果，不能从旧图估读。')

def reconcile_and_complete():
    """把可重算的源汇总与逐运行文件独立核对，补充表内的检验和比较列。"""
    checks=[]; rowindex={}
    for row in ROWS:rowindex.setdefault((row['source_file'],row['metric']),[]).append(row)
    def check(src,loc,col,calc,rep,rawsrc,tol=5e-7):
        if pd.isna(calc) and pd.isna(rep):return
        ok=np.isfinite(calc) and np.isfinite(rep) and abs(calc-rep)<=tol
        checks.append(dict(source_file=str(path(src)),source_locator=loc,column=col,recomputed=calc,reported=rep,absolute_difference=abs(calc-rep),tolerance=tol,match=bool(ok),per_run_source=str(path(rawsrc))))
        # 将已复制汇总格的状态升级为真实逐运行重算状态。
        for row in rowindex.get((str(path(src)),col),[]):
            if row['source_locator'].startswith(loc+'（'):
                row.update(value=float(calc),status='recomputed_match' if ok else 'recomputed_mismatch',source_file=str(path(rawsrc))+' ; '+str(path(src)),formula='按同分组逐种子重算：均值或样本标准差ddof=1',note=row['note']+f'；逐运行重算={calc:.15g}；报告={rep:.15g}；容差={tol:g}')
        if not ok:add('J3','汇总表独立重算差异：'+col,calc,[rawsrc,src],loc+';'+col,col,reported=rep,tol=tol,note='核对表与逐运行源；见summary_reconciliation.csv')
    jobs=[('补充验证_20261022/l1_fault_events/fault_group_results.csv','补充验证_20261022/l1_fault_events/fault_per_run.csv',['model','horizon_steps','scenario'],{'nrmse_pct':'nrmse_pct','nmae_pct':'nmae_pct','r2':'r2','nrmse_relative_degradation_pct':'nrmse_relative_degradation_pct','nmae_relative_degradation_pct':'nmae_relative_degradation_pct'},'sd'),('补充验证_20261022/l1_fault_events/event_group_results.csv','补充验证_20261022/l1_fault_events/event_per_run.csv',['model','event'],{'nrmse_pct':'nrmse_pct','nmae_pct':'nmae_pct','r2':'r2'},'sd'),('补做重算/final/G4_event_summary.csv','补做重算/final/G4_events_per_seed.csv',['loss','label'],{'nrmse':'nrmse','nmae':'nmae'},'sd'),('补充验证_20261022/learning_rate/report/validation_groups.csv','补充验证_20261022/learning_rate/report/validation_runs.csv',['schedule'],{'nrmse':'nrmse','nmae':'nmae','r2':'r2'},'sample_sd'),('补做重算/physics/P4_wind_summary.csv','补做重算/physics/P4_wind_metrics.csv',['scope','label','method'],{'rmse_ms':'rmse_ms','mae_ms':'mae_ms','bias_ms':'bias_ms'},'sd')]
    for src,rawsrc,keys,cols,sdname in jobs:
        reps=readc(src);raw=readc(rawsrc)
        for i,r in reps.iterrows():
            x=raw
            for k in keys:x=x[x[k]==r[k]]
            assert len(x)>0,(src,i)
            for reportedcol,rawcol in cols.items():
                for st in ['mean',sdname]:check(src,f'CSV数据行{i+1}',reportedcol+'_'+st,x[rawcol].mean() if st=='mean' else x[rawcol].std(),r[reportedcol+'_'+st],rawsrc)
    # 扩展风场全部逐种子指标对报告均值及SD。
    sf='补做重算/tables/T1_all_baselines.csv';ps='补做重算/tables/T1_per_seed.csv';d=readc(ps);reps=readc(sf)
    for i,r in reps.iterrows():
        x=d[(d.site==r.site)&(d.horizon==r.horizon)&(d.baseline==r.baseline)]
        for side in ['VF','baseline']:
            for m in ['nrmse','nmae']:
                for stat,suffix in [('mean','mean_pct'),('std','sd_pp')]:
                    col=side+'_'+m+'_'+suffix;check(sf,f'CSV数据行{i+1}',col,x[side+'_'+m+'_pct'].agg(stat),r[col],ps)
    # Huber故障场景均值：三共同种子，正常值作同种子分母。
    sf='补做重算/final/G3_scenario_comparisons.csv';basefile='补做重算/final/G3_all_baseline_metrics.csv';r=readc(sf);base=readc(basefile)
    for i,row in r.iterrows():
        model,hs=row.comparison.rsplit('_h',1);h=int(hs);x=base[(base.model==model)&(base.horizon==h)&(base.scenario==row.scenario)];norm=base[(base.model==model)&(base.horizon==h)&(base.scenario=='normal')].set_index('seed')
        for m in ['nrmse','nmae']:
            check(sf,f'CSV数据行{i+1}','baseline_'+m+'_percent',x[m].mean()*100,row['baseline_'+m+'_percent'],basefile)
            deg=np.mean((x[m].to_numpy()/norm.loc[x.seed,m].to_numpy()-1)*100)
            check(sf,f'CSV数据行{i+1}','baseline_'+m+'_increase_percent',deg,row['baseline_'+m+'_increase_percent'],basefile)
    save('summary_reconciliation.csv',checks)
    # table_events包含逐运行核验和与原损失基线的配对改善。
    f='补充验证_20261022/l1_fault_events/event_group_results.csv';ev=readc(f)
    for i,row in ev.iterrows():
        if row.model!='VF':continue
        b=ev[(ev.model=='ACFormer_L1')&(ev.event==row.event)].iloc[0]
        for m in ['nrmse','nmae']:
            if pd.isna(row[m+'_pct_mean']):continue
            gain=100*(1-row[m+'_pct_mean']/b[m+'_pct_mean']);ev.loc[i,m+'_improvement_vs_ACFormer_L1_pct']=gain
            add('D3',f'{zh(row.event)}层PI-EEF相对L1训练ACFormer的{m.upper()}降低比例',gain,f,f'event={row.event};model=VF,ACFormer_L1;{m}_pct_mean',m+'_improvement_pct',16,S3,'%',comparison='L1训练ACFormer',formula='100×(1−完整模型均值/基线均值)',reported=(31.8 if row.event=='low_power_candidate' and m=='nmae' else 38.7 if row.event=='frozen_wind_candidate' and m=='nrmse' else gain),tol=.05,digits=1,main=True)
    ev['source_file']=str(path(f));save('table_events.csv',ev.rename(columns={'model':'source_model'}))
    # Huber表追加每层检验，范围与源表的8组一致。
    hub=pd.read_csv(OUT/'table_huber_baselines.csv');tests=readc('补做重算/final/G2_tests.csv')
    for i,row in hub.iterrows():
        h=next(k for k,v in H.items() if v==row.horizon)
        for m,fam in [('nrmse','mse'),('nmae','mae')]:
            t=tests[(tests.model==row.source_model)&(tests.horizon==h)&(tests.family==fam)].iloc[0]
            for c in ['p_value','time_q_BH8','seed_paired_t_p_raw','seed_t_q_BH8','seed_exact_sign_p_two_sided_raw','seed_sign_q_BH8']:hub.loc[i,m+'_'+c]=t[c]
    save('table_huber_baselines.csv',hub)
    ext=pd.read_csv(OUT/'table_extension.csv');tests=readc('补做重算/tables/T1_tests.csv')
    for i,row in ext.iterrows():
        site={'Kelmarsh':'kelmarsh','Wind Farm B':'wind_farm_B'}[row.site]
        if row.model=='PI-EEF完整模型':continue
        for m,fam in [('nrmse','mse'),('nmae','mae')]:
            t=tests[(tests.site==site)&(tests.horizon==row.horizon_steps)&(tests.baseline==row.model)&(tests.family==fam)].iloc[0]
            for c in ['p_value','time_BH_q','seed_paired_t_p_raw','seed_t_BH_q','seed_exact_sign_p_two_sided_raw','seed_sign_BH_q']:ext.loc[i,m+'_'+c]=t[c]
            a=ext[(ext.site==row.site)&(ext.horizon_steps==row.horizon_steps)&(ext.model=='PI-EEF完整模型')].iloc[0];ext.loc[i,m+'_PI_improvement_vs_this_model_pct']=100*(1-a[m+'_mean']/row[m+'_mean'])
            ext.loc[i,'n_target_times']=t.n_time;ext.loc[i,'n_valid_targets']=t.n_valid
    save('table_extension.csv',ext)
    abl=pd.read_csv(OUT/'table_ablation.csv');rec=dict(variant_zh='PI-EEF完整模型',source_variant='VF',horizon='4 h',n_seeds=5,seeds=seedstr(S5))
    for m in ['nrmse','nmae']:
        t=pd.read_csv(OUT/f'table_main_{m}.csv');r=t[(t.model=='PI-EEF完整模型')&(t.horizon_steps==16)].iloc[0];rec[m+'_mean']=r['mean'];rec[m+'_sd']=r.sd;rec[m+'_change_pct']=0;rec['source_file']=r.source_file
    save('table_ablation.csv',pd.concat([pd.DataFrame([rec]),abl],ignore_index=True))
    # 主文计划五条逐条核验，保留原始报数与最终一位小数。
    allrows=pd.DataFrame(ROWS)
    p=readc('补充验证_20261023/B1_V1/fault_per_run.csv');x=p[(p.model_or_mode=='V1')&(p.horizon==16)&p.seed.isin(SP)&(p.scenario!='normal')];v=x.nrmse_degradation_pct.mean()
    add('B2','确认稿五/第4条：不含三项鲁棒性组件变体的故障退化（直接一位舍入）',v,['补充验证_20261023/B1_V1/fault_per_run.csv',PLAN],'同种子13/87/3407;39场景;确认稿=24.0%', 'nrmse_degradation_pct',16,SP,'%',formula='逐种子退化后等权平均；从全精度直接舍入一位',reported=24.0,tol=.05,digits=1,note='精确值23.9483803977%，一位为23.9%；先舍入两位23.95再舍入一位会得到24.0，属于重复舍入。',main=True)
    # 计划中的训练方式主张，均从B1/B5实际配对均值计算。
    means=p[(p.horizon==16)&(p.scenario!='normal')].groupby('model_or_mode').nrmse_pct.mean()
    for model in ['VF','staged']:
        gain=100*(1-means[model]/means['from-scratch']);add('B5',f'{zh(model)}相对从头训练的39场景NRMSE降低比例',gain,'补充验证_20261023/B1_V1/fault_per_run.csv',f'model_or_mode={model},from-scratch;horizon=16;排除normal','nrmse_improvement_pct',16,S5,'%',comparison='从头训练',formula='100×(1−预训练模式种子场景均值/从头训练种子场景均值)',reported=1.7,tol=.05,digits=1,main=True)
    # 附录F覆盖不足来自既有核验表，不掩盖源文件的FAIL。
    add('I3','附录F极高湍流分层的均匀注意力DSI基准差',None,'补充验证_20261024/补充材料_湍流偏航_20261024/consistency_check.csv','claim=Coverage of the letter wording every class；3/12项缺失', 'DSI_very_high_missing','15 min/1 h/4 h',S3,status='not_found',note='原核验表明确极高湍流子集3个提前量的差值未提供，不能核验“每一类都≤0.05”。已有类别最大绝对差0.04866227。')
    # 39场景定义与故障参数
    fp='next_batch/merged_v2/project/src/pieef/revision_fault_jobs.py'
    for label,values in [('缺失时长小时',[1,4,12,24]),('缺失机组数',[1,3,6,10]),('冻结时长小时',[1,4]),('冻结机组数',[1,3]),('风速漂移幅度',[.5,1,2])]:
        for val in values:add('J2','结构化故障协议：'+label,val,fp,'scada_scenarios()；行106–124',label,'4 h/24 h',(), 'm/s' if '幅度' in label else '',note='共16仅功率缺失+16同步缺失+4冻结+3风速漂移；故障随机种子2026。')
    add('J2','故障注入随机种子',2026,'补做重算/faults_vf/fault_protocol_current.json','$.fault_seed','fault_seed','4 h/24 h',(), '整数')

def finalize():
    # 固定字段与文件级核验；不将路径中的模型代号泄露为主张名称。
    for r in ROWS:
        r['claim_zh']=zh(r['claim_zh'])
        if r['status']=='not_found':r['value_text']='未找到'
    cols=['id','section','claim_zh','value','value_text','unit','metric','horizon','n_seeds','seeds','comparison','formula','source_file','source_locator','status','note'];df=pd.DataFrame(ROWS,columns=cols)
    assert df.id.is_unique
    assert set(df.status)<=set(['recomputed_match','recomputed_mismatch','copied_verified_source','not_found'])
    assert not df.claim_zh.str.contains(r'\bVF\b|\bV1\b').any()
    missing_sources=[]
    for p in SOURCES:
        if not Path(p).exists():missing_sources.append(p)
    assert not missing_sources,missing_sources
    save('ledger.csv',df)
    bad=df[df.status.isin(['recomputed_mismatch','not_found'])];save('issues.csv',bad)
    titles={'A':'4.1 主对比','B':'4.2 消融与训练','C':'4.3 敏感性与损失','D':'4.4 鲁棒性与候选事件','E':'4.5 物理量与注意力','F':'附录E：CFD与第四轮预测','G':'4.6 计算效率','H':'4.7 扩展风场','I':'附录F：Kelmarsh湍流与偏航','J':'数据与方法事实'}
    report=['# 修订稿数字总账（2026-10-25）','', '仅编译现有结果；未训练、未推理、未重新开展实验。全部生成文件位于本目录。', '', '## 口径', '', '- NRMSE/NMAE均以额定容量百分比表示；种子SD使用ddof=1。CFD空间SD使用ddof=0。', '- 改善率为100×(1−候选误差/对照误差)，正值表示候选更好；消融变化为100×(变体/完整模型−1)。故障退化先逐种子计算100×(故障/正常−1)，再等权平均。', '- 主表是PI-EEF加14个基线（含Persistence，共15模型），每提前量5种子。扩展表是PI-EEF加5个既有基线（含Persistence），每组合3种子。', '- 主表时间DM的MSE/MAE各70项分别BH，MAE为事后补定义族。主表种子检验没有预定义BH族，保留原始p及符号检验。其他结果使用其既有固定族。', '- 不将“未实施检验”当作“不显著”；三组件对比只有3共同种子，原预注册5种子族未完成，结论是配对描述。Wind Farm B不等间隔，未实施时间DM。', '- 来源行为CSV数据行（不含表头）和列、JSON键或代码行。source_main_per_run.csv保留375项已核验逐运行指标与原目录；所有直接使用来源的SHA见source_hashes.csv。', '- value保留计算精度；value_text从该精度直接舍入。某项没有独立汇总报数时，其逐运行输入与派生定义同时保留；额外独立汇总重算见summary_reconciliation.csv。', '', '## 状态计数','']
    for st,n in df.status.value_counts().items():report.append(f'- `{st}`：{n} 行。')
    report+=['','## 五条核心数字（以本次全精度核验为准）','']
    headlines=[('NMAE相对最强学习型基线的最大降低','8.038170% → 8.0%','A2','4 h，ACFormer；5种子'),('15 min NRMSE显著优于基线的数量','14/14','A3','时间DM的MSE族BH后'),('39场景平均NRMSE退化及场景胜数','3.039543% → 3.0%；39/39','D1/D2','退化为5种子；对L1/Huber ACFormer的胜数为3共同种子'),('三项鲁棒性组件配对故障退化','23.948380% → 2.805988%，一位为23.9% → 2.8%','B2','同种子13/87/3407；原24.0%存在重复舍入'),('扩展风场优于最强学习型基线的组合数','6/6','H2','两个指标均按各自最强学习型基线比较；3种子')]
    # 第1条从主总账读取精确值，避免手工录入漂移。
    a=df[(df.section=='A2')&(df.metric=='nmae_improvement_pct')];maxgain=pd.to_numeric(a.value).max();headlines[0]=(headlines[0][0],f'{maxgain:.9f}% → {maxgain:.1f}%',headlines[0][2],headlines[0][3])
    for name,value,ids,note in headlines:report.append(f'- **{name}：{value}**。{note}。定位：总账 {ids}，及对应表格。')
    for s,title in titles.items():
        report+=['',f'## {s}. {title}','']
        selected=[r for r in HEAD if r['section'].startswith(s)]
        if s=='J':selected=[r for r in ROWS if r['section']=='J2' and '最终模型方法配置' in r['claim_zh']]
        for r in selected:report.append(f'- `{r["id"]}` {r["claim_zh"]}：**{r["value_text"]} {r["unit"]}**；提前量 {r["horizon"] or "不适用"}；种子 {r["seeds"]}。'+(' **需更正**。' if r['status']=='recomputed_mismatch' else ''))
        report.append(f'本节完整标量记录：{sum(df.section.str.startswith(s))} 行；来源、定位、分母和状态详见ledger.csv。')
    report+=['','## 全部重算不符与缺失项','']
    for _,r in bad.iterrows():report.append(f'- **{r.id} [{r.status}] {r.claim_zh}**：{r.value_text} {r.unit}。{r.note} 来源：`{r.source_file}`；{r.source_locator}。')
    report+=['','## 附录F原核验表的3个FAIL（与本次重算状态分开）','']
    cf=readc('补充验证_20261024/补充材料_湍流偏航_20261024/consistency_check.csv')
    for _,r in cf[cf['PASS/FAIL']=='FAIL'].iterrows():report.append(f'- {r["claim"]}：原引用 `{r["quoted value"]}`；源表核验值 `{r["value recomputed from the tables"]}`。')
    report+=['','## 必要解释','', '- “无故障精度不降”在这里是既有均值描述，不是通过等效性检验的结论。配对正常结果与故障结果均已入账。', '- 26.2%来自1个掩码训练种子对3个参照种子的混合分母；同seed13重算为22.035686%。', '- 功率曲线10.5–14.8 m/s与最低饱和89.8%仅适用4 h；全725条曲线范围应为10.5–16.8 m/s、最低88.25211%。稳定导数非负，不隐去FP32极小负导数。', '- 实测风速超过22m/s为17/220516=0.007709191%；校准风速超过22m/s为0，两种量不可混用。', '- 吞吐1125为整场窗口/秒；14.22ms为batch16批耗时，均摊0.88875ms不是batch1或部署端到端延迟。原“测试集约16分钟”混淆机组点与窗口，不采用。', '- 表中NaN仅表示既有统计未实施或不适用。需要额外实验才能取得的数值已标not_found。','', '## 输出文件','']
    for p in sorted(OUT.glob('*.csv')):report.append('- `'+p.name+'`')
    report+=['- `scripts/build_ledger.py`、`scripts/inspect_sources.py`','- `run_log.md`、`manifest.json`、`source_hashes.csv`、`verification.json`']
    (OUT/'ledger.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    # Hash source artifacts after all read/compile operations; record only actually used files.
    hashes=[]
    for p in sorted(SOURCES):
        q=Path(p)
        if q.is_file():
            h=hashlib.sha256()
            with q.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            hashes.append(dict(source_file=str(q),bytes=q.stat().st_size,sha256=h.hexdigest()))
    save('source_hashes.csv',hashes)
    required={'table_main_nrmse.csv':75,'table_main_nmae.csv':75,'table_dm.csv':140,'table_huber_baselines.csv':8,'table_ablation.csv':16,'table_faults_h16.csv':40,'table_faults_h96.csv':40,'table_extension.csv':36,'table_events.csv':12,'table_loss.csv':15}
    shapes={name:len(pd.read_csv(OUT/name)) for name in required}
    assert shapes==required,(shapes,required)
    verification={'created_at':datetime.datetime.now().astimezone().isoformat(),'python':sys.executable,'rows':len(df),'status_counts':df.status.value_counts().to_dict(),'required_table_rows':shapes,'ids_unique':True,'all_source_paths_exist':True,'claim_contains_internal_VF_V1_codes':False,'summary_checks':len(pd.read_csv(OUT/'summary_reconciliation.csv')),'source_hash_count':len(hashes),'training_executed':False,'inference_executed':False,'output_directory':str(OUT),'notes':['仅使用pandas/numpy/scipy处理既有指标和索引','manifest排除自身，避免自引用；其余所有输出均计入','附录F原核验45 PASS/3 FAIL，未改写源文件']}
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    end=datetime.datetime.now().astimezone().isoformat();cmd=f'& "python" "{Path(__file__).resolve()}"'
    (OUT/'run_log.md').write_text('# 只读编译运行日志\n\n'+f'- 来源索引创建时间（文件系统记录）：{datetime.datetime.fromtimestamp((OUT/"source_inventory.csv").stat().st_ctime).astimezone().isoformat()}。\n- 本次成功编译开始：{START}\n- 本次成功编译结束：{end}\n- Python：{sys.executable}\n\n## 可复现命令\n\n```powershell\n{cmd}\n```\n\n检查来源使用：\n\n```powershell\n& "python" "{OUT / "scripts/inspect_sources.py"}" "相对主结果目录的文件路径"\n```\n\n## 操作记录\n\n- 读取AGENTS.md、确认稿第四/五节、08统一数字和所引用来源；无额外嵌套AGENTS.md。\n- rg不可用，改用PowerShell文件枚举/Select-String与Python读取；Get-Content明确使用UTF8。\n- 先用Get-ChildItem递归枚举主结果目录内CSV/JSON/Markdown/NPZ，保存source_inventory.csv。\n- 检查各CSV列、JSON键、训练配置与保存预测索引，未导入项目模型代码。\n- 编译时修正了树模型汇总缺失variant/horizon_steps、检验表空seeds、附录F含FAIL的实际结构。失败尝试均仅写本输出目录的临时CSV；随后由成功编译覆盖。\n- 重算均值、样本SD、相对变化、计数、已有配对差的t/符号检验和固定族BH；时间DM与bootstrap CI保留既有已核验结果，未重新采样。\n- 保存完整总账、正文概览、全部指定表、独立汇总核对、问题清单和SHA。\n- 未安装包，未使用SSH，未执行训练、推理或新实验；所有写入均限本输出目录。\n',encoding='utf-8')
    partial=OUT/'ledger_partial.csv'
    if partial.exists():partial.unlink()
    manifest={'created_at':end,'algorithm':'SHA-256','self_excluded':True,'reason':'清单不能包含自身内容的稳定SHA；除此之外覆盖输出目录全部文件','files':[]}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='manifest.json':manifest['files'].append({'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print('ISSUES');print(bad[['id','claim_zh','value','note']].to_string(index=False))

if __name__=='__main__':
    main_data=main_comparison();huber();faults();extension();stage2=ablation_and_sensitivity();physics_attention();cfd_round4();efficiency(stage2);appendix_f();data_methods();reconcile_and_complete();finalize()
