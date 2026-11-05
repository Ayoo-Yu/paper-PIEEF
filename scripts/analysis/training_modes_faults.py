import sys,json,math
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy import stats
from prepare import js,sha,dump,info
import reference_helpers as ref
L=Path(__file__).resolve().parent;P=L.parents[1];protocol=js(L/'protocol.json')
SEEDS=protocol['seeds'];metrics=['nrmse_pct','nmae_pct','nrmse_degradation_pct','nmae_degradation_pct']
def bh(values):
 p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1);return out

def test_pair(group,h,other,metric,refname='VF'):
 a=seedmeans[(seedmeans.model_or_mode==refname)&(seedmeans.horizon==h)].set_index('seed').loc[SEEDS,metric].to_numpy()
 b=seedmeans[(seedmeans.model_or_mode==other)&(seedmeans.horizon==h)].set_index('seed').loc[SEEDS,metric].to_numpy()
 diff=a-b;fav=int((diff<0).sum());against=int((diff>0).sum());ties=int((diff==0).sum());t=stats.ttest_rel(a,b)
 sign=stats.binomtest(min(fav,against),fav+against,p=.5,alternative='two-sided').pvalue if fav+against else 1.
 return dict(family=group,horizon=h,reference=refname,comparator=other,endpoint=metric,n_seeds=5,seeds_favour_reference=fav,seeds_favour_comparator=against,ties=ties,reference_mean=float(a.mean()),comparator_mean=float(b.mean()),difference_reference_minus_comparator_pp=float(diff.mean()),difference_sd=float(diff.std(ddof=1)),relative_change_pct=float(100*(a.mean()/b.mean()-1)),paired_t_statistic=float(t.statistic),paired_t_p=float(t.pvalue),exact_two_sided_sign_p=float(sign),**{f'difference_seed{s}':float(x) for s,x in zip(SEEDS,diff)})
def tests(family):
 x=pd.DataFrame(family);x['paired_t_bh_q']=bh(x.paired_t_p);x['sign_bh_q']=bh(x.exact_two_sided_sign_p)
 x['tier']=['support' if r.seeds_favour_reference==5 and r.paired_t_bh_q<.05 else 'trend' if r.seeds_favour_reference>=4 else 'none/unfavourable' for r in x.itertuples()]
 return x

def summaries(data):
 rows=[]
 for (mode,h),g in data.groupby(['model_or_mode','horizon'],sort=False):
  faults=g[g.scenario!='normal'];worst=faults.groupby('scenario').nrmse_pct.mean().idxmax()
  groups=[('normal',g[g.scenario=='normal']),('fault39_mean',seedmeans[(seedmeans.model_or_mode==mode)&(seedmeans.horizon==h)]),('worst_scenario',g[g.scenario==worst])]
  if mode in ['V1','VF']:
   groups += [(scene,g[g.scenario==scene]) for scene in ['missing_power_10t_24h','missing_both_10t_24h']]
  for condition,v in groups:
   row=dict(model_or_mode=mode,horizon=h,condition=condition,scenario=worst if condition=='worst_scenario' else condition,n_seeds=len(v))
   for m in metrics:row[m+'_mean']=float(v[m].mean());row[m+'_sd']=float(v[m].std(ddof=1))
   rows.append(row)
 return pd.DataFrame(rows)

rows=pd.read_csv(L/'vf_reused_per_run.csv').to_dict('records');normal_gates=[]
for task in protocol['sources']:
 d=L/'runs'/task['task_id'];c=js(d/'completed.json');assert c['status']=='completed' and c['natural_process_exit']['returncode']==0
 for f in c['outputs']:assert sha(f['path'])==f['sha256']
 assert c['normal_gate']['status']=='PASS' and c['normal_gate']['failed_cells']==0
 normal_gates.append(dict(task_id=task['task_id'],**c['normal_gate']))
 n=next(r for r in c['rows'] if r['scenario']=='normal')
 assert len(c['rows'])==40 and set(r['scenario'] for r in c['rows'])=={'normal'}|{s['name'] for s in protocol['scenarios']}
 for r in c['rows']:
  rows.append(dict(model_or_mode=task['model_or_mode'],horizon=task['horizon'],seed=task['seed'],scenario=r['scenario'],nrmse_pct=100*r['nrmse'],nmae_pct=100*r['nmae'],r2=r['r2'],n_targets=r['n'],nrmse_degradation_pct=100*(r['nrmse']/n['nrmse']-1),nmae_degradation_pct=100*(r['nmae']/n['nmae']-1),source_run=task['directory'],prediction_file=str(d/(r['scenario']+'.npz'))))
df=pd.DataFrame(rows);assert len(df)==1200 and not df.duplicated(['model_or_mode','horizon','seed','scenario']).any();assert (df.n_targets==220401).all()
df.to_csv(L/'fault_per_run.csv',index=False)
seedmeans=df[df.scenario!='normal'].groupby(['model_or_mode','horizon','seed'],as_index=False)[metrics].mean()
seedmeans.to_csv(L/'per_seed_fault39_summary.csv',index=False)
b1=df[(df.horizon==16)&df.model_or_mode.isin(['VF','staged','from-scratch'])];v1=df[df.model_or_mode.isin(['V1','VF'])]
bs=summaries(b1);vs=summaries(v1);bs.to_csv(L/'b1_summary.csv',index=False);vs.to_csv(L/'v1_summary.csv',index=False)
bt=tests([test_pair('B1',16,o,m) for o in ['staged','from-scratch'] for m in ['nrmse_pct','nmae_pct']]);bt.to_csv(L/'b1_tests.csv',index=False)
vt=tests([test_pair('V',h,'V1',m) for h in [16,96] for m in ['nrmse_pct','nrmse_degradation_pct']]);vt.to_csv(L/'v1_tests.csv',index=False)
wins=[]
sg=b1[b1.scenario!='normal'].groupby(['model_or_mode','scenario'])[metrics].mean()
for o in ['staged','from-scratch']:
 for m in ['nrmse_pct','nmae_pct']:
  d=sg.loc['VF',m]-sg.loc[o,m];wins.append(dict(comparator=o,metric=m,joint_wins=int((d<0).sum()),comparator_wins=int((d>0).sum()),ties=int((d==0).sum())))
pd.DataFrame(wins).to_csv(L/'b1_scenario_win_counts.csv',index=False)
historical_file=P/'补做重算/faults/F3_V1_historical_per_seed.csv';hist=pd.read_csv(historical_file)
assert len(hist)==200 and set(hist.seed)==set(SEEDS)
hist=hist.rename(columns={'nrmse_percent':'nrmse_pct','nmae_percent':'nmae_pct','nrmse_increase_percent':'nrmse_degradation_pct'})
new=df[(df.model_or_mode=='V1')&(df.horizon==16)]
hr=[]
for condition in ['normal','fault39_mean']:
 a=new[new.scenario=='normal'] if condition=='normal' else new[new.scenario!='normal'].groupby('seed',as_index=False)[metrics].mean()
 b=hist[hist.scenario=='normal'] if condition=='normal' else hist[hist.scenario!='normal'].groupby('seed',as_index=False)[['nrmse_pct','nmae_pct','nrmse_degradation_pct']].mean()
 for metric in ['nrmse_pct','nmae_pct','nrmse_degradation_pct']:
  for s in SEEDS+['mean']:
   na=float(a[metric].mean() if s=='mean' else a[a.seed==s][metric].iloc[0]);hb=float(b[metric].mean() if s=='mean' else b[b.seed==s][metric].iloc[0])
   hr.append(dict(seed=s,condition=condition,metric=metric,new_value=na,historical_value=hb,new_minus_historical_pp=na-hb))
hf=pd.DataFrame(hr);hf.to_csv(L/'v1_vs_historical.csv',index=False)
scenario_hist=new.merge(hist,on=['seed','scenario'],suffixes=('_new','_historical'))
for m in ['nrmse_pct','nmae_pct','nrmse_degradation_pct']:scenario_hist[m+'_new_minus_historical_pp']=scenario_hist[m+'_new']-scenario_hist[m+'_historical']
scenario_hist.to_csv(L/'v1_vs_historical_per_scenario.csv',index=False)
# Final immutable-input checks; no additional inferential tests.
for f in protocol['source_files']+protocol['data_files']+[protocol['helper']]:assert sha(f['path'])==f['sha256']
for task in protocol['sources']+protocol['vf_sources']:
 for f in task['files']+[task['config_file']]:assert sha(f['path'])==f['sha256']
gates=dict(status='PASS',operator_sha_comparison=protocol['operator_sha_comparison'],identity_checks=protocol['sources']+protocol['vf_sources'],normal_reproduction=normal_gates,vf_reuse_audit=js(L/'vf_reuse_audit.json'),data_sha_comparison=js(L/'vf_data_hash_comparison.json'),original_v1_fallback=js(L/'v1_fallback_manifest.json'),operational_repair=js(L/'operational_repair/repair.json'),recovered_sources=js(L/'operational_repair/retry_sources.json'),final_input_hashes_unchanged=True,expected_new_sources=20,completed_new_sources=20,expected_total_metric_rows=1200,actual_total_metric_rows=len(df),valid_targets=220401,invalid_targets=115,trained=False)
dump(L/'gates.json',gates)

def fmt(v,sd=None,digits=3):return f'{v:.{digits}f}' if sd is None else f'{v:.{digits}f} ± {sd:.{digits}f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def report_summary(x):
 return table(['模型/模式','h','条件（最差按五种子平均 NRMSE 选取）','NRMSE %','NMAE %','NRMSE 退化 %','NMAE 退化 %'],[[r.model_or_mode,r.horizon,r.scenario,fmt(r.nrmse_pct_mean,r.nrmse_pct_sd),fmt(r.nmae_pct_mean,r.nmae_pct_sd),fmt(r.nrmse_degradation_pct_mean,r.nrmse_degradation_pct_sd,2),fmt(r.nmae_degradation_pct_mean,r.nmae_degradation_pct_sd,2)] for r in x.itertuples()])
def report_tests(x):
 return table(['比较','h','终点','VF−比较组（百分点）','相对变化 %','利于 VF 种子数','t 检验 p','t-BH q','精确符号 p','符号-BH q','tier'],[[f'VF vs {r.comparator}',r.horizon,r.endpoint,fmt(r.difference_reference_minus_comparator_pp),fmt(r.relative_change_pct,digits=2),f'{r.seeds_favour_reference}/5',f'{r.paired_t_p:.6g}',f'{r.paired_t_bh_q:.6g}',f'{r.exact_two_sided_sign_p:.6g}',f'{r.sign_bh_q:.6g}',r.tier] for r in x.itertuples()])
text=[]
text.append('# B1 训练模式与 V1 严格配对故障复评\n')
text.append('本任务仅使用已有最佳权重开展推理；没有训练模型。B1 检验预注册 H-B1：joint（VF）的 39 场景平均 NRMSE 低于 staged。V1 复评用于检验 VF 相对去除 F2/F3/F6 的框架是否提高故障鲁棒性，并描述正常精度是否保持。所有旧结果和外部文件只读；本次新增内容均位于本目录。')
text.append('\n## 来源、身份与门槛\n')
text.append(f'20 个新推理源（staged h16、from-scratch h16、V1 h16/h96，各 5 种子）及 10 个只读 VF 源的身份全部通过。种子为 {SEEDS}；配置文件 experiment/data/model/train 字段与运行 metrics.json 逐字段一致，config_id、种子、horizon、stage3_mode、data_protocol 和权重 SHA-256 均已记录在 protocol.json/gates.json。from-scratch 的 Stage 1/2 轮数均为 0，Stage 3 模式为 joint；staged 的 Stage 3 模式为 staged。')
text.append(f'代码来自 `{protocol["source_files"][0]["original"].split("project")[0]}project`，逐字节复制到 `source_project`；规范化文件路径及 SHA 清单的快照总 SHA-256 为 `{protocol["source_snapshot_sha256"]}`。首选源用于通过其门槛的源；首选源失败的 V1 按指示回退到原始模型，仍使用完全相同的故障/数据/评估文件，由运行器接口适配忽略原始 V1 未使用的 F6 参数。最终选用的代码路径与快照 SHA 在各源 normal_equivalence.json，原始模型清单和回退快照在 v1_fallback_manifest.json；10 个复用 VF 源的原始科学源、运行脚本和清单 SHA 另记于 source_snapshots.json。完整每文件 SHA 在 protocol.json。')
text.append('故障算子 src/pieef/faults.py、数据处理 src/pieef/revision_data.py 和评估 scripts/evaluate_revision.py，与 VF h16/h96 清单及 L1 存档的 SHA-256 全部相同。scada_ecmwf_15min.csv、turbines.csv 与 L1 数据指纹相同；manufacturer_curve.csv 另行记录 SHA。数据预处理逐项重建校验通过。')
text.append(table(['源','正常门槛','最大逐单元差 / capacity','失败单元'],[[r['task_id'],r['status'],f'{r["max_abs_delta_pu"]:.9g}',r['failed_cells']] for r in normal_gates]))
text.append('10 个 VF 源未重新推理：验证原 completed.json、所有预测 SHA、场景定义与源权重身份，并将已保存 normal.npz 与原 predictions.npz 在全部 220,516 单元重新核对同一阈值，均通过。每个 h/seed 的正常及故障预测具有完全相同的目标时间、预报起点、风机、目标有效掩码、事件标签、真值及容量；h16 与 h96 分别检查其自身的 240 与 1,440 分钟起点间隔。所有组均排除同样的 115 个无效目标，保留 220,401 个有效目标。详见 vf_reuse_audit.json。')
text.append('\n## 协议与统计\n')
text.append('复用 L1 的 39 个结构化 SCADA 场景，另含 normal；fault_seed=2026。缺失的功率/风速按原算子前向填充（全历史窗被覆盖、无前值时回退为 0）并使用 PI 掩码，冻结与漂移不掩码；故障锚点按同一预报起点规则，NWP 与目标保持不变。未改变场景、批量大小或数值复现容差。正常复现要求每个预测单元满足 |Δ|/capacity ≤ 1e-5 + 1e-5·|saved/capacity|，通过后才运行故障。最多 3 个本地 CUDA 推理进程，完整命令、PID、UTC 起止时间、退出码及日志保存在 runs 下。已完成场景带 SHA 收据，完成源带 completed.json，重启时校验后复用。')
text.append('NRMSE/NMAE 按容量归一化并只取有效目标；R² 使用有效目标的 MW 真值与预测。退化率为 100×(故障误差/本源正常误差−1)。先在每个种子内平均 39 场景，再报告五种子均值 ± 样本标准差（ddof=1）；最差场景按五种子平均 NRMSE 最大值选取。正常退化为 0。')
text.append('仅执行指定的种子级配对 t 检验（双侧）及精确双侧符号检验；零差值从符号检验有效样本数剔除。B1 家族 4 个终点、V 家族 4 个终点，分别对 t 检验和符号检验的 p 值在各自四项家族内 BH 校正。B1 判级以 t-BH q 为准：5/5 利于 VF 且 q<0.05 为 support；否则 ≥4/5 为 trend；其余 none/unfavourable。V 家族额外列出的 tier 仅沿用同一描述规则便于阅读，不增设检验。五对同向差的精确双侧符号 p 最小为 0.0625。')
text.append('\n## B1 结果\n');text.append(report_summary(bs[bs.condition.isin(['normal','fault39_mean','worst_scenario'])]));text.append('\n'+report_tests(bt))
primary=bt[(bt.comparator=='staged')&(bt.endpoint=='nrmse_pct')].iloc[0]
text.append(f'预注册主终点：joint/VF 为 {primary.reference_mean:.3f}%，staged 为 {primary.comparator_mean:.3f}%，差值（joint−staged）为 {primary.difference_reference_minus_comparator_pp:.3f} 个百分点，相对变化 {primary.relative_change_pct:.2f}%；{primary.seeds_favour_reference}/5 种子利于 joint，BH q={primary.paired_t_bh_q:.6g}，判级 **{primary.tier}**。')
text.append(table(['比较组','指标','joint 场景胜数 /39','比较组胜数','平局'],[[r['comparator'],r['metric'],r['joint_wins'],r['comparator_wins'],r['ties']] for r in wins]))
text.append('场景胜数按五种子场景均值比较，仅为描述统计。两种误差的平均退化及最差场景全部列于上表，不对退化另加 B1 检验。')
text.append('\n## V1 与 VF 严格配对结果\n');text.append(report_summary(vs));text.append('\n'+report_tests(vt))
for h in [16,96]:
 a=vs[(vs.horizon==h)&(vs.model_or_mode=='VF')&(vs.condition=='normal')].iloc[0];b=vs[(vs.horizon==h)&(vs.model_or_mode=='V1')&(vs.condition=='normal')].iloc[0]
 text.append(f'h{h} 正常精度：VF−V1 的 NRMSE 为 {a.nrmse_pct_mean-b.nrmse_pct_mean:+.3f} 个百分点，NMAE 为 {a.nmae_pct_mean-b.nmae_pct_mean:+.3f} 个百分点；相对变化分别为 {100*(a.nrmse_pct_mean/b.nrmse_pct_mean-1):+.2f}% 和 {100*(a.nmae_pct_mean/b.nmae_pct_mean-1):+.2f}%。这是描述性比较，没有追加正常精度检验，也不能据此宣称已证明统计等效或非劣。')
text.append('\n## 新 V1 h16 与历史结果\n')
text.append('历史结果仅作描述，缺乏可核对故障目标索引，不用于 VF 配对显著性检验。不能把新旧差异单独归因于模型变化；本次 V1 使用相同已有权重，主要建立了与 VF 可核验的共同故障协议及索引。以下列出每种子正常值、39 场景均值及退化率的变化；完整 NMAE 对比见 v1_vs_historical.csv，逐场景差值见 v1_vs_historical_per_scenario.csv。')
sel=hf[(hf.metric=='nrmse_pct')|((hf.condition=='fault39_mean')&(hf.metric=='nrmse_degradation_pct'))]
text.append(table(['种子','条件','指标','新结果','历史结果','新−历史（百分点）'],[[r.seed,r.condition,r.metric,fmt(r.new_value,digits=2 if 'degradation' in r.metric else 3),fmt(r.historical_value,digits=2 if 'degradation' in r.metric else 3),fmt(r.new_minus_historical_pp)] for r in sel.itertuples()]))
text.append('\n## 全方向结果与限制\n')
for family,ts in [('B1',bt),('V',vt)]:
 for r in ts.itertuples():
  direction='更低（利于 VF）' if r.difference_reference_minus_comparator_pp<0 else '更高（不利于 VF）' if r.difference_reference_minus_comparator_pp>0 else '相同'
  text.append(f'- {family} h{r.horizon}、VF vs {r.comparator}、{r.endpoint}：VF {direction}，差值 {r.difference_reference_minus_comparator_pp:+.3f} 个百分点，{r.seeds_favour_reference}/5 同向，tier={r.tier}；符号检验 p={r.exact_two_sided_sign_p:.6g}。')
text.append('不以单个有利场景替代 39 场景主终点。每个场景、种子、R² 与目标数均完整保留在 fault_per_run.csv。五种子推断范围有限；符号检验与配对 t 检验对样本量和差值分布的敏感性不同，均报告而不择优选择。')
text.append('\n## 偏离与交付\n')
text.append('操作性偏离：首批部分源在写最终进度时重复传入 status，发生收尾日志异常；39 场景及门槛结果已经保存。保留原脚本、失败日志和完成候选文件，修复仅涉及状态记录。恢复时重新检查正常门槛并逐一核对已有故障预测的 SHA 收据，复用通过校验的场景；所有最终 completed.json 均对应自然退出码 0。详细恢复源清单见 operational_repair/retry_sources.json。科学协议无偏离：未训练、未重跑 VF、未放宽门槛、未改变故障算子、未追加统计检验、未安装软件包。代码副本和辅助审计/逐场景对比文件均仅写入指定输出目录。apply_patch 工具无法创建该中文路径下的文件，使用 PowerShell UTF-8 写入脚本；此工具层替代不改变科学内容。V 家族 tier 为报告性延伸，其判级规则已明确，推断仍限于指定四项检验。outputs_sha256.csv 列出本目录除该清单自身以外所有文件的 SHA-256；清单自身无法自引用。')
text.append(f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。')
(L/'报告.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')
print(bt.to_string(index=False));print(vt.to_string(index=False));print(hf[hf.seed=='mean'].to_string(index=False))
