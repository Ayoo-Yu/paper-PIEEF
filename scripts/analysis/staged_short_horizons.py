"""Only preregistered B5 endpoints, four paired tests per test family."""
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, binomtest
from b5_prepare_gate import OUT, EXPECTED, sha, write

RUN_FIELDS='mode horizon seed split nrmse_pct nmae_pct r2 best_epoch run_dir'.split()
TEST_FIELDS='horizon metric joint_mean staged_mean diff_pp rel_change_pct seeds_favouring_joint t_p t_bh_q sign_p sign_bh_q tier note'.split()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def csv_write(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def hashes():
    rows=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='outputs_sha256.csv' and '__pycache__' not in p.parts]
    csv_write(OUT/'outputs_sha256.csv',rows,['path','bytes','sha256'])
def bh(values):
    p=np.asarray(values); order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; result=np.empty_like(p);result[order]=np.minimum(1,adjusted);return result
def analyze():
    gates=read(OUT/'gates.json'); assert gates['identity_gate']['status']=='PASS'
    manifest=read(OUT/'config_manifest.json'); rows=[]; receipts=[]
    for item in manifest:
        receipt=read(OUT/'state/runs'/f"{item['config_id']}.json"); assert receipt['status']=='completed' and receipt['return_code']==0
        receipts.append(receipt)
        for mode,folder in [('joint',Path(item['joint_run'])),('staged',Path(receipt['run_dir']))]:
            m=read(folder/'metrics.json'); history=read(folder/'loss_history.json')
            for split,key in [('test','metrics'),('val','validation_metrics')]:
                a=m[key]
                rows.append(dict(mode=mode,horizon=item['horizon'],seed=item['seed'],split=split,nrmse_pct=a['nrmse']*100,nmae_pct=a['nmae']*100,r2=a['r2'] if split=='test' else '',best_epoch=history['best_epoch'],run_dir=str(folder)))
    assert len(rows)==40 and len(gates['index_checks'])==10 and all(x['status']=='PASS' for x in gates['index_checks'])
    csv_write(OUT/'b5_runs.csv',rows,RUN_FIELDS)
    tests=[]; descriptive=[]
    for h in (1,4):
        for metric in ('nmae_pct','nrmse_pct'):
            j=np.array([r[metric] for r in rows if r['horizon']==h and r['mode']=='joint' and r['split']=='test'])
            s=np.array([r[metric] for r in rows if r['horizon']==h and r['mode']=='staged' and r['split']=='test'])
            diff=j-s; wins=int((diff<0).sum()); nonzero=int((diff!=0).sum())
            tp=float(ttest_rel(j,s).pvalue) if np.any(diff!=0) else 1.
            sp=float(binomtest(wins,nonzero,p=.5,alternative='two-sided').pvalue) if nonzero else 1.
            tests.append(dict(horizon=h,metric=metric,joint_mean=float(j.mean()),staged_mean=float(s.mean()),diff_pp=float(diff.mean()),rel_change_pct=float(diff.mean()/s.mean()*100),seeds_favouring_joint=wins,t_p=tp,sign_p=sp))
        for mode in ('joint','staged'):
            for split in ('test','val'):
                part=[r for r in rows if r['horizon']==h and r['mode']==mode and r['split']==split]
                descriptive.append(dict(horizon=h,mode=mode,split=split,nmae_mean=float(np.mean([r['nmae_pct'] for r in part])),nmae_sd=float(np.std([r['nmae_pct'] for r in part],ddof=1)),nrmse_mean=float(np.mean([r['nrmse_pct'] for r in part])),nrmse_sd=float(np.std([r['nrmse_pct'] for r in part],ddof=1)),best_epochs=[r['best_epoch'] for r in part]))
    tq,sq=bh([t['t_p'] for t in tests]),bh([t['sign_p'] for t in tests])
    for t,q,z in zip(tests,tq,sq):
        t.update(t_bh_q=float(q),sign_bh_q=float(z))
        t['tier']='support' if t['seeds_favouring_joint']==5 and q<.05 else ('trend' if t['seeds_favouring_joint']>=4 else 'none/unfavourable')
        t['note']='按种子计数为趋势，但均值方向有利于 staged，不得视为有利结果。' if t['tier']=='trend' and t['diff_pp']>0 else ('均值有利于 joint' if t['diff_pp']<0 else '均值有利于 staged' if t['diff_pp']>0 else '均值相同')
    csv_write(OUT/'b5_tests.csv',tests,TEST_FIELDS)
    wall=sum(r['child_wall_clock_s'] for r in receipts)
    incidents=read(OUT/'operational_incidents.json') if (OUT/'operational_incidents.json').exists() else []
    summary={'preregistration_sha256':EXPECTED,'tests':tests,'descriptive':descriptive,'total_training_child_wall_clock_s':wall,'completed_runs':10,'scientific_protocol_deviations':[],'operational_incidents':incidents}
    write(OUT/'summary.json',summary)
    lines=['# B5：短提前量下联合与分阶段训练比较','', '目的：检验 h1（15 分钟）与 h4（1 小时）下 joint VF 的测试误差是否低于 staged，用于回复 R4-6。', '',f'预注册 SHA-256：`{EXPECTED}`。开跑前已记录，文件未修改。','', '## 门禁与执行', '',f"正常推理一致性门禁通过：{gates['identity_gate']['cells']:,} 个单元全部满足指定容差，最大 |Δ|/capacity = {gates['identity_gate']['max_abs_delta_over_capacity']:.10g}。10 份配置差异门禁通过。所有 test/val 的索引、目标、有效性掩码、机组及容量检查通过。",'', 'VF 的原验证预测未随现有运行保存，因此从各自 config 与最佳权重重建验证预测并保存在本目录 joint_validation 下；核对重建指标与已保存验证指标。正式分析仍采用原 metrics.json。未修改旧运行。', '', '仅使用本机 NVIDIA GeForce RTX 4070 SUPER；串行执行；每次训练前记录 CPU 和可见进程状态。未调用 Docker、SSH，未操作 CFD 目录或停止其他进程。线程数为 2，与学习率 runner 一致。', '',f'10 次训练子进程总墙钟：{wall/3600:.4f} 小时（含加载、训练、评估、保存；另行导出验证预测不计入）。', '', '## 逐种子结果', '', '| 模式 | h | seed | test NMAE % | test NRMSE % | test R² | val NMAE % | val NRMSE % | 第3阶段最佳epoch |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['split']!='test':continue
        v=next(v for v in rows if v['mode']==r['mode'] and v['horizon']==r['horizon'] and v['seed']==r['seed'] and v['split']=='val')
        lines.append(f"| {r['mode']} | {r['horizon']} | {r['seed']} | {r['nmae_pct']:.6f} | {r['nrmse_pct']:.6f} | {r['r2']:.6f} | {v['nmae_pct']:.6f} | {v['nrmse_pct']:.6f} | {r['best_epoch']} |")
    lines += ['', '## 主要终点与预注册判定', '', '差值为 joint − staged，单位为百分点；相对变化以 staged 均值为分母。所有 paired-t 均为双侧；精确双侧符号检验剔除零差值。两种检验分别在 2×2 共 4 项族内做 BH。', '', '| h | 指标 | joint均值 | staged均值 | 差值pp | 相对变化% | 利于joint | t p | t BH q | sign p | sign BH q | tier | 方向说明 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for t in tests:
        lines.append(f"| {t['horizon']} | {t['metric']} | {t['joint_mean']:.6f} | {t['staged_mean']:.6f} | {t['diff_pp']:.6f} | {t['rel_change_pct']:.4f} | {t['seeds_favouring_joint']}/5 | {t['t_p']:.6g} | {t['t_bh_q']:.6g} | {t['sign_p']:.6g} | {t['sign_bh_q']:.6g} | {t['tier']} | {t['note']} |")
    lines+=['','support：5/5 利于 joint 且 t-BH q<0.05；trend：至少4/5但不满足support；其余 none/unfavourable。上述表格完整报告有利与不利结果，没有筛选。', '', '## 描述性结果', '', '样本标准差采用 ddof=1；验证结果、标准差与最佳轮次不做额外检验。', '', '| h | 模式 | split | NMAE均值±SD % | NRMSE均值±SD % | 最佳epoch（seed 13/21/42/87/3407） |','|---|---|---|---:|---:|---|']
    for d in descriptive: lines.append(f"| {d['horizon']} | {d['mode']} | {d['split']} | {d['nmae_mean']:.6f} ± {d['nmae_sd']:.6f} | {d['nrmse_mean']:.6f} ± {d['nrmse_sd']:.6f} | {d['best_epochs']} |")
    lines+=['', '## 偏离、限制与可复现性', '', '科学协议无偏离：未更换终点、种子、提前量、训练设置、检验族或判定规则。验证预测补充导出是产物完整性处理，不参与重新选点。冻结源码无修改。队列支持跳过已完成且 SHA 校验通过的运行；原入口未提供优化器/RNG恢复，因此不宣称支持中断运行的逐epoch无损续训，失败时保留现场并停止。', '', 'VF 来自云端多种 GPU，staged 来自本机 GPU，同 seed 跨硬件不是严格随机轨迹配对；这是预注册已知局限。每组仅5个seed，精确符号检验最小双侧p=0.0625；不显著不能解释为等效。', '', '配置、源码/输入快照 SHA、原始日志、PID/UTC/GPU记录、四份权重与所有预测均位于本目录。outputs_sha256.csv 覆盖最终文件（不包含自身）。']
    if incidents:
        lines+=['','执行事件（完整保留原始错误与恢复证据）：']+[f"- {i['description']}" for i in incidents]
    (OUT/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

if __name__=='__main__':
    result=analyze();hashes();print(json.dumps(result,ensure_ascii=False,indent=2))
