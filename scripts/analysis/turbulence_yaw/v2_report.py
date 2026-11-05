"""E1-E3 report additions; all numbers are read from delivered tables."""
from common import *
from build_report import named_table,fmt,H,L
def get(name):return pd.read_csv(O/name)
def ms(r,c,n=4):return f'{r[c+"_mean"]:.{n}f} ± {r[c+"_sd"]:.{n}f}'

def corrigendum(k):
    text='## 勘误说明\n\n以下引用方案第 6A 节的 E1–E3；来源：`planning/第五轮补强方案_Kelmarsh湍流与偏航_20261024.md`（输入 SHA-256 见 `k0_identity_checks.json`）。\n\n'
    text+='> E1：“改为容量的 1e-5（2.05e-5 MW）”。\n>\n> E2：“修正后的预测 = 预测 + 验证集平均残差”。\n>\n> E3：“做两种修正：(a) 只按风速区间；(b) 按风速区间 × TI 层，其余规则相同。TI 带来的收益上限 = (b) 相对 (a) 的额外改善”；“(a) 单独报告，并注明它使用了预测时刻拿不到的目标时刻风速”。\n\n'
    text+='原残差定义仍为 y_true−y_pred；风速<4 m/s、验证样本不足50的格子不修正。其余阈值、分层、筛选、指标、种子及 bootstrap 全部沿用首轮最终脚本。减法结果只保留首轮已有6条，明确标为 `superseded_subtract`（已被 E2 取代），不纳入新版三种子汇总。\n\n### v1 → v2 对照\n\n'
    text+=named_table('v1/k0_identity_checks.json；k0_identity_checks.json；k2_oracle_bound.csv',['项目','v1','v2','原因'],[
        ['VF 身份容差（MW）','2.05e-6','2.05e-5','E1'],['VF 通过运行','6/9','9/9','原差值不变，只改容差'],['VF 三种子汇总','三个提前量均阻断','三个提前量全部可用','E1'],['K5 学得曲线','36/54','54/54','E1'],['K2 主结果','预测减验证均值残差','预测加验证均值残差；风速/风速×TI 双对照','E2/E3'],['K4 三个输出文件','首轮结果','逐字节完全一致','独立于 VF 门禁']])
    rows=[]
    for r in k['v1_v2_reproduction']['tables']:
        rows.append([r['file'],r.get('n_common_rows','—'),r.get('n_numeric','—'),r.get('n_differences',0),f'{r.get("max_abs_difference",0):.12g}',r.get('status','EXACT' if r.get('n_differences',0)==0 else 'CHANGED')])
    text+=named_table('k0_identity_checks.json → v1_v2_reproduction.tables；所列同名 v1/v2 文件',['对照文件/字段','共同结果行','已核对数值','差异数','最大绝对差','结论'],rows)
    details=k['v1_v2_reproduction']['differences']
    if details:
        text+='所有数值差异均保存在 `k0_identity_checks.json` 的 `v1_v2_reproduction.differences`（含行键、列名、v1、v2、绝对差）；下表按文件/列汇总，未把舍入差异隐去。\n'
        frame=pd.DataFrame(details);rows=[]
        for key,g in frame.groupby(['file','column'],dropna=False,sort=False):
            example=g.iloc[0];rows.append([*key,len(g),f'{g.abs_difference.max():.12g}',f'{example.v1:.17g} → {example.v2:.17g}'])
        text+=named_table('k0_identity_checks.json → v1_v2_reproduction.differences',['文件','列','差异数','最大绝对差','v1 → v2 示例'],rows)
    text+='对照覆盖首轮已有单种子 VF、全部基线、经验曲线、注意力、Wind Farm A历史行、K4 与 K5。K5 首轮最终文件曾经过 CSV 读取再写入及 crossing 刷新，重算保留计算精度，出现上述末位序列化差异；不把新结果强行写回旧值。首轮已阻断的空值不视为已有结果，新增三种子统计单独报告。首轮所有文件在启动时登记 SHA-256，在最终 manifest 再核查不可变性。文中 `v1/` 指本目录上一级 `K_turbulence_yaw/`，并非另建目录。\n'
    d=get('k2_oracle_bound.csv');rows=[]
    for _,r in d[d.correction=='superseded_subtract'].iterrows():
        same=d[(d.row_type=='seed')&(d.status=='PASS')&(d.horizon==r.horizon)&(d.seed==r.seed)].set_index('correction')
        rows.append([H[int(r.horizon)],r.seed,f'{r.nrmse_after_pct:.6f} / {r.nmae_after_pct:.6f}',f'{same.loc["ws_only","nrmse_after_pct"]:.6f} / {same.loc["ws_only","nmae_after_pct"]:.6f}',f'{same.loc["ws_x_ti","nrmse_after_pct"]:.6f} / {same.loc["ws_x_ti","nmae_after_pct"]:.6f}'])
    text+=named_table('v1/k2_oracle_bound.csv；k2_oracle_bound.csv',['提前量','种子','v1减法，已取代：NRMSE/NMAE（%）','v2(a)：NRMSE/NMAE（%）','v2(b)：NRMSE/NMAE（%）'],rows)
    return text

def k1_conclusion():
    d=get('k1_ti_strata_errors.csv');s=d[(d.row_type=='comparison_summary')&(d.stratum=='high')&(d.baseline!='Persistence')]
    good=(s.VF_improvement_pct>0)&(s.VF_favorable_available_seeds>=2)
    if good.all():text='按第4节条件，在高湍流条件下仍保持优势。'
    else:
        failed=[f'{H[int(r.horizon)]} {r.metric.upper()}（均值相对改善 {r.VF_improvement_pct:.4f}%，有利种子 {int(r.VF_favorable_available_seeds)}/3）' for _,r in s[~good].iterrows()]
        text='高 TI 层未满足第4节的完整优势条件；未达条件的是：'+'；'.join(failed)+'。'
    return text+'其余有利和不利结果均列于下表，不作检验性推断。（来源：`k1_ti_strata_errors.csv`。）\n'

def k2_section():
    text='修正采用 corrected=y_pred+mean_validation(y_true−y_pred)，不裁剪。两个版本使用相同正常运行测试掩码、相同风速箱 [4,5), …, [14,15), [15,∞)，验证格子 n≥50 才修正；风速<4 m/s不修正。(a)合并 TI 估计风速箱均值，(b)用低/中/高互斥 TI 层分别估计，描述性的很高层不重复修正。**(a) 已使用预测时刻不可得的目标时刻风速；(b) 还使用目标时刻 TI。两者均不可部署，(b) 的全部改善不能归因于 TI。**\n'
    text+='误差单位为容量百分比，绝对变化以百分点表示：Δ=修正后−修正前；相对变化=100×(后/前−1)，负值为改善。TI增量=(b)−(a)，相对增量分母为(a)的修正后误差；相对改善是相对变化的相反数。CSV 的 `nrmse_before/after`、`nmae_before/after` 与 `_pct` 列同义。\n'
    residual=get('k2_ti_residuals.csv');rows=[]
    for _,r in residual[(residual.row_type=='seed_summary')&(residual.status=='PASS')].iterrows():
        if pd.notna(r.mean_residual_mw_mean):rows.append([H[int(r.horizon)],L[r.stratum],f'[{int(r.wind_lower_ms)},{int(r.wind_lower_ms)+1})' if r.wind_lower_ms<15 else '[15,∞)',ms(r,'mean_residual_mw')])
    text+=named_table('k2_ti_residuals.csv',['提前量','TI 层','风速箱（m/s）','测试残差 MW：三种子均值±SD'],rows)
    d=get('k2_oracle_bound.csv');p=d[d.row_type=='seed'];rows=[]
    for _,r in p.iterrows():
        rows.append([H[int(r.horizon)],r.seed,{'ws_only':'(a) ws_only','ws_x_ti':'(b) ws_x_ti','superseded_subtract':'已取代：首轮减法，仅参考'}[r.correction],int(r.n_test),int(r.n_corrected),f'{r.nrmse_before_pct:.4f} → {r.nrmse_after_pct:.4f}',r.nrmse_change_pp,r.nrmse_relative_change_pct,f'{r.nmae_before_pct:.4f} → {r.nmae_after_pct:.4f}',r.nmae_change_pp,r.nmae_relative_change_pct])
    text+=named_table('k2_oracle_bound.csv',['提前量','种子','修正','测试 n','修正 n','NRMSE 前→后（%）','ΔNRMSE（百分点）','相对变化（%）','NMAE 前→后（%）','ΔNMAE（百分点）','相对变化（%）'],rows)
    summary=d[d.row_type=='seed_summary'];rows=[]
    for _,r in summary.iterrows():
        rows.append([H[int(r.horizon)],r.correction,ms(r,'nrmse_before_pct'),ms(r,'nrmse_after_pct'),ms(r,'nrmse_change_pp'),ms(r,'nrmse_relative_change_pct'),ms(r,'nmae_before_pct'),ms(r,'nmae_after_pct'),ms(r,'nmae_change_pp'),ms(r,'nmae_relative_change_pct')])
    text+=named_table('k2_oracle_bound.csv',['提前量','修正','NRMSE 前（%）','NRMSE 后（%）','ΔNRMSE（百分点）','NRMSE 相对变化（%）','NMAE 前（%）','NMAE 后（%）','ΔNMAE（百分点）','NMAE 相对变化（%）'],rows)
    text+='上表每格为三个种子的均值±样本SD，先算逐种子变化再汇总。以下为 TI 可归因的额外变化 (b)−(a)，既给逐种子也给三种子统计。\n'
    rows=[]
    for _,r in d[d.correction=='ws_x_ti'].iterrows():
        cols=[m+s for m in ['nrmse','nmae'] for s in ['_ti_increment_change_pp','_ti_increment_relative_change_pct']]
        values=[ms(r,c) for c in cols] if r.row_type=='seed_summary' else [r[c] for c in cols]
        rows.append([H[int(r.horizon)],'均值±SD' if r.row_type=='seed_summary' else r.seed,*values])
    text+=named_table('k2_oracle_bound.csv',['提前量','种子/汇总','NRMSE (b)−(a)（百分点）','NRMSE 相对增量（%）','NMAE (b)−(a)（百分点）','NMAE 相对增量（%）'],rows)
    for _,r in summary[summary.correction=='ws_x_ti'].iterrows():
        improvement=r.nmae_ti_increment_relative_improvement_pct_mean
        if improvement>=1:
            text+=f'{H[int(r.horizon)]}：TI 的 NMAE 额外相对改善为 {ms(r,"nmae_ti_increment_relative_improvement_pct")}%（三种子均值±SD），≥1%；按第6A节列为局限和未来工作，需要预测时可用的 TI 预报。\n'
        elif improvement>=0:
            text+=f'{H[int(r.horizon)]}：即使知道目标时刻的湍流强度，额外改善也不超过 {improvement:.4f}%（此处指本方案三种子均值，不是单种子最大值或普适上界）；逐种子范围及 SD 见上表。\n'
        else:
            text+=f'{H[int(r.horizon)]}：即使知道目标时刻的湍流强度，未获得额外 NMAE 改善，三种子平均反而增加 {-improvement:.4f}%；不将负收益改写为正向上限。\n'
    return text+'这些是固定分箱与验证均值修正规则下的 oracle 对照结果，不是所有可能修正方法的数学最优界。（来源：`k2_oracle_bound.csv`。）\n'

def k3_summaries():
    a=get('k3_attention_by_ti.csv');p=a[a.row_type=='seed_summary'];rows=[]
    for key,g in p.groupby(['horizon','stratum'],sort=False):
        ix=g.set_index('metric');rows.append([H[int(key[0])],L[key[1]],*[ms(ix.loc[m],'attention_mean',6) for m in ['normalized_entropy','maximum_weight','dsi']]])
    return '完整三种子均值±样本SD如下；查询数及有限/未定义数见上面的逐种子表。图显示全部九个运行的逐种子结果。\n'+named_table('k3_attention_by_ti.csv',['提前量','TI 层','归一化熵','最大权重','DSI'],rows)

def k5_summaries():
    a=get('k5_curve_shift.csv');rows=[]
    cols=['learned_v50_ms','learned_v90_ms','learned_minus_all_normal_v50_ms','learned_minus_aligned_lt2_v50_ms','learned_minus_all_normal_v90_ms','learned_minus_aligned_lt2_v90_ms']
    for (h,t),g in a.groupby(['horizon','turbine'],sort=False):
        assert len(g)==3
        rows.append([H[int(h)],int(t),g.training_abs_phi_median_deg.iloc[0],g.training_fraction_abs_phi_gt10.iloc[0],*[f'{g[c].mean():.4f} ± {g[c].std(ddof=1):.4f}' for c in cols]])
    return named_table('k5_curve_shift.csv（按提前量×机组，三个种子汇总；ddof=1）',['提前量','机组','训练 |φ| 中位数（°）','|φ|>10° 比例','学得 v50','学得 v90','Δv50 正常','Δv50 对中','Δv90 正常','Δv90 对中'],rows)
