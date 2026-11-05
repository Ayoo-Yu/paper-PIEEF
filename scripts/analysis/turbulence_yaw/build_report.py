"""Generate Chinese report directly from the delivered CSV/JSON evidence."""
from common import *
H={1:'15 分钟',4:'1 小时',16:'4 小时'}
L={'low':'低','medium':'中','high':'高','very_high':'很高（高层子集）','undefined':'无定义','lt5':'<5°','5to10':'[5°,10°)','10to20':'[10°,20°)','ge20':'≥20°'}
def fmt(v,n=4):
    if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return '—'
    if isinstance(v,(float,np.floating)):return f'{v:.{n}f}'
    return str(v).replace('|','\\|').replace('\n',' ')
def tab(headers,rows):
    return '\n'+'| '+' | '.join(fmt(x) for x in headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(x) for x in r)+' |\n' for r in rows)+'\n'
def get(name):return pd.read_csv(O/name)
def named_table(name,headers,rows):return f'数据来源：`{name}`。\n'+tab(headers,rows)
def errors_section(name):
    d=get(name);s=d[d.row_type=='seed_summary'];out=[]
    for _,r in s.iterrows():
        complete=r.status=='PASS'
        out.append([H[int(r.horizon)],r.model,L[r.stratum],r.n_valid_per_seed if complete else '门禁阻断',f'{r.nrmse_pct_mean:.4f} ± {r.nrmse_pct_sd:.4f}' if complete else '三种子汇总不可用',f'{r.nmae_pct_mean:.4f} ± {r.nmae_pct_sd:.4f}' if complete else '三种子汇总不可用'])
    text=named_table(name,['提前量','模型','分层','各种子有效单元数','NRMSE（%）均值 ± SD','NMAE（%）均值 ± SD'],out)
    a=d[(d.row_type=='seed')&(d.model=='VF')&(d.status=='PASS')]
    text+='VF 三个原定种子的逐种子结果如下，汇总采用 ddof=1。\n'
    text+=named_table(name,['提前量','种子','分层','有效单元数','NRMSE（%）','NMAE（%）'],[[H[int(r.horizon)],r.seed,L[r.stratum],int(r.n_valid),r.nrmse_pct,r.nmae_pct] for _,r in a.iterrows()])
    p=d[d.row_type=='paired_seed_comparison'];rows=[]
    for key,g in p.groupby(['horizon','seed','stratum','baseline'],sort=False):
        v=g.set_index('metric').VF_improvement_pct
        rows.append([H[int(key[0])],key[1],L[key[2]],key[3],v.get('nrmse'),v.get('nmae')])
    text+='相对改善 = 100 × (1 − VF/基线)，正值有利于 VF；仅同种子配对。\n'
    text+=named_table(name,['提前量','种子','分层','比较基线','NRMSE 相对改善（%）','NMAE 相对改善（%）'],rows)
    p=d[d.row_type=='comparison_summary'];rows=[]
    for key,g in p.groupby(['horizon','stratum','baseline'],sort=False):
        v=g.set_index('metric');n=int(v.iloc[0].n_paired_seeds)
        rows.append([H[int(key[0])],L[key[1]],key[2],f'{int(v.loc["nrmse","VF_favorable_available_seeds"])}/{n}',f'{int(v.loc["nmae","VF_favorable_available_seeds"])}/{n}',f'{v.loc["nrmse","VF_improvement_pct"]:.4f} / {v.loc["nmae","VF_improvement_pct"]:.4f}'])
    text+=named_table(name,['提前量','分层','比较基线','NRMSE 有利/可用种子','NMAE 有利/可用种子','三种子均值相对改善：NRMSE/NMAE（%）'],rows)
    return text

def main():
    checkpoint();start=begin('python -B scripts/build_report.py');k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf8'));inc=json.loads((O/'operational_incidents.json').read_text(encoding='utf8'))
    parts=['# Kelmarsh 湍流强度与偏航不对中补充分析（K0–K6）\n',
    '本次仅分析已有预测、注意力诊断、权重和 SCADA；未训练、微调或改动任何原有模型、阈值、预测及主表。所有计算使用 CPU。报告中的空缺是门禁阻断或未定义，不作补值。\n',
    'E1 勘误后九个 VF 运行全部通过身份门禁，完整报告三个原定种子的均值±样本标准差及全部有利、不利结果。下面所有数值来自所注明的 CSV 或 JSON。\n',
    '## K0：身份、指标、原始对齐与数据口径\n']
    import v2_report
    parts.insert(1,v2_report.corrigendum(k))
    checks=k['metric_checks'];raw=k['raw_alignment'];ident=k['vf_identity_checks']
    parts.append(named_table('k0_identity_checks.json',['门禁','通过数/核对数','最大绝对差','冻结容差'],[
        ['主表指标（包括重复 VF 行的逐值核对）',f'{sum(x["passed"] for x in checks)}/{len(checks)}',f'{max(x["abs_difference_pp"] for x in checks):.12g} 个百分点','0.001 个百分点'],
        ['VF 与 M_controls',f'{sum(x["passed"] for x in ident)}/{len(ident)}',f'{max(x["max_abs_prediction_difference_mw"] for x in ident):.12g} MW','2.05e-5 MW（E1）'],
        ['原始风速及标准差',f'{sum(x["passed"] for x in raw["comparisons"])}/{len(raw["comparisons"])}',f'{max(x["max_abs_difference"] for x in raw["comparisons"]):.12g}','1e-6']]))
    parts.append(named_table('k0_identity_checks.json',['提前量','种子','最大预测差（MW）','时间戳/目标/掩码/机组一致','后续 VF 分析'],[[H[x['horizon']],x['seed'],f'{x["max_abs_prediction_difference_mw"]:.12g}','是' if all(x['fields_equal'].values()) else '否','允许' if x['passed'] else '停止'] for x in ident]))
    b=k['split_grid'];parts.append(named_table('k0_identity_checks.json',['区间（UTC，左闭右开）','开始','结束'],[['训练',b['train_start'],b['train_end_exclusive']],['验证',b['val_start'],b['val_end_exclusive']],['测试',b['test_start'],b['test_end_exclusive']]]))
    parts.append(named_table('k0_identity_checks.json',['提前量','种子','验证目标数','验证首/末目标（UTC）','测试目标数','测试首/末目标（UTC）'],[[H[x['horizon']],x['seed'],x['n_val_targets'],x['val_first_saved_target_utc']+' / '+x['val_last_saved_target_utc'],x['n_test_targets'],x['test_first_saved_target_utc']+' / '+x['test_last_saved_target_utc']] for x in k['split_records']]))
    parts.append('训练边界直接读取 preprocessing 保存的结束索引和时间戳。验证结束边界没有单独序列化，因此用原生产代码的配置索引表达式恢复，并与全部运行保存的验证/测试目标首尾核对；该最小偏差记入 `operational_incidents.json`。不得用方案中的近似日期替代。\n')
    q=k['ti_threshold_check'];parts.append(named_table('k0_identity_checks.json',['阈值','冻结值','重算值','重算 − 冻结'],[[label,x,y,y-x] for label,x,y in zip(['下三分位','上三分位','90 分位'],q['frozen'],q['recomputed'])]))
    parts.append('TI = 风速标准差/机舱风速，仅在风速 ≥4 m/s 时定义。低层 <0.124；中层 [0.124,0.166)；高层 ≥0.166；很高层 ≥0.216，仅作描述且包含在高层之内；风速 <4 m/s 单列无定义。重算值均在允许差内，所有分析保留冻结阈值。机舱风速计位于转子之后，因此这里的 TI 是机舱湍流强度近似量。（定义与核对见 `k0_identity_checks.json`。）\n')
    parts.append('压缩包直接流式读取，未解压。重复时间戳沿用 prepare_kelmarsh 的字段最小/最大值核对规则，有冲突的字段设为缺失。15 分钟采样完全沿用生产代码：合并时间网格、逐字段向前填充后取样，避免使用未来观测；缺失字段可能来自更早记录，SCADA_observation_time 不能表达各字段实际观测年龄。φ 也按此规则取原始 vane 均值并环绕到 [−180°,180°)。原始 K4/K5 机理曲线使用未填充的 10 分钟记录。（来源：`k0_identity_checks.json` 的 raw_alignment。）\n')
    parts.append(named_table('k0_identity_checks.json',['机组','交叉核对单元数','中位 |φ − φ_alt|（°）','Pearson r'],[[x['turbine'],x['n'],x['median_abs_difference_deg'],x['pearson_r']] for x in k['yaw_crosscheck']]))
    parts.append('φ_alt = wrap(风向 − 机舱位置)。上表按两个已环绕角度的普通绝对差和 Pearson 相关系数报告，符合冻结定义；没有以 φ_alt 替换 vane。\n')
    sp=k['power_setpoint_check'];parts.append(named_table('k0_identity_checks.json',['setpoint 核对','值'],[['训练原始记录中有限值个数',sp['n_finite']],['低于 2000 kW 比例',sp['fraction_below_2000']],['与实测功率的描述性相关',sp['pearson_with_measured_power']]]))
    parts.append('导出文件仅给出 Turbine Power setpoint 列名，没有控制器语义说明。上述分布和与功率的对应关系不足以确认它是限功率指令，按冻结方案不用于筛选。没有据此追加阈值。\n')
    parts.append('误差计算均先按每台容量 2.05 MW 归一化，再在有效机组×目标时刻单元上取均值：NRMSE=100√mean(e²)，NMAE=100 mean(|e|)。三种子标准差为样本标准差（ddof=1）。K1/K6 使用所有 target_valid，不施加正常运行筛选。（来源：`k0_identity_checks.json`。）\n')
    parts.append('## K1：按目标时刻 TI 分层的预测误差\n')
    import v2_report
    parts.append(v2_report.k1_conclusion())
    parts.append(errors_section('k1_ti_strata_errors.csv'))
    parts.append('图：`figures/k1_ti_errors.png`、同名 SVG。所有模型均为三种子均值±SD；完整 n 见上表。\n')
    parts.append('## K2：经验曲线、残差及方案指定的上限实验\n')
    parts.append('训练经验曲线使用 15 分钟文件中实际可用功率目标，要求 target_valid、原生产 events=normal，且所对应原始记录不与停机/降额区间重叠。K2 的原事件类别保留生产代码已有的统计异常定义；K4/K5 按用户指定另用不含 3σ 的原始记录筛选。K2 验证残差估计和测试误差均限定同样的正常运行单元，前后误差使用同一测试掩码，不与 K1 全目标误差直接混算。\n')
    a=get('k2_ti_power_curves.csv');p=a[a.turbine.astype(str)=='pooled'];rows=[]
    for b,g in p.groupby('wind_lower_ms',sort=True):
        ix=g.set_index('stratum');rows.append([f'[{b},{b+1})']+[f'{fmt(ix.loc[l,"mean_power_mw"])} / {int(ix.loc[l,"n_records"])}' for l in TI_LAYERS])
    parts.append(named_table('k2_ti_power_curves.csv',['机舱风速（m/s）',*[L[l]+'：MW / n' for l in TI_LAYERS]],rows))
    parts.append('每台机组的同口径全部分箱结果均在该 CSV；`figures/k2_ti_power_curves.png`（及 SVG）同时显示合并和各机组曲线。没有平滑、插补或扩大风速分箱。\n')
    a=get('k2_ti_residuals.csv');p=a[(a.row_type=='seed')&(a.split=='test')&(a.status=='PASS')];runkeys=list(dict.fromkeys(zip(p.horizon,p.seed)));rows=[]
    for (b,l),g in p.groupby(['wind_lower_ms','stratum'],sort=False):
        if not (g.n_valid>0).any():continue
        lookup={(r.horizon,r.seed):r for _,r in g.iterrows()}
        rows.append([f'[{int(b)},{int(b)+1})' if b<15 else '[15,∞)',L[l]]+[f'{fmt(lookup[x].mean_residual_mw)} / {int(lookup[x].n_valid)}' for x in runkeys])
    parts.append(named_table('k2_ti_residuals.csv',['风速（m/s）','TI 层',*[H[int(h)]+' s'+str(s)+'：MW / n' for h,s in runkeys]],rows))
    parts.append('上表是测试期 mean(实测−预测)；无样本格子不在正文重复列出，但 CSV 保留。验证期格子数、均值及是否达到修正条件也在同一 CSV。三种子残差均值±SD 完整列于下表及 CSV。\n')
    parts.append(v2_report.k2_section())
    parts.append('## K3：按目标时刻查询机组 TI 分层的注意力\n')
    parts.append('熵为 −Σa log(a)/log(N)，最大权重为实际注意力行最大值，均包含自身机组；DSI=A_up/(A_up+A_down)，方向投影 >0.5 / <−0.5 分别定义上/下游邻居，分母为零时不定义。按保存的测试样本中全部查询机组分层，不再以 target_valid 限制注意力；每项按有限诊断值平均并单列缺失数。没有方向性或尾流因果解释。（来源：`k0_identity_checks.json` 中记录的诊断源码及 `k3_attention_by_ti.csv`。）\n')
    a=get('k3_attention_by_ti.csv');p=a[(a.row_type=='seed')&(a.status=='PASS')];rows=[]
    for key,g in p.groupby(['horizon','seed','stratum'],sort=False):
        ix=g.set_index('metric');rows.append([H[int(key[0])],key[1],L[key[2]],int(ix.iloc[0].n_queries),ix.loc['normalized_entropy','attention_mean'],ix.loc['maximum_weight','attention_mean'],ix.loc['dsi','attention_mean'],int(ix.loc['dsi','n_finite']),int(ix.loc['dsi','n_undefined'])])
    parts.append(named_table('k3_attention_by_ti.csv',['提前量','种子','TI 层','查询数','归一化熵','最大权重','DSI','有效 DSI 数','未定义 DSI 数'],rows))
    parts.append(v2_report.k3_summaries())
    p=a[a.row_type=='historical_reference'];rows=[]
    for key,g in p.groupby(['seed','stratum'],sort=False):
        ix=g.set_index('metric');rows.append([key[0],key[1],int(ix.iloc[0].n_queries),ix.loc['normalized_entropy','attention_mean'],ix.loc['maximum_weight','attention_mean'],'原分工况文件未保存'])
    parts.append('与Wind Farm A既有分工况结果并列如下，仅转录本任务相同种子编号的旧结果，未新增Wind Farm A推理或分层。其工况按每个时间点的多数机组事件标签分组，n 是时间点数，随后对全部查询机组平均；这与 Kelmarsh 逐查询机组 TI 分组的权重口径不同。旧文件没有分工况 DSI，不能补造或用全局 DSI 代替。\n')
    parts.append(named_table('k3_attention_by_ti.csv',['Wind Farm A既有种子','既有工况','时间点 n','归一化熵','最大权重','DSI'],rows))
    parts.append('## K4：偏航分布、功率比和参考风速敏感性\n')
    a=get('k4_yaw_distribution.csv');parts.append(named_table('k4_yaw_distribution.csv',['时期','机组','网格记录 n','φ 中位数（°）','φ IQR（°）','|φ| 中位数（°）','|φ| P90（°）','|φ|>10° 比例'],[[r.period,int(r.turbine),int(r.n_records),r.phi_median_deg,r.phi_iqr_deg,r.abs_phi_median_deg,r.abs_phi_p90_deg,r.fraction_abs_phi_gt10] for _,r in a.iterrows()]))
    parts.append('分布使用训练/测试各自全部有限的 15 分钟 vane 观测，不要求实际功率目标有效；功率比分析则使用训练原始 10 分钟正常记录。筛除缺失/非有限值或功率 ≤−20 kW、风速 >5 m/s 且功率 <10 kW 的候选、连续相同风速的第五条及后续记录、与 Stop/降额状态区间重叠的记录。**不使用 3σ 统计异常筛选**，以免删去本次研究的低功率偏航记录。每条 10 分钟记录按 [t,t+10 min) 与状态 [start,end) 判断重叠。各原因可能重叠，不应将列数直接相加。（来源：`k0_identity_checks.json`。）\n')
    parts.append(named_table('k0_identity_checks.json',['机组','训练原始记录','无效','低功率候选','冻结风速','状态重叠','保留正常'],[[x['turbine'],x['n_train'],x['invalid'],x['low_power_candidates'],x['frozen'],x['status_overlap'],x['retained_normal']] for x in k['raw_normal_filter']['per_turbine']]))
    cats={}
    for x in k['status_filter']['turbines']:
        for c in x['categories']:
            key=(c['status'],c['code'],c['message'],str(c['iec_category']));cats[key]=cats.get(key,0)+c['n_events']
    parts.append(named_table('k0_identity_checks.json',['状态类别','代码','原始描述','IEC 类别','全年事件数'],[[*key,n] for key,n in sorted(cats.items())]))
    parts.append('上述是实际被采用的状态代码全集。降额通过 Status/Message/合同类别/IEC 类别中的 derat、curtail、partial performance、reduced performance、power limit、power reduction 明文标签检索；本档案未找到独立降额标签事件，不能把没有标签视为已排除全部隐含限功率。未凭 setpoint 或含义不明确的 Warning 自行增加筛选。（来源：`k0_identity_checks.json` 的 status_filter。）\n')
    fit=json.loads((O/'k4_cos_fit.json').read_text(encoding='utf8'))
    parts.append('功率比以每台机组 |φ|<2°、0.5 m/s 箱且至少 30 条记录的均值功率为 P_ref；风速范围 [4,11) m/s。替代参考速度为同时间其他五台机组机舱风速的中位数，任一缺失则该替代速度缺失。两条分析分别按自身参考速度重新选风速箱和 P_ref，绝不混用。两度分箱样本数达到 200 才参与等权最小二乘，以箱中心作为 φ；不约束 p 的正负。按 UTC 自然日、全机组共同重采样，1000 次、随机种子 2026；每次重新估计参考功率，再计算比例和 p，采用百分位 95% 区间。不是逐记录独立抽样。（来源：`k4_cos_fit.json`。）\n')
    parts.append(named_table('k4_cos_fit.json',['参考风速','p','95% 下限','95% 上限','参与拟合的两度箱数','有效 bootstrap 次数'],[[x['reference_speed'],x['p'],x['ci95_lower'],x['ci95_upper'],len(x['eligible_two_degree_bins']),x['n_valid_bootstrap']] for x in fit['fits']]))
    a=get('k4_yaw_power_ratio.csv');parts.append(named_table('k4_yaw_power_ratio.csv',['参考风速','分箱类型','|φ| 分箱','n','平均 R','95% 下限','95% 上限','用于 p 拟合'],[[r.reference_speed,r.bin_type,r.stratum,int(r.n_records),r.mean_power_ratio,r.ci95_lower,r.ci95_upper,r.fit_eligible] for _,r in a.iterrows()]))
    parts.append('自身机舱风速的功率比并不呈简单单调下降；替代参考风速给出更明显的随 |φ| 增大而下降的形态，两种 p 差别较大。该结果对参考风速选择敏感，不将其中一条曲线单独写作已确认的偏航损失定律，也不据置信区间作假设检验。文献 p 值留给作者引用，本报告不填入外部数值。图：`figures/k4_yaw_power_ratio.png` 和同名 SVG。（来源：`k4_yaw_power_ratio.csv`、`k4_cos_fit.json`。）\n')
    parts.append('## K5：学得曲线及 v50/v90 偏移\n')
    x=k['k5_extraction'];parts.append(f'原计划 {x["expected_curves"]} 条曲线；E1 后完整提取 {x["executed_runs"]} 个运行的 {x["executed_curves"]} 条，稳定导数单调性通过 {x["monotonic_pass"]}/{x["executed_curves"]}。九个运行全部使用原有权重。（来源：`k0_identity_checks.json`。）\n')
    parts.append('从 M_controls 的原始权重副本，严格加载其生产代码的 LearnableBSplineCurve 子模块；只求已有函数值及导数，不更新参数。与 P1 一样，使用基准密度下、不含切入/切出门控的 B 样条基础曲线，在物理 m/s 轴上取值。生产代码先将风速还原为 m/s 再校准，样条输入已经是 m/s，不能再次乘 SCADA 标准差。稳定导数用相邻非负控制点增量公式核验，浮点前向微小下降另行保留。（来源：`k0_identity_checks.json`、`k5_curve_shift.csv`、`k5_learned_curves.csv`。）\n')
    parts.append('**下面的 v50/v90 比较存在坐标差异：学得曲线以校准风速为自变量，经验曲线以机舱风速计读数为自变量。偏移不能直接解释为偏航因果效应。** 经验曲线使用训练原始正常记录，一条取全部正常记录，一条取 |φ|<2°，保留全部有观测的 0.5 m/s 分箱，不追加计数门槛。以第一段跨越容量 50%/90% 的线性插值取 v50/v90，不平滑、不强行单调、不外推；不跨越则留空。（来源同上。）\n')
    a=get('k5_curve_shift.csv');p=a[a.status=='PASS'];rows=[]
    for _,r in p.iterrows():rows.append([H[int(r.horizon)],int(r.seed),int(r.turbine),r.training_abs_phi_median_deg,r.training_fraction_abs_phi_gt10,r.learned_v50_ms,r.all_normal_v50_ms,r.aligned_lt2_v50_ms,r.learned_minus_all_normal_v50_ms,r.learned_minus_aligned_lt2_v50_ms,r.learned_v90_ms,r.all_normal_v90_ms,r.aligned_lt2_v90_ms,r.learned_minus_all_normal_v90_ms,r.learned_minus_aligned_lt2_v90_ms])
    parts.append(named_table('k5_curve_shift.csv',['提前量','种子','机组','训练 |φ| 中位数（°）','训练 |φ|>10° 比例','学得 v50','正常 v50','对中 v50','Δv50 正常','Δv50 对中','学得 v90','正常 v90','对中 v90','Δv90 正常','Δv90 对中'],rows))
    parts.append(v2_report.k5_summaries())
    parts.append('上述风速及差值单位均为 m/s，Δ=学得−经验。所有可用曲线的偏移方向为正，但机组间偏移次序没有随偏航程度一致排列，九个运行也未形成可归因于偏航的一致排序；不写“模型学到了偏航效应”。仅作描述，不计算机组间相关性检验。（来源：`k5_curve_shift.csv`。）\n')
    parts.append(named_table('k5_curve_shift.csv',['提前量','种子','机组','最小控制点增量','稳定最小导数','FP32 负导数比例（%）','自动微分与稳定导数最大差','前向最大下降（pu）','核验'],[[H[int(r.horizon)],int(r.seed),int(r.turbine),f'{r.min_cp_increment:.8g}',f'{r.stable_min_derivative_pu_per_ms:.8g}',r.float32_negative_derivative_pct,f'{r.max_autograd_vs_stable_pu_per_ms:.8g}',f'{r.max_forward_decrease_pu:.8g}',r.passed] for _,r in p.iterrows()]))
    parts.append('## K6：按目标时刻 |φ| 分层的预测误差\n')
    parts.append('沿用 K1 的全部 target_valid 和容量归一化口径，分层只替换为冻结的偏航区间。全部逐种子配对结果及 VF、基线的完整三种子汇总如下；相对改善的正负均原样报告。\n')
    parts.append(errors_section('k6_yaw_strata_errors.csv'))
    parts.append('## 偏差、运行事件与结论边界\n')
    parts.append(named_table('operational_incidents.json',['事件编号','类型','原因','处置'],[[x['code'],x['kind'],x['reason'],x['action']] for x in inc]))
    parts.append('现有一年的 Kelmarsh 记录不足以检验老化；Wind Farm A既有记录时长的局限也不因本分析改变。本任务没有增加分层、训练种子、统计检验或模型选择。高 TI 主张、湍流修正收益和偏航曲线解释均按勘误及上述方法边界解释。\n')
    parts.append('## 复现、图表核查与输出文件\n')
    parts.append('在工作区执行以下命令，所有脚本自行把输出限制在本目录，Python 使用 -B 禁止在只读源码目录写字节码：\n\n```powershell\n& python -B 补充验证_20261024\\K_turbulence_yaw\\v2_勘误后\\scripts\\run_all.py\n```\n')
    parts.append('独立图脚本均读取本目录 CSV/JSON；字体、配色、可编辑 SVG 和 PNG 导出共用 figure_style.py。按用户要求保持首轮 PNG/SVG 文件集。静态技能检查不追踪导入的共用样式，且把宽度算式误识别为字面数；交付核查在组合源码和实际 SVG 上逐项复核，详见 `k0_identity_checks.json` 的 figure_qa。全部图片在最终尺寸检查了文字、图例、面板边界及数据方向。\n')
    parts.append('manifest.json 逐一记录输入及输出文件 SHA-256，并再次核查所有登记输入未变。文件不能包含自身完整字节的 SHA-256 固定点；manifest 自项采用“移除 self 项后的规范 JSON”摘要，明确区别于普通文件字节摘要。该唯一自引用例外记录在 operational_incidents.json。run_log.md 保存执行命令、开始/结束 UTC 和持续秒数。\n')
    files=sorted({p.relative_to(O).as_posix() for p in O.rglob('*') if p.is_file() and '.mplconfig' not in p.parts}|{'报告.md','manifest.json','scripts/build_report.py','scripts/finalize.py','scripts/run_all.py'})
    parts.append('\n'.join('- `'+x+'`' for x in files)+'\n')
    (O/'报告.md').write_text('\n'.join(parts),encoding='utf-8')
    end(start,'build_report')
if __name__=='__main__':main()
