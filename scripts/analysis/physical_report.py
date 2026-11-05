from run_physics import *
def table(headers,rows):
 def fmt(x):
  if x is None:return '—'
  if isinstance(x,(float,np.floating)):return f'{x:.6g}' if np.isfinite(x) else '未定义'
  return str(x).replace('|','/')
 return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(x) for x in row)+' |\n' for row in rows)+'\n'
def run():
 seeds=[13,21,42,87,3407];p1=pd.read_csv(OUT/'P1_curve_summary.csv');mono=pd.read_csv(OUT/'P1_monotonicity_crosscheck.csv');between=pd.read_csv(OUT/'P1_between_turbines.csv');p2=pd.read_csv(OUT/'P2_density_summary.csv');p3=pd.read_csv(OUT/'P3_cutout_summary.csv');p4=pd.read_csv(OUT/'P4_wind_metrics.csv');p4s=pd.read_csv(OUT/'P4_wind_summary.csv');p5=pd.read_csv(OUT/'P5_gradient_summary.csv');stat=json.loads((OUT/'P4_statistics.json').read_text(encoding='utf-8'));integrity=json.loads((OUT/'copy_integrity.json').read_text(encoding='utf-8'));meta=json.loads((OUT/'audit_metadata.json').read_text(encoding='utf-8'));p14=json.loads((OUT/'P14_completed.json').read_text(encoding='utf-8'));p5done=json.loads((OUT/'P5_completed.json').read_text(encoding='utf-8'));final=json.loads((OUT/'final_verification.json').read_text(encoding='utf-8'))
 lines=['# 任务 P 报告（补做重算）\n','**P1–P5 均完成。** 只使用已有最终权重进行推理、直接曲线计算和最终权重处求导；训练/优化步为 0。上一轮“目录拒绝访问、数值未完成”的状态已由本报告替代。未执行任务 F 或 T，未写入其输出目录。\n']
 lines+=['## 完成范围与复现\n','输入采用指定的只读副本 `补做重算/inputs/round3/`。h16 种子集合为 **13/21/42/87/3407**；P1 另完成 h1/h4/h48/h96 的全部五种子，共 25 个主模型运行、725 条机组曲线。发现主模型每个配置/种子均只有一个完成运行，没有选择或排除重复运行。\n',table(['项目','状态','实际计算范围'],[['P1 / B1','完成','25 运行 × 29 机组；0–30 m/s，0.1 m/s 网格；h16 五张图及全提前量总览'],['P2 / C1','完成','五种子完整测试样本；密度比、等效风速和有/无密度的物理功率差'],['P3 / C2','完成','五种子目标时刻 v_cal 和实际硬切出门'],['P4 / B8','完成','同目标时间/机组的有效原始 SCADA 风速；全体、29 机组、4 风速区间及两层检验'],['P5 / B7','完成','每种子 50 个共同训练 batch，batch=16；六个原始损失项、合成辅助项，四个参数组']])]
 lines+=['每次比较先核验目标时间、预测起点、机组、真值、有效掩码、容量；五种子全部预处理数组和数据/模型/训练配置一致。h16 的 metrics、config、predictions 三方提前量均为 16，目标与起点精确相差 240 分钟。全测试集 **7,604 个时刻 × 29 台 = 220,516 个位置**，有效功率目标 **220,401**。P4 使用独立的有效原始风速掩码，未以功率掩码代替。\n','正常复现门槛固定为 `|Δ|/容量 ≤ 1e-5 + 1e-5·|旧预测/容量|`；本次对全部 220,516 个预测位置检查，比仅查有效功率目标更严格。CPU 用于 P1–P4；P5 使用 GPU 前另完成一次全测试集正常复现。\n']
 gates=[]
 for s in seeds:
  a=json.loads((OUT/f'normal_equivalence_s{s}.json').read_text(encoding='utf-8'));b=json.loads((OUT/f'cuda_normal_equivalence_s{s}.json').read_text(encoding='utf-8'));gates.append([s,a['status'],a['max_abs_delta_pu'],b['status'],b['max_abs_delta_pu'],a['max_tolerance_fraction'],b['max_tolerance_fraction']])
 lines.append(table(['种子','CPU','最大Δ/容量','GPU','最大Δ/容量','CPU 最大误差/容许值','GPU 最大误差/容许值'],gates))
 lines+=['## P1：已学功率曲线\n','完整链为：目标 NWP 反标准化取 `nwp_wind_speed100_ms` → `delta_v=3*tanh(calib_net)` → `v_cal=relu(v_raw+delta_v)` → `v_eq=v_cal*(ρ/ρ0)^(1/3)` → 单调控制点的三次 B 样条 → 在 **v_cal≥3 且 v_cal<22 m/s** 时保留，否则归零。切入/切出是硬门，使用 v_cal 而非 v_eq。\n','控制点由 softplus 增量累加并截顶构成；本次所有已学控制点的相邻增量均非负。以下同时报告实际 FP32 自动求导的严格负值比例和用相邻控制点增量计算的等价稳定导数；**没有将 FP32 小负值改写为 0**。稳定导数将样条导数写成三个非负控制点增量的非负加权和；与自动求导逐点交叉核验最大差小于 1e-6 容量单位/(m/s)。\n']
 rows=[]
 for s in seeds:
  g=p1[(p1.horizon==16)&(p1.seed==s)];m=mono[(mono.horizon==16)&(mono.seed==s)];b=between[(between.horizon==16)&(between.seed==s)].iloc[0]
  rated=g.rated100_wind_ms.dropna();rows.append([s,m.stable_negative_derivative_pct.mean(),m.float32_negative_derivative_pct.mean(),m.float32_min_derivative_pu_per_ms.min(),f'{g.cap_pct.min():.4f}–{g.cap_pct.max():.4f}',f'{rated.min():.1f}–{rated.max():.1f}' if len(rated) else '未达到',int(g.rated100_wind_ms.isna().sum()),b.max_turbine_spread_capacity_pp])
 lines.append(table(['种子','稳定导数<0 (%)','FP32导数<0 (%)','最小FP32导数 (pu/(m/s))','截顶水平 (%容量)','首次达到100%风速范围 (m/s)','未达额定机组数','最大机组间差 (容量百分点)'],rows))
 lines.append('校准网络依赖全部目标 NWP 特征，不能仅给一个原始风速就定义唯一校准输出。因此完整物理功率曲线以 v_cal 为横轴，直接计算真实 power_at(v_cal,ρ/ρ0)；校准网络在真实测试输入上的行为另由 P3/P4 计算，没有人为拼造 NWP 网格。\n')
 lines.append('额定到达风速按网格上 FP32 基础输出≥1.0 的首点定义；另表 cap_wind_ms 以达到实际截顶值减1e-6 pu作为数值平台到达标准。该平台识别标准只用于曲线描述，与正常预测复现门槛无关。\n')
 lines+=['FP32 负导数及相邻网格微小下降主要出现在平台段；具体负值幅度、有限差分负比例、截顶到达风速、每台机组完整网格均保存在 P1 数据表中。切出门在 22 m/s 将功率直接归零，22–30 m/s 为 0；这一设计下降不计入基础样条单调性。额定风速配置 10 m/s 不强制样条在 10 m/s 达到额定。\n']
 rows=[]
 for h in [1,4,16,48,96]:
  g=p1[p1.horizon==h];m=mono[mono.horizon==h];b=between[between.horizon==h]
  rows.append([h,len(g),m.stable_negative_derivative_pct.mean(),m.float32_negative_derivative_pct.mean(),m.max_base_decrease_pu.max()*100,g.cap_pct.min(),g.cap_pct.max(),b.max_turbine_spread_capacity_pp.max()])
 lines.append(table(['提前量','机组曲线数','稳定负导数 (%)','FP32负导数 (%)','最大单步下降 (容量百分点)','最小截顶 (%)','最大截顶 (%)','最大机组间差 (容量百分点)'],rows))
 lines+=['每个 h16 图展示全部 29 台基础曲线与导数，以及测试密度 p05/p50/p95 下完整物理分支的机组均值及 min–max 带。全提前量图覆盖全部 25 运行；带宽是机组差异，**不是置信区间**。曲线横轴是样条输入或校准风速，图中已分别标注。\n','## P2：模型实际密度修正\n','五种子均使用 **ρ0=1.225 kg/m³**，`ρ=p/(287.05*T)`。p 来自目标行 `nwp_surface_pressure_pa`（Pa），T 来自目标行 `nwp_temperature_2m_k`（K）；使用训练统计量反标准化，无湿度或地形密度外推。预测物理分支使用目标时刻 NWP，历史重建先验使用历史 NWP，两者没有混算。测试 NWP 对全部机组广播，密度分布有 7,604 个不重复时刻，表内位置计数为 220,516。\n']
 q=p2[(p2.seed==13)&(p2.scope=='all')];rows=[]
 for key,label,mult in [('rho_ratio','ρ/ρ0（%参考密度）',100),('equivalent_wind_factor','(ρ/ρ0)^(1/3)（%原风速）',100)]:
  r=q[q.quantity==key].iloc[0];rows.append([label]+[r[k]*mult for k in ['min','p05','p50','p95','max']])
 lines.append(table(['量','min','p05','p50','p95','max'],rows))
 lines+=['功率等效影响定义为固定同一个 v_cal、同一已学曲线、同一硬门，计算 `100*[P_phys(ρ/ρ0)−P_phys(1)]`，单位为容量百分点；它是物理分支影响，不是最终预测影响。\n',table(['种子','min','p05','p50','p95','max'],[[s]+[p2[(p2.seed==s)&(p2.scope=='all')&(p2.quantity=='power_effect_capacity_pp')].iloc[0][k] for k in ['min','p05','p50','p95','max']] for s in seeds])]
 rr=q[q.quantity=='rho_ratio'].iloc[0];ff=q[q.quantity=='equivalent_wind_factor'].iloc[0]
 lines.append(f'当前测试密度比 p05–p95 的半跨度为 **{50*(rr.p95-rr.p05):.4f}%**；等效风速因子 p05–p95 相对 1 的变化为 **{100*(ff.p05-1):+.4f}% 至 {100*(ff.p95-1):+.4f}%**，全跨度 {100*(ff.p95-ff.p05):.4f} 个百分点。旧“6.21%”使用约 1.186 的参考密度且是密度比半跨度，不能解读为立方根风速因子的全跨度，也不能解读为功率变化。另在 P2 表中保存有效功率目标子集及 P_phys(1)>0 上的相对变化，零分母数量单列。\n')
 lines+=['## P3：校准风速与切出触发\n','以下是全部测试位置的 v_cal（m/s）；分位数以及有效功率目标子集均在 CSV 中保留。\n',table(['种子','min','p05','p50','p95','max','>22 (%)','>25 (%)','门<0.99 (%)','门<0.5 (%)'],[[s]+[p3[(p3.seed==s)&(p3.scope=='all')].iloc[0][k] for k in ['min','p05','p50','p95','max','above22_pct','above25_pct','gate_lt099_pct','gate_lt05_pct']] for s in seeds])]
 lines.append('实际切出门为 `1[v_cal<22]`，门关闭含 v_cal=22；它只有 0/1 两个值，因此两个门值阈值统计相同。这只描述本次正常测试集的实际触发，不能推断其他输入、极端风况或训练全程。\n')
 lines+=['## P4：TD-DCN 与 SCADA 风速的对齐误差\n','目标 NWP 字段为 **100 m 风速**，不是配置中 `wind_height_m=10` 所暗示的 10 m；该配置的 U/V 方向字段不改变 speed index=7 的实际取值。SCADA 使用原始 `turbine_ws1…29`；机组元数据的轮毂高度为 85 m，但现有元数据不足以独立证实 SCADA 风速计准确安装高度、自由来流/机舱语义或修正方式。因此本结果是同时间、同机组映射下的观测一致性比较，不能宣称同高度自由来流误差的校准收益。\n','时间由带时区 Timestamp 精确匹配，机组由 turbines.csv 的 T01–T29 顺序与保存索引逐项断言。只保留有限且非负的原始实测风速；不使用因果填补后的风速作真值。风速 RMSE/MAE/bias 必须用 **m/s**，不能除以功率容量。bias=预测−实测；改善率=100×(1−校准误差/原始 NWP 误差)，正值更好。\n']
 rows=[]
 for s in seeds:
  g=p4[(p4.seed==s)&(p4.scope=='all')];a=g[g.method=='raw_nwp'].iloc[0];b=g[g.method=='v_cal'].iloc[0]
  rows.append([s,int(a.n),a.rmse_ms,b.rmse_ms,100*(1-b.rmse_ms/a.rmse_ms),a.mae_ms,b.mae_ms,100*(1-b.mae_ms/a.mae_ms),a.bias_ms,b.bias_ms])
 lines.append(table(['种子','n','原始RMSE','校准RMSE','改善 (%)','原始MAE','校准MAE','改善 (%)','原始bias','校准bias'],rows))
 ag=p4s[(p4s.scope=='all')&(p4s.method=='v_cal')].iloc[0];bg=p4s[(p4s.scope=='all')&(p4s.method=='raw_nwp')].iloc[0]
 lines.append(f"全体五种子校准 RMSE 为 {ag.rmse_ms_mean:.6f}±{ag.rmse_ms_sd:.6f} m/s，MAE 为 {ag.mae_ms_mean:.6f}±{ag.mae_ms_sd:.6f} m/s；相对原始 NWP 的均值改善为 {100*(1-ag.rmse_ms_mean/bg.rmse_ms_mean):.4f}% / {100*(1-ag.mae_ms_mean/bg.mae_ms_mean):.4f}%。平均偏差由 {bg.bias_ms_mean:.6f} 变为 {ag.bias_ms_mean:.6f} m/s。RMSE/MAE与有符号偏差描述不同方面，不能只保留有利指标。\n")
 lines.append('分层边界按原始 SCADA 观测定义为 <3、[3,10)、[10,15]、>15 m/s；以下为五种子均值±样本标准差（ddof=1）。逐种子完整数值在 P4_wind_metrics.csv。\n')
 rows=[]
 for label in ['<3','[3,10)','[10,15]','>15']:
  for method in ['raw_nwp','v_cal']:
   g=p4s[(p4s.scope=='wind_bin')&(p4s.label==label)&(p4s.method==method)].iloc[0]
   rows.append([label,method,int(g.n)]+[f'{g[k+"_mean"]:.4f} ± {g[k+"_sd"]:.4f}' for k in ['rmse_ms','mae_ms','bias_ms']])
 lines.append(table(['风速区间','方法','n','RMSE (m/s)','MAE (m/s)','bias (m/s)'],rows))
 lines.append('逐机组五种子均值如下；各机组的种子标准差和每种子值均在 P4_wind_summary.csv / P4_wind_metrics.csv，未因不利方向或样本少而删去任何机组。\n')
 rows=[]
 for tid in sorted(p4s[p4s.scope=='turbine'].label.unique()):
  g=p4s[(p4s.scope=='turbine')&(p4s.label==tid)];a=g[g.method=='raw_nwp'].iloc[0];b=g[g.method=='v_cal'].iloc[0]
  rows.append([tid,int(a.n),a.rmse_ms_mean,b.rmse_ms_mean,a.mae_ms_mean,b.mae_ms_mean,a.bias_ms_mean,b.bias_ms_mean])
 lines.append(table(['机组','n','原始RMSE','校准RMSE','原始MAE','校准MAE','原始bias','校准bias'],rows))
 lines+=['**检验族在计算前写入 design_fixed.json，并沿用上一轮源码核查的定义**：全体校准 vs 原始 NWP 的 MSE 为一族、MAE 为一族，各只有一个比较；时间 DM、种子配对 t、精确符号检验分别作单项族 BH，因此 q=p。逐机组及区间仅描述，不补做选择性检验。先对各模型求损失，再按共同有效机组、五种子聚合为逐时间损失；没有先平均预测。\n','时间 DM/HAC/HLN 及移动时间块自助法直接复用 corrected_statistics.py；时间 DM 以这五个已训练模型为条件，种子层检验反映此固定测试期上训练随机性的差异，二者含义不同。精确符号检验调用该实现中的 SciPy binomtest。\n']
 rows=[]
 for loss,r in stat.items():
  t=r['time'];s=r['seed'];rows.append([loss,r['n_time'],t['mean_loss_difference'],f"[{t['ci_low']:.6g}, {t['ci_high']:.6g}]",t['p_value_bh'],s['paired_t_p_raw_bh'],s['exact_sign_p_two_sided_raw_bh'],f"{s['n_negative']}/5"])
 lines.append(table(['损失','时间n','校准−原始平均损失','移动块95%CI','DM q','种子t q','符号q','校准更低种子数'],rows))
 t=stat['MSE']['time'];lines.append(f"自助法 {t['n_bootstrap']} 次，块长 {t['block_length']} 个时间步，种子 13；HAC lag={t['hac_lag']}，HLN h=16。MSE 差单位 (m/s)²，MAE 差单位 m/s。只有五个种子时，双侧符号检验最小 p=0.0625；不能把非显著解释为等效。\n")
 lines+=['## P5：最终权重处梯度诊断\n','训练代码 Stage 3 joint 的实际损失为：\n\n`L = L_forecast + a*w_R*(L_reconstruction + λ_bound*L_boundary + λ_deriv*L_temporal) + a*w_C*(L_calibration_fit + 0.05*mean(delta_v²))`。\n\n其中 `a=exp[-0.1*(best_epoch−1)]`，w_R=w_C=0.05，λ_bound=0.1，λ_deriv=0.05。预测项为有效目标上的 Huber δ=0.1；重建项为掩码位置归一化功率 MSE；边界项为反 RevIN 后的 `relu(−p)+relu(p−1.05)`；时间差分物理项比较相邻时间步与先验的功率差分，不是 dP/dv 单调性罚项；校准拟合仅用有限且 y>0.01 的目标。没有额外虚构的单调性损失或切出边界损失。\n']
 rows=[]
 for s in seeds:
  a=json.loads((OUT/f'P5_audit_s{s}.json').read_text(encoding='utf-8'));rows.append([s,a['best_epoch'],a['decay'],a['corruption_counts']['corrupted_batches'],len(a['batches']),a['peak_allocated_bytes']/1024**3,a['weights_unchanged']])
 lines.append(table(['种子','best epoch','辅助衰减系数','F2损坏batch','batch总数','峰值已分配显存GiB','权重未变'],rows))
 lines+=['固定诊断种子 2026，从训练样本无放回随机排列取 800 个样本组成 50 个完整 batch（16/批），五个模型共用索引；另以 2026 初始化独立 F2 流和各模型的 torch 随机流。F2 完整复用 ForecastCorruption：批次损坏概率0.5，样本级随机机组/时段/通道；仅预测路径受损，辅助重建用干净历史。保存全部抽样索引、训练目标行号和 F2 场景。\n','dropout 与训练代码一致：预测编码 `_stage3_features` 临时 eval，关闭编码器 dropout 与随机预训练掩码但保留梯度；返回后恢复 train，图/预测路径保持训练模式，辅助重建启用训练 dropout 和随机掩码。冻结解码器参数，保留通过解码器向编码器反传。GPU 与 CPU 的随机算子不承诺相同位级随机序列；最终五种子诊断统一使用 GPU 与共同随机种子。\n','参数组：encoder=pi_embedding+position_embedding+mae_encoder；TD_DCN=dcn（校准网络及曲线参数）；graph=图层；all_shared=上述并集再加 RevIN 仿射参数。无梯度参数补零保留相同坐标。任一梯度范数为0时余弦未定义，明确报告数量；负比例分母仅为定义的余弦，不能将零梯度填成零余弦。范数比定义为 B/A，使用实际衰减和系数加权的梯度；另保存未加权范数。\n','以下为全部共享参数上预测项 A 对每项 B 的 250 个 batch 分布汇总。它们是诊断样本的描述统计，没有把 250 个 batch 当独立训练重复作显著性检验。\n']
 g=p5[(p5.seed.astype(str)=='pooled')&(p5.group=='all_shared')&(p5.kind=='pair')&(p5.term_a=='forecast')];rows=[]
 for _,r in g.iterrows():rows.append([r.term_b,r.cosine_mean,r.cosine_p05,r.cosine_p50,r.cosine_p95,int(r.n_defined),int(r.n_undefined),r.negative_pct_defined,r.norm_b_over_a_p50,r.norm_b_over_a_p05,r.norm_b_over_a_p95])
 lines.append(table(['B项','余弦均值','p05','p50','p95','定义n','未定义n','负余弦 (%)','加权范数B/A中位','p05','p95'],rows))
 lines.append('编码器、TD-DCN、图层的合成辅助项配对如下；完整六项及全部两两配对、逐种子分布、原始/加权范数见 P5_gradient_summary.csv。\n')
 rows=[]
 for group in ['encoder','TD_DCN','graph']:
  for term in ['reconstruction_total','calibration_total']:
   r=p5[(p5.seed.astype(str)=='pooled')&(p5.group==group)&(p5.kind=='pair')&(p5.term_a=='forecast')&(p5.term_b==term)].iloc[0]
   rows.append([group,term,r.cosine_mean,int(r.n_defined),int(r.n_undefined),r.negative_pct_defined,r.norm_b_over_a_p50])
 lines.append(table(['参数组','B项（A=预测）','平均余弦','定义n','未定义n','负余弦 (%)','加权范数B/A中位'],rows))
 lines.append('**这些只是最终验证集最佳权重处的局部梯度诊断，不能代表训练过程的梯度轨迹，也不能仅凭负余弦频率推断物理项导致最终预测退化。** 每个 batch 断言拆分损失之和与原训练总损失一致；每个种子结束后逐张量核验 state_dict 未变，未构造或执行优化器。\n')
 lines+=['## 副本、来源与耗时\n',f"MANIFEST.md 只提供归档 SHA，没有逐成员 SHA。本次先校验全部 6 个 round3/local_stage2 归档与清单一致，再从中只读读取 h16 种子 13/42/3407 的 weights_best.pt 和 predictions.npz，逐成员 SHA 与副本对照，**{integrity['covered']}/{integrity['requested']} 通过**。没有把文件数/总字节数相同当作 SHA 验证。明细见 copy_integrity.json。\n"]
 lines.append(table(['种子/文件','副本SHA256'],[[Path(x['copy']).parent.name+'/'+Path(x['copy']).name,x['sha256']] for x in integrity['samples']]))
 lines.append(f"源码仅来自 `{PROJECT}`；本次源码清单 SHA256：`{sha(OUT/'source_manifest.json')}`。历史运行 tracking.code_git_commit 为 `4844f3e1aae8513f358c6ad9af6f8cf98b536b93`；这是历史记录，和本次实际文件 SHA 分开列示，不以当前仓库 HEAD 冒充历史源码。\n")
 lines.append(table(['关键源码','SHA256'],[[str(Path(f['path']).relative_to(PROJECT)) if Path(f['path']).is_relative_to(PROJECT) else Path(f['path']).name,f['sha256']] for f in json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8')) if Path(f['path']).name in ['physics.py','e2e.py','train_main.py','pretrain.py','forecast_corruption.py','corrected_statistics.py']]))
 lines.append(table(['h16种子','只读运行目录（相对 inputs/round3）'],[[r['seed'],str(Path(r['path']).relative_to(OUT.parent/'inputs/round3'))] for r in records() if r['horizon']==16]))
 lines.append(f"P1–P4：CPU（{meta['cpu']}），torch {meta['torch']}，Python {meta['python']}，2 个计算线程，batch=16；五种子推理与主要分析耗时 {p14['seconds']:.1f} 秒（不含随后可选提前量曲线和绘图）。P5：NVIDIA GeForce RTX 4070 SUPER，batch=16 未减小，单进程显存上限设为48%；完整GPU正常复现+250 batch梯度及等待CPU门禁共 {p5done['seconds']:.1f} 秒。最初CPU诊断前5 batch用68.6秒，因计算耗时改用GPU，CPU试算未纳入结果；其日志保留。\n")
 lines.append(f"结束核验：{final['input_files_unchanged']} 个输入文件 SHA 未变，{final['source_files_unchanged']} 个源码文件 SHA 未变；全部25个曲线运行、250个梯度batch齐全。每项输出的源码清单SHA、运行目录、设备、种子集合及自身SHA统一记录在 output_manifest.json / 输出文件清单.csv，避免把共享来源重复嵌入数十万行数据。\n")
 lines+=['## 输出与限制\n','主要文件如下（均位于本目录）：\n',table(['输出','用途'],[['P1_curve_summary.csv / P1_monotonicity_crosscheck.csv / P1_between_turbines.csv','额定风速、截顶、严格及稳定导数、机组差异'],['curves_h*_s*.csv（25份）','全部0–30m/s网格、基础与三密度完整曲线'],['P1_h16_seed*.png/.svg/.pdf、P1_all_horizons.*','h16五种子图及全提前量概览'],['P2_density_summary.csv / P3_cutout_summary.csv','全部和有效功率子集的分布与门统计'],['P4_wind_metrics.csv / P4_wind_summary.csv / P4_statistics.json','全体、机组、风速层误差和两层检验'],['physical_samples_s*.npz','完整时间/起点/机组索引、真值/掩码/容量、实际物理中间量'],['P5_gradients.csv / P5_gradient_summary.csv / P5_losses.csv','全部梯度余弦、范数、范数比、损失及系数'],['P5_batch_indices.json / P5_audit_s*.json','共同batch索引、F2场景、模式和权重未变核验'],['normal_equivalence_s*.json / cuda_normal_equivalence_s*.json','CPU/GPU完整正常复现门禁'],['copy_integrity.json / input_manifest.json / source_manifest.json / final_verification.json','输入副本及源码溯源与不变性核验'],['run_physics.py / run_gradients_cuda.py / summarize_physics.py / plot_physics.py 等','复算脚本'],['output_manifest.json / 输出文件清单.csv','全输出绝对路径、SHA256及来源；报告SHA在外部清单中']])]
 lines.append('未完成的必做计算：**无**。保留的解释边界：SCADA 测风高度/传感器定义无法由现有元数据独立证实；额外提前量仅做直接曲线，不做其全测试预测复现；P5只对应最终权重，不能替代训练轨迹。P2物理分支功率差不能当作最终预测收益。全部不利分层、严格FP32负导数及零梯度未定义值均保留。\n')
 lines.append('主要输出的 SHA256 如下；完整绝对路径和全部输出 SHA 见 `输出文件清单.csv`。清单本身不自引用，报告自身SHA由 output_manifest.json 记录。\n')
 lines.append(table(['本目录相对路径','SHA256'],[[name,sha(OUT/name)] for name in ['P1_curve_summary.csv','P1_monotonicity_crosscheck.csv','P1_all_horizons.png','P2_density_summary.csv','P3_cutout_summary.csv','P4_wind_metrics.csv','P4_statistics.json','P5_gradients.csv','P5_gradient_summary.csv','P5_batch_indices.json','final_verification.json']]))
 (OUT/'报告.md').write_text('\n'.join(lines),encoding='utf-8')
 # Replace stale blocked state in the prior source note with actual execution evidence.
 note='''# 任务 P 源码核查（本轮已执行核验）

P1–P5 已完成。实际配置、权重、测试身份、CPU/GPU正常复现和最终权重处梯度均已核验；本文件替代上一轮只做静态核查的状态。数值、SHA和具体运行路径见报告及相应清单。

## P1：完整物理链

e2e.py 将目标 NWP 按训练统计量反标准化，取 index=7 的 nwp_wind_speed100_ms。physics.py 的 delta_v=3*tanh(calib_net(x_target))，v_cal=relu(v_raw+delta_v)。v_eq=v_cal*(rho/rho0)^(1/3) 输入 B 样条；最终门为 v_cal>=3 且 v_cal<22。密度不改变门的判定风速。

LearnableBSplineCurve 控制点由 softplus 非负增量累加后与 clamp(rated_power,0.8,1.0) 取最小值；cp_start 只被 zeros_like 使用，不构成可学习非零起点。额定风速配置10不强迫样条在10 m/s达到额定。样条输入上界截在 max_ws-0.01。

本次同时导出 FP32 自动求导与稳定等价导数。设四个相邻控制点p0…p3、局部坐标w、网格宽度g，导数为 [0.5*(1-w)^2*(p1-p0)+(-w^2+w+0.5)*(p2-p1)+0.5*w^2*(p3-p2)]/g。三个系数及控制点差分均非负。FP32平台小负值未删改；切出设计下降与基础曲线单调性分开报告。

## P2–P3：密度与门

五种子实际 reference_density=1.225；rho=p/(287.05*T)，目标行 nwp_surface_pressure_pa 为Pa、nwp_temperature_2m_k为K。预测分支用目标NWP；重建先验用历史NWP。功率影响固定v_cal及曲线，计算100*(P(rho/rho0)-P(1))，单位容量百分点，不等于最终预测变化。

实际切出门是1[v_cal<22]，<0.99和<0.5两种统计相同；严格>22另列。分布同时给全体输入位置和有效功率目标子集。旧6.21%/1.186口径已由实际测试数据重新计算，不沿用全CSV摘要。

## P4：对齐及统计

raw NWP来自100m speed字段，配置wind_height_m=10不覆盖该字段索引。SCADA原始turbine_ws1…29按目标Timestamp及T01…T29机组映射，有限且非负才有效，不用填补值作真值。轮毂元数据85m不能独立证明传感器高度或自由来流定义，保留这一限制。

风速误差单位m/s；分层<3、[3,10)、[10,15]、>15。MSE/MAE各为一个全体比较族，时间DM、种子t和符号检验分别BH单项族。实际复用corrected_statistics.py，先损失后聚合，无预测集成和p值平均；逐机组/区间只描述。

## P5：真实Stage 3损失及求导

L=forecast+a*wR*(reconstruction+lambda_bound*boundary+lambda_deriv*temporal)+a*wC*(calibration_fit+0.05*mean(delta_v^2))。

a=exp[-0.1*(best_epoch-1)]，wR=wC=0.05，lambda_bound=0.1，lambda_deriv=0.05。forecast为Huber(delta=0.1)；重建对掩码上的归一化功率计MSE；边界对反RevIN后的功率计算relu(-p)+relu(p-1.05)；temporal为相邻时间步功率差分与先验差分的MSE，不是dP/dv正则；calibration_fit仅用有效且y>0.01目标。不存在额外的显式单调性或切出边界损失。

_stage3_features临时eval关闭预测编码器dropout和随机掩码，保留joint梯度，然后恢复train。辅助重建启用训练dropout和随机块掩码。F2只损坏预测历史，辅助重建用干净历史。解码器参数冻结但允许向编码器反传。50个batch×5种子共用固定2026样本排列/F2流；最终统一GPU求导。CPU初始试算仅记录耗时，不纳入统计。

encoder包含embedding、position和mae_encoder；TD_DCN包含dcn；graph包含图层；all_shared还纳入RevIN仿射参数。未连接参数梯度补零，任一范数零则余弦未定义；不把未定义当零。原始/加权范数及两两余弦全部保存。逐batch校验损失拆分等价；逐seed验证所有state_dict张量不变；没有优化器step。

最终权重处诊断不能代表训练全程梯度轨迹。
'''
 (OUT/'源码核查.md').write_text(note,encoding='utf-8')
 meta.update(status='completed',finished=time.strftime('%Y-%m-%d %H:%M:%S'),P14_seconds=p14['seconds'],P5_seconds=p5done['seconds'],P5_device=p5done['device'],optimizer_steps=0);js('audit_metadata.json',meta)
 oldprogress=(OUT/'进度.md').read_text(encoding='utf-8')
 recent=[line for line in oldprogress.splitlines() if line.startswith('- ') and ('2026-10-21' in line or '2026-10-22' in line)]
 (OUT/'进度.md').write_text('# 任务 P 进度：全部完成\n\nP1、P2、P3、P4、P5 均已完成；本轮可读副本已替代此前访问受阻输入。详细数值与身份/复现门禁见报告.md。\n\n'+'\n'.join(recent)+'\n',encoding='utf-8')
 progress('任务 P 完成：P1–P5 全部重算，中文报告、图表、复现与身份核验记录已更新。')
 # Full output manifest excludes itself and its CSV mirror to avoid recursive hashes.
 manifest=[]
 for f in sorted(OUT.rglob('*')):
  if not f.is_file() or f.name in ['output_manifest.json','输出文件清单.csv'] or 'mplconfig' in f.parts:continue
  import re
  selected=records() if f.name.startswith(('P1','curves_','curve_summary_','run_inventory','input_manifest','source_manifest','copy_integrity','final_verification','报告','源码核查','进度','audit_metadata')) else [r for r in records() if r['horizon']==16]
  match=re.search(r'h(\d+)_s(\d+)',f.name)
  if match:selected=[r for r in records() if r['horizon']==int(match[1]) and r['seed']==int(match[2])]
  else:
   match=re.search(r'(?:_s|seed)(\d+)',f.name)
   if match:selected=[r for r in selected if r['seed']==int(match[1])]
  manifest.append(dict(path=str(f),relative_path=str(f.relative_to(OUT)),sha256=sha(f),bytes=f.stat().st_size,source_manifest_sha256=sha(OUT/'source_manifest.json'),run_directories=[r['path'] for r in selected],seeds=sorted(set(r['seed'] for r in selected)),device='CPU P1-P4; RTX4070SUPER P5; CPU summarization',provenance='input_manifest.json; audit_metadata.json; per-seed runtime and audit records'))
 js('output_manifest.json',manifest);csv('输出文件清单.csv',[{k:v for k,v in x.items() if k not in ['run_directories','seeds']} for x in manifest])
 print('REPORT_READY',len(manifest),sha(OUT/'报告.md'),flush=True)
if __name__=='__main__':run()
