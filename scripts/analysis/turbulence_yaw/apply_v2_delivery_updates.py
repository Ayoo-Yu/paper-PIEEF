"""One-time, auditable E1-dependent delivery updates to the copied v1 scripts."""
from pathlib import Path
S=Path(__file__).resolve().parent
def edit(name,old,new):
    p=S/name;s=p.read_text(encoding='utf8');assert old in s,(name,old[:100]);p.write_text(s.replace(old,new),encoding='utf8')

edit('figure_ti_attention.py',"marker=['o','s'][n]","marker=['o','s','^'][n]")
edit('figure_ti_attention.py','(n-.5)*.035','(n-1)*.035')
edit('figure_ti_attention.py','Three-seed summaries blocked; *very high is a subset of high','All 3 seeds per horizon; *very high is a subset of high')
p=S/'figure_ti_errors.py';s=p.read_text(encoding='utf8');a=s.index("                if model=='VF':");b=s.index('            ax.set_title',a)
s=s[:a]+'''                q=d[(d.row_type=='seed_summary')&(d.model==model)&(d.horizon==h)].set_index('stratum').reindex(TI_LAYERS)
                ax.errorbar(x+(mi-3)*.035,q[metric+'_pct_mean'],yerr=q[metric+'_pct_sd'],color=COLORS[model],linewidth=.85,marker='.',markersize=3,capsize=1.5,label=model)
'''+s[b:];s=s.replace('Target-time nacelle turbulence intensity | VF: 2 passed seeds per horizon','Target-time nacelle turbulence intensity | all models: 3 seeds').replace('Baselines: mean +/- SD of 3 seeds','Mean +/- sample SD of 3 seeds');p.write_text(s,encoding='utf8')

edit('finalize.py',"==3*len(layers)","==0")
edit('finalize.py',"assert (vf.status=='BLOCKED_INCOMPLETE_SEEDS').all();assert vf.nrmse_pct_mean.isna().all()","assert (vf.status=='PASS').all();assert vf.nrmse_pct_mean.notna().all()")
edit('finalize.py',"base=d[(d.row_type=='seed_summary')&(d.model!='VF')]","base=d[d.row_type=='seed_summary']")
edit('finalize.py',"a[(a.status=='PASS')&(a.model!='VF')]","a[a.status=='PASS']")
edit('finalize.py','assert len(a)==6','assert len(a)==18')
edit('finalize.py','6 eligible seed rows; exact absolute/relative changes and signs','18 corrected seed rows; exact absolute/relative changes and signs')
edit('finalize.py',"==3*5*3","==0")
edit('finalize.py',"(d.status=='PASS').sum()==36 and (d.status=='BLOCKED_GATE').sum()==18","(d.status=='PASS').sum()==54 and (d.status=='BLOCKED_GATE').sum()==0")
edit('finalize.py','len(cur)==36*301','len(cur)==54*301')
edit('finalize.py','36 curves x 301 points','54 curves x 301 points')
# Do not invent a repeated v1 incident in this run.
p=S/'finalize.py';s=p.read_text(encoding='utf8');s='\n'.join(line for line in s.splitlines() if "incident('read_during_recompute'" not in line)+'\n';s=s.replace("    k['delivery_validation']=check_tables(k);k['figure_qa']=figure_preflight()","    k['delivery_validation']=check_tables(k);k['figure_qa']=figure_preflight()\n    import v2_audit\n    v2_audit.verify(k)")
s=s.replace('User explicitly requires SVG/PNG only; actual editable SVG exists. PDF requirement is inapplicable.','Same v1 output set requested: editable SVG and PNG retained.')
s=s.replace('按用户仅 PNG/SVG 的约束豁免 PDF，渲染图已目视检查。','按用户保持首轮文件集的要求保留 PNG/SVG；渲染图单独目视核查。')
p.write_text(s,encoding='utf8')
edit('run_all.py',"    t=begin('python -B scripts/run_all.py')","    t=begin('python -B scripts/run_all.py')\n    import v2_audit\n    v2_audit.snapshot()")

# Keep the original report's unaffected methods, K4 and full tables.
edit('build_report.py','完整三种子 VF 结论目前不能给出：每个提前量均有一个 VF 运行未达到冻结身份容差。保留通过门禁的逐种子结果、全部基线，以及独立的 SCADA 机理分析；不另选种子或替换预测。下面所有数值来自所注明的 CSV 或 JSON。','E1 勘误后九个 VF 运行全部通过身份门禁，完整报告三个原定种子的均值±样本标准差及全部有利、不利结果。下面所有数值来自所注明的 CSV 或 JSON。')
edit('build_report.py',"'2.05e-6 MW'","'2.05e-5 MW（E1）'")
edit('build_report.py','VF 仅列通过门禁的逐种子值，不以剩余种子均值替代冻结的三种子汇总。','VF 三个原定种子的逐种子结果如下，汇总采用 ddof=1。')
edit('build_report.py',"'不可用'])","f'{v.loc[\"nrmse\",\"VF_improvement_pct\"]:.4f} / {v.loc[\"nmae\",\"VF_improvement_pct\"]:.4f}'])")
edit('build_report.py',"'冻结三种子比较'","'三种子均值相对改善：NRMSE/NMAE（%）'")
p=S/'build_report.py';s=p.read_text(encoding='utf8')
start=s.index("    parts.append('高 TI 层未满足");end=s.index("    parts.append(errors_section",start)
s=s[:start]+"    import v2_report\n    parts.append(v2_report.k1_conclusion())\n"+s[end:]
s=s.replace('基线为三种子均值±SD，VF 为通过门禁的逐种子点','所有模型均为三种子均值±SD')
s=s.replace('三种子残差均值±SD 的槽位保留为 BLOCKED_INCOMPLETE_SEEDS，不把未通过身份门禁的运行混入。','三种子残差均值±SD 完整列于下表及 CSV。')
start=s.index("    a=get('k2_oracle_bound.csv')");end=s.index("    parts.append('## K3：",start)
s=s[:start]+"    parts.append(v2_report.k2_section())\n"+s[end:]
start=s.index("    parts.append('各提前量的三种子均值±SD 仍受");end=s.index("    p=a[a.row_type=='historical_reference']",start)
s=s[:start]+"    parts.append(v2_report.k3_summaries())\n"+s[end:]
s=s.replace('原计划 {x["expected_curves"]} 条曲线；受身份门禁影响，提取','原计划 {x["expected_curves"]} 条曲线；E1 后完整提取').replace('其余运行保留阻断槽位，不加载后补模型或改选权重。','九个运行全部使用原有权重。')
s=s.replace('且完整运行一致性条件被门禁阻断；','九个运行也未形成可归因于偏航的一致排序；')
needle="    parts.append('上述风速及差值单位均为 m/s"
pos=s.index(needle);s=s[:pos]+"    parts.append(v2_report.k5_summaries())\n"+s[pos:]
s=s.replace('通过门禁的逐种子配对结果和全部基线三种子汇总如下。每个提前量的 VF 三种子结果仍不完整，不能由局部有利结果推广到全部运行。','全部逐种子配对结果及 VF、基线的完整三种子汇总如下；相对改善的正负均原样报告。')
s=s.replace('均受上述门禁与方法边界限制','均按勘误及上述方法边界解释')
s=s.replace('K_turbulence_yaw\\\\scripts\\\\run_all.py','K_turbulence_yaw\\\\v2_勘误后\\\\scripts\\\\run_all.py')
s=s.replace('用户指定仅 PNG/SVG，因此不额外生成技能默认建议的 PDF/TIFF。','按用户要求保持首轮 PNG/SVG 文件集。')
pos=s.index("    checks=k['metric_checks']");s=s[:pos]+"    import v2_report\n    parts.insert(1,v2_report.corrigendum(k))\n"+s[pos:]
p.write_text(s,encoding='utf8')
print('E1 delivery updates applied; unaffected analysis scripts preserved.')
