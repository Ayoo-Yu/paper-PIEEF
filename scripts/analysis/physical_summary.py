from run_physics import *
sys.path.insert(0,str(ROOT/'修复重算_20261021/statistics'))
import corrected_statistics as stats

def summarize_p4():
 cal=[];raw=[];seedrows=[];reference=None
 for seed in [13,21,42,87,3407]:
  z=dict(np.load(OUT/f'physical_samples_s{seed}.npz',allow_pickle=False))
  if reference is not None:
   for k in ['timestamp','origin_timestamp','turbine_ids','wind_valid','target_valid','capacity_mw','scada_wind','y_true_mw','raw_nwp']:
    assert np.array_equal(reference[k],z[k],equal_nan=True) if z[k].dtype.kind=='f' else np.array_equal(reference[k],z[k]),k
  reference=z;mask=z['wind_valid'];counts=mask.sum(axis=1);assert (counts>0).all()
  ec=np.where(mask,z['v_cal'].astype(float)-z['scada_wind'],0);er=np.where(mask,z['raw_nwp'].astype(float)-z['scada_wind'],0)
  cal.append({'MSE':(ec**2).sum(1)/counts,'MAE':np.abs(ec).sum(1)/counts})
  raw.append({'MSE':(er**2).sum(1)/counts,'MAE':np.abs(er).sum(1)/counts})
  seedrows.append(dict(seed=seed,MSE_cal=np.mean(ec[mask]**2),MSE_raw=np.mean(er[mask]**2),MAE_cal=np.mean(np.abs(ec[mask])),MAE_raw=np.mean(np.abs(er[mask]))))
 result={}
 for loss in ['MSE','MAE']:
  c=np.mean([x[loss] for x in cal],axis=0);r=np.mean([x[loss] for x in raw],axis=0)
  inf=stats.inference(r,c,16);supp=stats.seed_supplement([x[loss+'_cal']-x[loss+'_raw'] for x in seedrows])
  supp['multiplicity']='predeclared singleton global family; paired t and exact sign each adjusted separately with BH'
  # Predefined separate singleton families for each statistic and loss.
  for obj,key in [(inf,'p_value'),(supp,'paired_t_p_raw'),(supp,'exact_sign_p_two_sided_raw')]:
   obj[key+'_bh']=float(stats.established.bh_adjust([obj[key]])[0]) if obj[key] is not None else None
  result[loss]=dict(time=inf,seed=supp,n_time=len(c),n_valid=int(mask.sum()),family='singleton global comparison; each inference level adjusted separately')
 js('P4_statistics.json',result);csv('P4_seed_losses.csv',seedrows)
 w=pd.read_csv(OUT/'P4_wind_metrics.csv');summary=[]
 for (scope,label,method),g in w.groupby(['scope','label','method'],sort=False):
  row=dict(scope=scope,label=label,method=method,n=int(g.n.iloc[0]),n_seed=len(g))
  for metric in ['rmse_ms','mae_ms','bias_ms']:row[metric+'_mean']=g[metric].mean();row[metric+'_sd']=g[metric].std(ddof=1)
  summary.append(row)
 csv('P4_wind_summary.csv',summary)
 progress('P4 检验完成：复用 corrected_statistics 的时间 DM/HAC/HLN、移动时间块自助法和种子层检验；按预定义单项族 BH。')

def summarize_p5():
 df=pd.read_csv(OUT/'P5_gradients.csv');rows=[]
 for seed in [13,21,42,87,3407,'pooled']:
  sub=df if seed=='pooled' else df[df.seed==seed]
  for (group,kind,a,b),g in sub.fillna({'term_b':''}).groupby(['group','kind','term_a','term_b'],sort=False):
   row=dict(seed=seed,group=group,kind=kind,term_a=a,term_b=b,n_total=len(g))
   if kind=='pair':
    finite=g.cosine.dropna();row.update(cosine_mean=finite.mean(),cosine_sd=finite.std(),n_defined=len(finite),n_undefined=len(g)-len(finite),negative_count=int((finite<0).sum()),negative_pct_defined=100*(finite<0).mean() if len(finite) else np.nan)
    for key in ['cosine','norm_b_over_a']:
     q=quant(g[key]);row.update({key+'_'+k:v for k,v in q.items()})
   else:
    for key in ['raw_norm','weighted_norm']:
     row[key+'_mean']=g[key].mean();row.update({key+'_'+k:v for k,v in quant(g[key]).items()})
   rows.append(row)
 csv('P5_gradient_summary.csv',rows);progress('P5 汇总完成：五种子和合并分布、零梯度未定义计数、所有两两余弦及范数比已导出。')
if __name__=='__main__':
 summarize_p4()
 if (OUT/'P5_completed.json').exists():summarize_p5()
