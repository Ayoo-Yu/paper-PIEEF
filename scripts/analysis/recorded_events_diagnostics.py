from pathlib import Path
import os,json
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[k]='2'
import numpy as np
import pandas as pd
L=Path(__file__).resolve().parent
import reference_helpers as ref
def js(n):return json.loads((L/n).read_text(encoding='utf8'))
r=js('results.json');assert js('completion_gate.json')['status']=='PASS'
sources=[];arrays=[]
for seed in (13,42,3407):
 p=L/'runs'/f'ACFormer_L1_h16_seed{seed}'/'normal.npz';a=ref.load_npz(p);arrays.append(a);sources.append(dict(path=str(p),sha256=ref.sha256(p)))
for a in arrays[1:]:
 ref.same_indices(arrays[0],a);assert np.array_equal(arrays[0]['y_true_mw'],a['y_true_mw'],equal_nan=True) and np.array_equal(arrays[0]['capacity_mw'],a['capacity_mw'])
a=arrays[0];targets=[]
for label in ('all_valid','normal','low_power_candidate','power_outlier','frozen_wind_candidate'):
 mask=a['target_valid'] if label=='all_valid' else a['target_valid']&(a['events']==label)
 y=a['y_true_mw'][mask].astype(np.float64);mean=float(y.mean());sst=float(np.square(y-mean).sum())
 targets.append(dict(event=label,n_targets_once=int(len(y)),n_times_once=int(mask.any(axis=1).sum()),mean_mw=mean,variance_population_mw2=float(y.var(ddof=0)),sd_population_mw=float(y.std(ddof=0)),SST_mw2=sst,min_mw=float(y.min()),max_mw=float(y.max()),range_mw=float(y.max()-y.min()),median_mw=float(np.median(y)),zeros=int((y==0).sum()),negative=int((y<0).sum()),n_unique_values=int(np.unique(y).size),mean_forecaster_RMSE_mw=float(np.sqrt(sst/len(y)))))
residual_context=[]
for task in js('protocol.json')['sources']:
 if task['horizon']!=16:continue
 for model,folder in [('ACFormer_L1',L/'runs'/task['task_id']),('VF',Path(task['vf_directory']))]:
  z=ref.load_npz(folder/'normal.npz');ref.same_indices(a,z);assert np.array_equal(a['y_true_mw'],z['y_true_mw'],equal_nan=True)
  for dist in targets:
   label=dist['event'];mask=z['target_valid'] if label=='all_valid' else z['target_valid']&(z['events']==label)
   y=z['y_true_mw'][mask].astype(np.float64);pred=z['y_pred_mw'][mask].astype(np.float64);error=pred-y;sse=float(np.square(error).sum())
   row=dict(model=model,seed=task['seed'],event=label,RMSE_mw=float(np.sqrt(np.square(error).mean())),MAE_mw=float(np.abs(error).mean()),SSE_mw2=sse,SST_mw2=dist['SST_mw2'],R2=1-sse/(dist['SST_mw2']+1e-7),prediction_mean_mw=float(pred.mean()),prediction_min_mw=float(pred.min()),prediction_max_mw=float(pred.max()))
   reference=next(x for x in r['event_per_run'] if x['model']==model and x['seed']==task['seed'] and x['event']==label)
   assert np.isclose(row['R2'],reference['r2'],atol=1e-8,rtol=1e-10)
   residual_context.append(row)
f=pd.DataFrame(r['fault_per_run']);h=f[(f.horizon_steps==96)&(f.scenario!='normal')]
paired=[]
for (seed,name),g in h.groupby(['seed','scenario']):
 v=g[g.model=='VF'].iloc[0];b=g[g.model=='ACFormer_L1'].iloc[0]
 paired.append(dict(seed=int(seed),scenario=name,VF_nmae_pct=float(v.nmae_pct),L1_nmae_pct=float(b.nmae_pct),VF_minus_L1_nmae_percentage_points=float(v.nmae_pct-b.nmae_pct),relative_deterioration_pct=float(100*(v.nmae_pct/b.nmae_pct-1)),VF_nrmse_pct=float(v.nrmse_pct),L1_nrmse_pct=float(b.nrmse_pct)))
p=pd.DataFrame(paired);means=[]
for name,g in p.groupby('scenario'):
 means.append(dict(scenario=name,n_seeds=3,VF_nmae_pct_mean=float(g.VF_nmae_pct.mean()),L1_nmae_pct_mean=float(g.L1_nmae_pct.mean()),VF_minus_L1_nmae_percentage_points_mean=float(g.VF_minus_L1_nmae_percentage_points.mean()),VF_minus_L1_nmae_percentage_points_sd=float(g.VF_minus_L1_nmae_percentage_points.std(ddof=1)),contribution_to_39_scenario_mean_gap_percentage_points=float(g.VF_minus_L1_nmae_percentage_points.mean()/39)))
means.sort(key=lambda x:x['VF_minus_L1_nmae_percentage_points_mean'],reverse=True)
seeds=[]
for seed,g in p.groupby('seed'):
 top=g.sort_values('VF_minus_L1_nmae_percentage_points',ascending=False).iloc[0]
 seeds.append(dict(seed=int(seed),VF_nmae_pct_mean39=float(g.VF_nmae_pct.mean()),L1_nmae_pct_mean39=float(g.L1_nmae_pct.mean()),VF_minus_L1_nmae_percentage_points_mean39=float(g.VF_minus_L1_nmae_percentage_points.mean()),VF_better_scenarios=int((g.VF_minus_L1_nmae_percentage_points<0).sum()),worst_scenario=top.scenario,worst_scenario_gap_percentage_points=float(top.VF_minus_L1_nmae_percentage_points),worst_scenario_VF_nmae_pct=float(top.VF_nmae_pct),worst_scenario_L1_nmae_pct=float(top.L1_nmae_pct)))
positive=sum(x['contribution_to_39_scenario_mean_gap_percentage_points'] for x in means if x['VF_minus_L1_nmae_percentage_points_mean']>0)
negative=sum(x['contribution_to_39_scenario_mean_gap_percentage_points'] for x in means if x['VF_minus_L1_nmae_percentage_points_mean']<0)
assert len(means)==39 and len(seeds)==3 and np.isclose(positive+negative,p.VF_minus_L1_nmae_percentage_points.mean())
out=dict(status='PASS',target_arrays_identical_across_three_seeds=True,target_distribution_counted_once=True,source_prediction_hashes=sources,target_distributions=targets,event_residual_context=residual_context,h96_nmae_scenarios_ranked=means,h96_nmae_per_seed=seeds,h96_nmae_mean_gap_decomposition=dict(adverse_scenarios=sum(x['VF_minus_L1_nmae_percentage_points_mean']>0 for x in means),favourable_scenarios=sum(x['VF_minus_L1_nmae_percentage_points_mean']<0 for x in means),adverse_contribution_percentage_points=positive,favourable_contribution_percentage_points=negative,net_gap_percentage_points=positive+negative),worst_seed_by_mean_gap=max(seeds,key=lambda x:x['VF_minus_L1_nmae_percentage_points_mean39']),worst_seed_by_absolute_VF_error=max(seeds,key=lambda x:x['VF_nmae_pct_mean39']),new_statistical_tests=False)
ref.atomic_json(L/'additional_diagnostics.json',out)
for name,rows in [('event_target_distributions.csv',targets),('event_residual_context.csv',residual_context),('h96_nmae_scenarios_ranked.csv',means),('h96_nmae_per_seed.csv',seeds),('h96_nmae_all_seed_scenarios.csv',paired)]:pd.DataFrame(rows).to_csv(L/name,index=False,encoding='utf-8-sig')
print(json.dumps({k:v for k,v in out.items() if k not in ('h96_nmae_scenarios_ranked','event_residual_context')},ensure_ascii=False))
