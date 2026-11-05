from run_physics import *
from pieef.models.physics import LearnableBSplineCurve
def run():
 out=[];between=[]
 for r in records():
  h,seed=r['horizon'],r['seed'];d=pd.read_csv(OUT/f'curves_h{h}_s{seed}.csv');w=torch.load(Path(r['path'])/'weights_best.pt',map_location='cpu',weights_only=True)
  cpstate={k.removeprefix('dcn.power_curve.'):v for k,v in w.items() if k.startswith('dcn.power_curve.')}
  c=LearnableBSplineCurve(29);c.load_state_dict(cpstate,strict=True);cp=c.get_control_points().detach().double().numpy();assert np.all(np.diff(cp,axis=1)>=0)
  v=np.sort(d.wind_ms.unique());t=np.minimum(np.maximum(v,0),c.max_ws-.01)/c.grid_size;idx=np.floor(t).astype(int);z=t-idx
  with torch.no_grad():base_direct=c(torch.tensor(v,dtype=torch.float32).reshape(-1,1).expand(-1,29)).numpy()
  stable=[]
  for j,tid in enumerate(sorted(d.turbine.unique())):
   # Exact derivative in nonnegative adjacent-control-point differences.
   dc=np.diff(cp[j]);der=(.5*(1-z)**2*dc[idx]+(-z*z+z+.5)*dc[idx+1]+.5*z*z*dc[idx+2])/c.grid_size
   der[(v<0)|(v>c.max_ws-.01)]=0
   g=d[d.turbine==tid].sort_values('wind_ms');observed=g.derivative_pct_per_ms.to_numpy()/100
   assert np.max(abs(der-observed))<1e-6
   stable.extend(100*der)
   out.append(dict(horizon=h,seed=seed,turbine=tid,stable_negative_derivative_pct=100*np.mean(der<0),float32_negative_derivative_pct=100*np.mean(observed<0),float32_min_derivative_pu_per_ms=observed.min(),stable_min_derivative_pu_per_ms=der.min(),max_autograd_vs_stable_pu_per_ms=np.max(abs(der-observed)),min_adjacent_cp_increment=dc.min(),max_base_decrease_pu=max(0,float(-np.diff(base_direct[:,j]).min()))))
  d['stable_derivative_pct_per_ms']=stable;d.to_csv(OUT/f'curves_h{h}_s{seed}.csv',index=False,encoding='utf-8-sig')
  pv=d.pivot(index='wind_ms',columns='turbine',values='base_pct');spread=pv.max(axis=1)-pv.min(axis=1)
  between.append(dict(horizon=h,seed=seed,max_turbine_spread_capacity_pp=spread.max(),wind_at_max_spread_ms=spread.idxmax(),base_at_zero_min_pct=pv.loc[0].min(),base_at_zero_max_pct=pv.loc[0].max()))
 csv('P1_monotonicity_crosscheck.csv',out);csv('P1_between_turbines.csv',between)
 progress('P1 单调性交叉核验完成：保留 FP32 严格负值比例，同时用非负控制点增量公式核验导数。')
if __name__=='__main__':run()
