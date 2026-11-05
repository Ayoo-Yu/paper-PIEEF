from common import *
from ti_analysis import permitted
import torch,importlib.util
torch.set_num_threads(2)

def crossing(v,y,level):
    ok=np.isfinite(v)&np.isfinite(y);v=np.asarray(v)[ok];y=np.asarray(y)[ok]
    above=np.flatnonzero(y>=level)
    if not len(above):return np.nan
    i=above[0]
    if i==0:return float(v[0])
    return float(v[i-1]+(level-y[i-1])/(y[i]-y[i-1])*(v[i]-v[i-1]))

def run(k,runs,cap,d):
    if not d['yaw_gate']:
        csv('k5_learned_curves.csv',[dict(status='BLOCKED_RAW_ALIGNMENT')]);csv('k5_curve_shift.csv',[dict(status='BLOCKED_RAW_ALIGNMENT')]);return
    p=register(PROJECT/'src/pieef/models/physics.py');register(PROJECT/'src/pieef/models/e2e.py');register(PROJECT/'src/pieef/model_factory.py')
    sp=importlib.util.spec_from_file_location('k_frozen_physics',p);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
    raw=d['raw'];emp={};curves=[];shifts=[]
    for j in range(6):
        for label in ['all_normal','aligned_lt2']:
            good=raw['train']&raw['normal'][:,j]
            if label=='aligned_lt2':good &= abs(raw['phi'][:,j])<2
            v=[];pw=[]
            for b in np.arange(0,30,.5):
                mask=good&(raw['ws'][:,j]>=b)&(raw['ws'][:,j]<b+.5);n=int(mask.sum());mean=float(raw['power'][mask,j].mean()/1000) if n else np.nan
                curves.append(dict(curve_type='empirical_'+label,turbine=j+1,horizon=np.nan,seed=np.nan,wind_ms=b+.25,wind_lower_ms=b,wind_upper_ms=b+.5,n_records=n,power_mw=mean,power_pu=mean/cap[j],used_for_crossing=n>0,status='PASS',wind_coordinate='measured_nacelle_anemometer_ms'))
                if n>0:v.append(b+.25);pw.append(mean/cap[j])
            emp[j,label]=dict(v50=crossing(v,pw,.5),v90=crossing(v,pw,.9),n_records=int(good.sum()))
    dist=pd.read_csv(O/'k4_yaw_distribution.csv');dist=dist[dist.period=='train'].set_index('turbine')
    monotonic=[]
    for (model,h,s),z in runs.items():
        if model!='VF':continue
        if not permitted(z,model):
            for j in range(6):
                shifts.append(dict(horizon=h,seed=s,turbine=j+1,status='BLOCKED_GATE'))
                curves.append(dict(curve_type='learned',horizon=h,seed=s,turbine=j+1,status='BLOCKED_GATE',wind_coordinate='calibrated_wind_speed_ms'))
            continue
        path=register(control(h,s)/'source_weights_best.pt');source=readjson(control(h,s)/'source_control.json')
        assert sha(path)==source['original_checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        points={key.removeprefix('dcn.power_curve.'):value for key,value in state.items() if key.startswith('dcn.power_curve.')}
        cfg=source['config']['model'];maxws=max(25.,float(state['dcn.cut_out_speeds'].max()))
        curve=mod.LearnableBSplineCurve(6,int(cfg['num_control_points']),maxws);curve.load_state_dict(points,strict=True);curve.eval()
        # P1 calls the B-spline directly in physical m/s. The model's calibration
        # receives de-normalized physical wind; do not denormalize this axis twice.
        v=torch.linspace(0,30,301).reshape(-1,1).expand(-1,6).clone().requires_grad_(True)
        base=curve(v);der=torch.autograd.grad(base.sum(),v)[0].detach().numpy();values=base.detach().numpy();x=v.detach().numpy()[:,0]
        cp=curve.get_control_points().detach().double().numpy();t=np.clip(x.astype(float),0,curve.max_ws-.01)/curve.grid_size;idx=np.floor(t).astype(int);u=t-idx
        for j in range(6):
            dc=np.diff(cp[j]);stable=(.5*(1-u)**2*dc[idx]+(-u*u+u+.5)*dc[idx+1]+.5*u*u*dc[idx+2])/curve.grid_size;stable[(x<0)|(x>curve.max_ws-.01)]=0
            delta=float(np.max(abs(stable-der[:,j])));passed=bool(np.all(dc>=0)&np.all(stable>=0)&(delta<1e-6))
            mono=dict(horizon=h,seed=s,turbine=j+1,min_cp_increment=dc.min(),stable_min_derivative_pu_per_ms=stable.min(),stable_negative_derivative_pct=100*np.mean(stable<0),float32_negative_derivative_pct=100*np.mean(der[:,j]<0),float32_min_derivative_pu_per_ms=der[:,j].min(),max_autograd_vs_stable_pu_per_ms=delta,max_forward_decrease_pu=max(0,float(-np.diff(values[:,j]).min())),passed=passed)
            monotonic.append(mono)
            v50=crossing(x,values[:,j],.5);v90=crossing(x,values[:,j],.9)
            row=dict(horizon=h,seed=s,turbine=j+1,status='PASS',learned_v50_ms=v50,learned_v90_ms=v90,training_abs_phi_median_deg=dist.loc[j+1,'abs_phi_median_deg'],training_fraction_abs_phi_gt10=dist.loc[j+1,'fraction_abs_phi_gt10'],coordinate_caveat='learned: calibrated wind m/s; empirical: nacelle anemometer m/s; differences are not identified yaw effects',**{q:val for q,val in mono.items() if q not in ['horizon','seed','turbine']})
            for label in ['all_normal','aligned_lt2']:
                for level in ['v50','v90']:
                    ev=emp[j,label][level];lv=v50 if level=='v50' else v90
                    row[label+'_'+level+'_ms']=ev;row['learned_minus_'+label+'_'+level+'_ms']=lv-ev
            shifts.append(row)
            for i,xx in enumerate(x):curves.append(dict(curve_type='learned',horizon=h,seed=s,turbine=j+1,wind_ms=xx,power_pu=values[i,j],power_mw=values[i,j]*cap[j],autograd_derivative_pu_per_ms=der[i,j],stable_derivative_pu_per_ms=stable[i],status='PASS',wind_coordinate='calibrated_wind_speed_ms'))
    csv('k5_learned_curves.csv',curves);csv('k5_curve_shift.csv',shifts)
    k['k5_extraction']=dict(expected_runs=9,expected_curves=54,executed_runs=len(monotonic)//6,executed_curves=len(monotonic),monotonic_pass=sum(x['passed'] for x in monotonic),method='Frozen LearnableBSplineCurve submodule loaded strict=True from source_weights_best.pt; P1 physical 0..30 m/s grid, step 0.1; autograd and nonnegative control-point derivative cross-check; no optimization',wind_conversion='Direct curve input is already m/s. End-to-end code de-normalizes NWP then adds calibration; no second SCADA-standardization conversion applied.',empirical_method='training raw normal data, all populated 0.5 m/s bins; first piecewise-linear 50/90% capacity crossing, no smoothing or extrapolation; absent crossing = NaN',base_curve='P1 B-spline base curve at unit density, without cut-in/cut-out gates; not a forecast output',monotonicity=monotonic)
    save_k0(k);log(f'K5 extracted {len(monotonic)}/54 curves; stable monotonicity passed {sum(x["passed"] for x in monotonic)}/{len(monotonic)}')

def refresh_crossings_from_csv():
    """Recalculate only crossing summaries from complete saved curve points."""
    checkpoint();t=begin('python -B scripts/learned_curves.py --refresh-crossings')
    curves=pd.read_csv(O/'k5_learned_curves.csv');shifts=pd.read_csv(O/'k5_curve_shift.csv');k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf8'))
    empirical=curves.curve_type.str.startswith('empirical_');curves.loc[empirical,'used_for_crossing']=curves.loc[empirical,'n_records']>0
    changed=[]
    for j in range(1,7):
        for label in ['all_normal','aligned_lt2']:
            g=curves[(curves.turbine==j)&(curves.curve_type=='empirical_'+label)&(curves.n_records>0)].sort_values('wind_ms')
            for level,threshold in [('v50',.5),('v90',.9)]:
                value=crossing(g.wind_ms.to_numpy(),g.power_pu.to_numpy(),threshold);mask=(shifts.turbine==j)&(shifts.status=='PASS')
                old=shifts.loc[mask,label+'_'+level+'_ms'].to_numpy();changed.append(float(np.nanmax(abs(old-value))) if len(old) else 0)
                shifts.loc[mask,label+'_'+level+'_ms']=value;shifts.loc[mask,'learned_minus_'+label+'_'+level+'_ms']=shifts.loc[mask,'learned_'+level+'_ms']-value
    curves.to_csv(O/'k5_learned_curves.csv',index=False,encoding='utf-8-sig');shifts.to_csv(O/'k5_curve_shift.csv',index=False,encoding='utf-8-sig')
    k['k5_extraction']['empirical_method']='training raw normal data, all populated 0.5 m/s bins; first piecewise-linear 50/90% capacity crossing, no smoothing or extrapolation; absent crossing = NaN'
    incident('k5_empirical_support_repair','交付前检查发现脚本曾把 K4 P_ref 的至少30条条件延用于 K5 经验曲线，冻结方案未要求此限制。','已取消 K5 的额外计数门槛，使用全部有观测分箱；从完整曲线 CSV 重算 v50/v90，最大改变量记录于本事件。K4 的冻结30条规则不变。','execution_repair')
    next(x for x in INCIDENTS if x['code']=='k5_empirical_support_repair')['max_crossing_change_ms']=max(changed)
    js('operational_incidents.json',INCIDENTS);save_k0(k);end(t,'refresh_crossings')

if __name__=='__main__':
    if '--refresh-crossings' not in sys.argv:raise SystemExit('Use --refresh-crossings, or run_analysis.py for full extraction')
    refresh_crossings_from_csv()
