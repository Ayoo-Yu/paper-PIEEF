from common import *
from scipy.optimize import least_squares

def fit_p(ratio,eligible):
    x=np.log(np.cos(np.deg2rad(np.arange(1,30,2)[eligible])));y=ratio[eligible]
    if len(y)<1 or not np.isfinite(y).all():return np.nan
    fit=least_squares(lambda p:np.exp(x*p[0])-y,x0=[2.0],jac=lambda p:(x*np.exp(x*p[0]))[:,None],max_nfev=1000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    return float(fit.x[0]) if fit.success else np.nan

def distribution(d):
    rows=[]
    for period,select in [('train',np.arange(len(d['ts']))<d['train_end']),('test',np.arange(len(d['ts']))>=d['val_end'])]:
        for j in range(6):
            a=d['phi'][select,j];a=a[np.isfinite(a)];q=np.quantile(a,[.25,.5,.75]);ab=abs(a)
            rows.append(dict(period=period,turbine=j+1,n_records=len(a),phi_median_deg=q[1],phi_q25_deg=q[0],phi_q75_deg=q[2],phi_iqr_deg=q[2]-q[0],abs_phi_median_deg=np.median(ab),abs_phi_p90_deg=np.quantile(ab,.9),fraction_abs_phi_gt10=np.mean(ab>10),sampling='all_finite_15min_grid_vane_no_normal_filter'))
    csv('k4_yaw_distribution.csv',rows)

def power_ratio(k,d):
    raw=d['raw'];grid=raw['timestamp'];train=raw['train'];days=grid[train].normalize().unique();nd=len(days)
    day=days.get_indexer(grid.normalize());nb=14;nc=19
    rng=np.random.default_rng(2026)
    weights=rng.multinomial(nd,np.full(nd,1/nd),size=1000).astype(float)
    ratios=[];fitrows=[];refs=[]
    for name in ['own_nacelle','other_five_median']:
        wind=raw['ws'].copy() if name=='own_nacelle' else np.column_stack([np.median(np.delete(raw['ws'],j,axis=1),axis=1) for j in range(6)])
        nref=np.zeros((nd,6,nb));sref=np.zeros_like(nref)
        counts=np.zeros((nd,6,nb,nc));sums=np.zeros_like(counts)
        excluded=[]
        for j in range(6):
            a=abs(raw['phi'][:,j]);v=wind[:,j];p=raw['power'][:,j]/1000
            good=train&raw['normal'][:,j]&np.isfinite(a)&np.isfinite(v)&(v>=4)&(v<11)
            ix=np.flatnonzero(good);bi=np.floor((v[ix]-4)/.5).astype(int);di=day[ix]
            ref=a[ix]<2
            np.add.at(nref[:,j],(di[ref],bi[ref]),1);np.add.at(sref[:,j],(di[ref],bi[ref]),p[ix][ref])
            for c in range(15):
                m=(a[ix]>=2*c)&(a[ix]<2*c+2)
                np.add.at(counts[:,j,:,c],(di[m],bi[m]),1);np.add.at(sums[:,j,:,c],(di[m],bi[m]),p[ix][m])
            for c,m in enumerate(masks_yaw(raw['phi'][ix,j]).values()):
                np.add.at(counts[:,j,:,15+c],(di[m],bi[m]),1);np.add.at(sums[:,j,:,15+c],(di[m],bi[m]),p[ix][m])
            excluded.append(dict(turbine=j+1,n_normal_training=int((train&raw['normal'][:,j]).sum()),n_speed_range_with_phi=int(good.sum()),n_missing_reference_speed=int((train&raw['normal'][:,j]&~np.isfinite(v)).sum())))
        nr=nref.sum(0);sr=sref.sum(0);ref=np.divide(sr,nr,out=np.full_like(sr,np.nan),where=nr>=30);eligible=np.isfinite(ref)&(ref>0)
        ct=counts.sum(0);sp=sums.sum(0);den=(ct*eligible[...,None]).sum((0,1))
        numer=np.where(eligible[...,None],sp/np.where(eligible,ref,1)[...,None],0).sum((0,1));mean=np.divide(numer,den,out=np.full(nc,np.nan),where=den>0)
        for j in range(6):
            for b in range(nb):refs.append(dict(reference_speed=name,turbine=j+1,wind_lower_ms=4+.5*b,wind_upper_ms=4.5+.5*b,n_aligned_records=int(nr[j,b]),reference_power_mw=ref[j,b],eligible=eligible[j,b],n_ratio_records=int(ct[j,b,15:].sum()) if eligible[j,b] else 0))
        # Re-estimate reference curves in every day-block bootstrap replicate.
        brn=(weights @ nref.reshape(nd,-1)).reshape(1000,6,nb);brs=(weights @ sref.reshape(nd,-1)).reshape(1000,6,nb)
        br=np.divide(brs,brn,out=np.full_like(brs,np.nan),where=brn>=30)
        be=eligible[None]&np.isfinite(br)&(br>0)
        bc=(weights @ counts.reshape(nd,-1)).reshape(1000,6,nb,nc);bs=(weights @ sums.reshape(nd,-1)).reshape(1000,6,nb,nc)
        bden=(bc*be[...,None]).sum((1,2));bnum=np.where(be[...,None],bs/np.where(be,br,1)[...,None],0).sum((1,2));boot=np.divide(bnum,bden,out=np.full_like(bnum,np.nan),where=bden>0)
        for c in range(nc):
            finite=boot[:,c][np.isfinite(boot[:,c])];ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            ratios.append(dict(reference_speed=name,bin_type='two_degree' if c<15 else 'stratum',stratum=f'{2*c}to{2*c+2}' if c<15 else YAW_LAYERS[c-15],abs_phi_lower_deg=2*c if c<15 else [0,5,10,20][c-15],abs_phi_upper_deg=2*c+2 if c<15 else [5,10,20,np.inf][c-15],abs_phi_center_deg=2*c+1 if c<15 else np.nan,n_records=int(den[c]),mean_power_ratio=mean[c],ci95_lower=ci[0],ci95_upper=ci[1],n_valid_bootstrap=len(finite),fit_eligible=bool(c<15 and den[c]>=200)))
        fiteligible=(den[:15]>=200)&np.isfinite(mean[:15]);p=fit_p(mean[:15],fiteligible);bp=np.array([fit_p(b[:15],fiteligible) for b in boot]);finite=bp[np.isfinite(bp)];ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
        fitrows.append(dict(reference_speed=name,p=p,ci95_lower=ci[0],ci95_upper=ci[1],eligible_two_degree_bins=np.flatnonzero(fiteligible).tolist(),eligible_bin_centers_deg=np.arange(1,30,2)[fiteligible],eligible_bin_counts=den[:15][fiteligible].astype(int),n_valid_bootstrap=len(finite),unconstrained_p=True,least_squares='unweighted bin means vs cos(bin midpoint)^p; same original n>=200 eligible bins in bootstrap',filter_counts=excluded))
        log(f'K4 {name}: p={p:.6g}, day bootstrap CI={ci.tolist()}, eligible bins={fiteligible.sum()}')
    csv('k4_yaw_power_ratio.csv',ratios)
    js('k4_cos_fit.json',dict(status='completed',bootstrap=dict(block='UTC calendar day jointly across all six turbines',n_days=nd,replicates=1000,seed=2026,method='multinomial day weights, percentile 95% interval; re-estimate P_ref per replicate; original eligible speed-bin support retained',minimum_reference_count=30,minimum_fit_bin_count=200),reference_curves=refs,fits=fitrows,wind_range_ms='[4,11)',reference_speed_alternative='median of all other five nacelle wind speeds; missing if any of five nonfinite',normal_filter='raw valid, no low-power candidate, no frozen wind, no overlapping Stop/derating; NO 3-sigma filter'))

def run(k,d):
    if not d['yaw_gate']:
        csv('k4_yaw_distribution.csv',[dict(status='BLOCKED_RAW_ALIGNMENT')]);csv('k4_yaw_power_ratio.csv',[dict(status='BLOCKED_RAW_ALIGNMENT')]);js('k4_cos_fit.json',dict(status='BLOCKED_RAW_ALIGNMENT'));return
    distribution(d);power_ratio(k,d)
