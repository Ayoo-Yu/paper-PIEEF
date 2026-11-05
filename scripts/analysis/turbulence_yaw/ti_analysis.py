from common import *

def permitted(z,model):return z['pass_gate'] and (model!='VF' or z.get('identity_pass',False))
def locate(d,a):
    ix=d['ts'].get_indexer(pd.to_datetime(a['timestamp'],utc=True));assert (ix>=0).all();return ix
def seed_summaries(rows,keys,values,required=3):
    frame=pd.DataFrame(rows);out=[]
    for key,g in frame[frame.row_type=='seed'].groupby(keys,dropna=False,sort=False):
        if not isinstance(key,tuple):key=(key,)
        ok=g[g.status=='PASS'];row=dict(zip(keys,key));row.update(row_type='seed_summary',seed='summary',n_seeds=len(ok),status='PASS' if len(ok)==required else 'BLOCKED_INCOMPLETE_SEEDS')
        row['available_seeds']=','.join(str(x) for x in ok.seed)
        for v in values:
            row[v+'_mean']=float(ok[v].mean()) if len(ok)==required else np.nan
            row[v+'_sd']=float(ok[v].std(ddof=1)) if len(ok)==required else np.nan
        if 'n_valid' in ok:row['n_valid_per_seed']=','.join(str(int(x)) for x in ok.n_valid)
        out.append(row)
    return out

def stratified(runs,cap,d,yaw=False):
    layers=YAW_LAYERS if yaw else TI_LAYERS;rows=[]
    for (model,h,s),z in runs.items():
        a=z['a'];ix=locate(d,a)
        masks=masks_yaw(d['phi'][ix]) if yaw else masks_ti(d['ws'][ix],d['std'][ix])[1]
        ok=permitted(z,model) and (not yaw or d['yaw_gate'])
        for layer in layers:
            r=dict(row_type='seed',model=model,horizon=h,seed=s,stratum=layer,status='PASS' if ok else 'BLOCKED_GATE')
            if ok:r.update(metrics(a['y_true_mw'],a['y_pred_mw'],a['target_valid']&masks[layer],cap),n_time_with_valid=int((a['target_valid']&masks[layer]).any(axis=1).sum()))
            else:r.update(n_valid=np.nan,nrmse_pct=np.nan,nmae_pct=np.nan)
            rows.append(r)
    summaries=seed_summaries(rows,['model','horizon','stratum'],['nrmse_pct','nmae_pct'])
    lookup={(r['model'],r['horizon'],r['seed'],r['stratum']):r for r in rows};comparisons=[]
    for h in HORIZONS:
        for b in [('DLinear' if h in (1,4) else 'XGBoost'),'Persistence']:
            for layer in layers:
                for metric in ('nrmse','nmae'):
                    pairs=[]
                    for s in SEEDS:
                        v=lookup['VF',h,s,layer];base=lookup[b,h,s,layer]
                        if v['status']==base['status']=='PASS':
                            p=100*(1-v[metric+'_pct']/base[metric+'_pct']);pairs.append((s,v[metric+'_pct'],base[metric+'_pct'],p))
                            comparisons.append(dict(row_type='paired_seed_comparison',model='VF',baseline=b,horizon=h,seed=s,stratum=layer,metric=metric,VF_improvement_pct=p,VF_favorable=v[metric+'_pct']<base[metric+'_pct'],status='PASS'))
                    complete=len(pairs)==3
                    comparisons.append(dict(row_type='comparison_summary',model='VF',baseline=b,horizon=h,seed='summary',stratum=layer,metric=metric,status='PASS' if complete else 'BLOCKED_INCOMPLETE_SEEDS',n_paired_seeds=len(pairs),available_seeds=','.join(str(x[0]) for x in pairs),VF_favorable_available_seeds=sum(x[1]<x[2] for x in pairs),VF_improvement_pct=100*(1-np.mean([x[1] for x in pairs])/np.mean([x[2] for x in pairs])) if complete else np.nan))
    csv('k6_yaw_strata_errors.csv' if yaw else 'k1_ti_strata_errors.csv',rows+summaries+comparisons)

def power_curves(d):
    rows=[];ws=d['ws'];power=d['sc'][[f'power{i}' for i in range(1,7)]].to_numpy(float)/1000
    training=np.arange(len(ws))[:,None]<d['train_end'];base=d['normal']&training
    for tid in ['pooled',*range(1,7)]:
        turbine=np.ones_like(base) if tid=='pooled' else np.broadcast_to(np.arange(1,7)==tid,base.shape)
        for b in range(3,15):
            for layer,m in d['ti_masks'].items():
                good=base&turbine&(ws>=b)&(ws<b+1)&m
                rows.append(dict(turbine=tid,stratum=layer,wind_lower_ms=b,wind_upper_ms=b+1,wind_center_ms=b+.5,n_records=int(good.sum()),mean_power_mw=float(power[good].mean()) if good.any() else np.nan,sampling='15min_grid_actual_power_targets',normal_filter='original_events_normal_and_no_stop_derating'))
    csv('k2_ti_power_curves.csv',rows)

def residuals_oracle(k,runs,cap,d):
    rows=[];orows=[]
    for (model,h,s),z in runs.items():
        if model!='VF':continue
        if not permitted(z,model):
            rows.append(dict(row_type='seed',horizon=h,seed=s,split='test',stratum='BLOCKED',wind_lower_ms=np.nan,status='BLOCKED_GATE',mean_residual_mw=np.nan))
            orows.append(dict(row_type='seed',horizon=h,seed=s,status='BLOCKED_GATE'));continue
        a=z['a'];val=z['val'];it=locate(d,a);iv=locate(d,val)
        for pred,ix in [(a,it),(val,iv)]:
            assert np.array_equal(pred['target_valid'],d['valid'][ix]);assert np.array_equal(pred['events'],d['events'][ix])
        normal_t=a['target_valid']&d['normal'][it];normal_v=val['target_valid']&d['normal'][iv]
        t_masks=masks_ti(d['ws'][it],d['std'][it])[1];v_masks=masks_ti(d['ws'][iv],d['std'][iv])[1]
        correction=np.zeros_like(a['y_pred_mw']);corrected=np.zeros_like(normal_t)
        for split,pred,ix,normal,ms in [('val',val,iv,normal_v,v_masks),('test',a,it,normal_t,t_masks)]:
            residual=pred['y_true_mw']-pred['y_pred_mw']
            for b in range(3,16):
                wind=d['ws'][ix];speed=(wind>=b)&((wind<b+1) if b<15 else np.isfinite(wind))
                for layer in TI_LAYERS:
                    good=normal&speed&ms[layer];n=int(good.sum());mean=float(residual[good].mean()) if n else np.nan
                    usable=split=='val' and b>=4 and layer in ['low','medium','high'] and n>=50
                    rows.append(dict(row_type='seed',horizon=h,seed=s,split=split,stratum=layer,wind_lower_ms=b,wind_upper_ms=b+1 if b<15 else np.inf,n_valid=n,mean_residual_mw=mean,correction_eligible=usable,status='PASS',scope='target_valid_and_events_normal_and_no_stop_derating'))
                    if usable:
                        tw=d['ws'][it];hit=normal_t&t_masks[layer]&(tw>=b)&((tw<b+1) if b<15 else np.isfinite(tw))
                        correction[hit]=mean;corrected[hit]=True
        before=metrics(a['y_true_mw'],a['y_pred_mw'],normal_t,cap)
        # E3: same wind bins, normal masks and minimum count; pool TI only for (a).
        ws_correction=np.zeros_like(a['y_pred_mw']);ws_corrected=np.zeros_like(normal_t)
        vr=val['y_true_mw']-val['y_pred_mw'];vw=d['ws'][iv];tw=d['ws'][it]
        for b in range(4,16):
            vg=normal_v&(vw>=b)&((vw<b+1) if b<15 else np.isfinite(vw))
            if int(vg.sum())>=50:
                hit=normal_t&(tw>=b)&((tw<b+1) if b<15 else np.isfinite(tw))
                ws_correction[hit]=float(vr[vg].mean());ws_corrected[hit]=True
        after_a=metrics(a['y_true_mw'],a['y_pred_mw']+ws_correction,normal_t,cap)
        after_b=metrics(a['y_true_mw'],a['y_pred_mw']+correction,normal_t,cap)
        for label,after,mask in [('ws_only',after_a,ws_corrected),('ws_x_ti',after_b,corrected)]:
            row=dict(row_type='seed',horizon=h,horizon_min=15*h,seed=s,correction=label,status='PASS',n_valid=before['n_valid'],n_test=before['n_valid'],n_corrected=int(mask.sum()),n_uncorrected=int(normal_t.sum()-mask.sum()),formula='pred_corrected = pred + mean_validation(y_true - y_pred)',scope='normal_operation_test_targets',uses_unavailable_target_time_information=True)
            for metric in ('nrmse','nmae'):
                x=before[metric+'_pct'];y=after[metric+'_pct'];av=after_a[metric+'_pct']
                row.update({metric+'_before_pct':x,metric+'_after_pct':y,metric+'_before':x,metric+'_after':y,metric+'_change_pp':y-x,metric+'_relative_change_pct':100*(y/x-1),metric+'_relative_improvement_pct':100*(1-y/x),metric+'_ti_increment_change_pp':y-av if label=='ws_x_ti' else np.nan,metric+'_ti_increment_relative_change_pct':100*(y/av-1) if label=='ws_x_ti' else np.nan,metric+'_ti_increment_relative_improvement_pct':100*(1-y/av) if label=='ws_x_ti' else np.nan})
            orows.append(row)
    # Preserve blocked cells rather than averaging the two remaining seeds.
    rr=pd.DataFrame(rows)
    for key,g in rr[(rr.status=='PASS')&(rr.split=='test')].groupby(['horizon','stratum','wind_lower_ms'],sort=False):
        rows.append(dict(row_type='seed_summary',horizon=key[0],seed='summary',split='test',stratum=key[1],wind_lower_ms=key[2],n_seeds=len(g),status='PASS' if len(g)==3 else 'BLOCKED_INCOMPLETE_SEEDS',mean_residual_mw_mean=g.mean_residual_mw.mean() if len(g)==3 else np.nan,mean_residual_mw_sd=g.mean_residual_mw.std(ddof=1) if len(g)==3 else np.nan))
    values=[m+suff for m in ['nrmse','nmae'] for suff in ['_before_pct','_after_pct','_change_pp','_relative_change_pct','_relative_improvement_pct']]
    values += [m+suff for m in ['nrmse','nmae'] for suff in ['_ti_increment_change_pp','_ti_increment_relative_change_pct','_ti_increment_relative_improvement_pct']]
    summaries=seed_summaries(orows,['horizon','correction'],values)
    for r in summaries:r['horizon_min']=15*r['horizon']
    orows+=summaries
    # Reference only: retain the six actually reported v1 subtract rows, not new trials.
    old=pd.read_csv(register(V1/'k2_oracle_bound.csv'),dtype=str)
    for _,r in old[(old.row_type=='seed')&(old.status=='PASS')].iterrows():
        row=r.dropna().to_dict();row.update(correction='superseded_subtract',horizon_min=15*int(r.horizon),n_test=int(float(r.n_valid)),status='SUPERSEDED_REFERENCE',source='v1/k2_oracle_bound.csv')
        for m in ['nrmse','nmae']:
            row[m+'_before']=r[m+'_before_pct'];row[m+'_after']=r[m+'_after_pct']
        orows.append(row)
    csv('k2_ti_residuals.csv',rows);csv('k2_oracle_bound.csv',orows)
    k['k2_corrigendum']=dict(residual='y_true - y_pred',correction='prediction + validation cell mean residual',comparisons=['ws_only','ws_x_ti'],ti_increment_change='metric(b) - metric(a), percentage points',ti_increment_relative_change='100*(metric(b)/metric(a)-1)',positive_change_means='worse',relative_improvement='negative of relative change',reference='six existing v1 subtract rows only; superseded, excluded from summaries',minimum_validation_count=50,no_correction_below_ws_ms=4)

def attention(k,runs,d):
    rows=[]
    for (model,h,s),z in runs.items():
        if model!='VF':continue
        good=permitted(z,model)
        a=npz(control(h,s)/'test_attention_diagnostics.npz');ix=locate(d,a);ms=masks_ti(d['ws'][ix],d['std'][ix])[1]
        for layer in TI_LAYERS:
            for metric in ['normalized_entropy','maximum_weight','dsi']:
                mask=ms[layer];valid=mask&np.isfinite(a[metric])
                rows.append(dict(row_type='seed',site='Kelmarsh',horizon=h,seed=s,stratum=layer,metric=metric,status='PASS' if good else 'BLOCKED_GATE',n_queries=int(mask.sum()) if good else np.nan,n_finite=int(valid.sum()) if good else np.nan,n_undefined=int((mask&~np.isfinite(a[metric])).sum()) if good else np.nan,attention_mean=float(a[metric][valid].mean()) if good and valid.any() else np.nan,scope='all_query_turbines_in_saved_test_samples_not_restricted_by_target_valid'))
    rows+=seed_summaries(rows,['site','horizon','stratum','metric'],['attention_mean'])
    # Reprint existing wind_farm_A regime results for the same three declared seeds;
    # no inference or new wind_farm_A stratification is performed.
    register(W/'交付_20261021/01_实验代码/p4_regime_attention.py')
    for s in SEEDS:
        p=W/f'交付_20261021/02_实验结果/p4reg_vf_seed{s}.json';old=readjson(p)
        for regime,stats in old['regimes'].items():
            for metric,key in [('normalized_entropy','mean_entropy'),('maximum_weight','mean_max_weight')]:
                rows.append(dict(row_type='historical_reference',site='wind_farm_A',horizon=16,seed=s,stratum=regime,metric=metric,status='EXISTING_RESULT_NOT_REANALYZED',n_queries=stats['n'],attention_mean=stats[key],scope='historical time-sample majority event label; per-time all-query average; n is time samples, not turbine cells',source=str(p)))
    k['wind_farm_A_attention_comparison']=dict(source='existing p4reg_vf_seed13/42/3407.json, test',same_entropy_definition='-sum(a log a)/log(number of turbines), all neighbors including self; max actual weight',weighting_difference='wind_farm_A historical regimes use majority event label per timestamp and average over all turbines; Kelmarsh TI is per query turbine at target time',dsi_in_historical_regime_files=False,no_new_wind_farm_A_seeds_or_inference=True)
    csv('k3_attention_by_ti.csv',rows)

def run(k,runs,cap,d):
    stratified(runs,cap,d);power_curves(d);residuals_oracle(k,runs,cap,d);attention(k,runs,d);stratified(runs,cap,d,yaw=True)
    k['stratification_coverage']=[]
    for (model,h,s),z in runs.items():
        if model!='VF':continue
        ix=locate(d,z['a']);valid=z['a']['target_valid'];ti,ms=masks_ti(d['ws'][ix],d['std'][ix]);covered=ms['low']|ms['medium']|ms['high']|ms['undefined'];ym=masks_yaw(d['phi'][ix]);yc=np.logical_or.reduce(list(ym.values()))
        k['stratification_coverage'].append(dict(horizon=h,seed=s,n_target_valid=int(valid.sum()),n_missing_ti_covariates=int((valid&~covered).sum()),n_missing_yaw=int((valid&~yc).sum()),very_high_is_subset_of_high=True))
    save_k0(k);log('K1/K2/K3/K6 completed for eligible runs; blocked seeds retained as explicit empty slots.')
