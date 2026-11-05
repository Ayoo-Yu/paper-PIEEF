from common import *
def run():
    register(W/'planning/第五轮补强方案_Kelmarsh湍流与偏航_20261024.md')
    for p in [W/'补做重算/tables/analyze_tables.py',W/'补做重算/physics/run_physics.py',W/'补做重算/physics/monotonicity_check.py',Path('results/paper/source/prepare_kelmarsh.py'),PROJECT/'src/pieef/revision_data.py',PROJECT/'src/pieef/r4_attention.py',W/'round4_prediction_20261024/postprocess/control_inference.py']:
        register(p)
    t=pd.read_csv(register(W/'补做重算/tables/T1_per_seed.csv'));t=t[t.site=='kelmarsh']
    turbines=pd.read_csv(register(DATA/'turbines.csv'));cap=turbines.capacity_mw.to_numpy()
    assert np.array_equal(cap,np.full(6,2.05))
    runs={};checks=[]; identities=[]
    for _,r in t.iterrows():
        for side,model in [('VF','VF'),('baseline',r.baseline)]:
            d=Path(r[side+'_dir']);key=(model,int(r.horizon),int(r.seed))
            if key not in runs:
                a=npz(d/'predictions.npz');pre=npz(d/'preprocessing.npz');meta=readjson(d/'metrics.json');protocol=readjson(d/'data_protocol.json')
                assert np.array_equal(a['turbine_ids'],np.arange(1,7).astype(str))
                assert np.array_equal(pre['capacity_mw'],cap)
                assert a['y_true_mw'].shape==a['y_pred_mw'].shape==a['target_valid'].shape
                assert np.array_equal(np.isfinite(a['y_true_mw']),a['target_valid'])
                assert np.isfinite(a['y_pred_mw'][a['target_valid']]).all()
                assert int(a['seed'])==r.seed and int(a['horizon'])==r.horizon
                ts=pd.to_datetime(a['timestamp'],utc=True);orig=pd.to_datetime(a['origin_timestamp'],utc=True)
                assert ts.is_unique and ts.is_monotonic_increasing and np.all(ts-orig==pd.Timedelta(minutes=15*r.horizon))
                runs[key]=dict(a=a,pre=pre,meta=meta,directory=d,pass_gate=True)
            z=runs[key];m=metrics(z['a']['y_true_mw'],z['a']['y_pred_mw'],z['a']['target_valid'],cap)
            for metric in ('nrmse','nmae'):
                value=m[metric+'_pct'];reported=float(r[f'{side}_{metric}_pct']);delta=abs(value-reported);passed=delta<=.001
                checks.append(dict(model=model,horizon=int(r.horizon),seed=int(r.seed),T1_baseline_row=r.baseline,metric=metric,reported_pct=reported,reproduced_pct=value,abs_difference_pp=delta,passed=passed,n_valid=m['n_valid']))
                z['pass_gate'] &= passed
    for (model,h,s),z in runs.items():
        if not z['pass_gate']:incident(f'metric_{model}_{h}_{s}','T1 指标不能在 0.001 个百分点内复现','停止该运行的分层分析','gate_failure')
        if model!='VF':continue
        a=z['a'];d=control(h,s);b=npz(d/'predictions.npz');val=npz(d/'val_predictions.npz')
        cp=npz(d/'preprocessing.npz');source=readjson(d/'source_control.json');register(d/'inference.log');readjson(d/'metrics.json')
        fields={k:bool(np.array_equal(a[k],b[k])) for k in ['timestamp','origin_timestamp','turbine_ids','target_valid','events']}
        fields['y_true_mw']=bool(np.array_equal(a['y_true_mw'],b['y_true_mw'],equal_nan=True))
        delta=float(np.max(abs(a['y_pred_mw']-b['y_pred_mw'])));passed=all(fields.values()) and delta<=float(cap[0])*1e-5
        identities.append(dict(horizon=h,seed=s,fields_equal=fields,max_abs_prediction_difference_mw=delta,tolerance_mw=float(cap[0])*1e-5,passed=passed))
        z['identity_pass']=passed;z['val']=val
        if not passed:incident(f'identity_{h}_{s}','VF 与 M_controls 身份门禁不通过','停止该 VF 运行所有 K1–K6 分析','gate_failure')
        for split,ref in [('test',b),('val',val)]:
            att=npz(d/(split+'_attention_diagnostics.npz'))
            for k in ['timestamp','origin_timestamp','turbine_ids']:assert np.array_equal(att[k],ref[k])
        assert np.array_equal(cp['capacity_mw'],cap)
        register(W/f'next_batch/final_configs_d/energy_final_multifarm_kelmarsh_PI-EEF_h{h}_seed{s}.yaml')
    # Exact training boundary is serialized. No rounded calendar approximation.
    bounds=[]
    for (model,h,s),z in runs.items():
        if model!='VF':continue
        pre=z['pre'];v=z['val'];a=z['a'];cfg=z['meta']['config']['data']
        bounds.append(dict(horizon=h,seed=s,split_config=cfg['split'],train_end_index_exclusive=int(pre['power_training_end_index']),train_start_utc=str(pre['power_training_start_timestamp']),train_last_grid_utc=str(pre['power_training_end_timestamp']),val_first_saved_target_utc=str(v['timestamp'][0]),val_last_saved_target_utc=str(v['timestamp'][-1]),test_first_saved_target_utc=str(a['timestamp'][0]),test_last_saved_target_utc=str(a['timestamp'][-1]),n_val_targets=len(v['timestamp']),n_test_targets=len(a['timestamp'])))
    k=dict(metric_definition='100*sqrt(mean(((prediction-truth)/capacity)^2)) and 100*mean(abs((prediction-truth)/capacity)), pooled valid turbine-target cells; sample SD across seeds uses ddof=1',capacity_mw=cap,metric_checks=checks,vf_identity_checks=identities,split_records=bounds,figure_backend='python',analysis_only=True,training_performed=False)
    save_k0(k)
    log(f"K0 metric comparisons {sum(c['passed'] for c in checks)}/{len(checks)}; max diff pp {max(c['abs_difference_pp'] for c in checks):.12g}; VF identities {sum(x['passed'] for x in identities)}/9; max diff MW {max(x['max_abs_prediction_difference_mw'] for x in identities):.12g}")
    return k,runs,cap
if __name__=='__main__':
    checkpoint();t0=begin('python -B scripts/k0_gates.py');run();end(t0,'k0_gates')
