"""v1 immutability, keyed numerical reproduction, and E1-E3 delivery checks."""
from common import *

def snapshot():
    checkpoint()
    for p in sorted(V1.rglob('*')):
        if p.is_file() and not p.is_relative_to(O):register(p)
    js('k0_identity_checks.json',dict(input_files=list(INPUTS.values())))
    incident('v2_preflight_display','首次 PowerShell 读取未指定 UTF-8，终端显示乱码；rg 不可用。','使用显式 UTF-8 重读完整方案并用原生目录查询；无输入改动。','operational')
    log('v1 files registered before rerun; no v1 files will be written.')

def verify(k):
    oldk=readjson(V1/'k0_identity_checks.json')
    audit=[];details=[]
    def numeric(a,b,path):
        if isinstance(a,dict):
            for key,v in a.items():
                if key in b:numeric(v,b[key],path+'/'+str(key))
        elif isinstance(a,list):
            assert len(a)==len(b),(path,len(a),len(b))
            for i,(v,w) in enumerate(zip(a,b)):numeric(v,w,path+'/'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            stat['n_numeric']+=1
            if a!=b:details.append(dict(file=stat['file'],cell=path,v1=a,v2=b,abs_difference=abs(a-b),classification='changed'))
    for key in ['metric_checks','split_records','split_grid','ti_threshold_check','raw_alignment','yaw_crosscheck','status_filter','raw_normal_filter','power_setpoint_check','stratification_coverage']:
        stat=dict(file='k0_identity_checks.json#'+key,n_numeric=0);start=len(details);numeric(oldk[key],k[key],key);stat.update(n_differences=len(details)-start);audit.append(stat)
    for a,b in zip(oldk['vf_identity_checks'],k['vf_identity_checks']):
        assert a['max_abs_prediction_difference_mw']==b['max_abs_prediction_difference_mw']
        assert a['fields_equal']==b['fields_equal'] and b['passed']
    csv_keys={
        'k1_ti_strata_errors.csv':['row_type','model','horizon','seed','stratum','baseline','metric'],
        'k2_ti_power_curves.csv':['turbine','stratum','wind_lower_ms'],
        'k2_ti_residuals.csv':['row_type','horizon','seed','split','stratum','wind_lower_ms'],
        'k3_attention_by_ti.csv':['row_type','site','horizon','seed','stratum','metric'],
        'k4_yaw_distribution.csv':['period','turbine'],
        'k4_yaw_power_ratio.csv':['reference_speed','bin_type','stratum'],
        'k5_learned_curves.csv':['curve_type','horizon','seed','turbine','wind_ms'],
        'k5_curve_shift.csv':['horizon','seed','turbine'],
        'k6_yaw_strata_errors.csv':['row_type','model','horizon','seed','stratum','baseline','metric']}
    def cellkey(row,keys):
        return tuple('<NA>' if pd.isna(row[c]) else (format(float(row[c]),'.10g') if isinstance(row[c],(int,float,np.integer,np.floating)) else str(row[c])) for c in keys)
    for name,keys in csv_keys.items():
        a=pd.read_csv(register(V1/name),float_precision='round_trip');b=pd.read_csv(O/name,float_precision='round_trip')
        # Numeric keys must compare independently of CSV int/float serialization.
        for c in keys:
            if pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_numeric_dtype(b[c]):
                a[c]=pd.to_numeric(a[c],errors='coerce').astype(float);b[c]=pd.to_numeric(b[c],errors='coerce').astype(float)
        lookup={cellkey(r,keys):r for _,r in b.iterrows()};assert len(lookup)==len(b),(name,'duplicate keys')
        stat=dict(file=name,n_rows_v1=len(a),n_rows_v2=len(b),n_common_rows=0,n_numeric=0,n_differences=0,max_abs_difference=0.,n_expected_summary_changes=0)
        start=len(details)
        for _,r in a.iterrows():
            # Blocked slots had no result; expected seed-count/comparison changes are separated.
            if 'status' in a and str(r.status).startswith('BLOCKED'):continue
            key=cellkey(r,keys);assert key in lookup,(name,key);v=lookup[key];stat['n_common_rows']+=1
            for c in a.columns.intersection(b.columns):
                x=r[c];y=v[c]
                if (c in keys and c!='wind_ms') or pd.isna(x) or isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)):continue
                stat['n_numeric']+=1
                if x!=y:
                    delta=abs(float(x)-float(y));stat['n_differences']+=1;stat['max_abs_difference']=max(stat['max_abs_difference'],delta)
                    details.append(dict(file=name,cell=dict(zip(keys,key)),column=c,v1=float(x),v2=float(y),abs_difference=delta,classification='serialization_roundoff' if delta<=1e-12 else 'UNEXPECTED'))
        stat['status']='EXACT' if not stat['n_differences'] else ('ROUNDING_ONLY' if stat['max_abs_difference']<=1e-12 else 'FAIL')
        stat['byte_identical']=sha(V1/name)==sha(O/name);audit.append(stat)
        assert stat['status']!='FAIL',stat
    # Strict requirement: all K4 files must be byte-identical, including bootstrap arrays.
    for name in ['k4_yaw_distribution.csv','k4_yaw_power_ratio.csv','k4_cos_fit.json']:
        assert sha(register(V1/name))==sha(O/name),name
    audit.append(dict(file='k4_cos_fit.json',status='BYTE_IDENTICAL',byte_identical=True,n_differences=0))
    old=pd.read_csv(V1/'k2_oracle_bound.csv',float_precision='round_trip');new=pd.read_csv(O/'k2_oracle_bound.csv',float_precision='round_trip')
    ref=new[new.correction=='superseded_subtract'];assert len(ref)==6
    for _,r in old[(old.row_type=='seed')&(old.status=='PASS')].iterrows():
        rr=ref[(ref.horizon==r.horizon)&(ref.seed==r.seed)].iloc[0]
        for c in old.select_dtypes(include='number').columns:
            if pd.notna(r[c]):assert r[c]==rr[c],(c,r[c],rr[c])
    audit.append(dict(file='k2_oracle_bound.csv',status='E2_E3_NEW_CORRECTIONS_6_OLD_ROWS_PRESERVED',n_differences=0,n_common_rows=6))
    for h in HORIZONS:
        for seed in SEEDS:
            pair=new[(new.row_type=='seed')&(new.status=='PASS')&(new.horizon==h)&(pd.to_numeric(new.seed,errors='coerce')==seed)].set_index('correction')
            assert set(pair.index)=={'ws_only','ws_x_ti'}
            assert pair.n_test.nunique()==1
            for metric in ['nrmse','nmae']:
                av=pair.loc['ws_only',metric+'_after_pct'];bv=pair.loc['ws_x_ti',metric+'_after_pct']
                assert np.isclose(pair.loc['ws_x_ti',metric+'_ti_increment_change_pp'],bv-av,rtol=0,atol=1e-12)
                assert np.isclose(pair.loc['ws_x_ti',metric+'_ti_increment_relative_change_pct'],100*(bv/av-1),rtol=0,atol=1e-10)
    for (h,label),g in new[(new.row_type=='seed')&(new.status=='PASS')].groupby(['horizon','correction']):
        summary=new[(new.row_type=='seed_summary')&(new.horizon==h)&(new.correction==label)].iloc[0]
        assert len(g)==3
        for c in [x for x in new.columns if x.endswith('_pct') or x.endswith('_pp')]:
            if c+'_mean' not in new or g[c].isna().all():continue
            assert np.isclose(summary[c+'_mean'],g[c].mean(),rtol=0,atol=1e-12)
            assert np.isclose(summary[c+'_sd'],g[c].std(ddof=1),rtol=0,atol=1e-12)
    # Ensure only E1-E3-related analysis code changed.
    unchanged=['raw_data.py','yaw_analysis.py','learned_curves.py','figure_yaw_power_ratio.py','figure_ti_power_curves.py','run_analysis.py','figure_style.py']
    for name in unchanged:assert sha(V1/'scripts'/name)==sha(O/'scripts'/name),name
    k['v1_v2_reproduction']=dict(tables=audit,differences=details,k4_all_three_files_byte_identical=True,unchanged_scripts=unchanged,old_metric_values_exact=True,old_identity_differences_exact=True,expected_changes=['E1 tolerance 2.05e-6 -> 2.05e-5 MW; VF passed 6 -> 9','VF three-seed summaries become available; learned curves 36 -> 54','E2/E3: plus residual, ws_only vs ws_x_ti; old subtract rows superseded'],historical_incidents=readjson(V1/'operational_incidents.json'))
    if details:
        incident('v1_v2_serialization_roundoff',f'对照发现 {len(details)} 个已有数值的末位差异；每项完整数值及差值列于 k0 的 v1_v2_reproduction.differences。','保留重算值与全部对照；不覆写为旧数值。K4 三文件须逐字节一致。','reproduction_detail')
    log('v1-v2 numeric audit complete; K4 byte-identical; differences='+str(len(details)))

if __name__=='__main__':
    checkpoint();k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf8'));verify(k);save_k0(k)
