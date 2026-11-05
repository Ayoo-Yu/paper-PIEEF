from common import *
import zipfile, csv as csvlib, importlib.util

def wrap(x):return (x+180)%360-180

def header_info(z,name):
    with z.open(name) as f:
        lines=[f.readline() for _ in range(10)]
    blob=b''.join(lines)
    encoding='utf-8-sig'
    try:blob.decode(encoding)
    except UnicodeDecodeError:
        encoding='cp1252';blob.decode(encoding)
    lines=[x.decode(encoding).rstrip('\r\n') for x in lines]
    assert '# Time zone: UTC' in lines[:9]
    return encoding,next(csvlib.reader([lines[9]]))

def run(k):
    sc=pd.read_csv(register(DATA/'SCADA_15min.csv'));ts=pd.DatetimeIndex(pd.to_datetime(sc.Timestamp,utc=True))
    assert ts.is_unique and np.all(np.diff(ts.asi8)==900_000_000_000)
    end=int(k['split_records'][0]['train_end_index_exclusive'])
    assert all(x['train_end_index_exclusive']==end for x in k['split_records'])
    assert str(ts[end-1])==k['split_records'][0]['train_last_grid_utc']
    # Validation grid boundary is absent from serialized preprocessing. Recover
    # with the producer's exact configured indexing expression, then verify
    # against every saved validation/test target, never use approximate dates.
    split=k['split_records'][0]['split_config'];val_end=int(len(ts)*(split[0]+split[1]))
    incident('validation_boundary_not_serialized','preprocessing 仅保存训练结束索引；data_protocol/config 不保存验证结束时间','按原生产代码与配置恢复 val_end=int(N*(0.7+0.1))，并逐运行核对保存的验证/测试首尾；训练边界直接取保存值。','minimal_deviation')
    for x in k['split_records']:
        assert pd.Timestamp(x['val_last_saved_target_utc'])<ts[val_end]<=pd.Timestamp(x['test_first_saved_target_utc'])
        assert pd.Timestamp(x['val_first_saved_target_utc'])>=ts[end]
    k['split_grid']=dict(n_grid=len(ts),train_start=str(ts[0]),train_end_exclusive=str(ts[end]),val_start=str(ts[end]),val_end_exclusive=str(ts[val_end]),test_start=str(ts[val_end]),test_end_exclusive=str(ts[-1]+pd.Timedelta(minutes=15)),train_end_index=end,val_end_index=val_end,validation_boundary_method='exact original configured indexing expression; saved target endpoints verified')
    ws=sc[[f'turbine_ws{i}' for i in range(1,7)]].to_numpy(float)
    std=sc[[f'wind_speed_std{i}' for i in range(1,7)]].to_numpy(float)
    ti,tm=masks_ti(ws,std);q=np.quantile(ti[:end][np.isfinite(ti[:end])],[1/3,2/3,.9]);frozen=np.array([.124,.166,.216])
    k['ti_threshold_check']=dict(recomputed=q,frozen=frozen,difference=q-frozen,exceeds_002=abs(q-frozen)>.002,n_training_turbine_grid_cells=int(np.isfinite(ti[:end]).sum()),keep_frozen_thresholds=True)
    if np.any(abs(q-frozen)>.002):incident('ti_threshold_difference','重算 TI 分位数与冻结值相差超过 0.002','保留冻结阈值；差值见 k0_identity_checks.json','minimal_deviation')
    grid=pd.date_range('2024-01-01','2025-01-01',freq='10min',inclusive='left',tz='UTC')
    fields={'Power (kW)':'power','Wind speed (m/s)':'ws','Wind speed, Standard deviation (m/s)':'std','Wind direction (°)':'direction','Nacelle position (°)':'nacelle','Turbine Power setpoint (kW)':'setpoint','Vane position 1+2 (°)':'phi'}
    raw={v:np.full((len(grid),6),np.nan) for v in fields.values()};stop=np.zeros((len(grid),6),bool);status_metadata=[];parsing=[]
    register(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        for j in range(6):
            name=next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Kelmarsh_{j+1}_'))
            enc,header=header_info(z,name);assert all(c in header for c in fields)
            low=np.full((len(grid),len(fields)),np.inf);high=np.full_like(low,-np.inf);counts=np.zeros(len(grid),int);rows=0;outside=0
            with z.open(name) as f:
                for chunk in pd.read_csv(f,skiprows=9,usecols=['# Date and time',*fields],encoding=enc,chunksize=30000):
                    times=pd.DatetimeIndex(pd.to_datetime(chunk['# Date and time'],utc=True));ns=times.asi8
                    assert np.all((ns-grid[0].value)%600_000_000_000==0)
                    inside=(ns>=grid[0].value)&(ns<grid[-1].value+600_000_000_000);outside+=int((~inside).sum());ix=((ns[inside]-grid[0].value)//600_000_000_000).astype(int)
                    vals=chunk[list(fields)].to_numpy(float)[inside];np.add.at(counts,ix,1)
                    for c in range(vals.shape[1]):
                        ok=np.isfinite(vals[:,c]);np.minimum.at(low[:,c],ix[ok],vals[ok,c]);np.maximum.at(high[:,c],ix[ok],vals[ok,c])
                    rows+=len(chunk)
            have=np.isfinite(low);conflict=have&(low!=high);vals=np.where(have&~conflict,low,np.nan)
            for c,key in enumerate(fields.values()):raw[key][:,j]=vals[:,c]
            parsing.append(dict(turbine=j+1,member=name,encoding=enc,rows=rows,duplicate_rows=int(np.maximum(counts-1,0).sum()),missing_grid_times=int((counts==0).sum()),conflicts={key:int(conflict[:,c].sum()) for c,key in enumerate(fields.values())},outside_year_rows=outside))
            sn=next(n for n in z.namelist() if n.startswith(f'Status_Kelmarsh_{j+1}_'));se,_=header_info(z,sn)
            with z.open(sn) as f:st=pd.read_csv(f,skiprows=9,encoding=se)
            text=st[['Status','Message','Service contract category','IEC category','Global contract category']].fillna('').astype(str).agg(' | '.join,axis=1)
            derating=text.str.contains(r'derat|curtail|partial performance|reduced performance|power limit|power reduction',case=False,regex=True)
            selected=st.Status.str.casefold().eq('stop')|derating
            categories=[]
            for keys,g in st[selected].groupby(['Status','Code','Message','IEC category'],dropna=False):
                categories.append(dict(status=keys[0],code=int(keys[1]),message=keys[2],iec_category=keys[3],n_events=len(g)))
            starts=pd.to_datetime(st.loc[selected,'Timestamp start'],utc=True,errors='coerce');ends=pd.to_datetime(st.loc[selected,'Timestamp end'],utc=True,errors='coerce')
            missing=int((starts.isna()|ends.isna()).sum())
            if missing:incident(f'status_missing_end_T{j+1}','停机/降额状态缺少可用区间','该状态只排除包含其开始时刻的 10 分钟记录，记录局限。','minimal_deviation')
            for a,b in zip(starts,ends):
                if pd.isna(a):continue
                if pd.isna(b):b=a+pd.Timedelta(nanoseconds=1)
                if b==a:b=a+pd.Timedelta(nanoseconds=1)
                stop[:,j]|=(grid<b)&(grid+pd.Timedelta(minutes=10)>a)
            status_metadata.append(dict(turbine=j+1,member=sn,encoding=se,status_counts=st.Status.value_counts().to_dict(),selected_events=int(selected.sum()),derating_events=int(derating.sum()),missing_intervals=missing,categories=categories,overlap_records=int(stop[:,j].sum())))
            log(f'Raw turbine {j+1}: {rows} rows; encoding={enc}; status excluded records={stop[:,j].sum()}')
    raw['phi']=wrap(raw['phi']);raw['stop']=stop;raw['timestamp']=grid
    ix=grid.get_indexer(ts,method='pad');assert (ix>=0).all();obs=pd.to_datetime(sc.SCADA_observation_time,utc=True)
    assert np.array_equal(grid[ix].asi8,pd.DatetimeIndex(obs).asi8)
    # The original align_scada performs field-wise forward fill before sampling.
    # SCADA_observation_time tracks the nominal record, not each field's age.
    register(Path('src/pieef/site_alignment.py'))
    sampled={key:pd.DataFrame(raw[key],index=grid).reindex(grid.union(ts)).sort_index().ffill().reindex(ts).to_numpy() for key in ['ws','std','phi']}
    alignment=[]
    for j in range(6):
        for key,target in [('ws',ws),('std',std)]:
            a=sampled[key][:,j];b=target[:,j];finite=np.isfinite(a)&np.isfinite(b);md=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else np.nan
            same=bool(np.array_equal(np.isfinite(a),np.isfinite(b)))
            alignment.append(dict(turbine=j+1,field=key,max_abs_difference=md,n_compared=int(finite.sum()),finite_masks_equal=same,passed=same and md<=1e-6))
    gate=all(x['passed'] for x in alignment)
    k['raw_alignment']=dict(passed=gate,tolerance=1e-6,observation_times_equal=True,comparisons=alignment,parsing=parsing,sampling='Original site_alignment.align_scada: union 10min and 15min grids, field-wise forward-fill, reindex 15min; observation_time is nominal 10min record time')
    if not gate:incident('raw_alignment_failure','原始数据无法在冻结容差内复现 15 分钟风速/标准差','停止 K4–K6；K1–K3 继续用已有 15 分钟 SCADA。','gate_failure')
    phi=sampled['phi'];alt=wrap(sc[[f'wind_direction{i}' for i in range(1,7)]].to_numpy()-sc[[f'nacelle_position{i}' for i in range(1,7)]].to_numpy());cross=[]
    for j in [None,*range(6)]:
        a=phi.ravel() if j is None else phi[:,j];b=alt.ravel() if j is None else alt[:,j];ok=np.isfinite(a)&np.isfinite(b)
        cross.append(dict(turbine='pooled' if j is None else j+1,n=int(ok.sum()),median_abs_difference_deg=np.median(abs(a[ok]-b[ok])),pearson_r=np.corrcoef(a[ok],b[ok])[0,1]))
    k['yaw_crosscheck']=cross;k['status_filter']=dict(interval_convention='10-min record [timestamp,timestamp+10min), status [start,end); exclude any overlap',derating_text_rule='derat|curtail|partial performance|reduced performance|power limit|power reduction',turbines=status_metadata)
    # K4/K5 raw normal operation: no 3-sigma filter. Match producer's frozen
    # reading semantics: the fifth identical reading and every later one.
    valid=np.isfinite(raw['power'])&(raw['power']>-20)&np.isfinite(raw['ws']);low=(raw['ws']>5)&(raw['power']<10)
    repeat=np.zeros_like(valid);runlength=np.zeros(6,int)
    for i in range(1,len(grid)):
        same=np.isfinite(raw['ws'][i])&(raw['ws'][i]==raw['ws'][i-1]);runlength=np.where(same,runlength+1,0);repeat[i]=runlength>=4
    raw['normal']=valid&~low&~repeat&~stop
    raw['train']=grid<ts[end];raw['test']=(grid>=ts[val_end])
    k['raw_normal_filter']=dict(no_three_sigma_filter=True,freeze='5th and later consecutive identical finite readings; producer semantics',per_turbine=[dict(turbine=j+1,n_train=int(raw['train'].sum()),invalid=int((raw['train']&~valid[:,j]).sum()),low_power_candidates=int((raw['train']&valid[:,j]&low[:,j]).sum()),frozen=int((raw['train']&repeat[:,j]).sum()),status_overlap=int((raw['train']&stop[:,j]).sum()),retained_normal=int((raw['train']&raw['normal'][:,j]).sum())) for j in range(6)])
    # K2 uses the original events normal class, including its original anomaly
    # definition; K4 explicitly differs by excluding no 3-sigma anomalies.
    path=PROJECT/'src/pieef/revision_data.py';spec=importlib.util.spec_from_file_location('k_revision_data',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    powers=sc[[f'power{i}' for i in range(1,7)]].to_numpy(np.float32)/1000
    valid15,events15=module._power_events(powers,ws.astype(np.float32),end,np.full(6,2.05))
    normal15=valid15&(events15=='normal')&~stop[ix]&np.isfinite(ws)
    sett=raw['setpoint'][raw['train']];pw=raw['power'][raw['train']];ok=np.isfinite(sett)&np.isfinite(pw)
    k['power_setpoint_check']=dict(source_column='Turbine Power setpoint (kW)',n_finite=int(np.isfinite(sett).sum()),fraction_below_2000=float((sett[np.isfinite(sett)]<2000).mean()),pearson_with_measured_power=float(np.corrcoef(sett[ok],pw[ok])[0,1]),semantic_conclusion='Export only supplies column name, not controller definition. High correspondence with measured power cannot establish it as a curtailment command; not used for filtering.')
    save_k0(k);log(f'Raw alignment passed={gate}; max difference={max(x["max_abs_difference"] for x in alignment):.12g}; TI quantiles={q.tolist()}')
    return dict(sc=sc,ts=ts,ws=ws,std=std,ti=ti,ti_masks=tm,phi=phi,raw=raw,raw_index=ix,normal=normal15,valid=valid15,events=events15,train_end=end,val_end=val_end,yaw_gate=gate)
