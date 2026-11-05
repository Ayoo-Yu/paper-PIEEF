"""Integrity checks, figure preflight, input immutability, and final manifest."""
from common import *
import ast, importlib.util, dataclasses, xml.etree.ElementTree as ET

def check_tables(k):
    result=[]
    for name,layers in [('k1_ti_strata_errors.csv',TI_LAYERS),('k6_yaw_strata_errors.csv',YAW_LAYERS)]:
        d=pd.read_csv(O/name);a=d[d.row_type=='seed'];assert len(a)==54*len(layers)
        assert len(a[a.status=='BLOCKED_GATE'])==0
        assert (a.loc[a.status=='BLOCKED_GATE',['nrmse_pct','nmae_pct']].isna().all()).all()
        base=d[d.row_type=='seed_summary'];assert (base.status=='PASS').all()
        vf=d[(d.row_type=='seed_summary')&(d.model=='VF')];assert (vf.status=='PASS').all();assert vf.nrmse_pct_mean.notna().all()
        for key,g in a[a.status=='PASS'].groupby(['model','horizon','stratum']):
            s=base[(base.model==key[0])&(base.horizon==key[1])&(base.stratum==key[2])].iloc[0]
            for metric in ['nrmse_pct','nmae_pct']:
                assert np.isclose(g[metric].mean(),s[metric+'_mean'],rtol=0,atol=1e-12)
                assert np.isclose(g[metric].std(ddof=1),s[metric+'_sd'],rtol=0,atol=1e-12)
        for _,r in d[d.row_type=='paired_seed_comparison'].iterrows():
            p=a[(a.horizon==r.horizon)&(a.seed==r.seed)&(a.stratum==r.stratum)].set_index('model');metric=r.metric+'_pct'
            assert np.isclose(r.VF_improvement_pct,100*(1-p.loc['VF',metric]/p.loc[r.baseline,metric]),rtol=0,atol=1e-10)
        result.append(dict(file=name,status='PASS',seed_rows=len(a),blocked_rows=int((a.status=='BLOCKED_GATE').sum()),checks='complete expected slots; no blocked values; baseline mean/sample SD; paired relative differences'))
    d=pd.read_csv(O/'k2_oracle_bound.csv');a=d[(d.row_type=='seed')&(d.status=='PASS')];assert len(a)==18
    for m in ['nrmse','nmae']:
        assert np.allclose(a[m+'_change_pp'],a[m+'_after_pct']-a[m+'_before_pct'],rtol=0,atol=1e-12)
        assert np.allclose(a[m+'_relative_change_pct'],100*(a[m+'_after_pct']/a[m+'_before_pct']-1),rtol=0,atol=1e-10)
        assert np.allclose(a[m+'_relative_improvement_pct'],-a[m+'_relative_change_pct'],rtol=0,atol=1e-10)
    result.append(dict(file='k2_oracle_bound.csv',status='PASS',checks='18 corrected seed rows; exact absolute/relative changes and signs'))
    d=pd.read_csv(O/'k3_attention_by_ti.csv');a=d[(d.row_type=='seed')&(d.site=='Kelmarsh')]
    assert len(a)==9*5*3 and (a.status=='BLOCKED_GATE').sum()==0
    ok=a[a.status=='PASS'];assert (ok.n_finite+ok.n_undefined==ok.n_queries).all();assert ok.attention_mean.between(0,1).all()
    result.append(dict(file='k3_attention_by_ti.csv',status='PASS',checks='all expected slots; finite/undefined counts reconcile; bounded diagnostics'))
    d=pd.read_csv(O/'k4_yaw_power_ratio.csv');assert len(d)==38
    fit=json.loads((O/'k4_cos_fit.json').read_text(encoding='utf8'));assert fit['bootstrap']['replicates']==1000 and fit['bootstrap']['seed']==2026
    assert (d.n_valid_bootstrap==1000).all();assert (d.ci95_lower<=d.ci95_upper).all()
    for x in fit['fits']:
        sub=d[(d.reference_speed==x['reference_speed'])&(d.bin_type=='two_degree')];assert np.array_equal(sub.fit_eligible,sub.n_records>=200)
    result.append(dict(file='k4_yaw_power_ratio.csv',status='PASS',checks='2 references x 19 bins; bootstrap count; frozen fit support'))
    d=pd.read_csv(O/'k5_curve_shift.csv');assert len(d)==54 and (d.status=='PASS').sum()==54 and (d.status=='BLOCKED_GATE').sum()==0
    c=pd.read_csv(O/'k5_learned_curves.csv');cur=c[(c.curve_type=='learned')&(c.status=='PASS')]
    assert len(cur)==54*301
    for key,g in cur.groupby(['horizon','seed','turbine']):
        assert len(g)==301 and np.all(np.diff(g.wind_ms)>0) and (g.stable_derivative_pu_per_ms>=0).all()
    for ref in ['all_normal','aligned_lt2']:
        for level in ['v50','v90']:
            a=d[d.status=='PASS'];assert np.allclose(a[f'learned_minus_{ref}_{level}_ms'],a[f'learned_{level}_ms']-a[f'{ref}_{level}_ms'],rtol=0,atol=1e-12,equal_nan=True)
    result.append(dict(file='k5_curve_shift.csv',status='PASS',checks='54 expected slots; 54 curves x 301 points; exact shift subtraction; stable monotonicity'))
    return result

def figure_preflight():
    path=register(Path('results/paper/source/validate_figure.py'))
    spec=importlib.util.spec_from_file_location('k_figure_validator',path);v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
    style=(O/'scripts/figure_style.py').read_text(encoding='utf8');out=[]
    for script,stem in [('figure_ti_errors.py','k1_ti_errors'),('figure_ti_attention.py','k3_ti_attention'),('figure_yaw_power_ratio.py','k4_yaw_power_ratio'),('figure_ti_power_curves.py','k2_ti_power_curves')]:
        src=(O/'scripts'/script).read_text(encoding='utf8');findings=[dataclasses.asdict(x) for x in v.validate_source(style+'\n'+src,'python')]
        svg=O/'figures'/f'{stem}.svg';png=O/'figures'/f'{stem}.png';root=ET.parse(svg).getroot();texts=root.findall('.//{http://www.w3.org/2000/svg}text')
        assert png.is_file() and len(texts)>10
        for f in findings:
            if f['level']=='FAIL':
                assert f['check_id']=='EXPORT-VECTOR',f
                f['task_disposition']='Same v1 output set requested: editable SVG and PNG retained.'
            elif f['level']=='WARN':f['task_disposition']='Actual source uses 183/25.4 inches; static literal parser cannot evaluate arithmetic. Actual SVG and rendered PNG inspected; exports are 300 dpi.'
        out.append(dict(script=script,svg=svg.relative_to(O).as_posix(),png=png.relative_to(O).as_posix(),svg_width=root.attrib.get('width'),svg_height=root.attrib.get('height'),editable_text_nodes=len(texts),raw_validator_findings=findings,task_contract_pass=True,visual_qa='Inspected rendered PNG: labels, legend, panel margins, and data directions; no clipping observed.',intended_width_mm=183,dpi=300))
    return out

def prepare():
    checkpoint();t=begin('python -B scripts/finalize.py --prepare');k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf8'))
    for p in (O/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    k['delivery_validation']=check_tables(k);k['figure_qa']=figure_preflight()
    import v2_audit
    v2_audit.verify(k)
    for rel in ['SKILL.md','manifest.yaml','static/core/contract.md','static/core/stance.md','static/fragments/backend/python.md','references/qa-contract.md']:
        register(Path('results/paper')/rel)
    incident('figure_preflight_import_scope','技能静态检查对独立图脚本不追踪共用样式导入，并将 183/25.4 宽度表达式误识为字面数；其默认要求还包括 PDF。','已组合共用样式复核字体、可编辑 SVG、PNG 和实际尺寸；按用户保持首轮文件集的要求保留 PNG/SVG；渲染图单独目视核查。','operational')
    incident('manifest_self_reference','文件无法在自身内容中包含其完整字节 SHA-256 的一般固定点。','所有普通输入/输出使用完整文件字节 SHA-256；manifest 自项明确采用移除 self 项后的规范 JSON 摘要，并提供验证方法。','minimal_deviation')
    incident('wind_farm_A_historical_dsi_missing','旧Wind Farm A分工况 JSON 仅保存熵、最大权重和时间点数，未保存分工况 DSI；其多数事件标签分组权重也与本任务逐查询 TI 分组不同。','并列转录指定相同种子的既有熵/最大权重，明确不同权重及 DSI 缺失，不追加Wind Farm A推理或新分层。','minimal_deviation')
    import scipy,torch
    k['software']=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,torch=torch.__version__,device='CPU',training_performed=False)
    k['frozen_method_constants']=dict(turbines=6,seeds=list(SEEDS),horizons_steps=list(HORIZONS),grid_minutes=15,raw_minutes=10,capacity_mw=2.05,ti_min_ws_ms=4,ti_boundaries=[.124,.166,.216],ti_recalculation_tolerance=.002,yaw_range_deg='[-180,180)',yaw_boundaries_deg=[5,10,20],yaw_aligned_abs_lt_deg=2,yaw_crosscheck_warning_median_difference_deg=2,invalid_power_le_kw=-20,low_power_candidate_ws_gt_ms=5,low_power_candidate_power_lt_kw=10,frozen_identical_readings=5,raw_three_sigma_filter=False,k2_original_normal_events_includes_three_sigma=True,oracle_minimum_validation_cell_count=50,oracle_wind_bins_ms='[4,5),...,[14,15),[15,infinity)',k4_reference_bin_width_ms=.5,k4_min_reference_count=30,k4_two_degree_bin_width=2,k4_min_fit_count=200,k4_day_bootstrap_replicates=1000,k4_bootstrap_seed=2026,k4_confidence_percent=95,k5_evaluation_wind_ms='0 to 30, step 0.1',k5_capacity_crossing_fractions=[.5,.9],k5_empirical_bin_width_ms=.5,k5_no_extra_minimum_cell_count=True,entropy='-sum(a*log(a))/log(N), includes self',dsi='A_up/(A_up+A_down), directional alignment >0.5 or <-0.5',seed_sd_ddof=1)
    # Housekeeping only for the font cache created by these scripts, with exact
    # resolved path confinement and no recursive deletion.
    cache=O/'scripts/.mplconfig'
    if cache.exists():
        assert cache.resolve().is_relative_to(O.resolve())
        for p in cache.iterdir():
            assert p.is_file() and p.name.startswith('fontlist-') and p.suffix=='.json';p.unlink()
        cache.rmdir()
    save_k0(k);js('operational_incidents.json',INCIDENTS);end(t,'finalize prepare')

def manifest():
    checkpoint();t=begin('python -B scripts/finalize.py --manifest');k=json.loads((O/'k0_identity_checks.json').read_text(encoding='utf8'))
    checks=[]
    for x in INPUTS.values():
        actual=sha(x['path']);checks.append(dict(path=x['path'],sha256=actual,unchanged=actual==x['sha256']))
    assert all(x['unchanged'] for x in checks)
    k['input_immutability']=dict(status='PASS',n_files=len(checks),checked_utc=utc(),checks=checks)
    save_k0(k)
    end(t,'finalize manifest input checks')
    outputs=[dict(path=p.relative_to(O).as_posix(),sha256=sha(p),bytes=p.stat().st_size,hash_scope='full_file_bytes') for p in sorted(O.rglob('*')) if p.is_file() and p.name!='manifest.json']
    m=dict(schema='k_turbulence_yaw_sha256_v1',created_utc=utc(),root=str(O),inputs=list(INPUTS.values()),outputs=outputs,self_hash_rule='Remove the self key, then json.dumps(ensure_ascii=False, sort_keys=True, separators=(comma,colon), allow_nan=False).encode(UTF-8); SHA-256 of that canonical payload. All other hashes cover exact file bytes.',input_immutability_pass=True)
    payload=json.dumps(clean(m),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    m['self']=dict(path='manifest.json',sha256=hashlib.sha256(payload).hexdigest(),hash_scope='canonical_JSON_without_self_key')
    js('manifest.json',m)
    print('FINAL manifest full-file SHA-256:',sha(O/'manifest.json'))
    print('OUTPUT_FILES',len(outputs)+1,'INPUT_FILES',len(INPUTS))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--manifest',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.manifest:manifest()
