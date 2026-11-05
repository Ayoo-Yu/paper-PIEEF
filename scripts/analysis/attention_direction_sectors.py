"""E3 frozen-model inference, preregistered sector diagnostics and draft maps.

Adapted from the provided attention inference pipeline; source files stay read-only.
Run --prepare once to preflight/freeze, then --run once. --render-only never infers.
"""
from pathlib import Path
import os, sys, json, hashlib, time, importlib.util, copy, argparse, traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PLAN = OUT.parents[1]
PROJECT = PLAN / 'next_batch/merged_v2/project'
DATA = Path('data/wind_farm_A')
ORIGINAL = PLAN / '回复补齐_20261021/integration/attention_completion.py'
FIG_SOURCE = Path('results/paper/source/make_fig16_attention_maps.py')
VALIDATOR = Path('results/paper/source/validate_figure.py')
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  LOKY_MAX_CPU_COUNT='2', PYTHONDONTWRITEBYTECODE='1',
                  MPLCONFIGDIR=str(OUT/'mplconfig'), TORCH_HOME=str(OUT/'torch_cache'),
                  NATURE_FIGURE_CONFIG=str(OUT/'backend_config.json'))
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
SEEDS = [13, 21, 42, 87, 3407]
SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
CURRENT_SEED = None
GATES = []


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def js(name, value, exclusive=False):
    with (OUT/name).open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def log(message):
    line = f'[{utc()}] {message}'
    print(line, flush=True)
    with (OUT/'run_log.txt').open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def records():
    rows = [r for r in json.loads((PLAN/'补做重算/physics/run_inventory.json').read_text(encoding='utf-8')) if r['horizon'] == 16]
    assert sorted(r['seed'] for r in rows) == SEEDS
    return sorted(rows, key=lambda r: r['seed'])


def prepare():
    assert not (OUT/'design_frozen.json').exists(), 'Design already frozen; do not overwrite'
    assert not (OUT/'inference_once.json').exists(), 'Inference already attempted'
    (OUT/'run_log.txt').write_text('', encoding='utf-8')
    log('E3 preflight; prior stopped preflight used the singular plotting filename and ran no inference.')
    rows = records()
    required = [ORIGINAL, FIG_SOURCE, VALIDATOR, PROJECT/'scripts/evaluate_revision.py',
                DATA/'scada_ecmwf_15min.csv', DATA/'turbines.csv', DATA/'manufacturer_curve.csv',
                PLAN/'补做重算/physics/run_inventory.json']
    required += list((PROJECT/'src/pieef').rglob('*.py'))
    for r in rows:
        required.extend(Path(r['path'])/n for n in ['metrics.json', 'predictions.npz', 'preprocessing.npz', 'weights_best.pt'])
    for p in required:
        assert p.is_file(), f'Missing input: {p}'
    for r in rows:
        m = json.loads((Path(r['path'])/'metrics.json').read_text(encoding='utf-8'))
        assert m['status'] == 'completed' and m['evaluation_split'] == 'test'
        assert m['config']['tracking']['implementation_id'] == 'energy-r3-vf-F2F3F6'
        assert m['seed'] == r['seed'] and m['horizon'] == 16
        assert m['config']['train']['batch_size'] == 16
        prot = m['config']['data']['revision_protocol']
        assert prot['wind_height_m'] == 10
        assert prot['nwp_feature_columns'][4:6] == ['nwp_u10_ms', 'nwp_v10_ms']
        with np.load(Path(r['path'])/'predictions.npz', allow_pickle=False) as z:
            assert z['y_pred_mw'].shape == z['target_valid'].shape
            assert z['y_pred_mw'].shape[1] == 29
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    load_eval()  # Dependency import only; no model creation or forward call.
    import subprocess
    validation = subprocess.run([sys.executable,'-B',str(VALIDATOR),str(Path(__file__)),'--json'],
        capture_output=True,text=True,encoding='utf-8',check=False)
    validation = json.loads(validation.stdout)
    overrides = {'FONT-FAMILY':'Requested Fig16 house style explicitly uses Times New Roman; installed font verified.',
                 'EXPORT-RASTER':'Requested raster draft is600dpi PNG; editable PDF/SVG also exported.'}
    assert not [f for f in validation['findings'] if f['level']=='FAIL' and f['check_id'] not in overrides]
    js('source_preflight.json',dict(raw_report=validation,reviewed_overrides=overrides,contextual_result='PASS'))
    previous = OUT.parents[1]/'回复补齐_20261021/integration/attention'
    recorded = []
    for n in ['completed.json', 'source_manifest.json']:
        payload = json.loads((previous/n).read_text(encoding='utf-8'))
        for key in ['interpreter', 'python_executable', 'executable']:
            if payload.get(key):
                recorded.append(payload[key])
    selected = Path(recorded[0]) if recorded else Path('python')
    assert Path(sys.executable).resolve() == selected.resolve(), 'Interpreter differs from required environment'
    js('backend_config.json', {'backend': 'python'})
    sources = [dict(path=str(p), sha256=sha(p)) for p in required]
    js('source_manifest.json', dict(created_at_utc=utc(), interpreter=sys.executable,
        interpreter_selection='prior metadata' if recorded else 'specified pi-eef env; prior completed/manifest contain no interpreter field',
        python_version=sys.version, packages=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, matplotlib=matplotlib.__version__),
        font=font, files=sources, runs=rows))
    design = dict(task='E3 attention by wind-direction sector', frozen_at_utc=utc(),
        purpose='post-hoc inference-only diagnostic of frozen final PI-EEF models; no training, selection or changes to existing results',
        runs=rows, implementation_id='energy-r3-vf-F2F3F6', seeds=SEEDS, horizon_steps=16,
        split='test', batch_size=16, precision='FP32', threads=2, cuda_memory_fraction=.55,
        interpreter=sys.executable, training=False, inference_passes_per_seed=1,
        reproduction_gate=dict(metadata_and_turbine_ids='exact', preprocessing='original evaluate_revision assertions',
            truth='array_equal(equal_nan=True)', weights='torch.equal every state_dict tensor; file hashes unchanged',
            predictions='abs(new-saved)/capacity <= 1e-5 + 1e-5*abs(saved/capacity), all predictions finite',
            sole_allowed_exception='CUDA numerical nondeterminism <=0.001 rated power at every point, only after every other gate passes',
            larger_failure='stop and report; no inference rerun'),
        rows=dict(unit='query turbine x forecast origin', target_valid=True, event='normal',
            history='five finite non-negative raw turbine_ws observations ending at origin (60 min span), mean >=3 m/s; original finite sample-CV eligibility assertion retained',
            variability_split=False, downsampling=False),
        direction=dict(model_last_two='sin(atan2(u,v)), cos(atan2(u,v)); positive wind-to unit vector, restored by raw_nwp',
            actual_u='nwp_u10_ms', actual_v='nwp_v10_ms', height_m=10,
            sector_source='model.raw_nwp(target)[...,-2:].mean(1), converted with atan2(-w_x,-w_y) degrees modulo360',
            crosscheck='raw target-time 100m atan2(-u100,-v100); also raw10 sign verification; median/mean/P90 angular errors on all and included origins',
            source_choice='converted model vector preserves same-NWP requirement and meteorological from convention; raw100 is an independent height-mismatched check',
            alignment='retain original wind-to w, rel_coords=r_i-r_j; no sign flip for alignment'),
        sectors=dict(order=SECTORS, centres_deg=list(range(0,360,45)), assignment='floor(((from_deg+22.5)%360)/45); half-open intervals; N=[337.5,360) union [0,22.5)'),
        aggregation=dict(matrix='per seed/sector/query: average attention over eligible origin rows; row sum1',
            received_share='100 * equal mean over all 29 query row means, self included; no origin-count pooling across queries',
            entropy='row H/log(29), all29 keys including self; then equal mean of eligible query-origin rows',
            maximum_weight='maximum of29 in each eligible row, then equal mean across rows',
            upwind_index='per row sum(att[align>0])/sum(att[align!=0]); self/zero alignment excluded; then equal mean across eligible rows',
            uniform_expectation='per row count(align>0)/count(align!=0), then equal mean across same rows',
            undefined='empty sector/query represented by NaN with counts; zero nonself denominator stops rather than inventing value',
            sample_sd='ddof=1 across five seeds; descriptive only, no tests/CI'),
        sparse=dict(threshold_origins=200, definition='fewer than200 distinct eligible origins in any seed', exclude_from_decision=True, still_report_and_draw=True),
        decision=dict(d='received_share(s,k,j) minus seed k equal mean over non-sparse sectors',
            consistency='Pearson pooled over all non-sparse sectors and29 turbines vs other-four-seed mean d, >=0.5 for all5 seeds',
            size='max absolute five-seed mean d across non-sparse sectors/turbines >=2.0 percentage points',
            main_text='both consistency and size PASS; otherwise Fig16 stays as is',
            wake='per sector count of seeds with upwind_index > uniform_expectation; allowed only5 of5; necessary descriptive condition, not causal proof'),
        figure=dict(core_conclusion='Show five-seed mean spatial received attention by meteorological wind-from sector, whatever the verdict.',
            archetype='quantitative grid', panel_evidence='eight fixed sectors; all reported including sparse',
            width_in=7, height_in=2.65, layout='2x4', backend='Python/matplotlib',
            node_fill='five-seed mean received share, self included; one common white-red scale including sparse sectors',
            arrows='two strongest non-self keys per query, stable argsort of five-seed mean matrix; key->query; 58 links/panel',
            adaptation='house serif/palette/geographic axes; smaller marks and arrow geometry for4 panels across; no change to link selection',
            exports=['png600dpi','pdf editable text','svg editable text']),
        deviations=[], source_manifest_sha256=sha(OUT/'source_manifest.json'), script_sha256=sha(__file__))
    js('design_frozen.json', design, exclusive=True)
    js('input_preflight.json', dict(status='PASS', checked_at_utc=utc(), inputs=len(required),
        corrected_plotting_source_exists=True, inference_executed=False, design_sha256=sha(OUT/'design_frozen.json')))
    log(f'Preflight PASS; design frozen sha256={sha(OUT/"design_frozen.json")}; GPU available={torch.cuda.is_available()}.')


def verify_sources():
    manifest = json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
    for source in manifest['files']:
        assert sha(source['path']) == source['sha256'], f'Input changed: {source["path"]}'


def angular(a, b):
    return np.abs((a-b+180)%360-180)


def angle_stats(a):
    return dict(n=int(len(a)), median_abs_deg=float(np.median(a)), mean_abs_deg=float(a.mean()),
                p90_abs_deg=float(np.quantile(a,.9)), max_abs_deg=float(a.max()))


def load_eval():
    sys.path.insert(0, str(PROJECT/'src'))
    spec = importlib.util.spec_from_file_location('e3_attention_eval', PROJECT/'scripts/evaluate_revision.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def infer():
    global CURRENT_SEED
    started = time.time()
    design = json.loads((OUT/'design_frozen.json').read_text(encoding='utf-8'))
    assert sha(__file__) == design['script_sha256'], 'Script changed since design freeze'
    assert sha(OUT/'source_manifest.json') == design['source_manifest_sha256']
    assert design['seeds'] == SEEDS and design['decision']['size'].endswith('>=2.0 percentage points')
    verify_sources()
    js('inference_once.json', dict(started_at_utc=utc(), pid=os.getpid(), design_sha256=sha(OUT/'design_frozen.json'),
        instruction='Exclusive sentinel: no second inference attempt allowed.'), exclusive=True)
    ev = load_eval()
    raw = pd.read_csv(DATA/'scada_ecmwf_15min.csv')
    ts = pd.DatetimeIndex(pd.to_datetime(raw.Timestamp)).tz_localize('Asia/Shanghai').tz_convert('UTC')
    assert np.all(np.diff(ts.asi8) == pd.Timedelta(minutes=15).value)
    windraw = raw[[f'turbine_ws{i+1}' for i in range(29)]]
    winds = windraw.where(np.isfinite(windraw) & (windraw>=0))
    rollmean = winds.rolling(5,min_periods=5).mean().to_numpy()
    rolls = winds.rolling(5,min_periods=5).std(ddof=1).to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        cv = rolls/rollmean
    eligible = np.isfinite(cv) & (rollmean>=3.)
    assert np.array_equal(eligible, np.isfinite(rollmean)&(rollmean>=3.))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.55)
    log(f'One inference pass/seed begins; device={device}; FP32; threads=2; memory_fraction=.55.')
    summaries, matrices, query_counts, directions, convention = [], [], [], [], []
    bundle = None
    for r in records():
        seed = CURRENT_SEED = r['seed']
        d = Path(r['path'])
        m = json.loads((d/'metrics.json').read_text(encoding='utf-8'))
        cfg = copy.deepcopy(m['config'])
        assert m['status']=='completed' and m['evaluation_split']=='test'
        assert cfg['tracking']['implementation_id']=='energy-r3-vf-F2F3F6'
        cfg['data'].update(scada_path=str(DATA/'scada_ecmwf_15min.csv'),turbines_path=str(DATA/'turbines.csv'))
        cfg['model']['manufacturer_curve_path'] = str(DATA/'manufacturer_curve.csv')
        rec = dict(source_dir=d,config=cfg,seed=seed,horizon=16,model='PI-EEF')
        if bundle is None:
            bundle = ev._bundle(rec)
        else:
            # Rebuild/validate saved preprocessing for each seed, as part of the gate.
            other = ev._bundle(rec)
            assert np.array_equal(bundle.test_loader.dataset.indices, other.test_loader.dataset.indices)
            del other
        assert bundle.test_loader.batch_size == 16
        z = dict(np.load(d/'predictions.npz',allow_pickle=False))
        meta = bundle.handler.metadata['test']
        for k in ('timestamp','origin_timestamp','target_valid','events'):
            a = np.asarray(meta[k]); b = z[k]
            assert np.array_equal(a.astype(str),b.astype(str)) if a.dtype.kind in 'OU' else np.array_equal(a,b), k
        assert np.array_equal(np.asarray(bundle.handler.turbine_ids).astype(str),z['turbine_ids'].astype(str))
        ix = ts.get_indexer(pd.to_datetime(z['origin_timestamp'],utc=True)); assert (ix>=0).all()
        assert np.array_equal(ix+16,bundle.test_loader.dataset.indices)
        ti = ts.get_indexer(pd.to_datetime(z['timestamp'],utc=True)); assert (ti>=0).all()
        assert np.array_equal(ti, ix+16)
        ok = eligible[ix] & z['target_valid'].astype(bool)
        mask = ok & (z['events']=='normal')
        if seed == SEEDS[0]:
            js('row_selection_counts.json',dict(test_origins=len(ix),test_query_origin_rows=int(mask.size),
                target_valid_rows=int(z['target_valid'].sum()),normal_target_rows=int((z['target_valid'].astype(bool)&(z['events']=='normal')).sum()),
                valid_target_and_history_eligible_rows=int(ok.sum()),included_rows=int(mask.sum()),
                included_origins=int(mask.any(1).sum()),excluded_rows=int(mask.size-mask.sum()),
                predicate='target_valid AND events==normal AND five finite nonnegative origin wind observations AND mean>=3m/s'))
        model,kind,_ = ev._predictor(rec,bundle,device)
        assert not model.training
        assert all(v.dtype==torch.float32 for v in model.parameters() if v.is_floating_point())
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        pred, truth, from_angles = [], [], []
        ms = np.zeros((8,29,29),dtype=np.float64)
        mc = np.zeros((8,29),dtype=np.int64)
        ns = np.zeros(8,dtype=np.int64)
        sums = np.zeros((8,4),dtype=np.float64)
        offset = 0
        with torch.no_grad():
            for bi,b in enumerate(bundle.test_loader):
                pp = ev._predict(model,kind,b,torch.zeros_like(b[0]),torch.zeros_like(b[1]),device)
                pred.append(pp); truth.append(b[-1].numpy())
                att = model.graph.last_attn_weights.cpu().double().numpy(); bs = len(att)
                assert att.shape == (bs,29,29) and np.isfinite(att).all() and (att>=0).all()
                assert np.allclose(att.sum(-1),1,rtol=1e-6,atol=1e-6)
                e = -(att*np.log(np.maximum(att,1e-300))).sum(-1)/np.log(29)
                mx = att.max(-1)
                target = b[3].to(device)
                wind = model.raw_nwp(target)[...,-2:].mean(1)
                w = torch.nn.functional.normalize(wind,dim=-1).cpu().double().numpy()
                rel = torch.nn.functional.normalize(model.rel_coords,dim=-1).cpu().double().numpy()
                align = np.einsum('bd,ijd->bij',w,rel)
                assert np.isfinite(w).all() and (np.linalg.norm(w,axis=-1)>0).all()
                from_deg = np.degrees(np.arctan2(-w[:,0],-w[:,1]))%360
                from_angles.append(from_deg)
                sector = np.floor(((from_deg+22.5)%360)/45).astype(int)
                nz = (align!=0) & (~np.eye(29,dtype=bool))[None,:,:]
                positive = (align>0) & nz
                den = (att*nz).sum(-1)
                assert (den>0).all() and (nz.sum(-1)>0).all()
                ui = (att*positive).sum(-1)/den
                expected = positive.sum(-1)/nz.sum(-1)
                for s in range(8):
                    mm = mask[offset:offset+bs] & (sector==s)[:,None]
                    ms[s] += (att*mm[:,:,None]).sum(0)
                    mc[s] += mm.sum(0)
                    ns[s] += mm.any(1).sum()
                    sums[s] += [v[mm].sum() for v in [e,mx,ui,expected]]
                offset += bs
                if bi%150 == 0:
                    log(f'seed={seed} batch={bi}/{len(bundle.test_loader)}')
        assert offset == len(mask)
        preds = np.maximum(np.concatenate(pred)*bundle.capacity_mw,0)
        tr = np.concatenate(truth)*bundle.capacity_mw
        assert np.array_equal(tr,z['y_true_mw'],equal_nan=True)
        delta = np.abs(preds-z['y_pred_mw'])/bundle.capacity_mw
        allowed = 1e-5+1e-5*np.abs(z['y_pred_mw']/bundle.capacity_mw)
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
        assert np.isfinite(preds).all() and np.isfinite(delta).all()
        gate = dict(seed=seed, status='PASS', max_delta_pu=float(delta.max()), points=int(preds.size),
                    original_tolerance_pass=bool(np.all(delta<=allowed)), points_outside_original_tolerance=int((delta>allowed).sum()),
                    indices_targets_capacity_exact=True, metadata_exact=True, preprocessing_pass=True, weights_unchanged=True,
                    weights_file_unchanged=sha(d/'weights_best.pt')==next(x['sha256'] for x in json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))['files'] if x['path']==str(d/'weights_best.pt')),
                    allowed_gpu_exception_used=False)
        assert gate['weights_file_unchanged']
        try:
            # Original prediction assertion retained; the sole exception is explicitly preregistered.
            assert np.all(delta<=allowed) and np.isfinite(preds).all()
        except AssertionError:
            if device.type=='cuda' and np.all(delta<=.001):
                gate.update(status='PASS_WITH_PREREGISTERED_GPU_EXCEPTION', allowed_gpu_exception_used=True)
                log(f'seed={seed}: original numerical tolerance exceeded; CUDA-only exception <=.001 pu applies; all other gates passed.')
            else:
                gate['status']='FAIL'; GATES.append(gate); js('reproduction_gates.json',GATES)
                raise
        GATES.append(gate); js('reproduction_gates.json',GATES)
        matrix = np.divide(ms,mc[:,:,None],out=np.full_like(ms,np.nan),where=mc[:,:,None]>0)
        for s in range(8):
            nrows = int(mc[s].sum())
            # No invisible change in equal-query weighting if a query has zero observations.
            assert nrows==0 or np.all(mc[s]>0), f'Sector {SECTORS[s]} has a missing query mean'
            vals = sums[s]/nrows if nrows else np.full(4,np.nan)
            summaries.append(dict(seed=seed,sector=SECTORS[s],origins=int(ns[s]),rows=nrows,
                query_count_min=int(mc[s].min()),query_count_max=int(mc[s].max()),
                entropy=vals[0],max_weight=vals[1],upwind_index=vals[2],uniform_expectation=vals[3]))
        fd = np.concatenate(from_angles)
        raw100 = np.degrees(np.arctan2(-raw['nwp_u100_ms'].to_numpy()[ti],-raw['nwp_v100_ms'].to_numpy()[ti]))%360
        raw10 = np.degrees(np.arctan2(-raw['nwp_u10_ms'].to_numpy()[ti],-raw['nwp_v10_ms'].to_numpy()[ti]))%360
        included = mask.any(1)
        c = dict(seed=seed, model_from_vs_raw100_all=angle_stats(angular(fd,raw100)),
            model_from_vs_raw100_included=angle_stats(angular(fd[included],raw100[included])),
            model_from_vs_raw10_all=angle_stats(angular(fd,raw10)),
            unconverted_model_to_vs_raw100_all=angle_stats(angular((fd+180)%360,raw100)))
        assert c['model_from_vs_raw10_all']['max_abs_deg']<.001, 'Model sign/feature convention did not reproduce raw10'
        convention.append(c)
        js('convention_check.json',dict(source_choice=design['direction'],per_seed=convention))
        matrices.append(matrix); query_counts.append(mc); directions.append(fd)
        pd.DataFrame(summaries).to_csv(OUT/'sector_summary.csv',index=False,encoding='utf-8-sig')
        log(f'seed={seed} reproduction {gate["status"]}; max_delta={delta.max():.9g} pu; included_rows={mask.sum()}; origins={included.sum()}.')
        del model,before
        if device.type=='cuda':
            torch.cuda.empty_cache()
    verify_sources()
    mats = np.stack(matrices)
    assert np.allclose(mats.sum(-1),1,atol=1e-6,rtol=1e-6,equal_nan=False)
    np.savez_compressed(OUT/'sector_matrices.npz', matrices=mats, seeds=SEEDS,sectors=SECTORS,
        turbine_ids=z['turbine_ids'],query_counts=np.stack(query_counts),coords_m=bundle.handler.coords,
        model_from_direction_deg=np.stack(directions),origin_timestamp=z['origin_timestamp'],timestamp=z['timestamp'],
        eligible_normal_rows=mask)
    CURRENT_SEED = None
    decide()
    render()
    report()
    js('completed.json',dict(status='completed', completed_at_utc=utc(),device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,seconds=time.time()-started,
        training=False,inference_passes_per_seed=1,inputs_unchanged=True,script_sha256=sha(__file__),
        design_sha256=sha(OUT/'design_frozen.json'),gates=GATES))
    log('E3 complete; all source/run/data hashes unchanged; no training or model selection.')


def decide():
    frame = pd.read_csv(OUT/'sector_summary.csv')
    with np.load(OUT/'sector_matrices.npz',allow_pickle=False) as z:
        mats = z['matrices']; tids=z['turbine_ids'].astype(str)
    sparse = np.array([bool((frame[frame.sector==s].origins<200).any()) for s in SECTORS])
    keep = ~sparse
    shares = mats.mean(axis=2)*100
    assert np.allclose(shares.sum(-1),100,atol=1e-5)
    d = shares-shares[:,keep,:].mean(axis=1,keepdims=True) if keep.any() else np.full_like(shares,np.nan)
    received = [dict(seed=seed,sector=s,turbine=t,received_share_pct=shares[k,i,j],d_pp=d[k,i,j],sparse=bool(sparse[i]))
                for k,seed in enumerate(SEEDS) for i,s in enumerate(SECTORS) for j,t in enumerate(tids)]
    pd.DataFrame(received).to_csv(OUT/'sector_received_share.csv',index=False,encoding='utf-8-sig')
    loo = []
    for k,seed in enumerate(SEEDS):
        a = d[k,keep,:].ravel(); b = np.delete(d,k,axis=0).mean(0)[keep,:].ravel()
        r = float(np.corrcoef(a,b)[0,1]) if a.size>=2 and np.std(a)>0 and np.std(b)>0 else np.nan
        loo.append(dict(seed=seed,pearson_r=r,pooled_elements=len(a),threshold=.5,status='PASS' if r>=.5 else 'FAIL'))
    mean_d = d.mean(0)
    candidate = np.where(keep[:,None],np.abs(mean_d),-np.inf)
    size = float(np.max(candidate)) if keep.any() else np.nan
    pos = np.unravel_index(np.argmax(candidate),candidate.shape) if keep.any() else (0,0)
    consistency = all(x['status']=='PASS' for x in loo)
    size_pass = bool(size>=2.)
    wake = []
    means = []
    for i,s in enumerate(SECTORS):
        g = frame[frame.sector==s]
        n = int((g.upwind_index>g.uniform_expectation).sum())
        wake.append(dict(sector=s,seeds_above_expectation=n,total_seeds=5,description=f'{n} of 5 seeds above expectation',
            strict_all_five_status='PASS' if n==5 else 'FAIL',physical_wake_interpretation_allowed=n==5,sparse=bool(sparse[i])))
        row = dict(sector=s,seeds=5,sparse=bool(sparse[i]))
        for metric in ['origins','rows','entropy','max_weight','upwind_index','uniform_expectation']:
            row[metric+'_mean']=g[metric].mean(); row[metric+'_sd']=g[metric].std(ddof=1)
            row[metric+'_mean_sd']=f'{g[metric].mean():.6g} ± {g[metric].std(ddof=1):.6g}'
        row['seeds_above_expectation']=n
        means.append(row)
    frame['sparse']=frame.sector.map(dict(zip(SECTORS,sparse)))
    frame.to_csv(OUT/'sector_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(means).to_csv(OUT/'sector_summary_five_seed_mean_sd.csv',index=False,encoding='utf-8-sig')
    decision = dict(created_at_utc=utc(),sparse_sectors=[s for s,b in zip(SECTORS,sparse) if b],
        non_sparse_sectors=[s for s,b in zip(SECTORS,keep) if b],
        consistency=dict(status='PASS' if consistency else 'FAIL',threshold=.5,required_seeds=5,per_seed=loo),
        size=dict(status='PASS' if size_pass else 'FAIL',value_pp=size,threshold_pp=2.,sector=SECTORS[pos[0]],
                  turbine=tids[pos[1]],signed_five_seed_mean_d_pp=mean_d[pos]),
        main_text_candidate=dict(status='PASS' if consistency and size_pass else 'FAIL',qualifies=consistency and size_pass),
        final_verdict='MAIN_TEXT_CANDIDATE' if consistency and size_pass else 'DOES_NOT_QUALIFY_KEEP_FIG16',
        wake_reading=wake,deviations=[])
    js('decision.json',decision)
    log(f'Decision={decision["final_verdict"]}; correlations={[x["pearson_r"] for x in loo]}; max_abs_mean_d={size:.6g} pp; sparse={decision["sparse_sectors"]}.')


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.patches import FancyArrowPatch
    from matplotlib import font_manager
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'grid.linestyle':':','grid.color':'#cdd1d4','grid.linewidth':.6,'axes.axisbelow':True,
        'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'savefig.facecolor':'white'})
    with np.load(OUT/'sector_matrices.npz',allow_pickle=False) as z:
        matrices = z['matrices'].mean(0)
        xy = z['coords_m'].astype(float); xy=(xy-xy.min(0))/1000
        tids=z['turbine_ids'].astype(str)
    summary = pd.read_csv(OUT/'sector_summary.csv')
    dec = json.loads((OUT/'decision.json').read_text(encoding='utf-8'))
    shares = matrices.mean(1)*100
    vmax = max(10.,float(np.ceil(shares.max())))
    cmap = LinearSegmentedColormap.from_list('white_red',['white','#d62728'])
    norm = Normalize(0,vmax)
    fig = plt.figure(figsize=(7,2.65))
    edges = []
    for s,name in enumerate(SECTORS):
        col=s%4; row=s//4
        ax = fig.add_axes([([.42,2.06,3.70,5.34][col])/7,([1.66,.77][row])/2.65,1.46/7,(1.46/3)/2.65])
        a = matrices[s]
        scores=a.copy();np.fill_diagonal(scores,-np.inf)
        idx=np.argsort(scores,axis=1,kind='stable')[:,-2:]
        selected={(q,int(k)) for q in range(29) for k in idx[q]}
        assert len(selected)==58
        for q,k in sorted(selected,key=lambda pair:a[pair]):
            weight=float(a[q,k]);alpha=.30+.55*np.clip(weight/.15,0,1)
            arrow=FancyArrowPatch(xy[k],xy[q],arrowstyle='-|>',mutation_scale=2.6,
                linewidth=.55*(.3+10*weight),color='#275f86',alpha=alpha,shrinkA=.9,shrinkB=.9,
                connectionstyle='arc3,rad=0',zorder=1)
            ax.add_patch(arrow)
            edges.append(dict(sector=name,query_turbine=tids[q],key_turbine=tids[k],mean_weight=weight))
        ax.scatter(xy[:,0],xy[:,1],s=7.5,c=shares[s],cmap=cmap,norm=norm,edgecolors='#222222',linewidths=.3,zorder=3)
        ax.set(xlim=(-.3,9.45),ylim=(-.25,3.),aspect='equal',xticks=[0,4,8],yticks=[0,2])
        ax.grid(True);ax.tick_params(which='both',direction='in',top=True,right=True,length=2,pad=2)
        if col==0:
            ax.set_ylabel('North (km)',labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        if row==1:
            ax.set_xlabel('East (km)',labelpad=2)
        counts=summary[summary.sector==name].origins
        count_text=f'{int(counts.iloc[0]):,}' if counts.nunique()==1 else f'{int(counts.min()):,}–{int(counts.max()):,}'
        tag=' [sparse]' if name in dec['sparse_sectors'] else ''
        ax.text(0,1.40,f'({chr(97+s)}) {name}{tag}',transform=ax.transAxes,fontsize=8,fontweight='bold',ha='left',va='bottom')
        ax.text(0,1.09,f'n = {count_text} origins',transform=ax.transAxes,fontsize=7.5,ha='left',va='bottom')
    cax=fig.add_axes([1.85/7,.25/2.65,3.4/7,.075/2.65])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    cb.set_label('Received share (%)',fontsize=8,labelpad=2)
    cb.ax.tick_params(direction='in',length=2,labelsize=7.5,pad=2)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            box=t.get_window_extent(renderer)
            inside=box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1
            texts.append(dict(text=t.get_text(),inside_canvas=inside,font_pt=t.get_fontsize()))
    assert all(t['inside_canvas'] for t in texts), [t for t in texts if not t['inside_canvas']]
    # Preserve exact 7-inch physical width; no bbox_inches='tight'.
    fig.savefig(OUT/'fig_e3_sector_maps.png',dpi=600)
    fig.savefig(OUT/'fig_e3_sector_maps.pdf')
    fig.savefig(OUT/'fig_e3_sector_maps.svg')
    pd.DataFrame(edges).to_csv(OUT/'figure_displayed_links.csv',index=False,encoding='utf-8-sig')
    js('figure_qa.json',dict(status='PASS',width_in=7,height_in=2.65,dpi=600,font=font_manager.findfont('Times New Roman'),
        source_style_sha256=sha(FIG_SOURCE),edge_selection='58 strongest-two non-self links/panel, stable argsort, key->query',
        scale_pct=[0,vmax],statistics='five-seed mean; sparse panels kept; n=distinct included origins per seed',
        adaptation='same house palette/serif/geographic aspect; s7.5 nodes, .55 arrow width scaling, smaller arrowheads/shrink for tiny panels',
        text_bounds=texts,visual_inspection='pending',svg_fonttype='none',pdf_fonttype=42))
    plt.close(fig)


def report():
    d=json.loads((OUT/'decision.json').read_text(encoding='utf-8'))
    c=json.loads((OUT/'convention_check.json').read_text(encoding='utf-8'))['per_seed'][0]
    gates=json.loads((OUT/'reproduction_gates.json').read_text(encoding='utf-8'))
    means=pd.read_csv(OUT/'sector_summary_five_seed_mean_sd.csv')
    rs=', '.join(f'{x["seed"]}: {x["pearson_r"]:.6f}' for x in d['consistency']['per_seed'])
    max_delta=max(x['max_delta_pu'] for x in gates)
    sparse=', '.join(d['sparse_sectors']) or 'None / 无'
    table=['|Sector|Origins / seed|Rows / seed|Entropy mean ± SD|Max weight mean ± SD|Upwind mean ± SD|Uniform mean ± SD|Above expectation|Sparse|',
           '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r,w in zip(means.itertuples(),d['wake_reading']):
        table.append(f'|{r.sector}|{r.origins_mean:.0f} ± {r.origins_sd:.0f}|{r.rows_mean:.0f} ± {r.rows_sd:.0f}|{r.entropy_mean:.4f} ± {r.entropy_sd:.4f}|{r.max_weight_mean:.4f} ± {r.max_weight_sd:.4f}|{r.upwind_index_mean:.4f} ± {r.upwind_index_sd:.4f}|{r.uniform_expectation_mean:.4f} ± {r.uniform_expectation_sd:.4f}|{w["description"]}|{r.sparse}|')
    verdict_cn='符合正文候选条件' if d['main_text_candidate']['qualifies'] else '不符合正文候选条件；Fig. 16 保持原样'
    text=f'''# E3: wind-direction sectors / 风向扇区诊断

**结论：{verdict_cn}。** 一次推理流程覆盖五个冻结 h16 测试运行（seeds 13, 21, 42, 87, 3407）；无训练、模型选择或现有模型/结果修改。

冻结设计在推理前写入。保留原脚本的元数据、目标、预处理、预测和权重断言；五个运行的复现状态为 {', '.join(str(x['seed'])+': '+x['status'] for x in gates)}。最大逐点预测差为 {max_delta:.9g} 额定功率。GPU 例外使用次数为 {sum(x['allowed_gpu_exception_used'] for x in gates)}。全部输入及源代码 SHA-256 前后一致。

模型最后两个特征为 sin(atan2(u,v))、cos(atan2(u,v))，是风吹去的方向，未带负号。模型恢复标准化后在风机维度求均值；扇区按 atan2(−w_x,−w_y) 转为从北顺时针的气象来向。冻结配置实际使用 10 m U/V。采用转换后的同一模型向量划分扇区，保留原 wind-to 向量计算 alignment。与目标时刻原始 100 m 来向比较，全部 {c['model_from_vs_raw100_all']['n']} 个测试起点的绝对角差中位数为 {c['model_from_vs_raw100_all']['median_abs_deg']:.6f}°，均值 {c['model_from_vs_raw100_all']['mean_abs_deg']:.6f}°，P90 {c['model_from_vs_raw100_all']['p90_abs_deg']:.6f}°；纳入起点中位数为 {c['model_from_vs_raw100_included']['median_abs_deg']:.6f}°。与原始 10 m 的最大差为 {c['model_from_vs_raw10_all']['max_abs_deg']:.8f}°。未转换方向与 100 m 的中位差为 {c['unconverted_model_to_vs_raw100_all']['median_abs_deg']:.6f}°，验证了符号转换；100 m 剩余差异反映高度不同。

仅纳入目标有效、normal、起点末五次原始风速有限且非负、均值≥3 m/s的 query×origin 行；不按变异性分层。每个 query 分别平均形成矩阵，再等权平均29个 query 得到含自连接的 received share。熵、最大权重、上风指标及其均匀期望均按纳入行等权平均。上风指标为正 alignment 邻居的注意力占非零 alignment 邻居总注意力的比例，自连接不参与；均匀期望为相应邻居个数比例。跨种子 SD 使用 ddof=1，无显著性检验。

稀疏扇区（任一种子少于200个纳入起点）：{sparse}。它们保留在图表中，排除出 d 的基线和判据。跨种子留一相关为 {rs}；一致性判据 **{d['consistency']['status']}**。五种子平均 d 的最大绝对值为 **{d['size']['value_pp']:.6f} pp**（{d['size']['sector']}, {d['size']['turbine']}, 带符号 {d['size']['signed_five_seed_mean_d_pp']:.6f} pp）；2.0 pp大小判据 **{d['size']['status']}**。最终判定 **{d['final_verdict']}**。

逐扇区上风检验如下。只有5/5种子超过期望的扇区允许作物理尾流解读；满足此必要条件也不能证明因果尾流机制。扇区图展示五种子均值；颜色含自连接且共用色标；箭头按每个 query 两条最强非自连接，方向为 key→query。图是7英寸宽的草稿，不替换现有图件。

{chr(10).join(table)}

**偏离：无分析阈值或统计规则偏离。** 10 m/100 m高度差已明确记录；使用模型10 m向量保留同一NWP的设计。4列小图缩小节点、箭头宽度/箭头头部，保持原Fig.16筛边规则、配色、字体和地理比例。此前缺少“s”的路径已按本次明确路径修正；此前未执行推理。

**English. Verdict: {d['final_verdict']}.** One inference pass was performed for each of the five frozen h16 test models, in FP32, batch16, two threads, with CUDA memory fraction0.55. No training, model selection or existing result changes occurred. All five reproduction gates passed (maximum pointwise difference {max_delta:.9g} rated power); {sum(x['allowed_gpu_exception_used'] for x in gates)} preregistered GPU exceptions were used. Metadata, truth and weights passed the original checks; all input hashes stayed unchanged.

The last two restored NWP features are sin/cos of atan2(u,v), a wind-to vector from the configured10 m components. Meteorological wind-from sectors use atan2(−w_x,−w_y), clockwise from north. The model vector is retained for sector assignment after conversion, and its original wind-to sign is retained for alignment. Against raw100 m target-time wind-from direction, the median absolute difference is {c['model_from_vs_raw100_all']['median_abs_deg']:.6f}° over all test origins ({c['model_from_vs_raw100_included']['median_abs_deg']:.6f}° over included origins); raw10 m maximum difference is {c['model_from_vs_raw10_all']['max_abs_deg']:.8f}°. The height mismatch explains residual disagreement and is explicitly reported.

Rows require a valid normal target and five finite nonnegative raw wind observations ending at origin with mean≥3 m/s; no variability split is used. Matrices average eligible origins separately for each query. Received shares equally average29 query means, including self. Entropy, maximum weight, conditional upwind share and uniform expectation average eligible query-origin rows. The table reports five-seed mean±sample SD. Sparse sectors: {sparse}.

Leave-one-seed-out correlations: {rs}; consistency **{d['consistency']['status']}** at0.5 for every seed. Largest absolute five-seed mean d: **{d['size']['value_pp']:.6f} percentage points**, size **{d['size']['status']}** at2.0. Main-text candidacy: **{d['main_text_candidate']['status']}**. Wake eligibility is reported per sector above as “n of5 seeds above expectation”; only5/5 permits a wake reading, without establishing causality.

No analytical thresholds or selection rules were changed. The draft uses the supplied house style and exact top-two nonself link rule, with smaller marks for the2×4 layout. Outputs include per-seed summaries, a separate five-seed mean±SD CSV, full sector matrices, received shares and d, convention checks, reproduction gates and the frozen decision.
'''
    (OUT/'report_E3.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare:
            prepare()
        elif args.run:
            infer()
        else:
            render()
    except Exception as exc:
        status='STOPPED_FAILED_GATE' if (OUT/'inference_once.json').exists() else 'STOPPED_PREFLIGHT'
        if args.run and CURRENT_SEED is not None and not any(g['seed']==CURRENT_SEED and g['status']=='FAIL' for g in GATES):
            GATES.append(dict(seed=CURRENT_SEED,status='FAIL',error=str(exc)))
            js('reproduction_gates.json',GATES)
        js('stopped.json',dict(status=status,at_utc=utc(),seed=CURRENT_SEED,error=str(exc),traceback=traceback.format_exc()))
        log(f'{status}: {exc}')
        (OUT/'report_E3.md').write_text(f'# E3 stopped / 停止\n\n{status}\n\n{exc}\n\n按要求停止，未重试推理。Stopped without rerunning inference.\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
