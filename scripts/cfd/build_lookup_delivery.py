"""Package the fixed 38-case CFD evidence. Never trains, solves or re-summarizes.

The controlling scientific acceptance is the existing strict summary. This
writer independently checks its complete finite identities, small D/E report
bytes, table keys, vectors and paired arithmetic. Archive SHA values in the
source index retain their original verified-evidence origin; archives are not
re-read here. Before formal38 the only output is PREPARED status beside source.
"""
from pathlib import Path
import argparse, collections, contextlib, csv, datetime, hashlib, io, json
import math, os, shutil, statistics, sys, tempfile, uuid

HERE=Path(__file__).resolve().parent
WORK=next(p for p in HERE.parents if (p/'scripts/summarize_cfd_results.py').is_file())
OUT=WORK.parent/'补充验证_20261023'
EBASE=Path('results/paper')
EPREP=EBASE/'closeout_20261024/report_prepared'
DEST=OUT/'交付_20261024'
EDEST=EBASE/'delivery_20261024'
SETTINGS_SHA='e955155a03911e4266bb8a171e664f0ae98c828d99d16b8572bae7d588e613cc'
SUMMARY_SOURCE_SHA='094de6c82447478271dc0f09747df861b59cb63d3675a5f6539eaa62b41b90b3'
MPI_KIND='mpi_finalize_post_only_recovery'
LOADED_KIND='loaded_ct08_applicable43_count_correction'
MPI_RIDS=frozenset('energy-r4-wind_farm_A-sector-a'+x+'-z0p3-ct0-v1' for x in ('022p50','225p00','292p50','337p50'))
LOADED_RIDS=frozenset('energy-r4-wind_farm_A-wake_pair-a'+x+'-z0p3-ct0p8-v1' for x in ('090p00','112p50','135p00','337p50'))
SAMPLING_RID='energy-r4-wind_farm_A-sector-a090p00-z0p3-ct0-v1'
KEL_RECOVERY_RID='energy-r4-kelmarsh-sector-a240p00-z0p05-ct0-v1'
TABLES={
 'cfd_wind_farm_A_turbine_sector.csv':(464,'vector','wind_farm_A','sector'),
 'cfd_kelmarsh_turbine_sector.csv':(72,'vector','kelmarsh','sector'),
 'cfd_wind_farm_A_intersector_direct.csv':(116,'vector','wind_farm_A','intersector_check'),
 'cfd_wind_farm_A_intersector_errors.csv':(116,'interpolation',None,None),
 'cfd_roughness_sensitivity_sources.csv':(35,'vector',None,'roughness_sensitivity'),
 'cfd_roughness_paired_changes.csv':(35,'roughness',None,None),
 'cfd_wind_farm_A_wake_Ct08.csv':(116,'vector','wind_farm_A','wake_pair'),
 'cfd_wind_farm_A_wake_paired_changes.csv':(116,'wake',None,None),
}
METADATA=('cfd_wind_farm_A_metadata.json','cfd_kelmarsh_metadata.json')
VECTOR_NUMBERS=('sector_deg','hub_height_m','speedup','local_dir_deg','delta_dir_deg',
 'u_east_mps','v_north_mps','w_mps','horizontal_speed_mps','speed3d_mps','inlet_at_hub_mps','z0_m','ct')
PROFILE_COUNTS={'native':28,'sampling_recovery':1,'existing_post_only_recovery':1,
 'mpi_finalize_post_only_recovery':4,'loaded43_count_correction':4}

class GateError(ValueError):pass
class Pending(GateError):pass
def need(value,message):
 if not value:raise GateError(message)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_json(path):return json.loads(Path(path).read_bytes(),parse_constant=lambda v:(_ for _ in ()).throw(GateError('Nonfinite JSON '+v)))
def descriptor(path):
 p=Path(path);need(p.is_file() and not p.is_symlink(),'Regular file required: '+str(p));raw=p.read_bytes()
 return {'path':str(p),'bytes':len(raw),'sha256':digest(raw)}
def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
def atomic(path,raw):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
 with t.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def wrap(value):return (value+180.)%360.-180.
def number(value,name='number'):
 try:result=float(value)
 except (ValueError,TypeError) as ex:raise GateError('Invalid '+name) from ex
 need(math.isfinite(result),'Nonfinite '+name);return result
def truth(value):return value is True or isinstance(value,str) and value.lower()=='true'
def close(a,b):return math.isclose(number(a),number(b),rel_tol=1e-8,abs_tol=1e-8)
def csv_rows(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
def csv_bytes(rows,fields=None):
 stream=io.StringIO(newline='');fields=fields or (list(rows[0]) if rows else [])
 writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 return stream.getvalue().encode('utf-8-sig')
def exact_keys(rows,key,expected,label):
 keys=[key(row) for row in rows]
 need(len(keys)==len(set(keys)),label+' duplicate keys')
 need(set(keys)==set(expected),label+' missing or unexpected keys')
def summary_final(summary,settings_raw):
 need(digest(settings_raw)==SETTINGS_SHA,'Frozen settings SHA differs')
 need(summary.get('settings_sha256')==SETTINGS_SHA and summary.get('summary_script_sha256')==SUMMARY_SOURCE_SHA,'Strict summary source/settings identity differs')
 count=summary.get('complete_cases');need(type(count) is int and 0<=count<=38 and summary.get('required_cases')==38,'Invalid finite case count')
 if count<38:raise Pending('Formal strict summary is '+str(count)+'/38; no final report generated')
 need(summary.get('rejected_cases')==[],'Formal38 must have zero rejected cases')
 need(summary.get('source_profile_counts')==PROFILE_COUNTS,'Final accepted source profile counts differ')

def case_infos(plan):
 cases=plan.get('case_queue');need(isinstance(cases,list) and len(cases)==38,'Exactly frozen38 cases required')
 need(len({x['run_id'] for x in cases})==38,'Duplicate frozen run_id')
 expected=collections.Counter({('wind_farm_A','sector'):16,('kelmarsh','sector'):12,('wind_farm_A','intersector_check'):4,('wind_farm_A','wake_pair'):4,('wind_farm_A','roughness_sensitivity'):1,('kelmarsh','roughness_sensitivity'):1})
 need(collections.Counter((x['site'],x['kind']) for x in cases)==expected,'Frozen scientific scope differs')
 infos={};site_ids={}
 for case in cases:
  p=Path(case['bundle'])/'frozen_info.json';raw=p.read_bytes();need(digest(raw)==case['input_sha256'],'Frozen input science SHA differs: '+case['run_id'])
  info=read_json(p);need(info['key']==case['run_id'] and info['site']==case['site'] and info['case_kind']==case['kind'],'Frozen input identity differs')
  need(close(info['wind']['angle'],case['angle_deg']) and close(info['wind']['roughnessLength'],case['z0_m']),'Frozen physical settings differ')
  turbines={str(x['id']):x for x in info['turbines']};need(len(turbines)==(29 if case['site']=='wind_farm_A' else 6),'Wrong physical turbine count')
  if case['site'] in site_ids:need(set(turbines)==site_ids[case['site']],'Turbine identities changed between cases')
  site_ids[case['site']]=set(turbines);infos[case['run_id']]=info
 return {x['run_id']:x for x in cases},infos

def source_category(source):
 if source.get('source_kind')==LOADED_KIND:return 'loaded43_count_correction'
 if source.get('source_kind')==MPI_KIND:return MPI_KIND
 if source.get('sampling_recovery') is True:return 'sampling_recovery'
 if source.get('recovery') is True:return 'existing_post_only_recovery'
 return 'native'

def check_small_pair(desc):
 need(isinstance(desc,dict) and all(k in desc for k in ('D_path','E_path','bytes','sha256')),'D/E descriptor incomplete')
 p,q=Path(desc['D_path']),Path(desc['E_path']);raw=p.read_bytes();need(q.read_bytes()==raw and len(raw)==desc['bytes'] and digest(raw)==desc['sha256'],'D/E small evidence differs')
 return json.loads(raw)

def validate_sources(summary,cases):
 sources=summary.get('all_case_sources');need(isinstance(sources,dict) and set(sources)==set(cases),'Exactly38 unique accepted source identities required')
 categories=collections.Counter()
 for rid,source in sources.items():
  case=cases[rid];category=source_category(source);categories[category]+=1
  need(source.get('strict_full_file_stream_verified') is True,'Prior complete strict D/E stream evidence missing: '+rid)
  if source.get('run_id') is not None:need(source['run_id']==rid,'Accepted source RID differs')
  if source.get('scientific_identity_sha256') is not None:need(source['scientific_identity_sha256']==case['input_sha256'],'Accepted source science differs')
  if category==MPI_KIND:
   need(rid in MPI_RIDS and source.get('source_profile')=='mpi_finalize_postprocessing_recovery' and source.get('recovery_accepted') is True,'Unknown MPI recovery exception')
   need(source.get('original_execution_exit_code')==1 and source.get('original_quality_status')=='FAIL' and source.get('original_failed_checks')==['actual_process_exit_zero'] and source.get('native_process_success_claimed') is False,'MPI original failure must remain')
   gate=read_json(source['root_acceptance']);receipt=read_json(source['D_E_verification'])
   signature=source['mpi_finalize_verification_signature']
   for key,path in [('root_acceptance',source['root_acceptance']),('D_E_verification',source['D_E_verification'])]:
    need(descriptor(path)==signature[key],'Actual MPI small evidence differs from strict accepted signature')
   need(gate.get('status')=='ROOT_ACCEPTED_DERIVED_RECOVERY' and gate.get('run_id')==rid and gate.get('science_sha256')==case['input_sha256'] and gate.get('native_process_success_claimed') is False,'MPI root acceptance identity differs')
   need(receipt.get('status')=='PASS' and receipt.get('run_id')==rid and receipt.get('scientific_identity_sha256')==case['input_sha256'] and receipt.get('other_44_original_checks_all_pass') is True and receipt.get('post_only_actual_exit_code')==0 and receipt.get('raw_assessor_actual_exit_code')==1 and receipt.get('original_execution_exit_code')==1 and receipt.get('native_process_success_claimed') is False and receipt.get('independent_D_E_full_member_SHA_CRC_verified') is True,'Exact MPI derived44/post0 evidence absent')
   copies=gate.get('copies',[]);need(len(copies)==2 and {x['disk'] for x in copies}=={'D','E'},'MPI receipt needs original D/E copies')
   for copy in copies:
    p=Path(copy['path']);need(descriptor(p)=={k:copy[k] for k in ('path','bytes','sha256')},'MPI small receipt bytes differ')
  elif category=='loaded43_count_correction':
   need(rid in LOADED_RIDS and case['kind']=='wake_pair' and close(case['ct'],.8),'Unknown loaded43 exception')
   need(source.get('original_worker_gate_status')=='FAIL' and source.get('original_assessor_status')=='PASS' and source.get('applicable_check_count')==43 and source.get('all_applicable_checks_pass') is True and source.get('native_natural_exit_code')==0 and source.get('assessor_natural_exit_code')==0 and source.get('not_a_solver_or_postprocessing_rerun') is True,'Loaded43 is only exact count correction after real native0/assessor0')
   gate=check_small_pair(source['loaded43_verification_signature']['root_source'])
   need(gate.get('kind')=='root_finite_loaded43_count_correction_acceptance' and gate.get('status')=='ROOT_ACCEPTED_LOADED43_APPLICABLE_CHECKS' and gate.get('run_id')==rid and (gate.get('scientific_identity_sha256') or gate.get('science_sha256'))==case['input_sha256'],'Loaded43 root science/status differs')
  elif category=='sampling_recovery':
   need(rid==SAMPLING_RID and source.get('original_quality_status')=='FAIL' and source.get('original_failed_checks')==['all_AGL_layers_complete'],'Only original exact wind_farm_A sampling recovery permitted')
  elif category=='existing_post_only_recovery':need(rid==KEL_RECOVERY_RID,'Only original exact Kelmarsh post/sampling recovery permitted')
  else:need(source.get('source_kind') in (None,'guarded_g4_finite_four'),'Unknown native source profile')
 need(dict(categories)==PROFILE_COUNTS,'Actual finite source distribution differs')
 need({rid for rid,s in sources.items() if source_category(s)==MPI_KIND}==MPI_RIDS,'All four original MPI recoveries must remain')
 need({rid for rid,s in sources.items() if source_category(s)=='loaded43_count_correction'}==LOADED_RIDS,'All four exact loaded43 sources required')
 return sources

def validate_vector(row,case,info,source):
 rid=case['run_id'];need(row.get('run_id')==rid and row.get('site')==case['site'] and row.get('case_kind')==case['kind'],'Table site/run/kind identity differs')
 turbines={str(x['id']):x for x in info['turbines']};need(str(row.get('turbine_id')) in turbines,'Unexpected turbine')
 if not truth(row.get('quality_pass')):
  need(rid in MPI_RIDS and case['kind']=='sector' and case['site']=='wind_farm_A' and source_category(source)==MPI_KIND and truth(row.get('recovery_accepted')) and row.get('source_kind')==MPI_KIND,'RawFalse row lacks exact legitimate MPI recovery')
 need(truth(row.get('interpolation_supported','True')),'Unsupported vector')
 r=dict(row)
 for name in VECTOR_NUMBERS:r[name]=number(row[name],name)
 need(close(r['sector_deg'],case['angle_deg']) and close(r['ct'],case['ct']) and close(r['z0_m'],case['z0_m']),'Table angle/Ct/z0 identity differs')
 hub=number(turbines[str(row['turbine_id'])]['hub']);wind=info['wind'];z0=number(wind['roughnessLength']);ref=number(wind['referenceHeight'])
 need(close(r['hub_height_m'],hub),'Hub height differs')
 inlet=number(wind['speed'])*math.log1p(hub/z0)/math.log1p(ref/z0)
 need(all(r[k]>0 for k in ('speedup','horizontal_speed_mps','inlet_at_hub_mps')),'Nonpositive vector ratio')
 need(close(r['inlet_at_hub_mps'],inlet) and close(math.hypot(r['u_east_mps'],r['v_north_mps']),r['horizontal_speed_mps']) and close(math.sqrt(r['u_east_mps']**2+r['v_north_mps']**2+r['w_mps']**2),r['speed3d_mps']) and close(r['horizontal_speed_mps']/inlet,r['speedup']),'Vector/ABL denominator/ratio mismatch')
 direction=math.degrees(math.atan2(-r['u_east_mps'],-r['v_north_mps']))%360
 need(0<=r['local_dir_deg']<360 and -180<=r['delta_dir_deg']<180 and abs(wrap(r['local_dir_deg']-direction))<1e-7 and abs(wrap(r['delta_dir_deg']-wrap(direction-r['sector_deg'])))<1e-7,'Meteorological direction/wrapping mismatch')
 return r

def validate_tables(raw_tables,cases,infos,sources):
 tables={};vectors={}
 for name,(count,kind,site,case_kind) in TABLES.items():
  rows=csv_rows(raw_tables[name]);need(len(rows)==count,name+' wrong row count')
  if kind=='vector':
   selected=[c for c in cases.values() if c['kind']==case_kind and (site is None or c['site']==site)]
   expected={(c['run_id'],str(t['id'])) for c in selected for t in infos[c['run_id']]['turbines']}
   exact_keys(rows,lambda r:(r['run_id'],str(r['turbine_id'])),expected,name)
   rows=[validate_vector(r,cases[r['run_id']],infos[r['run_id']],sources[r['run_id']]) for r in rows]
   for r in rows:vectors[(r['run_id'],r['turbine_id'])]=r
  tables[name]=rows
 baseline={(r['site'],r['turbine_id'],r['sector_deg']):r for n in ('cfd_wind_farm_A_turbine_sector.csv','cfd_kelmarsh_turbine_sector.csv') for r in tables[n]}
 pairspec=[('cfd_wind_farm_A_wake_paired_changes.csv','cfd_wind_farm_A_wake_Ct08.csv','wake'),('cfd_roughness_paired_changes.csv','cfd_roughness_sensitivity_sources.csv','roughness')]
 for dest,source,mode in pairspec:
  expected={(r['site'],r['turbine_id'],r['sector_deg']) for r in tables[source]};rows=tables[dest]
  exact_keys(rows,lambda r:(r['site'],r['turbine_id'],number(r['sector_deg'])),expected,dest)
  for r in rows:
   angle=number(r['sector_deg']);b=baseline[(r['site'],r['turbine_id'],angle)];altkey='loaded_run_id' if mode=='wake' else 'alternative_run_id';a=vectors[(r[altkey],r['turbine_id'])]
   need(a['site']==r['site'] and close(a['sector_deg'],angle),'Paired site/direction differs')
   if mode=='wake':
    need(r['terrain_run_id']==b['run_id'] and a['case_kind']=='wake_pair','Wake baseline/loaded run differs')
    for k,v in {'terrain_horizontal_speed_mps':b['horizontal_speed_mps'],'loaded_horizontal_speed_mps':a['horizontal_speed_mps'],'loaded_minus_terrain_mps':a['horizontal_speed_mps']-b['horizontal_speed_mps'],'direction_change_deg':wrap(a['local_dir_deg']-b['local_dir_deg'])}.items():need(close(r[k],v),'Wake paired arithmetic differs: '+k)
   else:
    need(r['baseline_run_id']==b['run_id'] and a['case_kind']=='roughness_sensitivity','Roughness baseline/alternative differs')
    for k,v in {'baseline_z0_m':b['z0_m'],'alternative_z0_m':a['z0_m'],'speedup_difference':a['speedup']-b['speedup'],'direction_difference_deg':wrap(a['local_dir_deg']-b['local_dir_deg'])}.items():need(close(r[k],v),'Roughness paired arithmetic differs: '+k)
   for k in r:
    if k.endswith(('_mps','_deg')) or k in ('speedup_difference','baseline_z0_m','alternative_z0_m'):number(r[k],k)
 direct=tables['cfd_wind_farm_A_intersector_direct.csv'];errors=tables['cfd_wind_farm_A_intersector_errors.csv'];expected={(r['turbine_id'],r['sector_deg']) for r in direct}
 exact_keys(errors,lambda r:(r['turbine_id'],number(r['check_direction_deg'])),expected,'Intersector errors')
 for r in errors:
  a=vectors[(r['direct_run_id'],r['turbine_id'])];angle=a['sector_deg'];need(a['case_kind']=='intersector_check' and close(angle,r['check_direction_deg']),'Wrong direct intersector run/angle');low=math.floor(angle/22.5)*22.5;high=(low+22.5)%360.;l=baseline[('wind_farm_A',r['turbine_id'],low)];h=baseline[('wind_farm_A',r['turbine_id'],high)];w=((angle-low)%360.)/22.5
  need(r['left_run_id']==l['run_id'] and r['right_run_id']==h['run_id'],'Wrong circular interpolation bracket')
  pd=wrap(l['delta_dir_deg']+w*wrap(h['delta_dir_deg']-l['delta_dir_deg']));ps=l['speedup']+w*(h['speedup']-l['speedup'])
  for k,v in {'direct_delta_deg':a['delta_dir_deg'],'interpolated_delta_deg':pd,'angle_error_deg':wrap(pd-a['delta_dir_deg']),'direct_speedup':a['speedup'],'interpolated_speedup':ps,'speedup_error':ps-a['speedup']}.items():need(close(r[k],v),'Intersector arithmetic differs: '+k)
 raw_false=[r for r in tables['cfd_wind_farm_A_turbine_sector.csv'] if not truth(r['quality_pass'])]
 need(len(raw_false)==116 and {r['run_id'] for r in raw_false}==MPI_RIDS,'Exact retained four MPI rawFalse rows differ')
 return tables

def quantile(values,q):
 values=sorted(number(x) for x in values);need(values and 0<=q<=1,'Empty quantile or invalid q');x=(len(values)-1)*q;i=math.floor(x);j=math.ceil(x)
 return values[i]+(x-i)*(values[j]-values[i])
def describe(rows,key,absolute=False):
 need(rows,'Empty descriptive population');vals=[abs(number(r[key])) if absolute else number(r[key]) for r in rows]
 i=min(range(len(vals)),key=vals.__getitem__);j=max(range(len(vals)),key=vals.__getitem__)
 def at(n):return {k:rows[n][k] for k in ('site','turbine_id','sector_deg','check_direction_deg','run_id','loaded_run_id','alternative_run_id','direct_run_id') if k in rows[n]}
 return {'n_spatial_rows':len(vals),'mean':statistics.fmean(vals),'population_sd':statistics.pstdev(vals),'median':quantile(vals,.5),'p90':quantile(vals,.9),'minimum':vals[i],'maximum':vals[j],'minimum_at':at(i),'maximum_at':at(j),'negative_count':sum(v<0 for v in vals),'positive_count':sum(v>0 for v in vals),'zero_count':sum(v==0 for v in vals),'absolute_values':absolute,'weighting':'equal_turbine_x_direction_rows_not_wind_frequency','independent_seed_replicates':False,'CI_or_p_value_computed':False}
def grouped_descriptions(rows,group_keys,metrics):
 groups=collections.defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in group_keys)].append(r)
 out=[]
 for keys,rs in sorted(groups.items(),key=lambda kv:tuple(str(x) for x in kv[0])):
  record=dict(zip(group_keys,keys));record['n_rows']=len(rs)
  for name,key,absolute in metrics:
   v=describe(rs,key,absolute)
   for stat in ('mean','population_sd','median','p90','minimum','maximum'):record[name+'_'+stat]=v[stat]
  out.append(record)
 return out
def statistics_bundle(tables):
 result={'methods':{'population':'fixed deterministic turbine-by-direction CFD response, not training seeds or stochastic replicates','weights':'equal turbine x equal direction; no observed-frequency weighting','quantile':'linear interpolation at (n-1)*q','paired_sign':'alternative/loaded minus baseline/terrain','direction_change':'wrapped to [-180,180) degrees','spatial_sd':'population SD (ddof0); no CI/p values or hypothesis test'},'default_lut':{}}
 for site in ('wind_farm_A','kelmarsh'):
  r=tables['cfd_'+site+'_turbine_sector.csv'];result['default_lut'][site]={'speedup':describe(r,'speedup'),'signed_deflection_deg':describe(r,'delta_dir_deg'),'absolute_deflection_deg':describe(r,'delta_dir_deg',True)}
 result['interpolation']={k:describe(tables['cfd_wind_farm_A_intersector_errors.csv'],field,absolute) for k,field,absolute in [('signed_direction_error_deg','angle_error_deg',False),('absolute_direction_error_deg','angle_error_deg',True),('signed_speedup_error','speedup_error',False),('absolute_speedup_error','speedup_error',True)]}
 result['wake']={'signed_speed_change_mps':describe(tables['cfd_wind_farm_A_wake_paired_changes.csv'],'loaded_minus_terrain_mps'),'signed_direction_change_deg':describe(tables['cfd_wind_farm_A_wake_paired_changes.csv'],'direction_change_deg'),'absolute_direction_change_deg':describe(tables['cfd_wind_farm_A_wake_paired_changes.csv'],'direction_change_deg',True)}
 result['roughness']={}
 for site in ('wind_farm_A','kelmarsh'):
  r=[r for r in tables['cfd_roughness_paired_changes.csv'] if r['site']==site];result['roughness'][site]={'signed_speedup_change':describe(r,'speedup_difference'),'signed_direction_change_deg':describe(r,'direction_difference_deg'),'absolute_direction_change_deg':describe(r,'direction_difference_deg',True)}
 return result

def read_snapshot():
 paths=[OUT/'cfd_results_summary.json',OUT/'cfd_settings_frozen.json']+[OUT/n for n in TABLES]+[OUT/n for n in METADATA]
 raw={p.name:p.read_bytes() for p in paths};summary=json.loads(raw['cfd_results_summary.json']);summary_final(summary,raw['cfd_settings_frozen.json'])
 for p in paths:
  ep=EBASE/'reports'/p.name
  if not ep.is_file() or ep.read_bytes()!=raw[p.name]:raise Pending('Waiting for same-byte E/reports copy: '+p.name)
 plan=json.loads(raw['cfd_settings_frozen.json']);cases,infos=case_infos(plan);sources=validate_sources(summary,cases)
 tables=validate_tables({n:raw[n] for n in TABLES},cases,infos,sources)
 for site,count,rows in [('wind_farm_A',16,464),('kelmarsh',12,72)]:
  meta=json.loads(raw['cfd_'+site+'_metadata.json']);need(meta.get('site')==site and meta.get('complete') is True and meta.get('completed_sector_cases')==count and meta.get('required_sector_cases')==count and meta.get('accepted_turbine_rows')==rows and meta.get('unique_turbine_sector_rows')==rows,'Default LUT metadata incomplete')
 return {'raw':raw,'summary':summary,'plan':plan,'cases':cases,'infos':infos,'sources':sources,'tables':tables,'input_sha256':{n:digest(b) for n,b in raw.items()}}

def evidence_index(cases,sources):
 rows=[]
 def add(rid,role,path,recorded_sha='',recorded_bytes='',origin='recorded_prior_strict_evidence'):
  if not isinstance(path,str) or not (':' in path[:3] or path.startswith('/')):return
  p=Path(path);local=p.is_absolute() and bool(p.drive);exists=p.exists() if local else False
  rows.append({'run_id':rid,'site':cases[rid]['site'],'case_kind':cases[rid]['kind'],'json_role':role,'path':path,'recorded_sha256':recorded_sha,'recorded_bytes':recorded_bytes,'SHA_origin':origin,'currently_local_path_exists':exists,'fresh_archive_rehash_performed':False})
 def walk(rid,value,pointer):
  if isinstance(value,dict):
   if isinstance(value.get('path'),str):add(rid,pointer,value['path'],value.get('sha256',''),value.get('bytes',''))
   if isinstance(value.get('D_path'),str):
    for k in ('D_path','E_path'):add(rid,pointer+'/'+k,value.get(k),value.get('sha256',''),value.get('bytes',''))
   for k,v in value.items():
    if isinstance(k,str) and (k.startswith('/') or ':/' in k or ':\\' in k) and isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v):add(rid,pointer+'/'+k,k,v)
    if k not in ('path','D_path','E_path'):walk(rid,v,pointer+'/'+k)
  elif isinstance(value,list):
   for i,v in enumerate(value):walk(rid,v,pointer+'/'+str(i))
  elif isinstance(value,str):add(rid,pointer,value)
 for rid,source in sources.items():
  walk(rid,source,'all_case_sources/'+rid)
  for key in ('D_archive','E_archive'):
   if source.get(key):add(rid,key,source[key],source.get('archive_sha256',''))
  for key in ('D_archives','E_archives'):
   for p in source.get(key,[]):
    h=source.get('quality_archive_sha256' if '.quality_addendum.' in p else 'base_archive_sha256','');add(rid,key,p,h)
  p=Path(cases[rid]['bundle'])/'frozen_info.json';add(rid,'frozen_input',str(p),cases[rid]['input_sha256'],p.stat().st_size,'fresh_small_frozen_input_SHA_verified')
 unique={tuple(str(r[k]) for k in ('run_id','json_role','path','recorded_sha256')):r for r in rows}
 return list(unique.values())

def package_case_rows(snapshot):
 out=[];failures=[]
 for rid,c in snapshot['cases'].items():
  s=snapshot['sources'][rid];category=source_category(s);root=s.get('root_acceptance',s.get('root_acceptance_gate',''))
  if category=='loaded43_count_correction':root=s['loaded43_verification_signature']['root_source']['D_path']
  r={'run_id':rid,'site':c['site'],'case_kind':c['kind'],'inlet_direction_deg':c['angle_deg'],'z0_m':c['z0_m'],'ct':c['ct'],'turbine_rows':len(snapshot['infos'][rid]['turbines']),'scientific_identity_sha256':c['input_sha256'],'acceptance_profile':category,'strict_full_D_E_verified':s['strict_full_file_stream_verified'],'original_native_exit_code':s.get('original_execution_exit_code',s.get('native_natural_exit_code','')),'original_quality_status':s.get('original_quality_status',s.get('original_worker_gate_status','')),'original_failed_checks':json.dumps(s.get('original_failed_checks',[]),ensure_ascii=False),'native_success_claimed':s.get('native_process_success_claimed',''),'root_acceptance':root,'quality':s.get('quality',''),'source_provenance_copy':'sources/'+rid+'.json'}
  out.append(r)
  if category!='native':failures.append(dict(r,explanation={'sampling_recovery':'原AGL采样不全FAIL保留，独立采样恢复验收','existing_post_only_recovery':'Kelmarsh既有后处理/采样恢复，原记录保留','mpi_finalize_post_only_recovery':'原native1/raw45 FAIL保留，另经44项+post0+root及D/E验收','loaded43_count_correction':'原worker固定计数FAIL保留，真实native0/assessor0且43项适用检查通过'}[category]))
 return out,failures

def markdown_report(snapshot,stats):
 s=snapshot['summary'];d=stats['default_lut']['wind_farm_A'];k=stats['default_lut']['kelmarsh'];inter=stats['interpolation'];wake=stats['wake']['signed_speed_change_mps'];tables=snapshot['tables']
 def f(v):return format(v,'.6f')
 def loc(v):return '、'.join(str(v[x]) for x in ('turbine_id','sector_deg','check_direction_deg','run_id','loaded_run_id','direct_run_id') if x in v)
 lines=['# 两风场CFD有限验证完整结果报告（38工况）','',f'本报告基于正式严格汇总截至 {s["updated_utc"]} 的38/38个工况及D/E同字节报告文件。Wind Farm A默认查找表为16个入流方向×29台机组，共464行；Kelmarsh为12个方向×6台机组，共72行。夹角直接计算4项、带风机对照4项、粗糙度敏感性2项单列，不混入默认Ct=0查找表。','',f'在固定仿真假设下，Wind Farm A局地风向绝对偏转的等机组×等方向均值为 {f(d["absolute_deflection_deg"]["mean"])}°，最大值 {f(d["absolute_deflection_deg"]["maximum"])}°；Kelmarsh对应为 {f(k["absolute_deflection_deg"]["mean"])}°和 {f(k["absolute_deflection_deg"]["maximum"])}°。这些是确定性空间描述统计，不是实测风频加权结果、训练种子均值或独立重复试验。','', '本批没有训练或重新评估预测模型，因此不据CFD结果宣称预测精度改善、最优粗糙度或现场验证成立。','', '## 1. 默认查找表与局地差异','', '| 风场 | 方向/机组/行数 | z0 (m) | 加速比最小/均值/最大 | 绝对偏转中位数/p90/最大 (°) |','|---|---|---:|---|---|']
 location_lines=[]
 for site,label,directions,turbines,z0 in [('wind_farm_A','Wind Farm A',16,29,.3),('kelmarsh','Kelmarsh',12,6,.05)]:
  a=stats['default_lut'][site];v=a['speedup'];b=a['absolute_deflection_deg'];lines.append(f'| {label} | {directions}/{turbines}/{directions*turbines} | {z0} | {f(v["minimum"])}/{f(v["mean"])}/{f(v["maximum"])} | {f(b["median"])}/{f(b["p90"])}/{f(b["maximum"])} |')
  location_lines+=['',f'{label}加速比最小位置：{loc(v["minimum_at"])}；最大位置：{loc(v["maximum_at"])}。绝对偏转最大位置：{loc(b["maximum_at"])}。']
 lines+=location_lines
 lines+=['','加速比为机组处水平风速除以同机组轮毂高度的冻结ABL入流风速；风向采用气象来向（北0°、顺时针），偏转及方向差均环绕到[-180°,180°)。带符号偏转、逐方向和逐机组汇总见 `summary_statistics.json`、`per_direction_descriptive.csv`、`per_turbine_descriptive.csv`。','', '## 2. 独立夹角计算与插值核对','',f'4个预先冻结夹角、每个29台机组，共116行。方向误差定义为“环形插值偏转−直接计算偏转”；加速比误差为“插值−直接计算”。方向绝对误差均值为 {f(inter["absolute_direction_error_deg"]["mean"])}°，最大 {f(inter["absolute_direction_error_deg"]["maximum"])}°（{loc(inter["absolute_direction_error_deg"]["maximum_at"])}）；加速比绝对误差均值 {f(inter["absolute_speedup_error"]["mean"])}，最大 {f(inter["absolute_speedup_error"]["maximum"])}（{loc(inter["absolute_speedup_error"]["maximum_at"])}）。','',f'方向有符号平均误差为 {f(inter["signed_direction_error_deg"]["mean"])}°，加速比有符号平均误差为 {f(inter["signed_speedup_error"]["mean"])}；绝对误差与有符号偏差不互换。所有直接结果、左右邻扇区和逐机组误差均保留在对应CSV。','', '## 3. Ct=0.8与纯地形Ct=0的配对对照','',f'4个冻结方向×29台机组，共116个同机组、同入流方向配对。“带风机−纯地形”的水平风速变化均值为 {f(wake["mean"])} m/s，最小 {f(wake["minimum"])} m/s（{loc(wake["minimum_at"])}），最大 {f(wake["maximum"])} m/s（{loc(wake["maximum_at"])}）。其中负值 {wake["negative_count"]} 行、正值 {wake["positive_count"]} 行、零值 {wake["zero_count"]} 行；正负结果全部保留。负值表示该局地采样风速较低，正值也不删除或改写为无效结果。','', '| 入流方向 (°) | 配对数 | 风速差均值 (m/s) | 最小/最大 (m/s) | 负值/正值行数 |','|---:|---:|---:|---|---|']
 for angle in sorted({number(r['sector_deg']) for r in tables['cfd_wind_farm_A_wake_paired_changes.csv']}):
  rows=[r for r in tables['cfd_wind_farm_A_wake_paired_changes.csv'] if close(r['sector_deg'],angle)];a=describe(rows,'loaded_minus_terrain_mps');lines.append(f'| {angle:g} | {len(rows)} | {f(a["mean"])} | {f(a["minimum"])}/{f(a["maximum"])} | {a["negative_count"]}/{a["positive_count"]} |')
 lines+=['','方向变化另存逐配对CSV，使用环形差而非跨0°的直接减法。上述差异反映这两类CFD工况的局地响应，不能直接解释为功率损失、预测收益或实测尾流验证。','', '## 4. 两项有限粗糙度敏感性','', '| 风场 | 固定方向 (°) | 基准/替代z0 (m) | 机组数 | 加速比差均值/最小/最大 | 方向差均值 (°) |','|---|---:|---|---:|---|---:|']
 for site,label in [('wind_farm_A','Wind Farm A'),('kelmarsh','Kelmarsh')]:
  rows=[r for r in tables['cfd_roughness_paired_changes.csv'] if r['site']==site];a=stats['roughness'][site];v=a['signed_speedup_change'];r=rows[0];lines.append(f'| {label} | {number(r["sector_deg"]):g} | {r["baseline_z0_m"]}/{r["alternative_z0_m"]} | {len(rows)} | {f(v["mean"])}/{f(v["minimum"])}/{f(v["maximum"])} | {f(a["signed_direction_change_deg"]["mean"])} |')
 lines+=['','差值均为“替代z0−基准z0”。这是每个风场一个预声明方向的敏感性检查，不是粗糙度搜索或最优值识别；基准z0=0.3 m和0.05 m没有因结果更有利而替换。方法依据及其适用限制见复制的方法说明。','', '## 5. 验收来源与原失败记录','', '正式38项包含28项原生来源、1项Wind Farm A采样恢复、1项Kelmarsh既有后处理/采样恢复、4项MPI收尾后处理恢复，以及4项Ct=0.8适用检查计数纠正。各类来源分别保留，未把恢复结果包装成原生成功。','', '四项MPI恢复的原生进程退出码1及原45项raw FAIL始终保留；正式接收依据是其余44项检查、真实post0、独立完整D/E成员与数值/索引核验及root接收。wind_farm_A默认LUT中116行仍为 `quality_pass=False`，准确对应这4个工况各29台机组，不是116次新增失败运行；`recovery_accepted=True`与来源字段也原样保留。其余348行和Kelmarsh72行保持原标志。','', '4项带风机工况保留原worker固定45项计数造成的FAIL；其原生求解、assessor均自然退出0，实际适用43项均通过。这属于精确有限范围的计数纠正，不是降低物理阈值或补跑求解。wind_farm_A90°采样层不全的原失败与Kelmarsh既有恢复也保留在其源目录。逐工况、全部来源和失败/恢复入口分别见 `case_index.csv`、`source_evidence_index.csv`、`failure_and_recovery_index.csv`。','', '## 6. 复核、图组与适用边界','', '8张原正式表、2份风场metadata、冻结settings和原严格汇总均以同字节复制进入本包；38份冻结输入和逐工况来源快照另存。完整原生场、网格、CFD检查点、全部日志和失败尝试仍保存在原D/E目录及归档中，通过索引链接访问，本步骤没有搬移或删除它们。归档SHA来自原严格验收证据，本打包步骤没有再次流式解压；小报告文件和复制输出SHA已重新核对。','', '统计均为固定网格、固定入流与粗糙度假设下的确定性空间描述。总体标准差仅表示这些机组×方向行之间的离散程度；未计算置信区间、显著性或多重检验。收敛与质量检查不能替代网格独立性试验、现场流场验证或机舱风向验证，当前地形表面数据、均一粗糙度假设和其他冻结设置的适用限制仍成立。','', '图组由独立绘图脚本按 `methods/figure_contract.json` 生成；图像、图注与图稿核验以 `figures/figure_bundle_manifest.json`、`figures/figure_captions.md` 和 `figures/figure_QA.json` 为准。本脚本不生成、替换或宣称独立图组已完成。已有训练期D2辅助材料在 `auxiliary/D2/` 单列，不计入38项CFD完成，也不据其额外宣称现场验证。','', '## 7. 文件入口','', '| 文件 | 内容 |','|---|---|']
 for n in TABLES:lines.append(f'| [{n}]({n}) | 原正式表，同字节保留 |')
 lines += ['| [summary_statistics.json](summary_statistics.json) | 描述统计、正负对照与极值位置 |','| [case_index.csv](case_index.csv) | 38工况身份与验收来源 |','| [source_evidence_index.csv](source_evidence_index.csv) | 配置、归档、原失败和恢复的路径/SHA来源索引 |','| [failure_and_recovery_index.csv](failure_and_recovery_index.csv) | 原失败/恢复与计数纠正记录 |','| [delivery_manifest.json](delivery_manifest.json) | 本包输出文件与SHA |','', '本报告不修改任何预测模型、原实验协议、原始测量目标或已完成实验。']
 return '\n'.join(lines)+'\n'

def status_record(status,message,extra=None):
 value={'status':status,'observed_utc':now(),'message':message,'final_report_generated':False,'scientific_source_or_service_modified':False};value.update(extra or {})
 raw=json_bytes(value)
 for p in (HERE/'prepared_status.json',EPREP/'prepared_status.json'):atomic(p,raw)
 return value

def reproduce_readme():
 entry=WORK/'closeout_20261024/report_prepared/build_final_cfd_delivery.py'
 return '\n'.join(['# 复核与重建说明','',
  '这是轻量结果交付目录，含正式表、中文报告、小型配置/来源快照和索引。它不包含全部GB级原生场、日志或整个求解环境；这些仍在原D/E档案目录，通过case_index.csv、source_evidence_index.csv及sources/逐工况记录定位。打包时没有再复制或流式解压大归档。','',
  '实际执行入口在原项目布局中：', '',
  '```powershell',f'& "python" -X utf8 "{entry.as_posix()}"',
  f'& "python" -X utf8 "{entry.as_posix()}" --build','```','',
  '第一条仅检查正式38、D/E报告文件、身份与数值；第二条在同一门禁通过后生成/复核本包。只有原项目WORK/closeout_20261024/...中的入口是已验证路径。本包scripts/下的.py是对应源码快照，按项目相对布局定位WORK/OUT，不能声称复制到任意目录即可直接运行。恢复项目时须保留原scripts/、补充验证目录及配置的相对布局。','',
  '从E档案恢复单个工况时：','',
  '1. 在新的空恢复目录操作，不使用、不覆盖原实验源目录。先按case_index.csv定位对应工况，再从source_evidence_index.csv及其原验收receipt取得精确E归档路径和SHA；恢复前独立重算压缩文件SHA并核对。',
  '2. 对base加quality_addendum的工况，先展开base，再将同工况quality overlay展开到该新目录；使用已保全的archive_overlay_index.json确定覆盖层与每成员SHA。只有单一归档的采样/既有恢复来源不人为添加第二层。不得混用不同RID或不同恢复版本的归档。',
  '3. 展开时仅允许索引列出的安全相对常规文件路径，不跟随符号链接、不越出恢复目录；逐成员大小/SHA与原index核验后再使用。原raw质量失败、原生退出1和后处理恢复的各份凭据都保留，不用overlay把原失败事实改写成原生成功。',
  '4. 完整CFD环境、网格及求解器来源见原档案和frozen settings中的image/source信息。表格打包脚本不是CFD求解入口，不会重新训练或仿真。若需要从原项目再执行严格总汇总，那是一次完整D/E扫描，应与现有维护服务协调，不能同时重复启动。','',
  'D/E交付目录内命名文件的SHA见delivery_manifest.json；归档SHA索引明确标注来自原严格验收的记录，而非本打包脚本再次读大归档。辅助D2训练期材料与38项CFD范围分开，图组由独立figure_contract及绘图脚本管理。',''])

@contextlib.contextmanager
def own_lock():
 HERE.mkdir(parents=True,exist_ok=True);p=HERE/'build.lock'
 with p.open('a+b') as f:
  f.seek(0,2)
  if f.tell()==0:f.write(b'0');f.flush()
  f.seek(0)
  if os.name=='nt':
   import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:
   if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(f,fcntl.LOCK_UN)

def build(snapshot):
 stats=statistics_bundle(snapshot['tables']);source_key=digest(json_bytes(snapshot['input_sha256']));builder=descriptor(__file__)
 existing=DEST/'delivery_manifest.json'
 if existing.is_file():
  m=read_json(existing);need(m.get('source_snapshot_sha256')==source_key and m.get('builder_sha256')==builder['sha256'],'Existing package differs; preserve it for explicit review')
  for d in m['files']:
   p,q=DEST/d['relative_path'],EDEST/d['relative_path'];need(p.read_bytes()==q.read_bytes() and descriptor(p)['sha256']==d['sha256'],'Existing delivered file D/E/SHA differs')
  return {'status':'EXISTING_IDENTICAL_FINAL_DATA_REPORT_PACKAGE_VERIFIED','D_path':str(DEST),'E_path':str(EDEST),'new_report_generated':False}
 stage=HERE/('staging_'+uuid.uuid4().hex);stage.mkdir()
 def emit(name,raw):p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 for name,raw in snapshot['raw'].items():emit(name,raw)
 emit('summary_statistics.json',json_bytes(stats));emit('完整中文CFD结果报告.md',markdown_report(snapshot,stats).encode('utf8'));emit('复核与重建说明.md',reproduce_readme().encode('utf8'))
 cases,failures=package_case_rows(snapshot);emit('case_index.csv',csv_bytes(cases));emit('failure_and_recovery_index.csv',csv_bytes(failures));emit('source_evidence_index.csv',csv_bytes(evidence_index(snapshot['cases'],snapshot['sources'])))
 vectors=[r for name,spec in TABLES.items() if spec[1]=='vector' for r in snapshot['tables'][name]]
 metrics=[('speedup','speedup',False),('signed_deflection_deg','delta_dir_deg',False),('absolute_deflection_deg','delta_dir_deg',True)]
 emit('per_case_descriptive.csv',csv_bytes(grouped_descriptions(vectors,['site','case_kind','run_id','sector_deg','z0_m','ct'],metrics)))
 defaults=snapshot['tables']['cfd_wind_farm_A_turbine_sector.csv']+snapshot['tables']['cfd_kelmarsh_turbine_sector.csv']
 emit('per_direction_descriptive.csv',csv_bytes(grouped_descriptions(defaults,['site','sector_deg'],metrics)));emit('per_turbine_descriptive.csv',csv_bytes(grouped_descriptions(defaults,['site','turbine_id'],metrics)))
 for rid,case in snapshot['cases'].items():emit('inputs/'+rid+'/frozen_info.json',(Path(case['bundle'])/'frozen_info.json').read_bytes());emit('sources/'+rid+'.json',json_bytes(snapshot['sources'][rid]))
 emit('scripts/build_final_cfd_delivery.py',Path(__file__).read_bytes())
 methods=WORK/'research/粗糙度调研与场地核查_20261023.md'
 if methods.is_file():emit('methods/'+methods.name,methods.read_bytes())
 contract=WORK/'ops/lab_deployment_20261023/final_delivery_preparation_20261024/figure_contract.json'
 if contract.is_file():emit('methods/figure_contract.json',contract.read_bytes())
 for p in (OUT/'D2').glob('*.md'):emit('auxiliary/D2/'+p.name,p.read_bytes())
 for name,h in snapshot['input_sha256'].items():
  if digest((OUT/name).read_bytes())!=h or digest((EBASE/'reports'/name).read_bytes())!=h:raise Pending('Source report changed during packaging; no final files published')
 files=[]
 for p in sorted(stage.rglob('*')):
  if p.is_file():d=descriptor(p);files.append({'relative_path':p.relative_to(stage).as_posix(),'bytes':d['bytes'],'sha256':d['sha256']})
 manifest={'schema_version':1,'status':'FORMAL38_DATA_AND_CHINESE_REPORT_PACKAGE','evidence_cutoff_utc':snapshot['summary']['updated_utc'],'created_utc':now(),'settings_sha256':SETTINGS_SHA,'strict_summary_source_sha256':SUMMARY_SOURCE_SHA,'source_snapshot_sha256':source_key,'source_report_sha256':snapshot['input_sha256'],'builder_sha256':builder['sha256'],'case_count':38,'source_profile_counts':PROFILE_COUNTS,'figure_generation_and_QA_are_separate':True,'figure_readiness_not_claimed_by_this_packager':True,'archive_hash_origin':'recorded_original_strict_full_D_E_evidence_not_rehashed_by_packager','files':files}
 emit('delivery_manifest.json',json_bytes(manifest))
 # Check all named collisions before publishing; preserve any existing edits.
 for p in sorted(stage.rglob('*')):
  if p.is_file():
   for root in (DEST,EDEST):
    target=root/p.relative_to(stage)
    if target.exists():need(target.is_file() and target.read_bytes()==p.read_bytes(),'Existing delivery file differs; no overwrite: '+str(target))
 # Publish only our named report/table/index/source files; never figures.
 for p in sorted(stage.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(stage);need(rel.parts[0]!='figures','Figure writer boundary')
  raw=p.read_bytes()
  for root in (DEST,EDEST):
   target=root/rel
   if target.exists():need(target.is_file() and target.read_bytes()==raw,'Existing delivery file differs; no overwrite: '+str(target))
   else:atomic(target,raw)
 for d in files:need((DEST/d['relative_path']).read_bytes()==(EDEST/d['relative_path']).read_bytes() and digest((DEST/d['relative_path']).read_bytes())==d['sha256'],'Published D/E SHA differs')
 result={'status':'FORMAL38_DATA_REPORT_DELIVERED_D_E_BYTE_SHA_VERIFIED','D_path':str(DEST),'E_path':str(EDEST),'manifest':descriptor(DEST/'delivery_manifest.json'),'copied_file_count':len(files),'final_report_generated':True,'figures_not_written':True,'archives_not_restreamed':True}
 for p in (DEST/'delivery_validation.json',EDEST/'delivery_validation.json'):atomic(p,json_bytes(result))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true',help='Publish requested file package only after every formal38 gate passes');args=parser.parse_args()
 try:
  with own_lock():
   snapshot=read_snapshot()
   if not args.build:value=status_record('READY_FORMAL38_VALIDATED_NOT_YET_PACKAGED','All finite identities, table math and same-byte D/E reports pass',{'formal_count':38,'statistics':statistics_bundle(snapshot['tables'])})
   else:
    value=build(snapshot)
    for p in (HERE/'actual_delivery_receipt.json',EPREP/'actual_delivery_receipt.json'):atomic(p,json_bytes(value))
   print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0
 except Pending as ex:print(json.dumps(status_record('PREPARED_WAITING_FOR_COMPLETE_FORMAL38',str(ex)),ensure_ascii=False));return 0
 except Exception as ex:print(json.dumps(status_record('BLOCKED_REPORT_GATE_FAILED',type(ex).__name__+': '+str(ex)),ensure_ascii=False));return 2

if __name__=='__main__':raise SystemExit(main())
