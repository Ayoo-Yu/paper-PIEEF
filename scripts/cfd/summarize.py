"""Aggregate only strict-quality CFD whose complete D/E artifacts are verified.

Never blend intersector checks/roughness experiments into the sector LUT.
Reads native result CSVs or retained remote archives; starts no computation.
"""
from pathlib import Path,PurePosixPath
import csv,json,io,tarfile,math,datetime,hashlib,os,importlib.util,re,gzip,uuid
WORK=Path(__file__).resolve().parents[1]
OUT=WORK.parent/'补充验证_20261023'
EBACK=Path('results/paper')
SETTINGS_SHA='e955155a03911e4266bb8a171e664f0ae98c828d99d16b8572bae7d588e613cc'
ASSESSOR_SHA='c049b8be4f035343a8a032ae4dc0399f215e1fa762770edc11a55a4a62856c9b'
QUALITY_SOURCE_SHA='3ed900f32e083048be38cf1cbf73b56f0454015efd094f38f5499e46741ba311'
IMAGE_ID='sha256:fb7471345c2dad77c19464e475e8e7a5c2eab8710381726cd15c4f45a967d02e'
SAMPLING_RID='energy-r4-wind_farm_A-sector-a090p00-z0p3-ct0-v1'
SAMPLING_SCI='c3e4bccfdc36d94f17197f6064bd7d5fc81c51b1f802a39f771d459a8a65da5a'
SAMPLING_PRODUCER_SHA='5846f37f7c0328fc9fe6de6e5ed428196e6afd7a699b162ea4eb2c55ed4e0659'
SAMPLING_VALIDATOR_SHA='ec6f47da20f6b3eb46f9930c8af9b618bba6fa046024a2625559b82f95432c72'
SAMPLING_VALIDATOR_NAME='validate_sampling_recovery_v5_local_post_v3.py'
SAMPLING_BINARY_SHA='2ee0b22f795f1f4a6fa15d5ea38fae8ab10c0f202c4948b51a6943d9a58fef05'
ROUGH_SHA='05eff906f587e289979c8bf33ce3084bf82d26741019c9e88a37669a49738f6c'
VERIFY_CACHE={}
ACTIVE_METADATA_SESSION=None
def require(value,message):
 if not value:raise ValueError(message)
def load(p):
 return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))
def dump(p,x):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(q,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def writecsv(p,rows):
 if not rows:
  # Preserve earlier derived output while preventing a stale table from looking
  # like evidence accepted by this fresh verification pass.
  if p.exists():
   folder=OUT/'superseded_tables';folder.mkdir(exist_ok=True)
   target=folder/(p.stem+'.'+sha(p)+'.csv')
   if target.exists():require(sha(target)==sha(p),'Superseded table conflict');p.unlink()
   else:os.replace(p,target)
  return
 q=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 with q.open('w',encoding='utf-8-sig',newline='') as f:
  fields=list(dict.fromkeys(key for row in rows for key in row))
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(q,p)
def wrap(x):return (x+180)%360-180
def percentile(values,q):
 a=sorted(values);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def module(name):
 path=WORK/'scripts'/(name+'.py');spec=importlib.util.spec_from_file_location('summary_'+name,path)
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def safe_member(name):
 require(isinstance(name,str) and name and '\\' not in name and ':' not in name
         and not any(ord(c)<32 for c in name),'Unsafe manifest member')
 parts=PurePosixPath(name).parts
 require(not name.startswith('/') and all(p not in ('','.','..') for p in name.split('/')),'Unsafe manifest member')
 return '/'.join(parts)

def manifest_index(files):
 require(isinstance(files,list) and files,'Empty complete-file manifest');index={}
 for row in files:
  name=safe_member(row['path'])
  require(name not in index and type(row.get('bytes')) is int and row['bytes']>=0
          and re.fullmatch('[a-f0-9]{64}',row.get('sha256','')),'Invalid or duplicate manifest entry')
  index[name]=row
 return index

def verify_file(path,row):
 path=Path(path)
 require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
         and sha(path)==row['sha256'],'Source file differs from verified archive: '+str(path))

def tar_bytes(path,name,expected):
 # Called only after the whole archive was streamed and its member index bound.
 require(name in expected,'Required assessment artifact absent from complete archive: '+name)
 if ACTIVE_METADATA_SESSION is not None:return ACTIVE_METADATA_SESSION.member_bytes(path,name,expected,max_bytes=16*1024**2)
 with tarfile.open(path,'r:gz') as stream:
  member=stream.getmember(name);require(member.isreg() and member.size<=16*1024**2,'Invalid assessment member')
  data=stream.extractfile(member).read()
 row=expected[name]
 require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Assessment member changed after archive verification')
 return data

def verify_completion(q,completed,get,item,expected_assessor):
 require(completed.get('status')=='PASS' and completed.get('quality_pass') is True,'Assessment not completed successfully')
 files=completed.get('files',{})
 require({'quality.json','turbine_sector.csv','foamToVTK_quality.log'}<=set(files),'Incomplete assessment completion manifest')
 for name,digest in files.items():
  require(safe_member(name)==name and '/' not in name,'Unexpected assessment completion path')
  require(hashlib.sha256(get(name)).hexdigest()==digest,'Assessment completion SHA mismatch: '+name)
 require(q.get('status')=='PASS' and q.get('quality_pass') is True and q.get('run_id')==item['run_id'],'Wrong/nonpassed assessment identity')
 require(q.get('input_hashes',{}).get('frozen_info_sha256')==item['input_sha256'],'Assessment frozen input identity differs')
 require(q.get('assessor_sha256')==expected_assessor and q.get('original_quality_source_sha256')==QUALITY_SOURCE_SHA,'Assessment source identity differs')
 checks=q.get('checks',{})
 require(checks and all(v is True for v in checks.values()) and q.get('failed_checks')==[],'Incomplete or failed strict quality checks')
 require(q.get('kind')==('loaded_Ct08' if float(item['ct'])==.8 else 'terrain_zero_Ct'),'Quality Ct kind mismatch')

def same_info(current,frozen):
 runtime={'calculationStatus','startedAt','completedAt','updatedAt'}
 require({k:v for k,v in current.items() if k not in runtime}=={k:v for k,v in frozen.items() if k not in runtime},'Runtime scientific configuration changed')

def exited(inspect,container_id):
 state=inspect['State']
 require(inspect.get('Id')==container_id and inspect.get('Image')==IMAGE_ID
         and state.get('Running') is False and type(state.get('ExitCode')) is int
         and state['ExitCode']==0 and state.get('OOMKilled') is False and not state.get('Error'),'Actual container did not naturally complete on frozen image')

def selected_local_source(item):
 rid=item['run_id'];group=OUT/'C3'
 default=dict(case=group/'runtime/uploads'/rid,quality=group/'runtime/uploads'/rid/'assessment_energy_v1/quality.json',
              archive_verified=group/'results'/rid/'archive_verified.json',backup=EBACK/'C3'/rid,assessor_sha256=ASSESSOR_SHA,recovery=False)
 registry=group/'recovery_registry.json'
 if not registry.exists():return default
 record=load(registry)
 require(record.get('schema_version')==1 and record.get('settings_sha256')==SETTINGS_SHA,'Recovery registry schema/settings differs')
 entries=record.get('runs',{});require(isinstance(entries,dict),'Recovery registry runs must be an explicit ID map')
 if rid not in entries:return default
 row=entries[rid]
 if row.get('status')!='PASS':return default
 root=group/'recovery_postprocess_v2';case=Path(row['case']);quality=Path(row['quality_json']);verified=Path(row['archive_verified_json'])
 require(case.resolve()==(root/'runtime/uploads'/rid).resolve(),'Recovery case is not the explicit independent v2 copy')
 require(quality.resolve().is_relative_to(case.resolve()) and verified.resolve()==(group/'recovered_results'/rid/'archive_verified.json').resolve(),'Recovery artifact path outside approved copy')
 require(Path(row['archives']['D']['path']).resolve()==(verified.parent/(rid+'.tar.gz')).resolve()
         and Path(row['archives']['E']['path']).resolve()==(EBACK/'C3/recovered'/rid/(rid+'.tar.gz')).resolve(),'Recovery D/E archive locations differ from explicit preservation layout')
 # One shared implementation verifies the original failure, native protection,
 # actual post-only container/source, completion files and both complete archives.
 gate=module('assessment_recovery').validate_recovery_registry(registry,OUT/'cfd_settings_frozen.json')
 require(gate.get('status')=='PASS' and gate.get('run_id')==rid
         and gate.get('scientific_identity_sha256')==item['input_sha256'],'Recovery gate scientific identity differs')
 return dict(case=case,quality=quality,archive_verified=verified,backup=Path(row['archives']['E']['path']).parent,
             assessor_sha256=gate['assessor_sha256'],recovery=True,entry=row,registry_sha256=sha(registry),already_stream_verified=True)

def local_rows(item):
 source=selected_local_source(item);verified=source['archive_verified']
 if not verified.exists():return None
 a=load(verified)
 # A failed original case is preserved, but never load its nonexistent quality.
 if a.get('status')!='PASS' or a.get('quality_status')!='PASS':return None
 rid=item['run_id'];case=source['case'];backup=source['backup'];qpath=source['quality'];assessment=qpath.parent
 require(a.get('run_id')==rid and a.get('archive_operational_version')==2
         and Path(a['original_case']).resolve()==case.resolve(),'Local archive receipt/source identity mismatch')
 require(sha(backup/'archive_verified.json')==sha(verified),'D/E local archive receipts differ')
 files=manifest_index(a['files'])
 require(a.get('file_count')==len(files) and a.get('uncompressed_bytes')==sum(r['bytes'] for r in files.values()),'Local manifest cardinality differs')
 require(a.get('archive_name')==rid+'.tar.gz','Unexpected local archive name')
 d=verified.parent/a['archive_name'];e=backup/a['archive_name'];mirror=module('mirror_cfd_workstation')
 for archive in (d,e):mirror.verify_archive(archive,a['archive_bytes'],a['archive_sha256'])
 # Bind the live assessment and every native/source file to that immutable list.
 actual=set()
 for p in case.rglob('*'):
  require(not p.is_symlink(),'Symlink in complete original case')
  if p.is_file():actual.add(p.relative_to(case).as_posix())
 require(actual==set(files),'Current native case adds or omits archived files')
 for name,row in files.items():verify_file(case/name,row)
 key=('local',str(d.resolve()),str(e.resolve()),sha(verified),a['archive_sha256'])
 if key not in VERIFY_CACHE and not source.get('already_stream_verified'):
  queue=module('continue_local_cfd')
  for archive in (d,e):queue.verify_archive_stream(archive,rid,list(files.values()))
  # Also consume gzip CRC/trailer; the queue validator stops at tar zero blocks.
  for archive in (d,e):
   with gzip.open(archive,'rb') as f:
    while f.read(4*1024**2):pass
  VERIFY_CACHE[key]=True
 get=lambda name:(assessment/name).read_bytes()
 q=load(qpath);completed=load(assessment/'completed_assessment.json')
 verify_completion(q,completed,get,item,source['assessor_sha256'])
 require(sha(case/'frozen_info.json')==item['input_sha256'],'Archived local frozen input differs')
 for name,keyname in [('info.json','runtime_info_sha256'),('terrain.tif','terrain_sha256'),('execution_receipt.json','execution_receipt_sha256')]:
  require(sha(case/name)==q['input_hashes'][keyname],'Quality input/source SHA differs: '+name)
 x=load(case/'execution_receipt.json')
 same_info(load(case/'info.json'),load(case/'frozen_info.json'))
 require(x.get('run_id')==rid and x.get('frozen_input_sha256')==item['input_sha256'] and x.get('settings_sha256')==SETTINGS_SHA,'Local execution scientific identity differs')
 require(x.get('image_id')==IMAGE_ID and x.get('container_oom_killed') is False,'Local execution image/OOM differs')
 if source['recovery']:
  evidence=q.get('recovery_evidence',{});receipt=load(case/'postprocess_recovery_receipt.json')
  require(q.get('assessor_version')=='post-only-recovery-v2' and q.get('base_assessor_sha256')==ASSESSOR_SHA,'Recovery assessor identity differs')
  require(evidence.get('status')=='PASS' and evidence.get('scientific_run_id')==rid
          and evidence.get('scientific_identity_sha256')==item['input_sha256'] and evidence.get('settings_sha256')==SETTINGS_SHA
          and evidence.get('original_and_clone_native_files_unchanged') is True and evidence.get('parent_D_E_archives_stream_verified') is True,'Recovery evidence gate differs')
  require(evidence.get('recovery_receipt_sha256')==source['entry']['recovery_receipt_sha256']
          and receipt.get('scientific_run_id')==rid and receipt.get('solver_rerun') is False and receipt.get('mesh_rebuilt') is False
          and receipt.get('scientific_input_changed') is False,'Recovery not bound to unchanged post-only receipt')
  require(x.get('exit_code')==1 and x.get('container_exit_code')==1,'Recovery must retain truthful failed parent receipt')
 else:
  require(type(x.get('exit_code')) is int and x['exit_code']==0 and type(x.get('container_exit_code')) is int and x['container_exit_code']==0,'Local native process not naturally successful')
  queue=module('continue_local_cfd');queue.verify_exited_container(load(case/'container_inspect_final.json'),x,case,IMAGE_ID)
  ae=load(case/'assessment_execution_receipt.json');ai=ae['container']
  require(type(ae.get('exit_code')) is int and ae['exit_code']==0 and ae.get('assessor_sha256')==ASSESSOR_SHA,'Actual local assessor source/exit differs')
  exited(ai,ai['Id'])
  cmd=' '.join(ai['Config']['Cmd'])
  require('assess_energy_cfd.py' in cmd and '/uploads/'+rid in cmd,'Actual assessment command/case differs')
 return list(csv.DictReader(io.StringIO(get('turbine_sector.csv').decode('utf-8-sig')))),q,dict(quality=str(qpath),archive_verified=str(verified),
        D_archive=str(d),E_archive=str(e),archive_sha256=a['archive_sha256'],full_file_count=len(files),
        strict_full_file_stream_verified=True,recovery=source['recovery'],registry_sha256=source.get('registry_sha256'))

def sampling_descriptor(value,expected_sha=None):
 require(isinstance(value,dict) and {'path','bytes','sha256'}<=set(value),'Missing explicit sampling descriptor')
 path=Path(value['path']);require(path.is_absolute(),'Sampling descriptor path must be absolute')
 verify_file(path,value)
 if expected_sha:require(value['sha256']==expected_sha,'Unapproved sampling source/producer descriptor')
 return path

def sampling_recovery_source(item,deep=True,expected_provenance=None):
 """Explicit native-exit0 branch; never reuse the Kelmarsh status overrides.

 A progress read may reuse an already deeply verified summary only after all
 registry. and D/E archive hashes still bind.
 """
 if item.get('run_id')!=SAMPLING_RID:return None
 registry=OUT/'sampling_recovery_registry.json'
 if not registry.exists():return None
 registered=load(registry)
 require(registered.get('schema_version')==1 and registered.get('kind')=='accepted_sampling_query_recovery_registry_v5'
         and registered.get('settings_sha256')==SETTINGS_SHA,'Sampling registry protocol/settings differ')
 entries=registered.get('runs');require(isinstance(entries,dict) and set(entries)<={SAMPLING_RID},'Unexpected sampling recovery registry scope')
 if SAMPLING_RID not in entries:return None
 entry=entries[SAMPLING_RID]
 if entry.get('status')!='PASS':return None
 require(item.get('site')=='wind_farm_A' and item.get('group')=='C1' and item.get('input_sha256')==SAMPLING_SCI,'Sampling recovery scientific identity differs')
 names=('root_acceptance_gate','validation_request','validation_result','validator','producer_manifest')
 paths={name:sampling_descriptor(entry[name],SAMPLING_PRODUCER_SHA if name=='producer_manifest' else
                                SAMPLING_VALIDATOR_SHA if name=='validator' else None) for name in names}
 require(paths['validator'].resolve()==(WORK/'postprocess_cellsearch_v3'/SAMPLING_VALIDATOR_NAME).resolve(),'Unexpected sampling validator path')
 gate=load(paths['root_acceptance_gate'])
 require(gate.get('status')=='PASS' and gate.get('root_accepted') is True and gate.get('run_id')==SAMPLING_RID
         and gate.get('settings_sha256')==SETTINGS_SHA and gate.get('scientific_identity_sha256')==SAMPLING_SCI,'Actual root sampling acceptance identity differs')
 for key in ('strict_v1_all_checks_pass','original_native_bytes_unchanged','all_original_supported_AGL_points_unchanged','D_E_full_file_stream_verified'):
  require(gate.get(key) is True,'Root sampling acceptance incomplete: '+key)
 require(gate.get('patched_postFoam_sha256')==SAMPLING_BINARY_SHA and gate.get('roughFoam_sha256')==ROUGH_SHA,'Accepted sampler/solver source differs')
 for name in names[1:]:require(gate.get(name)==entry[name],'Root gate does not bind registry descriptor: '+name)
 request=load(paths['validation_request']);saved=load(paths['validation_result'])
 require(request.get('run_id')==SAMPLING_RID and request.get('scientific_identity_sha256')==SAMPLING_SCI and request.get('settings_sha256')==SETTINGS_SHA,'Sampling validation request identity differs')
 request_producer=request['post_only']['approved_producer_manifest']
 request_producer_path=sampling_descriptor(request_producer,SAMPLING_PRODUCER_SHA)
 require(request_producer['bytes']==entry['producer_manifest']['bytes'],'Request and registry producer bytes differ')
 require(request_producer_path.resolve()==paths['producer_manifest'].resolve()
         or request_producer_path.resolve().is_relative_to(Path(request['recovery_case']).resolve()),'Producer copy is outside the accepted full recovery case')
 require(saved.get('status')=='PASS' and saved.get('kind')=='sampling_post_only_recovery_branch_verified_v5_local_post'
         and saved.get('request_sha256')==entry['validation_request']['sha256'] and saved.get('validator_sha256')==SAMPLING_VALIDATOR_SHA,'Actual independent sampling result/source binding differs')
 require(saved.get('actual_execution_receipt_schema')=='exact_local_post_and_original_v1','Unapproved actual post/strict receipt schema')
 require(saved.get('run_id')==SAMPLING_RID and saved.get('scientific_identity_sha256')==SAMPLING_SCI and saved.get('settings_sha256')==SETTINGS_SHA,'Independent recovery result scientific identity differs')
 require(saved.get('approved_producer_manifest_sha256')==SAMPLING_PRODUCER_SHA and saved.get('post_only_binary_sha256')==SAMPLING_BINARY_SHA
         and saved.get('strict_quality_status_override_used') is False and saved.get('unchanged_v1_all_checks_PASS') is True,'Sampling recovery cannot override original strict checks')
 require(saved.get('original_CFD_exit_code')==0 and saved.get('original_strict_failure_retained')==['all_AGL_layers_complete'],'Original exit0/strict FAIL evidence not retained')
 require(set(request.get('archives',{}))=={'D','E'},'Both complete sampling recovery archives required')
 archive_paths={disk:sampling_descriptor(request['archives'][disk]) for disk in ('D','E')}
 require(archive_paths['D'].resolve().drive.upper()=='D:' and archive_paths['E'].resolve().drive.upper()=='E:'
         and archive_paths['D'].resolve()!=archive_paths['E'].resolve(),'Sampling archives must be on independent D/E volumes')
 require(request['archives']['D']['sha256']==request['archives']['E']['sha256']
         and request['archives']['D']['bytes']==request['archives']['E']['bytes'],'Sampling D/E complete archives differ')
 signature={'registry_sha256':sha(registry),'descriptors':{name:entry[name] for name in names},
            'archives':request['archives'],'quality':request['strict_assessment']['quality'],
            'completed':request['strict_assessment']['completed_assessment'],'full_case_manifest':request['full_case_manifest']}
 for value in (signature['quality'],signature['completed'],signature['full_case_manifest']):sampling_descriptor(value)
 if not deep:
  require(expected_provenance and expected_provenance.get('sampling_recovery') is True
          and expected_provenance.get('sampling_verification_signature')==signature
          and expected_provenance.get('strict_full_file_stream_verified') is True,'Sampling progress lacks the same deeply verified summary signature')
 else:
  cache_key=('sampling-recovery-v5',json.dumps(signature,sort_keys=True))
  if cache_key not in VERIFY_CACHE:
   spec=importlib.util.spec_from_file_location('summary_sampling_local_post_v5',paths['validator'])
   validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
   actual=validator.validate_request_v5_local_post(request,SAMPLING_PRODUCER_SHA)
   require(actual.get('status')=='PASS' and all(saved.get(k)==v for k,v in actual.items()),'Saved sampling result differs from actual full independent validation')
   VERIFY_CACHE[cache_key]=True
 qpath=sampling_descriptor(request['strict_assessment']['quality']);completed_path=sampling_descriptor(request['strict_assessment']['completed_assessment'])
 require(str(qpath.resolve())==saved.get('quality_json') and str(completed_path.resolve())==saved.get('completed_assessment_json'),'Recovered quality paths differ from independent result')
 q=load(qpath);complete=load(completed_path);get=lambda n:(qpath.parent/n).read_bytes()
 verify_completion(q,complete,get,item,ASSESSOR_SHA)
 require(not any(k in q for k in ('assessor_version','base_assessor_sha256','recovery_original_v1_pipeline_checks')),'Kelmarsh status override is forbidden for native sampling recovery')
 provenance={'quality':str(qpath),'sampling_recovery':True,'recovery':True,'recovery_kind':'native-sampling-post-only-v5',
  'sampling_registry':str(registry),'sampling_registry_sha256':sha(registry),'sampling_verification_signature':signature,
  'root_acceptance_gate':entry['root_acceptance_gate'],'validation_request':entry['validation_request'],'validation_result':entry['validation_result'],
  'producer_manifest':entry['producer_manifest'],'validator':entry['validator'],
  'D_archive':str(archive_paths['D']),'E_archive':str(archive_paths['E']),'archive_sha256':request['archives']['D']['sha256'],
  'full_file_count':saved['archives']['D']['stream_verified']['file_count'],'strict_full_file_stream_verified':True,
  'original_quality_status':'FAIL','original_failed_checks':['all_AGL_layers_complete'],
  'original_quality_sha256':saved['parent_original_quality_sha256']}
 if deep:VERIFY_CACHE[('sampling-provenance',sha(registry))]=provenance
 return list(csv.DictReader(io.StringIO(get('turbine_sector.csv').decode('utf-8-sig')))),q,provenance

# Prepared source fragment, not an executable CFD or queue operation.
FUTURE_FROZEN_QUEUE_SHA='7a84e95138922ed6bee68f22856204ba07918a7b6cfafafd146e4e599b2711b9'
FUTURE_HANDOFF_SHA='9387fb404ee539e14b7a3d0958e2c1447d1a654c4930c3836d4e8cbe886df2af'
FUTURE_LOCAL_HOST='local'
FROZEN_CWD_WRAPPER_SHA='d350074e2e4d1547a5ab01019118014d4bd729c5ea0bfe994ca7206009146d2b'
FROZEN_CWD_LAUNCHER_SHA='2ae59e8cd4fd02d76614c803ea6746b671b9757c2c1c1c4aa8892a1d77e9e386'
FROZEN_CWD_SHELL=('source /usr/lib/openfoam/openfoam2212/etc/bashrc; '
 'export FOAM_USER_APPBIN=/opt/physics/backend/bin; export PATH=$FOAM_USER_APPBIN:$PATH; '
 'mkdir -p /tmp/energy-home /tmp/energy-mpl; set -e; test "$WM_PROJECT_VERSION" = v2212; exec bash /opt/physics/backend/base/run_energy_preserve.sh')

def native_command_case_binding(item,request,ci,from_overlay):
 """Retain legacy argv binding; explicitly verify the frozen cwd launcher."""
 rid=item['run_id'];cmd=ci.get('Config',{}).get('Cmd',[])
 nativecmd=' '.join(cmd)
 if '/uploads/'+rid in nativecmd and 'run_energy_preserve.sh' in nativecmd:
  return {'native_case_binding':'legacy_explicit_command_case'}
 cwd='/physics-runtime/uploads/'+rid
 require(ci.get('Image')==request.get('image_id')==IMAGE_ID and ci.get('Id')==request.get('container_id'),'Cwd native container/image identity differs')
 require(request.get('operational_launcher_version')==2 and request.get('operational_launcher_sha256')==FROZEN_CWD_LAUNCHER_SHA,
         'Unapproved cwd-bound native launcher')
 require(request.get('post_only_utility_sha256')=='34394e5ff46047aa11b245897745013eacecc0da6c280993024f10394b040e15','Cwd launcher utility identity differs')
 wrapper=WORK/'ops/lab_master1_access/runtime_validation_20261023/run_energy_preserve.frozen.sh'
 require(sha(wrapper)==FROZEN_CWD_WRAPPER_SHA,'Approved image wrapper bytes differ')
 # This wrapper derives CASE_ROOT_DIR from pwd -P and refuses a case outside
 # backend/uploads. Its bytes came from the exact already-frozen image export.
 specpath=WORK/'ops/lab_master1_access/runtime_validation_20261023/runtime_spec.json'
 require(sha(specpath)=='a7a6f157230f5716e1b4e61c64ba606ac3a2dd183d1398d3927c4572b6da7863','Frozen image export provenance differs')
 spec=load(specpath)
 require(spec.get('image_id')==IMAGE_ID and spec.get('preserving_runner_sha256')==FROZEN_CWD_WRAPPER_SHA,'Wrapper/image byte provenance differs')
 require(ci.get('Path')=='/bin/bash' and ci.get('Args')==cmd==['-lc',FROZEN_CWD_SHELL]
         and ci['Config'].get('Entrypoint')==['/bin/bash'],'Actual cwd launcher Path/Args/Cmd differs')
 require(ci['Config'].get('WorkingDir')==request.get('container_cwd')==cwd,'Native cwd does not bind exact case')
 labels=ci['Config'].get('Labels',{})
 require(labels.get('org.energy.round4')=='20261023' and labels.get('org.energy.run_id')==rid
         and labels.get('org.energy.scientific_identity')==item['input_sha256'],'Native cwd container scientific labels differ')
 mount={'Type':'bind','Source':'results/paper','Destination':'/physics-runtime','Mode':'','RW':True,'Propagation':'rprivate'}
 require(ci.get('Mounts')==[mount],'Native cwd runtime bind mount differs')
 expected=['docker','create','--name','energy-r4-'+rid,'--restart','no','--network','none','--cpus','8','--memory','10g','--pids-limit','2048',
  '--label','org.energy.round4=20261023','--label','org.energy.run_id='+rid,'--label','org.energy.scientific_identity='+item['input_sha256'],
  '--mount','type=bind,source='+mount['Source']+',target=/physics-runtime','--workdir',cwd,
  '--env','HOME=/tmp/energy-home','--env','MPLCONFIGDIR=/tmp/energy-mpl','--env','OPENBLAS_NUM_THREADS=1','--env','OMP_NUM_THREADS=1',
  '--env','MKL_NUM_THREADS=1','--env','WINDSIM_PYTHON=/usr/bin/python3','--entrypoint','/bin/bash',IMAGE_ID,'-lc',FROZEN_CWD_SHELL]
 require(request.get('create_command')==expected,'Archived exact cwd create command differs')
 for name in ('container_inspect_created.json','container_inspect_started.json'):
  other=json.loads(from_overlay('runtime/jobs/'+rid+'/'+name))
  require(other.get('Id')==ci.get('Id') and other.get('Image')==IMAGE_ID and other.get('Path')==ci['Path']
          and other.get('Args')==ci['Args'] and other.get('Mounts')==ci['Mounts'],'Created/started cwd container identity differs')
  require(all(other.get('Config',{}).get(k)==ci['Config'].get(k) for k in ('Cmd','Entrypoint','WorkingDir','Labels','Env')),'Created/started cwd command or case changed')
 return {'native_case_binding':'exact_frozen_v2_launcher_cwd','native_wrapper_sha256':FROZEN_CWD_WRAPPER_SHA,'native_launcher_sha256':FROZEN_CWD_LAUNCHER_SHA,'native_container_cwd':cwd}

def future_local_provenance(item,request,ci,from_overlay):
 """Accept only the actually handed-off original 24, on the fixed local host."""
 frozen_path=WORK/'ops/local/finite_queue_frozen.json'
 require(sha(frozen_path)==FUTURE_FROZEN_QUEUE_SHA,'Original finite queue source changed')
 frozen=load(frozen_path);rows=frozen['cases'];future=rows[1:]
 require(frozen.get('settings_sha256')==SETTINGS_SHA and len(rows)==25 and len(future)==24
         and rows[0]['run_id']==SAMPLING_RID,'Original local continuation scope differs')
 ids=[r['run_id'] for r in future];science={r['run_id']:r['input_sha256'] for r in future}
 rid=item['run_id'];require(len(set(ids))==24 and rid in ids and science[rid]==item['input_sha256'],'Unapproved future local task/scientific identity')
 registry=OUT/'sampling_recovery_registry.json'
 require(registry.is_file(),'Future local CFD requires actual accepted first recovery')
 registry_sha=sha(registry)
 first=VERIFY_CACHE.get(('sampling-provenance',registry_sha))
 if first is None:
  first_value=sampling_recovery_source(rows[0],deep=True)
  require(first_value is not None,'First recovery no longer accepted');first=first_value[2]
 root_gate_sha=first['root_acceptance_gate']['sha256']
 jobs='runtime/jobs/'+rid+'/'
 overlay=json.loads(from_overlay(jobs+'sampling_producer_overlay.json'))
 handoff_raw=from_overlay(jobs+'owner_handoff.json')
 require(hashlib.sha256(handoff_raw).hexdigest()==FUTURE_HANDOFF_SHA,'Actual archived physical-host handoff differs')
 handoff=json.loads(handoff_raw);deployment=overlay.get('deployment',{})
 require(overlay.get('status')=='PASS' and overlay.get('run_id')==rid and overlay.get('container_id')==ci.get('Id')
         and request.get('container_id')==ci.get('Id'),'Actual local sampler overlay/container differs')
 require(request.get('image_id')==ci.get('Image')==deployment.get('image_id')==deployment.get('runtime_image_id')==IMAGE_ID
         and deployment.get('runtime_host_name')==FUTURE_LOCAL_HOST,'Unapproved physical host/runtime image')
 require(deployment.get('settings_sha256')==SETTINGS_SHA and deployment.get('first_case_id')==SAMPLING_RID
         and deployment.get('only_future_case_ids')==ids,'Local deployment changes frozen scope')
 require(deployment.get('first_case_root_gate',{}).get('sha256')==root_gate_sha
         and deployment.get('producer_manifest',{}).get('sha256')==SAMPLING_PRODUCER_SHA
         and deployment.get('owner_handoff',{}).get('sha256')==FUTURE_HANDOFF_SHA
         and deployment.get('original_frozen_queue',{}).get('sha256')==FUTURE_FROZEN_QUEUE_SHA,'Local deployment acceptance/producer/handoff/queue binding differs')
 require(handoff.get('status')=='safe_unstarted_owner_handoff' and handoff.get('original_host_name')=='local'
         and handoff.get('new_host_name')==FUTURE_LOCAL_HOST and handoff.get('run_ids')==ids
         and handoff.get('scientific_identities')==science and handoff.get('frozen_queue_sha256')==FUTURE_FROZEN_QUEUE_SHA
         and handoff.get('first_case_root_gate_sha256')==root_gate_sha,'Archived owner handoff scope/source differs')
 for key in ('original_lane_remains_disabled','original_worker_and_CFD_inactive','all_handed_off_atomic_jobs_absent','scientific_identity_and_inputs_unchanged'):
  require(handoff.get(key) is True,'Missing actual safe handoff evidence: '+key)
 require(handoff.get('original_settings_or_task_configs_changed') is False
         and handoff.get('original_control',{}).get('enable_remaining') is False
         and handoff.get('resource_policy')==dict(cpu_cores=8,memory_gib=10,one_CFD_at_a_time=True),'Handoff changes inputs/old disabled lane/resources')
 require(overlay.get('before_postFoam_sha256')=='4647de60c09d3b095aa5d53a966b4f26159695c93184cf811602403444288826'
         and overlay.get('after_postFoam_sha256')==deployment.get('binary',{}).get('sha256')==SAMPLING_BINARY_SHA
         and overlay.get('roughFoam_sha256')==deployment.get('roughFoam_sha256')==ROUGH_SHA,'Local sampling overlay changes solver/producer')
 require(overlay.get('container_not_started_during_overlay') is True and overlay.get('CFD_configuration_changed') is False
         and overlay.get('postFoam_mode')=='0755','Sampler overlay is not created-only/science-preserving')
 require(overlay.get('inputs_before_after_identical',{}).get('frozen_info.json')==item['input_sha256'],'Initial actual input integrity evidence differs')
 require(ci.get('HostConfig',{}).get('NanoCpus')==8_000_000_000 and ci['HostConfig'].get('Memory')==10*1024**3,'Local CFD resource cap differs from original task')
 return dict(runtime_host_name=FUTURE_LOCAL_HOST,runtime_image_id=IMAGE_ID,
             physical_owner_handoff_sha256=FUTURE_HANDOFF_SHA,first_case_root_gate_sha256=root_gate_sha,
             sampling_producer_sha256=SAMPLING_BINARY_SHA,original_logical_owner='local',actual_physical_owner='local',
             original_scientific_identity_and_inputs_unchanged=True)


def remote_rows(item):
 global ACTIVE_METADATA_SESSION
 verifier=module('mirror_cfd_workstation');reader=module('archive_metadata_reader');previous=ACTIVE_METADATA_SESSION
 with reader.ArchiveMetadataSession(verifier) as metadata:
  ACTIVE_METADATA_SESSION=metadata
  try:return _remote_rows_in_metadata_scope(item,verifier)
  finally:ACTIVE_METADATA_SESSION=previous

def _remote_rows_in_metadata_scope(item,mirror):
 rid=item['run_id'];d=OUT/item['group']/'remote_results'/rid;e=EBACK/item['group']/rid;ap=d/'D_E_verified_ack.json'
 if not ap.exists():return sampling_recovery_source(item)
 a=load(ap)
 if a.get('verified') is not True or a.get('status')!='PASS':return None
 require(a.get('run_id')==rid and a.get('group')==item['group'] and a.get('settings_sha256')==SETTINGS_SHA
         and a.get('scientific_identity_sha256')==item['input_sha256'],'Remote ACK identity/settings differs')
 require(sha(e/'D_E_verified_ack.json')==sha(ap),'D/E ACK bytes differ')
 qr=load(d/'quality_archive_receipt.json')
 b,qr,m,gate,x,_,_=mirror.validate_receipts(d,item,qr)
 require(gate.get('assessor_sha256')==ASSESSOR_SHA,'Remote gate assessor differs')
 archives=[d/(rid+'.tar.gz'),d/(rid+'.quality_addendum.tar.gz')]
 for folder in (d,e):
  ACTIVE_METADATA_SESSION.pin_expected(folder/archives[0].name,b['sha256']);ACTIVE_METADATA_SESSION.pin_expected(folder/archives[1].name,qr['sha256'])
 for folder in (d,e):
  for path,receipt,prefix in [(archives[0],b,'base_archive'),(archives[1],qr,'quality_archive')]:
   require(a[prefix+'_sha256']==receipt['sha256'] and a[prefix+'_bytes']==receipt['bytes'],'ACK archive binding differs')
   mirror.verify_archive(folder/path.name,receipt['bytes'],receipt['sha256'])
  for name in mirror.META_NAMES+('archive_overlay_index.json',):require(sha(folder/name)==sha(d/name),'D/E metadata differs: '+name)
  require(sha(folder/'archive_overlay_index.json')==a['D_E_overlay_index_sha256']
          and sha(folder/'post_assessment_artifact_manifest.json')==a['post_assessment_artifact_manifest_sha256'],'ACK manifest/index binding differs')
 idx=load(d/'archive_overlay_index.json');required=manifest_index(m['files'])
 require(idx['native_full_file_count']==a['native_full_file_count']==len(required)
         and idx['required_file_bytes']==a['native_full_file_bytes']==sum(r['bytes'] for r in required.values()),'ACK complete-file totals differ')
 key=('remote',str(d.resolve()),str(e.resolve()),sha(ap),b['sha256'],qr['sha256'])
 if key not in VERIFY_CACHE:
  di=mirror.validate_overlay(*archives,m,rid);ei=mirror.validate_overlay(e/archives[0].name,e/archives[1].name,load(e/'post_assessment_artifact_manifest.json'),rid)
  require(di==ei==idx,'Actual D/E streamed index differs from ACK index')
  mirror.compare_direct_metadata(d,di,rid);mirror.compare_direct_metadata(e,ei,rid);VERIFY_CACHE[key]=True
 index={('runtime/'+r['path']):r for r in idx['files']}
 prefix='runtime/quality/'+rid+'/strict_v1/'
 def get(name):
  full=prefix+name;row=index.get(full);require(row is not None,'Assessment artifact not in final overlay')
  archive=archives[1] if row['source_layer']=='quality_addendum' else archives[0]
  return tar_bytes(archive,full,index)
 def from_overlay(name):
  row=index.get(name);require(row is not None,'Missing actual execution/source artifact: '+name)
  archive=archives[1] if row['source_layer']=='quality_addendum' else archives[0]
  return tar_bytes(archive,name,index)
 q=json.loads(get('quality.json'));complete=json.loads(get('completed_assessment.json'))
 verify_completion(q,complete,get,item,ASSESSOR_SHA)
 require(hashlib.sha256(get('completed_assessment.json')).hexdigest()==gate['completed_assessment_sha256'],'Remote completion gate SHA differs')
 sourceprefix='runtime/uploads/'+rid+'/'
 require(index[sourceprefix+'frozen_info.json']['sha256']==item['input_sha256'],'Remote archived frozen info differs')
 for name,keyname in [('info.json','runtime_info_sha256'),('terrain.tif','terrain_sha256')]:
  require(index[sourceprefix+name]['sha256']==q['input_hashes'][keyname],'Remote quality source binding differs: '+name)
 execution=index['runtime/jobs/'+rid+'/execution_receipt.json']['sha256']
 require(q['input_hashes']['execution_receipt_sha256']==execution,'Remote quality execution binding differs')
 same_info(json.loads(from_overlay(sourceprefix+'info.json')),json.loads(from_overlay(sourceprefix+'frozen_info.json')))
 jobs='runtime/jobs/'+rid+'/'
 request=json.loads(from_overlay(jobs+'launch_request.json'));ci=json.loads(from_overlay(jobs+'container_inspect_final.json'))
 require(request.get('run_id')==rid and request.get('scientific_identity_sha256')==item['input_sha256']
         and request.get('settings_sha256')==SETTINGS_SHA and request.get('image_id')==IMAGE_ID,'Remote actual launch source differs')
 future_provenance={}
 if rid!=SAMPLING_RID:future_provenance=future_local_provenance(item,request,ci,from_overlay)
 exited(ci,request['container_id']);require(x.get('docker_container_id')==ci['Id'],'Native receipt/container ID differs')
 future_provenance.update(native_command_case_binding(item,request,ci,from_overlay))
 qi=json.loads(from_overlay(jobs+'quality_container_inspect.json'));exited(qi,gate['quality_container_id'])
 qualitycmd=' '.join(qi['Config']['Cmd'])
 require('assess_energy_cfd.py' in qualitycmd and '/uploads/'+rid in qualitycmd,'Remote actual assessor command/case differs')
 return list(csv.DictReader(io.StringIO(get('turbine_sector.csv').decode('utf-8-sig')))),q,dict(D_E_ack=str(ap),D_archives=[str(p) for p in archives],
      E_archives=[str(e/p.name) for p in archives],base_archive_sha256=b['sha256'],quality_archive_sha256=qr['sha256'],
      full_file_count=len(required),strict_full_file_stream_verified=True,**future_provenance)

def validate_rows(rows,item,info):
 expected={str(t['id']):t for t in info['turbines']}
 require(len(rows)==len(expected) and {str(r['turbine_id']) for r in rows}==set(expected),'Incomplete or duplicate turbine rows')
 wind=info['wind'];z0=float(wind['roughnessLength']);ref=float(wind['referenceHeight']);speed=float(wind['speed'])
 require(wind.get('profile')=='abl_log' and float(wind.get('displacementHeight',0))==0,'Unexpected frozen ABL profile')
 for row in rows:
  require(row['site']==item['site'] and row['run_id']==item['run_id'] and str(row['quality_pass']).lower()=='true','Wrong site/run/nonpassed CFD row')
  require(str(row.get('interpolation_supported','True')).lower()=='true','Unsupported vector cannot enter LUT')
  require(float(row['sector_deg'])==float(item['angle_deg']),'Sector identity mismatch')
  for key in ('hub_height_m','speedup','local_dir_deg','delta_dir_deg','u_east_mps','v_north_mps','w_mps','horizontal_speed_mps','speed3d_mps','inlet_at_hub_mps'):
   row[key]=float(row[key]);require(math.isfinite(row[key]),'Nonfinite LUT value')
  hub=float(expected[str(row['turbine_id'])]['hub'])
  require(math.isclose(row['hub_height_m'],hub,abs_tol=1e-9,rel_tol=0),'Wrong individual turbine hub height')
  inlet=speed*math.log((hub+z0)/z0)/math.log((ref+z0)/z0)
  require(row['speedup']>0 and row['inlet_at_hub_mps']>0 and row['horizontal_speed_mps']>0,'Nonpositive vector/speedup/denominator')
  close=lambda a,b:math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-8)
  require(close(row['inlet_at_hub_mps'],inlet),'Inlet denominator is not the same-hub frozen ABL speed')
  require(close(math.hypot(row['u_east_mps'],row['v_north_mps']),row['horizontal_speed_mps'])
          and close(math.hypot(row['horizontal_speed_mps'],row['w_mps']),row['speed3d_mps'])
          and close(row['horizontal_speed_mps']/row['inlet_at_hub_mps'],row['speedup']),'Inconsistent vector speed/ratio')
  direction=math.degrees(math.atan2(-row['u_east_mps'],-row['v_north_mps']))%360
  require(0<=row['local_dir_deg']<360 and -180<=row['delta_dir_deg']<180
          and abs(wrap(row['local_dir_deg']-direction))<1e-7
          and abs(wrap(row['delta_dir_deg']-wrap(direction-float(item['angle_deg']))))<1e-7,'Direction convention/wrapping/deflection differs')
  row.update(sector_deg=float(row['sector_deg']),z0_m=item['z0_m'],ct=item['ct'],case_kind=item['kind'])
 return rows

def lab_namespace_rows(item):
 # A separate immutable deployment/owner handoff is required. No native
 # namespace evidence can enter the Windows/Docker acceptance branch.
 contract=WORK/'ops/lab_deployment_20261023/lab_mirror_contract.json'
 if not contract.is_file():return None
 raw=load(contract);ids={r.get('run_id') for r in raw.get('cases',[]) if isinstance(r,dict)}
 if item['run_id'] not in ids:return None
 consumer=module('lab_cfd_evidence');context,rows,digest=consumer.lab.contract(contract)
 return consumer.lab_rows(item,context,deep=True)

def actual_rows(item):
 frozen=Path(item['bundle'])/'frozen_info.json'
 require(sha(frozen)==item['input_sha256'],'Current frozen input bundle SHA differs')
 info=load(frozen)
 if item['site']=='wind_farm_A':
  derived=module('mpi_finalize_cfd_evidence').recovery_rows(item,deep=True)
  if derived is not None:return derived
  parallel=module('guarded_g4_entry')
  if parallel.handles(item):return parallel.rows(item)
  guarded=module('guarded_g2_entry')
  if guarded.handles(item):return guarded.rows(item)
  lab_value=lab_namespace_rows(item)
  if lab_value is not None:
   # The native namespace consumer has already verified quality and returns
   # (rows, provenance). Its public pair also serves the separate lab report.
   rows,provenance=lab_value
   return validate_rows(rows,item,info),provenance
 value=local_rows(item) if item['site']=='kelmarsh' else remote_rows(item)
 if value is None:return None
 rows,q,provenance=value
 return validate_rows(rows,item,info),provenance
def main():
 settings=OUT/'cfd_settings_frozen.json'
 require(sha(settings)==SETTINGS_SHA,'Formal CFD settings changed')
 plan=load(settings);accepted={};provenance={};rejected=[]
 require(len(plan['case_queue'])==38 and len({t['run_id'] for t in plan['case_queue']})==38,'Frozen 38-case scope differs')
 for item in plan['case_queue']:
  try:
   value=actual_rows(item)
   if value:accepted[item['run_id']],provenance[item['run_id']]=value
  except (ValueError,KeyError,OSError,RuntimeError,tarfile.TarError,EOFError) as error:
   rejected.append(dict(run_id=item['run_id'],error_type=type(error).__name__,reason=str(error)))
 summary=[]
 for site,n in [('wind_farm_A',16),('kelmarsh',12)]:
  tasks=[r for r in plan['case_queue'] if r['site']==site and r['kind']=='sector']
  rows=[r for item in tasks for r in accepted.get(item['run_id'],[])]
  completed=sum(item['run_id'] in accepted for item in tasks)
  expected_rows={'wind_farm_A':464,'kelmarsh':72}[site]
  rowkeys={(r['site'],str(r['turbine_id']),float(r['sector_deg'])) for r in rows}
  final=(completed==n and len(tasks)==n and len(rows)==expected_rows and len(rowkeys)==expected_rows
         and all(r['site']==site and r['case_kind']=='sector' and float(r['ct'])==0
                 and float(r['z0_m'])==float(plan['roughness'][site]['baseline_m']) for r in rows))
  if not final:
   stale=OUT/f'cfd_{site}_turbine_sector.csv'
   if stale.exists():
    archive=OUT/'superseded_tables';archive.mkdir(exist_ok=True)
    old=archive/(stale.stem+'.'+sha(stale)+'.csv')
    if old.exists():require(sha(old)==sha(stale),'Superseded table conflict');stale.unlink()
    else:os.replace(stale,old)
  name=f'cfd_{site}_turbine_sector'+('' if final else '.partial')+'.csv'
  writecsv(OUT/name,rows)
  values=[abs(r['delta_dir_deg']) for r in rows]
  info={'site':site,'completed_sector_cases':completed,'required_sector_cases':n,'complete':final,
   'accepted_turbine_rows':len(rows),'required_turbine_rows':expected_rows,'unique_turbine_sector_rows':len(rowkeys),
   'quality_definition':'strict native45 or exact root-approved MPI-finalize post-only44+post0; original native1/rawFAIL retained; full D/E streams',
   'source_profile_counts':module('mpi_finalize_cfd_evidence').source_counts({t['run_id']:provenance[t['run_id']] for t in tasks if t['run_id'] in provenance}),
   'table':name if rows else None,'descriptive_abs_direction_deg':({'median':percentile(values,.5),'p90':percentile(values,.9),'maximum':max(values)} if values else None),
   'case_sources':{t['run_id']:provenance[t['run_id']] for t in tasks if t['run_id'] in provenance},
   'roughness_m':plan['roughness'][site]['baseline_m'],'assumptions':plan['turbulence'],
   'limitations':['uniform roughness is not site calibrated','DSM contains canopy/buildings','no grid independence or observational agreement claimed by a convergence pass']}
  dump(OUT/f'cfd_{site}_metadata.json',info);summary.append(info)
 for kind,name in [('intersector_check','cfd_wind_farm_A_intersector_direct.csv'),('wake_pair','cfd_wind_farm_A_wake_Ct08.csv'),('roughness_sensitivity','cfd_roughness_sensitivity_sources.csv')]:
  rows=[r for t in plan['case_queue'] if t['kind']==kind for r in accepted.get(t['run_id'],[])]
  writecsv(OUT/name,rows)
 # Baseline vs alternative roughness: same turbine/angle only, never change LUT baseline.
 baseline={(r['site'],r['turbine_id'],r['sector_deg']):r for t in plan['case_queue'] if t['kind']=='sector' for r in accepted.get(t['run_id'],[])}
 sensitivity=[];wake=[];inter=[]
 for t in plan['case_queue']:
  for r in accepted.get(t['run_id'],[]):
   b=baseline.get((r['site'],r['turbine_id'],r['sector_deg']))
   if t['kind']=='roughness_sensitivity' and b:
    sensitivity.append({'site':r['site'],'turbine_id':r['turbine_id'],'sector_deg':r['sector_deg'],'baseline_z0_m':b['z0_m'],'alternative_z0_m':r['z0_m'],
     'baseline_run_id':b['run_id'],'alternative_run_id':r['run_id'],'speedup_difference':r['speedup']-b['speedup'],
     'direction_difference_deg':wrap(r['local_dir_deg']-b['local_dir_deg'])})
   if t['kind']=='wake_pair' and b:
    wake.append({'site':r['site'],'turbine_id':r['turbine_id'],'sector_deg':r['sector_deg'],'terrain_run_id':b['run_id'],'loaded_run_id':r['run_id'],
     'terrain_horizontal_speed_mps':b['horizontal_speed_mps'],'loaded_horizontal_speed_mps':r['horizontal_speed_mps'],
     'loaded_minus_terrain_mps':r['horizontal_speed_mps']-b['horizontal_speed_mps'],'direction_change_deg':wrap(r['local_dir_deg']-b['local_dir_deg'])})
   if t['kind']=='intersector_check':
    low=math.floor(r['sector_deg']/22.5)*22.5;high=(low+22.5)%360
    l=baseline.get(('wind_farm_A',r['turbine_id'],low));h=baseline.get(('wind_farm_A',r['turbine_id'],high))
    if l and h:
     weight=((r['sector_deg']-low)%360)/22.5
     pd=wrap(l['delta_dir_deg']+weight*wrap(h['delta_dir_deg']-l['delta_dir_deg']))
     ps=l['speedup']+weight*(h['speedup']-l['speedup'])
     inter.append({'turbine_id':r['turbine_id'],'check_direction_deg':r['sector_deg'],'direct_run_id':r['run_id'],'left_run_id':l['run_id'],'right_run_id':h['run_id'],
      'direct_delta_deg':r['delta_dir_deg'],'interpolated_delta_deg':pd,'angle_error_deg':wrap(pd-r['delta_dir_deg']),
      'direct_speedup':r['speedup'],'interpolated_speedup':ps,'speedup_error':ps-r['speedup']})
 writecsv(OUT/'cfd_roughness_paired_changes.csv',sensitivity);writecsv(OUT/'cfd_wind_farm_A_wake_paired_changes.csv',wake);writecsv(OUT/'cfd_wind_farm_A_intersector_errors.csv',inter)
 result={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'settings_sha256':SETTINGS_SHA,'summary_script_sha256':sha(Path(__file__)),
  'complete_cases':len(accepted),'required_cases':38,'rejected_cases':rejected,'sites':summary,
  'source_profile_counts':module('mpi_finalize_cfd_evidence').source_counts(provenance),
  'original_failure_sources':{rid:s for rid,s in provenance.items() if s.get('recovery') or s.get('sampling_recovery')},
  'sensitivity_paired_rows':len(sensitivity),'wake_paired_rows':len(wake),'intersector_error_rows':len(inter),'all_case_sources':provenance,
  'no_predictions_or_model_selection_used':True}
 dump(OUT/'cfd_results_summary.json',result)
 lines=['# 两风场 CFD 结果汇总','',f"已核验并完成D/E保存：{len(accepted)}/38个工况。未通过或未齐全不写入正式LUT。",'',
  '| 风场 | 已通过扇区/所需 | 实际机位行数 | 表状态 |','|---|---:|---:|---|']
 for s in summary:lines.append(f"| {s['site']} | {s['completed_sector_cases']}/{s['required_sector_cases']} | {s['accepted_turbine_rows']} | {'完整' if s['complete'] else '阶段性'} |")
 lines+=['', '验收来源分类：'+json.dumps(result['source_profile_counts'],ensure_ascii=False)+'。MPI收尾恢复保留原native exit1与原45项raw FAIL，只以独立44项通过+真实post0及root验收计入。','粗糙度敏感性、夹角直接计算和Ct=0.8尾流对照分别存表，均不混入默认扇区LUT。所有数字只来自真实向量采样。',
  '风向偏转和加速比属于当前中性、均一粗糙壁和DSM假设下的仿真结果；收敛不能替代网格独立或实测验证。',
  '完整原生场、输入、实际退出凭据、网格和质量日志及逐文件备份清单见各工况的results/remote_results目录。']
 (OUT/'CFD结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'accepted':len(accepted),'required':38,'formal_lut_complete':all(s['complete'] for s in summary)}),flush=True)
if __name__=='__main__':main()
