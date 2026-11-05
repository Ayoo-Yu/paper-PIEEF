"""Strict read-only CFD assessment and supported per-turbine vector extraction.

Only foamToVTK postprocessing may be invoked; this script never starts a solver,
rebuilds a mesh, changes inputs, fills missing values, or selects a new attempt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.spatial import Delaunay,QhullError

HERE=Path(__file__).resolve()
ENGINE=HERE.parents[1]/'engine_snapshot/backend/physics_engine'
if not (ENGINE/'quality.py').is_file():
    ENGINE=Path('/opt/physics/backend')
sys.path.insert(0,str(ENGINE))
from quality import assess_case_quality

FIELDS=('U','p','k','epsilon','nut','SourceT')
RUNTIME_KEYS={'calculationStatus','completedAt','startedAt','updatedAt'}
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def strict_json(path):
    def invalid(value):
        raise ValueError(f'Non-finite JSON token: {value}')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)


def frozen_equivalent(current,frozen):
    return ({k:v for k,v in current.items() if k not in RUNTIME_KEYS}
            =={k:v for k,v in frozen.items() if k not in RUNTIME_KEYS})


def rotation_to_geographic(us,vs,theta_deg):
    theta=math.radians(theta_deg)
    east=-math.sin(theta)*us+math.cos(theta)*vs
    north=-math.cos(theta)*us-math.sin(theta)*vs
    return east,north


def direction_from(east,north):
    if not np.isfinite([east,north]).all() or math.hypot(east,north)<=0:
        raise ValueError('Wind direction is undefined for absent or zero horizontal wind')
    return math.degrees(math.atan2(-east,-north))%360.


def abl_speed(height,wind):
    if wind.get('profile')!='abl_log' or float(wind.get('displacementHeight',0))!=0:
        raise ValueError('This frozen protocol requires neutral ABL, displacement=0')
    z0=float(wind['roughnessLength']);reference=float(wind['referenceHeight']);speed=float(wind['speed'])
    if not all(math.isfinite(value) and value>0 for value in (height,z0,reference,speed)):
        raise ValueError('Invalid ABL speed/height parameters')
    return speed*math.log((height+z0)/z0)/math.log((reference+z0)/z0)


def interpolate_vectors(samples,query_xy_model):
    """Linear component interpolation, no extrapolation or bridging NaN holes."""
    data=np.asarray(samples,dtype=float)
    if data.ndim!=2 or data.shape[1]<7 or not np.isfinite(data[:,:3]).all():
        raise ValueError('Expected seven-column native XYZ,Ux,Uy,Uz,p with finite coordinates')
    if np.isinf(data[:,3:7]).any():
        raise ValueError('Infinite sampled field')
    xy,inverse=np.unique(data[:,:2],axis=0,return_inverse=True)
    count=np.bincount(inverse)
    values=np.column_stack([np.bincount(inverse,weights=data[:,index])/count for index in range(3,7)])
    if len(xy)<3:
        raise ValueError('Insufficient 2D sample support')
    try:
        triangles=Delaunay(xy)
    except QhullError as error:
        raise ValueError('Collinear sample support') from error
    queries=np.asarray(query_xy_model,dtype=float).reshape(-1,2)
    if not np.isfinite(queries).all():
        raise ValueError('Non-finite query coordinates')
    simplex=triangles.find_simplex(queries)
    result=np.full((len(queries),4),np.nan)
    supported=np.zeros(len(queries),dtype=bool)
    for index,(point,cell) in enumerate(zip(queries,simplex)):
        if cell<0:
            continue
        delta=point-triangles.transform[cell,2]
        first=triangles.transform[cell,:2].dot(delta)
        weights=np.r_[first,1-first.sum()]
        if (weights < -1e-10).any():
            continue
        weights=np.clip(weights,0.,1.);weights/=weights.sum()
        contributors=values[triangles.simplices[cell]]
        finite=np.isfinite(contributors)
        if not (finite | (weights[:,None]<=1e-12)).all():
            continue
        result[index]=(np.where(finite,contributors,0.)*weights[:,None]).sum(axis=0)
        supported[index]=True
    return result,supported


def check_export_arrays(arrays,zero_source):
    """Internal-volume field gates; signed pressure and zero nut remain legal."""
    checks={};statistics={}
    for name in FIELDS:
        if name not in arrays:
            checks[f'{name}_present']=False
            continue
        values=np.asarray(arrays[name])
        finite=bool(values.size and np.isfinite(values).all())
        checks[f'{name}_finite']=finite
        if finite:
            statistics[name]=dict(min=float(values.min()),max=float(values.max()),size=int(values.size))
            if name in ('k','epsilon'):
                checks[f'{name}_positive']=bool((values>0).all())
            if name=='nut':
                checks['nut_nonnegative']=bool((values>=0).all())
            if name=='SourceT' and zero_source:
                checks['terrain_SourceT_identically_zero']=bool((values==0).all())
    return checks,statistics


def source_log_checks(text,zero_source):
    raw=re.findall(r'UEqn Sources Check -> Max \|RoughT\|:\s*(\S+)\s*,\s*Max \|SourceT\|:\s*(\S+)',text)
    try:
        values=np.asarray([(float(a),float(b)) for a,b in raw],dtype=float).reshape(-1,2)
    except ValueError:
        return dict(source_log_present=False,source_log_finite=False)
    checks=dict(source_log_present=bool(len(values)),source_log_finite=bool(len(values) and np.isfinite(values).all()))
    if checks['source_log_finite']:
        checks['canopy_source_zero']=bool((values[:,0]==0).all())
        if zero_source:
            checks['terrain_source_log_zero']=bool((values[:,1]==0).all())
    return checks


def check_actuators(rows,n_turbines,zero_source):
    checks={};summary={}
    needed={'stage','iteration','turbine','upstream_cells','upstream_volume_model','disk_cells','disk_volume_model',
            'wind_speed_m_s','ct','accel_integral_target_model','accel_integral_discretized_model'}
    if not rows or needed-set(rows[0]):
        return {'actuator_diagnostics_complete':False},summary
    try:
        for row in rows:
            for key in needed-{'stage'}:
                if not math.isfinite(float(row[key])):
                    raise ValueError(f'Non-finite {key}')
        checks['actuator_diagnostics_finite']=True
        checks['all_reference_and_disk_support_positive']=all(
            float(row[key])>0 for row in rows for key in ('upstream_cells','upstream_volume_model','disk_cells','disk_volume_model'))
        ids=set(range(1,n_turbines+1))
        checks['actuator_turbine_identity_complete']={int(float(row['turbine'])) for row in rows}==ids
        final=[row for row in rows if row['stage']=='final_resampled']
        checks['final_actuator_rows_exact']=len(final)==n_turbines and {int(float(row['turbine'])) for row in final}==ids
        if zero_source:
            checks['all_terrain_ct_and_integrals_zero']=all(float(row[key])==0 for row in rows
                for key in ('ct','accel_integral_target_model','accel_integral_discretized_model'))
        else:
            checks['all_loaded_ct_08']=all(float(row['ct'])==.8 for row in rows)
        errors=[abs(float(row['accel_integral_target_model'])-float(row['accel_integral_discretized_model']))
                  /max(abs(float(row['accel_integral_target_model'])),1e-15) for row in rows]
        checks['thrust_integral_relative_error_le_1e_9']=max(errors)<=1e-9
        summary['maximum_relative_thrust_integral_error']=max(errors)
        pre=[row for row in rows if row['stage']=='pre_solve']
        iterations=sorted({float(row['iteration']) for row in pre})
        checks['tail20_actuator_iterations_available']=len(iterations)>=20
        tail=iterations[-20:]
        relevant=[row for row in pre if float(row['iteration']) in tail]
        tail_ids={(float(row['iteration']),int(float(row['turbine']))) for row in relevant}
        checks['tail20_all_turbines_complete']=len(tail)==20 and len(relevant)==20*n_turbines and len(tail_ids)==20*n_turbines
        spans={}
        if checks['tail20_all_turbines_complete']:
            for turbine in ids:
                speed=np.asarray([float(row['wind_speed_m_s']) for row in relevant if int(float(row['turbine']))==turbine])
                mean=float(np.mean(speed))
                spans[str(turbine)]=float((speed.max()-speed.min())/max(abs(mean),1e-12))
            checks['tail20_Uref_relative_span_le_1percent']=max(spans.values())<=.01
        else:
            checks['tail20_Uref_relative_span_le_1percent']=False
        summary.update(tail20_iterations=tail,tail20_Uref_relative_span_by_turbine=spans,
                       tail20_status='assessed' if checks['tail20_all_turbines_complete'] else 'not_assessed')
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as error:
        checks['actuator_diagnostics_finite']=False
        summary['error']=str(error)
    return checks,summary


def export_fields(run,output):
    """Separate native final-field export; intentionally retains its raw log."""
    export_name='VTK_quality_'+hashlib.sha256(str(output).encode('utf-8')).hexdigest()[:10]
    target=run/export_name
    if target.exists():
        raise ValueError('This assessment export directory already exists; use a new explicit assessment output directory')
    command=['foamToVTK','-latestTime','-fields','('+ ' '.join(FIELDS) +')','-no-point-data','-name',export_name]
    with (output/'foamToVTK_quality.log').open('w',encoding='utf-8') as handle:
        completed=subprocess.run(command,cwd=run,stdout=handle,stderr=subprocess.STDOUT,check=False)
    if completed.returncode:
        raise ValueError(f'Native final-field export failed with exit {completed.returncode}')
    candidates=list(target.glob('*/internal.vtu'))
    if not candidates:
        raise ValueError('No full-field native internal.vtu export')
    def export_time(path):
        match=re.search(r'_('+NUMBER+r')$',path.parent.name)
        return float(match.group(1)) if match else float('-inf')
    selected=max(candidates,key=export_time)
    import pyvista as pv
    volume=pv.read(selected)
    arrays={name:np.asarray(volume.cell_data[name]) for name in FIELDS if name in volume.cell_data}
    checks,statistics=check_export_arrays(arrays,False)
    # Read every exported boundary too, ensuring no finite-volume patch hides
    # a nonfinite value. Strict positivity is an internal-field requirement.
    boundaries={}
    for path in sorted((selected.parent/'boundary').glob('*.vtp')):
        surface=pv.read(path)
        combined={name:np.asarray(surface.cell_data[name]) if name in surface.cell_data
                  else np.asarray(surface.point_data[name]) for name in FIELDS
                  if name in surface.cell_data or name in surface.point_data}
        boundaries[path.stem]={name:bool(data.size and np.isfinite(data).all()) for name,data in combined.items()}
    checks['all_exported_boundary_fields_present_and_finite']=bool(boundaries) and all(
        len(value)==len(FIELDS) and all(value.values()) for value in boundaries.values())
    return arrays,dict(command=command,volume_path=str(selected),volume_sha256=sha(selected),
                       cell_count=volume.n_cells,exported_time=export_time(selected),
                       fields=statistics,boundaries=boundaries,checks=checks)


def find_layer(case,height):
    directory=case/'run/Output/plt'
    matches=[]
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            equal=math.isclose(float(path.name),height,rel_tol=1e-12,abs_tol=1e-9)
        except ValueError:
            equal=False
        if equal:
            matches.append(path)
    if len(matches)!=1:
        raise ValueError(f'Expected exactly one native AGL layer {height}; found {len(matches)}')
    return matches[0]


def evaluate(case,output):
    output.mkdir(parents=True,exist_ok=False)
    info_path=case/'info.json';frozen_path=case/'frozen_info.json'
    if not frozen_path.is_file():
        raise ValueError('case/frozen_info.json is mandatory; current status JSON is not immutable evidence')
    info=strict_json(info_path);frozen=strict_json(frozen_path)
    checks={'frozen_input_identity':frozen_equivalent(info,frozen),
            'pipeline_completed':info.get('calculationStatus')=='completed',
            'neutral_ABL':frozen.get('wind',{}).get('profile')=='abl_log',
            'displacement_zero':float(frozen.get('wind',{}).get('displacementHeight',0))==0,
            'canopy_disabled':float(frozen.get('roughness',{}).get('vege_times',0))==0,
            'native_deltaT_one':float(frozen.get('simulation',{}).get('pseudoTimeStep',frozen.get('simulation',{}).get('deltaT',0)))==1}
    receipt_path=case/'execution_receipt.json'
    receipt=strict_json(receipt_path) if receipt_path.is_file() else {}
    checks['actual_process_exit_zero']=receipt.get('exit_code')==0 and not isinstance(receipt.get('exit_code'),bool) and receipt.get('run_id')==case.name
    info=frozen
    log=case/'calculation.log'
    text=log.read_text(encoding='utf-8',errors='strict')
    q=assess_case_quality(log,case)
    checks['SIMPLE_converged_with_initial_residuals']=q['convergence']=='converged' and q['solver_finished']
    controls=q['residual_control']
    turbulence_controls=[pattern for pattern in controls if pattern not in ('p','U')]
    checks['residual_control_1e_4']=('p' in controls and 'U' in controls
        and any(re.fullmatch(pattern,'k') and re.fullmatch(pattern,'epsilon') for pattern in turbulence_controls)
        and all(float(value)==1e-4 for value in controls.values()))
    mesh=q.get('mesh_check',{})
    evidence=mesh.get('evidence',[])
    checks['standard_mesh_pass']=any(row.get('source')=='run/checkMesh.log' and row.get('status')=='passed' and row.get('normal_end') for row in evidence)
    checks['extended_mesh_pass']=any(row.get('extended_scope') and row.get('status')=='passed' and row.get('normal_end') for row in evidence)
    continuity=q.get('continuity_errors') or {}
    checks['continuity_sum_local_le_1e_3']=bool(continuity) and math.isfinite(float(continuity['sum_local'])) and 0<=float(continuity['sum_local'])<=1e-3
    checks['continuity_abs_global_le_1e_6']=bool(continuity) and abs(float(continuity['global']))<=1e-6
    recent=q.get('recent_iterations',[])
    tail={}
    for row in recent:
        for field,value in row.get('initial_residuals',{}).items():
            tail[field]=max(float(value),tail.get(field,0.))
    terrain=q.get('terrain_quality') or {}
    checks['terrain_no_outside_padding']=q.get('terrain_quality_status')=='available' and terrain.get('outside_dem_bottom_node_count')==0 and terrain.get('full_terrain_coverage') is True
    clearances=terrain.get('turbine_clearance',[])
    checks['all_turbines_geometric_clearance_positive']=len(clearances)==len(info['turbines']) and all(
        float(row['upper_blade_tip_roof_clearance_m'])>0 and float(row['minimum_sampled_lower_rotor_terrain_clearance_m'])>0
        and row['rotor_and_reference_support_checked'] is True for row in clearances)
    anchors_path=case/'run/Output/terrain-anchor-diagnostics.tsv'
    # Preserve the exact current source contract, without newest-file search.
    anchors=[]
    if anchors_path.is_file():
        with anchors_path.open(encoding='utf-8',newline='') as handle:
            anchors=list(csv.DictReader(handle,delimiter='\t'))
    checks['anchor_turbine_identity_complete']=len(anchors)==len(info['turbines']) and {int(row['turbine']) for row in anchors}==set(range(1,len(info['turbines'])+1))
    anchors=sorted(anchors,key=lambda row:int(row['turbine']))
    checks['anchor_AGL_all_exact']=len(anchors)==len(info['turbines']) and all(
        row.get('anchor_version')=='vertical-bottom-triangle-v1'
        and math.isfinite(float(row['realized_hub_agl_m']))
        and abs(float(row['realized_hub_agl_m'])-float(turbine['hub']))<=1e-6
        and abs(float(row['requested_hub_agl_m'])-float(turbine['hub']))<=1e-6
        for row,turbine in zip(anchors,info['turbines']))
    curves=[]
    for path in sorted((case/'customCurves').glob('*-U-P-Ct.txt')):
        values=np.loadtxt(path,ndmin=2)
        valid=values.shape[1]==3 and 2<=len(values)<=200 and np.isfinite(values).all() and (values>=0).all() and (np.diff(values[:,0])>0).all()
        if not valid:
            raise ValueError(f'Invalid actual Ct curve: {path}')
        curves.append((path,values))
    if not curves:
        raise ValueError('Missing actual Ct curves')
    zero_source=all((values[:,2]==0).all() for path,values in curves)
    checks['curve_Ct_frozen_zero_or_08']=zero_source or all((values[:,2]==.8).all() for path,values in curves)
    checks.update(source_log_checks(text,zero_source))
    actuator_path=case/'run/Output/actuator-diagnostics.tsv'
    with actuator_path.open(encoding='utf-8',newline='') as handle:
        actuator_rows=list(csv.DictReader(handle,delimiter='\t'))
    actuator_checks,actuator=check_actuators(actuator_rows,len(info['turbines']),zero_source)
    checks.update(actuator_checks)
    arrays,export=export_fields(case/'run',output)
    field_checks,field_stats=check_export_arrays(arrays,zero_source)
    checks.update(field_checks);checks.update(export['checks'])
    checks['final_export_time_matches_solver']=math.isfinite(export['exported_time']) and q.get('last_iteration') is not None and abs(export['exported_time']-float(q['last_iteration']))<=1.
    sampling=q.get('agl_sampling_quality') or {}
    layers=sampling.get('layers',[])
    expected=info['post'].get('heights',[info['post']['dh']*(index+1) for index in range(info['post']['numh'])] if 'dh' in info['post'] else [])
    checks['all_AGL_layers_complete']=len(layers)==len(expected) and all(
        math.isclose(float(layer['height_m']),float(height),rel_tol=1e-12,abs_tol=1e-9)
        and int(layer['valid_point_count'])==int(layer['requested_point_count'])
        and int(layer['outside_domain_count'])==0 and int(layer['non_finite_count'])==0
        for layer,height in zip(layers,expected))
    theta=float(info['wind']['angle']);scale=float(info['mesh']['scale'])
    angle=math.radians(theta+90.)
    turbine_rows=[];layer_hashes={};loaded={}
    for turbine in info['turbines']:
        hub=float(turbine['hub'])
        if hub not in loaded:
            path=find_layer(case,hub)
            loaded[hub]=np.loadtxt(path,ndmin=2)
            layer_hashes[str(hub)]=dict(path=str(path.relative_to(case)),sha256=sha(path))
        east=float(turbine['x']);north=float(turbine['y'])
        solver_x=(east*math.cos(angle)-north*math.sin(angle))*scale
        solver_y=(east*math.sin(angle)+north*math.cos(angle))*scale
        values,supported=interpolate_vectors(loaded[hub],[(solver_x,solver_y)])
        row=dict(site=info.get('site',info.get('site_code','unknown')),turbine_id=turbine['id'],sector_deg=theta,
                 hub_height_m=hub,speedup=None,local_dir_deg=None,delta_dir_deg=None,
                 run_id=case.name,quality_pass=False,interpolation_supported=bool(supported[0]),
                 u_east_mps=None,v_north_mps=None,w_mps=None,horizontal_speed_mps=None,
                 speed3d_mps=None,inlet_at_hub_mps=abl_speed(hub,info['wind']))
        if supported[0]:
            us,vs,w,p=(float(value) for value in values[0])
            u,v=rotation_to_geographic(us,vs,theta)
            try:
                direction=direction_from(u,v)
            except ValueError:
                row['interpolation_supported']=False
            else:
                speed=math.hypot(u,v)
                row.update(u_east_mps=u,v_north_mps=v,w_mps=w,horizontal_speed_mps=speed,
                           speed3d_mps=math.sqrt(speed*speed+w*w),speedup=speed/row['inlet_at_hub_mps'],
                           local_dir_deg=direction,delta_dir_deg=(direction-theta+180.)%360.-180.)
        turbine_rows.append(row)
    checks['every_turbine_vector_supported_and_positive']=all(row['interpolation_supported'] and row['speedup'] is not None and math.isfinite(row['speedup']) and row['speedup']>0 for row in turbine_rows)
    passed=bool(checks) and all(value is True for value in checks.values())
    for row in turbine_rows:
        row['quality_pass']=passed
    with (output/'turbine_sector.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(turbine_rows[0]));writer.writeheader();writer.writerows(turbine_rows)
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),case=str(case),run_id=case.name,
                status='PASS' if passed else 'FAIL',quality_pass=passed,checks=checks,
                failed_checks=[key for key,value in checks.items() if value is not True],
                kind='terrain_zero_Ct' if zero_source else 'loaded_Ct08',native_quality=q,
                tail20_max_initial_residuals=tail,tail20_residual_status='assessed' if len(recent)>=20 else 'not_assessed',
                continuity_gates=dict(sum_local_max=1e-3,abs_global_max=1e-6,native_deltaT_required=1),
                actuator=actuator,field_export=export,all_native_field_stats=field_stats,
                speedup_definition='horizontal speed magnitude / neutral ABL inlet speed at the same turbine AGL height',
                interpolation='native bottom-face-centre AGL vectors; horizontal linear barycentric interpolation; no extrapolation/NaN filling',
                local_direction_definition='meteorological from-direction after solver-to-East/North rotation',
                input_hashes=dict(frozen_info_sha256=sha(frozen_path),runtime_info_sha256=sha(info_path),
                                  execution_receipt_sha256=sha(receipt_path) if receipt_path.is_file() else None,
                                  calculation_log_sha256=sha(log),terrain_sha256=sha(case/'terrain.tif')),
                native_hub_layer_hashes=layer_hashes,assessor_sha256=sha(HERE),
                original_quality_source_sha256=sha(ENGINE/'quality.py'),
                limitations=['Mesh pass and convergence do not establish grid independence or observed site accuracy.',
                             'This assessor does not start CFD, change convergence thresholds, or select another run.'])
    save(output/'quality.json',result)
    save(output/'completed_assessment.json',dict(status=result['status'],quality_pass=passed,
         files={name:sha(output/name) for name in ('quality.json','turbine_sector.csv','foamToVTK_quality.log')}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        print(json.dumps(dict(status='REFUSED',error='Assessment output already exists; preserve it and choose a fresh output directory'),ensure_ascii=False),flush=True)
        return 2
    try:
        result=evaluate(args.case.resolve(),args.output.resolve())
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        failure=dict(status='FAIL',quality_pass=False,error_type=type(error).__name__,error=str(error),
                     case=str(args.case),created_utc=datetime.now(timezone.utc).isoformat())
        save(args.output/'assessment_failed.json',failure)
        print(json.dumps(failure,ensure_ascii=False),flush=True)
        return 2
    print(json.dumps(dict(status=result['status'],failed_checks=result['failed_checks']),ensure_ascii=False),flush=True)
    return 0 if result['quality_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
