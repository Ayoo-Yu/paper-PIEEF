from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'engine_snapshot/backend/physics_engine/tests/run_abl_validation.py'
text=src.read_text(encoding='utf-8')
text=text.replace('ENGINE = Path(__file__).resolve().parents[1]', 'ENGINE = Path("/opt/physics/backend")')
text=text.replace('parser.add_argument("--output",type=Path,required=True)', 'parser.add_argument("--output",type=Path,required=True)\n    parser.add_argument("--z0",type=float,required=True)\n    parser.add_argument("--reference-height",type=float,required=True)\n    parser.add_argument("--reference-speed",type=float,default=8.)')
old='wind=dict(profile="abl_log",speed=5.,referenceHeight=80.,roughnessLength=.03,displacementHeight=0.),'
new='wind=dict(profile="abl_log",speed=args.reference_speed,referenceHeight=args.reference_height,roughnessLength=args.z0,displacementHeight=0.),'
assert old in text
text=text.replace(old,new)
dst=ROOT/'scripts/run_energy_abl_validation.py'
dst.write_text(text,encoding='utf-8')
(ROOT/'audit/abl_parameterization_manifest.json').write_text(json.dumps({
 'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'derived':str(dst),'derived_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
 'changes':['explicit engine path','parameterize physical z0, reference height, reference speed'],
 'unchanged_gates':{'SIMPLE':1e-4,'U_relative_drift':0.02,'k_relative_drift':0.05,'epsilon_relative_drift':0.05},
 'purpose':'numerical neutral flat-fetch preservation; not site calibration'},indent=2)+'\n',encoding='utf-8')
print(dst)
