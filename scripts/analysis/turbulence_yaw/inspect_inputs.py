import sys, json, zipfile, csv, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parents[4]
t = pd.read_csv(W/'补做重算/tables/T1_per_seed.csv')
r = t[t.site=='kelmarsh'].iloc[0]
for d in [Path(r.VF_dir),W/'补充验证_20261023/M_controls/VF_Kelmarsh_h1_seed13']:
    print('DIRECTORY',d)
    print([p.name for p in d.iterdir()])
    for name in ['preprocessing.npz','data_protocol.json','predictions.npz','test_attention_diagnostics.npz','val_attention_diagnostics.npz','val_predictions.npz']:
        p=d/name
        if not p.exists():continue
        if p.suffix=='.json': print(name,p.read_text(encoding='utf-8-sig'));continue
        z=np.load(p,allow_pickle=False)
        print(name,{k:(str(z[k].dtype),z[k].shape) for k in z.files})
        for k in z.files:
            if z[k].size<=30 or 'timestamp' in k: print(k,z[k] if z[k].size<=30 else (z[k][0],z[k][-1]))
print('matplotlib',importlib.util.find_spec('matplotlib'))
with zipfile.ZipFile('data/kelmarsh/Kelmarsh_SCADA_2024_5962.zip') as z:
    for n in z.namelist():
        if 'Kelmarsh_1_' not in n:continue
        print('ZIP',n)
        with z.open(n) as f:
            for i in range(13):
                s=f.readline().decode('utf-8-sig')
                if len(s)>2000: print(s[:2000], '... [wide row truncated]')
                else: print(s.rstrip())
        if 'Status_' in n:
            with z.open(n) as f:
                q=pd.read_csv(f,skiprows=9)
            print(q.columns.tolist());print(q.head(5).to_string(index=False))
            for c in q.columns:
                if q[c].nunique()<100:print(c,q[c].value_counts().to_dict())
