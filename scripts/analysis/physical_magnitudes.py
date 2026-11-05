
"""Physical-correction magnitudes: how large are the density and cut-out adjustments actually?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

wind_farm_A = Path(r'data/wind_farm_A/scada_ecmwf_15min.csv')
df = pd.read_csv(wind_farm_A, nrows=5)
cols = list(df.columns)
press = [c for c in cols if 'pressure' in c.lower()]
temp = [c for c in cols if 'temperature' in c.lower()]
u = [c for c in cols if 'U_wind' in c or 'u_wind' in c]
v = [c for c in cols if 'V_wind' in c or 'v_wind' in c]
print('pressure cols:', press[:3])
print('temperature cols:', temp[:3])
print('u/v cols:', u[:2], v[:2])

df = pd.read_csv(wind_farm_A, usecols=[c for c in (press[:1]+temp[:1]+u[:1]+v[:1]) if c])
P = pd.to_numeric(df[press[0]], errors='coerce')
T = pd.to_numeric(df[temp[0]], errors='coerce')
out = {}
if len(P) and len(T):
    Ppa = P*100.0 if P.median() < 2000 else P       # hPa -> Pa when needed
    Tk = T + 273.15 if T.median() < 100 else T
    rho = Ppa/(287.05*Tk)
    ref = float(np.nanmedian(rho))
    ratio = rho/ref
    out['density'] = dict(n=int(np.isfinite(ratio).sum()), reference_kgm3=ref,
                          mean=float(np.nanmean(ratio)), sd=float(np.nanstd(ratio)),
                          p05=float(np.nanpercentile(ratio,5)), p95=float(np.nanpercentile(ratio,95)),
                          min=float(np.nanmin(ratio)), max=float(np.nanmax(ratio)),
                          wind_speed_shift_pct=float((np.nanpercentile(ratio,95)-np.nanpercentile(ratio,5))*100/2))
if len(u) and len(v):
    U = pd.to_numeric(df[u[0]], errors='coerce'); V = pd.to_numeric(df[v[0]], errors='coerce')
    ws = np.hypot(U, V)
    out['wind'] = dict(n=int(np.isfinite(ws).sum()), mean=float(np.nanmean(ws)),
                       max=float(np.nanmax(ws)),
                       frac_above_cutout_22=float((ws > 22).mean()),
                       frac_above_25=float((ws > 25).mean()),
                       frac_below_cutin_3=float((ws < 3).mean()))
OUT = Path(r'results/paper/source/physical_magnitudes.json')
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
