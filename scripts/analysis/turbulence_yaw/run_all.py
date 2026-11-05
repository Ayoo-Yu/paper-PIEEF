"""Reproduce K0-K6 and all requested artifacts without training."""
from common import *
import subprocess
def main():
    t=begin('python -B scripts/run_all.py')
    import v2_audit
    v2_audit.snapshot()
    names=['run_analysis.py','figure_ti_errors.py','figure_ti_attention.py','figure_yaw_power_ratio.py','figure_ti_power_curves.py']
    for name in names:subprocess.run([sys.executable,'-B',str(O/'scripts'/name)],check=True,cwd=W)
    subprocess.run([sys.executable,'-B',str(O/'scripts/finalize.py'),'--prepare'],check=True,cwd=W)
    subprocess.run([sys.executable,'-B',str(O/'scripts/build_report.py')],check=True,cwd=W)
    end(t,'run_all analysis and report; final manifest follows')
    subprocess.run([sys.executable,'-B',str(O/'scripts/finalize.py'),'--manifest'],check=True,cwd=W)
if __name__=='__main__':main()
