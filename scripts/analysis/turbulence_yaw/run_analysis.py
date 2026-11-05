from common import *
import argparse, traceback
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-only',action='store_true');args=parser.parse_args()
    checkpoint();t=begin(' '.join([sys.executable,'-B',*sys.argv]))
    try:
        import k0_gates,raw_data
        k,runs,cap=k0_gates.run();d=raw_data.run(k)
        if not args.raw_only:
            import ti_analysis,yaw_analysis,learned_curves
            ti_analysis.run(k,runs,cap,d)
            yaw_analysis.run(k,d)
            learned_curves.run(k,runs,cap,d)
        save_k0(k);js('operational_incidents.json',INCIDENTS)
        end(t,'run_analysis')
    except Exception as exc:
        incident('execution_'+utc(),traceback.format_exc(),'保留完整报错，修复脚本后重新运行；不改已有输入。','execution_failure')
        end(t,'run_analysis','failed');raise
if __name__=='__main__':main()
