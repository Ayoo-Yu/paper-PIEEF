#!/usr/bin/env python3
"""Recompute a CPU subset from one extracted frozen PI-EEF release run."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pieef.data import build_data_bundle
from pieef.model_factory import build_main_model
from pieef.train_main import evaluate_multistep

def public_path(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Extracted asset directory containing weights_best.pt')
    parser.add_argument('--windows',type=int,default=208)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.windows < 1:raise ValueError('windows must be positive')
    torch.set_num_threads(args.threads)
    run=args.run_dir
    record=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    cfg=record['config'];data=cfg['data']
    bundle=build_data_bundle(public_path(data['scada_path']),public_path(data['turbines_path']),
        int(data['seq_len']),int(record['horizon']),16,float(data['split'][0]),float(data['split'][1]),
        data.get('capacity_mw',2.),revision_protocol=data['revision_protocol'])
    with np.load(run/'preprocessing.npz',allow_pickle=False) as stored:
        for key,attribute in [('nwp_mean','nwp_mean'),('nwp_std','nwp_std'),
                              ('scada_wind_mean','scada_w_mean'),('scada_wind_std','scada_w_std')]:
            if not np.array_equal(stored[key],getattr(bundle.handler,attribute)):
                raise ValueError(f'Frozen preprocessing differs: {key}')
    options=copy.deepcopy(cfg['model'])
    if options.get('manufacturer_curve_path'):
        options['manufacturer_curve_path']=str(public_path(options['manufacturer_curve_path']))
    model=build_main_model(bundle,int(data['seq_len']),options,torch.device('cpu'))
    model.load_state_dict(torch.load(run/'weights_best.pt',map_location='cpu',weights_only=True),strict=True)
    count=min(args.windows,len(bundle.test_loader.dataset))
    loader=DataLoader(Subset(bundle.test_loader.dataset,range(count)),batch_size=16,shuffle=False)
    _,truth,prediction=evaluate_multistep(model,loader,torch.device('cpu'),bundle.capacity_mw)
    with np.load(run/'predictions.npz',allow_pickle=False) as stored:
        difference=np.abs((prediction-stored['y_pred_mw'][:count])/np.asarray(bundle.capacity_mw))
        targets_equal=np.array_equal(truth,stored['y_true_mw'][:count],equal_nan=True)
        timestamps_equal=np.array_equal(bundle.handler.metadata['test']['timestamp'][:count],stored['timestamp'][:count])
    maximum=float(difference.max())
    passed=maximum<=1e-4 and targets_equal and timestamps_equal
    print(json.dumps(dict(passed=passed,device='cpu',windows=count,max_abs_difference_pu=maximum,
        targets_equal=targets_equal,timestamps_equal=timestamps_equal),indent=2))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
