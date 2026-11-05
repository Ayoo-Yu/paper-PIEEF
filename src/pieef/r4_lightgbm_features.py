"""R4 F1: preserve original tabular features and append three shared CFD values."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

def use_stored_wind_farm_A_preprocessing(bundle,data_path,preprocessing_path):
    """Apply the published control's training-only statistics to native raw fields.

    The original weather fields and derived feature formulas are unchanged.
    Windows/Linux float32 trig implementations can still differ by ulps; that
    observed prediction drift is recorded separately, never relabelled exact.
    """
    h=bundle.handler
    if h.protocol.get('nwp_timing_rule')!='author_prepared_ecmwf':
        raise ValueError('F1 is the declared wind_farm_A prepared-weather experiment only')
    frame=pd.read_csv(data_path)
    columns=list(h.feature_columns[:-3]);values=frame[columns].to_numpy(dtype=np.float32)
    u,v=values[:,int(h.nwp_u_index)],values[:,int(h.nwp_v_index)]
    angle=np.arctan2(u,v)
    weather=np.column_stack([values,np.hypot(u,v),np.sin(angle),np.cos(angle)])
    with np.load(preprocessing_path,allow_pickle=False) as z:
        if not np.array_equal(z['turbine_ids'],h.turbine_ids):raise ValueError('Published control turbine order differs')
        if not np.array_equal(z['scada_wind_mean'],h.scada_w_mean) or not np.array_equal(z['scada_wind_std'],h.scada_w_std):
            raise ValueError('Published causal SCADA wind normalization differs')
        mean,std=z['nwp_mean'].copy(),z['nwp_std'].copy()
    if mean.shape!=(bundle.nwp_dim,) or not np.isfinite(std).all() or not (std>0).all():raise ValueError('Invalid frozen preprocessing')
    normalized=(weather-mean)/std
    if not np.isfinite(normalized).all():raise ValueError('Nonfinite weather under original frozen normalization')
    shared=torch.as_tensor(normalized,dtype=torch.float32)
    for split in ('train','val','test'):
        ds=getattr(bundle,split+'_loader').dataset
        if len(ds.nwp)!=len(frame) or ds.nwp.shape[-1]!=bundle.nwp_dim:raise ValueError('Original weather layout differs')
        ds.nwp=shared;ds.future_nwp=shared
    h.nwp_mean,h.nwp_std=mean,std
    train_end=int(h.power_training_end_index);val_end=train_end+len(h.val_n_norm)
    h.train_n_norm,h.val_n_norm,h.test_n_norm=normalized[:train_end],normalized[train_end:val_end],normalized[val_end:]
    return {'kind':'published_control_train_fitted_statistics','preprocessing_path':str(preprocessing_path),
            'preprocessing_sha256':hashlib.sha256(Path(preprocessing_path).read_bytes()).hexdigest(),
            'raw_weather_formula':'original columns + hypot(u,v), sin(arctan2(u,v)), cos(arctan2(u,v))',
            'raw_data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            'no_target_split_or_original_feature_added_removed':True}

class AppendedCFDWindows(Dataset):
    def __init__(self,base,extra):
        self.base=base;self.extra=torch.as_tensor(extra,dtype=torch.float32)
        if self.extra.shape!=(len(base),base.power.shape[1],3):raise ValueError('Exactly3 target CFD features required')
        if not torch.isfinite(self.extra).all():raise ValueError('Nonfinite appended CFD features')
    def __len__(self):return len(self.base)
    def __getattr__(self,name):return getattr(self.base,name)
    def __getitem__(self,item):
        p,w,n,target,y=self.base[item]
        return p,w,n,torch.cat([target,self.extra[item]],dim=-1),y

def append_shared_cfd_features(bundle,data_cfg,model_cfg):
    from .cfd_features import compute_local_target_features
    features=compute_local_target_features(bundle,data_cfg,model_cfg)
    features['provenance']['normalization']='published LightGBM control training-only statistics retained; original base features unchanged'
    for split in ('train','val','test'):
        loader=getattr(bundle,split+'_loader');f=features[split]
        if not np.array_equal(f['indices'],loader.dataset.indices):raise ValueError('CFD features target indices differ')
        extra=np.stack([f['flow_sin'],f['flow_cos'],f['raw_corrected_speed']],axis=-1)
        ds=AppendedCFDWindows(loader.dataset,extra)
        # Reuse the exact original sampler: no new shuffle/RNG construction.
        wrapped=DataLoader(ds,batch_size=loader.batch_size,sampler=loader.sampler,num_workers=0,pin_memory=False,drop_last=False)
        setattr(bundle,split+'_loader',wrapped)
    bundle.handler.r4_fairness_feature_columns=['target_local_flow_sin','target_local_flow_cos','target_cfd_corrected_speed_mps']
    return features['provenance']
