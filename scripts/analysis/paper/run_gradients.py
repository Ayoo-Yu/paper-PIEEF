from run_physics import *
import math,itertools
from types import SimpleNamespace
from torch.utils.data._utils.collate import default_collate
from pieef.train_main import _configure_stage3,_stage3_features,_forecast_loss
from pieef.forecast_corruption import ForecastCorruption
import torch.nn.functional as F

def run():
 started=time.perf_counter();allrows=[];lossrows=[];audit=[];bundle=None
 for r in sorted([r for r in records() if r['horizon']==16],key=lambda x:x['seed']):
  seed=r['seed'];gatepath=OUT/f'normal_equivalence_s{seed}.json'
  waitstart=time.perf_counter()
  while not gatepath.exists():
   assert time.perf_counter()-waitstart<3600,'normal gate unavailable after one hour'
   if 'Traceback (most recent call last)' in (OUT/'P14_run.log').read_text(encoding='utf-16',errors='ignore'):raise RuntimeError('P14 failed')
   time.sleep(5)
  assert json.loads(gatepath.read_text())['status']=='PASS'
  rec,meta=record(r)
  if bundle is None:bundle=ev._bundle(rec)
  model,_,_=ev._predictor(rec,bundle,torch.device('cpu'))
  before={k:v.detach().clone() for k,v in model.state_dict().items()}
  tc=SimpleNamespace(**rec['config']['train']);_configure_stage3(model,tc.stage3_mode,tc.stage3_decoder_policy,tc.reconstruction_weight);model.train()
  hist=json.loads((Path(r['path'])/'loss_history.json').read_text());epoch=hist['best_epoch']
  best=next(x for x in hist['stage3'] if x['epoch']==epoch)
  assert np.isclose(best['val_nrmse'],meta['validation_metrics']['nrmse'],atol=1e-7)
  decay=math.exp(-tc.auxiliary_decay*(epoch-1));lb=model.loss_fn_pi.lambda_bound;ld=model.loss_fn_pi.lambda_deriv
  factors=dict(forecast=1.,reconstruction=decay*tc.reconstruction_weight,boundary=decay*tc.reconstruction_weight*lb,temporal_physics=decay*tc.reconstruction_weight*ld,calibration_fit=decay*tc.calibration_weight,delta_regularizer=decay*tc.calibration_weight*.05)
  params=[(n,p) for n,p in model.named_parameters() if p.requires_grad and (n.startswith(('pi_embedding.','mae_encoder.','dcn.','graph.','revin.')) or n=='position_embedding')]
  groups={'encoder':[i for i,(n,p) in enumerate(params) if n.startswith(('pi_embedding.','mae_encoder.')) or n=='position_embedding'],'TD_DCN':[i for i,(n,p) in enumerate(params) if n.startswith('dcn.')],'graph':[i for i,(n,p) in enumerate(params) if n.startswith('graph.')],'all_shared':list(range(len(params)))}
  def gradients(loss,retain=True):
   gs=torch.autograd.grad(loss,[p for n,p in params],allow_unused=True,retain_graph=retain) if loss.requires_grad else [None]*len(params)
   return {g:torch.cat([(torch.zeros_like(params[i][1]) if gs[i] is None else gs[i]).detach().reshape(-1) for i in idx]).double() for g,idx in groups.items()}
  # A common random permutation of training examples: exactly 50 full batches.
  chosen=np.random.default_rng(2026).permutation(len(bundle.train_loader.dataset))[:50*tc.batch_size].reshape(50,tc.batch_size)
  js('P5_batch_indices.json',dict(seed=2026,batch_size=tc.batch_size,indices=chosen,target_row_indices=bundle.train_loader.dataset.indices[chosen]))
  corruption=ForecastCorruption('f2_block',bundle.handler.coords,bundle.handler.scada_w_std,2026,bundle.num_turbines,rec['config']['data']['seq_len'])
  torch.manual_seed(2026)
  captured={}
  def hook(m,args,kw,result):captured.update(args=args,kw=kw,result=result)
  handle=model.loss_fn_pi.register_forward_hook(hook,with_kwargs=True)
  for bi,idx in enumerate(chosen):
   bt=time.perf_counter();batch=default_collate([bundle.train_loader.dataset[int(i)] for i in idx]);original=tuple(x.clone() for x in batch)
   corrupt,pm,wm=corruption.corrupt(batch);p,w,nh,nt,y=corrupt
   assert all(torch.allclose(a,b,atol=0,rtol=0,equal_nan=True) for a,b in zip(batch,original))
   feat=_stage3_features(model,p,w,nh,'joint',force_power_mask=pm,force_wind_mask=wm)
   v,phys=model.forward_stage2_calibration(w,nt);pred=model.forward_stage3_forecast(feat,v,phys,nt,hist_power=p,hist_power_mask=pm)
   fl=_forecast_loss(pred,y,tc);grads={'forecast':gradients(fl,False)};values={'forecast':float(fl.detach())}
   del feat,v,phys,pred,fl
   recon,_,_,_=model.forward_stage1_reconstruct(*batch[:3])
   a=captured['args'];kw=captured['kw'];predr,target,mask,prior=a;physical=kw['physical_pred'];pv=kw['prior_valid_mask']
   lrec=((predr-target).square()*mask).sum()/(mask.sum()+1e-6)
   lbound=(F.relu(-physical)+F.relu(physical-1.05)).mean()
   dm=mask[:,:,1:]*pv[:,:,1:]*pv[:,:,:-1];ldiff=(((physical[:,:,1:]-physical[:,:,:-1])-(prior[:,:,1:]-prior[:,:,:-1])).square()*dm).sum()/(dm.sum()+1e-6)
   assert torch.allclose(recon,lrec+lb*lbound+ld*ldiff,atol=1e-7,rtol=1e-6)
   for term,loss in [('reconstruction',lrec),('boundary',lbound),('temporal_physics',ldiff)]:grads[term]=gradients(loss);values[term]=float(loss.detach())
   _,ap,dv=model.forward_stage2_training(nt);valid=torch.isfinite(y)&(y>.01)
   fit=F.mse_loss(ap.squeeze(-1)[valid],y[valid]) if valid.any() else dv.sum()*0
   reg=dv.square().mean()
   for term,loss in [('calibration_fit',fit),('delta_regularizer',reg)]:grads[term]=gradients(loss);values[term]=float(loss.detach())
   original_total=values['forecast']+decay*(tc.reconstruction_weight*float(recon.detach())+tc.calibration_weight*(float(fit.detach())+.05*float(reg.detach())))
   assert np.isclose(sum(factors[k]*values[k] for k in values),original_total,rtol=1e-6)
   for term in values:lossrows.append(dict(seed=seed,batch=bi,term=term,raw_loss=values[term],factor=factors[term],weighted_loss=values[term]*factors[term],best_epoch=epoch))
   for group in groups:
    weighted={k:grads[k][group]*factors[k] for k in grads}
    weighted['reconstruction_total']=weighted['reconstruction']+weighted['boundary']+weighted['temporal_physics']
    weighted['calibration_total']=weighted['calibration_fit']+weighted['delta_regularizer']
    weighted['auxiliary_total']=weighted['reconstruction_total']+weighted['calibration_total']
    for term,g in weighted.items():
     allrows.append(dict(seed=seed,batch=bi,group=group,kind='norm',term_a=term,term_b='',weighted_norm=float(g.norm()),raw_norm=float(grads[term][group].norm()) if term in grads else np.nan))
    for (a,ga),(b,gb) in itertools.combinations(weighted.items(),2):
     na=float(ga.norm());nb=float(gb.norm());den=na*nb
     allrows.append(dict(seed=seed,batch=bi,group=group,kind='pair',term_a=a,term_b=b,cosine=float(ga.dot(gb)/den) if den else np.nan,norm_b_over_a=nb/na if na else np.nan,zero_a=na==0,zero_b=nb==0))
   audit.append(dict(seed=seed,batch=bi,best_epoch=epoch,seconds=time.perf_counter()-bt,corrupted=pm is not None,scenarios=corruption.last_scenarios,dropout_aux_train=model.training))
   captured.clear();del recon,lrec,lbound,ldiff,grads,weighted,a,kw,predr,target,mask,prior,physical,pv,ap,dv,fit,reg,loss
   if (bi+1)%5==0:print(f'P5 seed {seed} batch {bi+1}/50 seconds {time.perf_counter()-started:.1f}',flush=True)
  handle.remove()
  assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),'weights changed'
  csv(f'P5_gradients_s{seed}.csv',[x for x in allrows if x['seed']==seed]);csv(f'P5_losses_s{seed}.csv',[x for x in lossrows if x['seed']==seed]);js(f'P5_audit_s{seed}.json',dict(status='completed',weights_unchanged=True,optimizer_steps=0,best_epoch=epoch,decay=decay,groups={g:[params[i][0] for i in idx] for g,idx in groups.items()},corruption_counts=corruption.counts,batches=[x for x in audit if x['seed']==seed]))
  progress(f'P5：seed {seed} 的 50 个训练 batch 梯度计算完成；权重逐张量未变，优化步 0。')
 csv('P5_gradients.csv',allrows);csv('P5_losses.csv',lossrows);js('P5_completed.json',dict(seconds=time.perf_counter()-started,batches=250,optimizer_steps=0))
if __name__=='__main__':run()
