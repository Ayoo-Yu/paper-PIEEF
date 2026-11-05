"""Portable redraws of the frozen, published figure source values.

No training, model inference, record sampling, or new hypothesis tests occur.
Paper exports are retained in figures/paper. Redraws use figures/reproduced.
The scientific observations and stored statistics are unchanged; layout can differ.
"""
from pathlib import Path
import os,sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from PIL import Image
SRC=ROOT/'results/paper/source'
PAPER=ROOT/'figures/paper'
OUT=ROOT/'figures/reproduced'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8,'axes.labelsize':9,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in','svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'savefig.facecolor':'white'})
BLUE='#2166AC';RED='#B2182B';GREY='#BABABA'
PALETTE=[BLUE,RED,GREY,'#4393C3','#D6604D','#4D4D4D']
FILES={'fig_8':'fig7_benchmark_heatmap_data.csv','fig_9':'fig8_robustness_components_data.csv','fig_10':'fig09_physics_boxplots_plotted_values.csv','fig_11':'fig10_gradient_densities_plotted_values.csv','fig_12':'faults_values.csv','fig_13':'Fig12_events_black_frame_plotted_values.csv','fig_14':'fig9_physical_curves_v3_data.csv','fig_15':'fig15_figS1_node_data.csv','fig_s1':'fig15_figS1_node_data.csv','fig_s2':'figS_ti_power_curves_data.csv','fig_s3':'figS_ti_errors_data.csv','fig_s4':'figS_ti_attention_data.csv','fig_s5':'figS_yaw_power_data.csv','fig_s6':'figS_cfd_lookup_data.csv','fig_s7':'figS_cfd_checks_data.csv'}
TITLES={'fig_8':'Benchmark errors at Wind Farm A','fig_9':'Robustness component results','fig_10':'Physical ablations by operating regime','fig_11':'Gradient alignment','fig_12':'Structured SCADA faults','fig_13':'Candidate event classes','fig_14':'Physical power curves','fig_s2':'TI-conditioned power curves','fig_s3':'Errors by turbulence intensity','fig_s4':'Attention by turbulence intensity','fig_s5':'Yaw misalignment and power','fig_s6':'CFD lookup values','fig_s7':'CFD quality checks'}

def read(name):return pd.read_csv(SRC/name,encoding='utf-8',float_precision='round_trip')
def finish(fig,item,paper=False):
 out=PAPER if paper else OUT;out.mkdir(parents=True,exist_ok=True)
 fig.savefig(out/f'{item}.png',dpi=600,metadata={})
 fig.savefig(out/f'{item}.svg',metadata={'Creator':'PI-EEF public release'})
 fig.savefig(out/f'{item}.pdf',metadata={'Creator':'PI-EEF public release'})
 plt.close(fig)
 # Re-encode raster without metadata.
 with Image.open(out/f'{item}.png') as im:im.save(out/f'{item}.png')
 return [str((out/f'{item}.{ext}').relative_to(ROOT)) for ext in ['png','svg','pdf']]

def fig6(paper=False):
 """Layout and all-record power density; elevation differs by a constant offset."""
 t=read('fig06_plotted_values.csv');b=read('fig06_power_binned_counts.csv')
 assert len(t)==29 and list(t.turbine_id)==[f'T{i:02}' for i in range(1,30)]
 assert min(t.x_coord_m)==min(t.y_coord_m)==min(t.relative_elevation_m)==0
 assert not any('latitude' in c or 'longitude' in c for c in t.columns)
 fig=plt.figure(figsize=(7,5.25));a=fig.add_axes([.095,.596,.79,.359]);p=fig.add_axes([.095,.09,.79,.394])
 cmap=LinearSegmentedColormap.from_list('terrain_relative',['#F7FBFF','#D1E5F0','#92C5DE','#4393C3','#2166AC','#053061'])
 art=a.scatter(t.x_coord_m/1000,t.y_coord_m/1000,c=t.relative_elevation_m,cmap=cmap,marker='^',s=40,edgecolors='#222222',linewidths=.55)
 a.set(xlim=(-.3,9.45),ylim=(-.25,3),xlabel='East from minimum (km)',ylabel='North from minimum (km)',aspect='equal');a.set_title('(a) Turbine layout',loc='left',fontweight='bold',fontsize=10);a.grid(True,linestyle=':',color='#E0E0E0')
 cb=fig.colorbar(art,cax=fig.add_axes([.915,.62,.013,.31]));cb.set_label('Relative elevation (m)')
 count=b.pivot(index='power_bin_index',columns='wind_bin_index',values='records').to_numpy()
 xe=np.r_[b.groupby('wind_bin_index',sort=True).wind_left_m_s.first(),b.wind_right_m_s.max()];ye=np.r_[b.groupby('power_bin_index',sort=True).power_bottom_mw.first(),b.power_top_mw.max()]
 assert int(count.sum())==int(b.records.sum()) and count.max()>0
 h=p.pcolormesh(xe,ye,np.ma.masked_equal(count,0),norm=LogNorm(vmin=1,vmax=count.max()),cmap=cmap,shading='flat',rasterized=False)
 p.set(xlim=(xe[0],xe[-1]),ylim=(ye[0],ye[-1]),xlabel='Mean wind speed (m/s)',ylabel='Farm total power (MW)');p.set_title('(b) Measured wind speed and farm power',loc='left',fontweight='bold',fontsize=10)
 cb=fig.colorbar(h,cax=fig.add_axes([.915,.115,.013,.34]));cb.set_label('Records')
 return finish(fig,'fig_6',paper)

def timeline():
 fig,axes=plt.subplots(2,1,figsize=(7,3.8));fig.subplots_adjust(hspace=.9,left=.1,right=.97,top=.88,bottom=.16)
 ax=axes[0];ax.hlines(1,0,33,color=BLUE,lw=5);ax.plot([0],[1],'o',color=RED);ax.set(xlim=(-1,36),ylim=(0,2),xticks=[0,3,9,15,21,27,33],yticks=[],xlabel='Hours from daily 00 UTC initialisation');ax.set_title('(a) Kelmarsh GFS: saved leads 3–33 h',loc='left',fontsize=9);ax.text(3,1.35,'Use the latest run available at the forecast origin',fontsize=8);ax.text(3,.35,'Availability: original S3 LastModified',fontsize=8)
 ax=axes[1];ax.hlines(1,0,24,color=GREY,lw=6);ax.set(xlim=(-1,25),ylim=(0,2),xticks=[],yticks=[]);ax.set_title('(b) Wind Farms A and B: ECMWF fields aligned to valid time',loc='left',fontsize=9);ax.text(0,1.45,'The files do not record issue or availability times.',fontsize=8);ax.text(0,.35,'Operational schedule: manuscript Section 3.1 and Supplementary Section S3.4.',fontsize=8)
 return finish(fig,'fig_7')

def attention(item):
 d=read(FILES[item]);strata=list(dict.fromkeys(d.stratum));strata=strata[:1]+strata[1:] if item=='fig_15' else strata[1:]
 fig,axes=plt.subplots(len(strata),1,figsize=(7,2.25*len(strata)),squeeze=False);fig.subplots_adjust(hspace=.5,right=.85)
 for i,(s,ax) in enumerate(zip(strata,axes[:,0])):
  z=d[d.stratum==s];diff=item=='fig_15' and i>0;c=z.change_from_typical_pp if diff else z.received_share_percent
  art=ax.scatter(z.x_km,z.y_km,c=c,cmap='RdBu_r' if diff else 'Reds',vmin=-1.3 if diff else 0,vmax=1.3 if diff else 10,s=30,edgecolors='#222222',linewidth=.5)
  ax.set(xlabel='East from minimum (km)',ylabel='North from minimum (km)',aspect='equal',title=s.replace('_',' '));fig.colorbar(art,ax=ax,label='Change (percentage points)' if diff else 'Received share (%)')
 return finish(fig,item)

def source_redraw(item):
 d=read(FILES[item]);panels=list(dict.fromkeys(d.panel.dropna())) if 'panel' in d else ['all']
 fig,axes=plt.subplots(len(panels),1,figsize=(7,max(3,2.7*len(panels))),squeeze=False);fig.subplots_adjust(hspace=.65,bottom=.12,top=.94,left=.12,right=.94)
 fig.suptitle(TITLES.get(item,item),fontsize=10)
 for ax,panel in zip(axes[:,0],panels):
  z=d[d.panel==panel].copy() if 'panel' in d else d.copy();ax.set_title(f'({panel})',loc='left',fontsize=9)
  if item=='fig_8':
   h=z.pivot(index='model',columns='horizon',values='mean');art=ax.imshow(h.to_numpy(),aspect='auto',cmap='Blues');ax.set_yticks(range(len(h)),h.index,fontsize=6);ax.set_xticks(range(len(h.columns)),h.columns);fig.colorbar(art,ax=ax,label=str(z.metric.iloc[0])+' (%)')
  elif item=='fig_10':
   zz=z[z.record_type=='seed_point'];groups=list(zz.groupby(['variant','regime','bin_label'],sort=False));positions=np.arange(len(groups));values=[g.value.to_numpy() for _,g in groups]
   ax.boxplot(values,positions=positions,widths=.55,patch_artist=True,boxprops={'facecolor':'#D1E5F0'});ax.set_xticks(positions,[f'{k[0]}\n{k[2]}' for k,_ in groups],rotation=40,ha='right',fontsize=5);ax.set_ylabel('Error increase (percentage points)');ax.axhline(0,color=GREY,lw=.7)
  elif item=='fig_11':
   zz=z[z.record_type.astype(str).str.contains('density')]
   if len(zz) and zz.x_cosine.notna().any():ax.plot(zz.x_cosine,zz.value,color=BLUE)
   else:ax.hist(z[z.record_type=='input_probe'].value,bins=25,density=True,color=BLUE,alpha=.8)
   ax.set(xlabel='Gradient cosine',ylabel='Density')
  elif item=='fig_14':
   for j,(key,g) in enumerate(z.groupby('curve',sort=False)):ax.plot(g.wind_ms,g.plotted_value,color=PALETTE[j%len(PALETTE)],lw=.8,label=str(key))
   ax.set(xlabel='Wind speed (m/s)',ylabel='Power (p.u.)');ax.legend(fontsize=5,ncol=5)
  elif item in ['fig_s2','fig_s5']:
   x='wind_center_ms' if item=='fig_s2' else 'abs_yaw_center_deg';group='ti_class' if item=='fig_s2' else 'reference_speed'
   for j,(key,g) in enumerate(z.groupby(group,sort=False)):
    g=g[g[x].notna() & g.value.notna()];ax.plot(g[x],g.value,'.-',ms=3,color=PALETTE[j%len(PALETTE)],label=str(key))
   ax.set_xlabel('Wind speed (m/s)' if item=='fig_s2' else 'Absolute yaw (degrees)');ax.set_ylabel('Power (MW)' if item=='fig_s2' else 'Power deficit');ax.legend(fontsize=6)
  elif item=='fig_s6':
   zz=z[z.record_type=='plotted_cell'];h=zz.pivot_table(index='turbine',columns='inflow_direction_deg',values='value',aggfunc='first');art=ax.imshow(h.to_numpy(),aspect='auto',cmap='RdBu_r');ax.set_yticks(range(len(h)),h.index,fontsize=5);ax.set_xticks(range(len(h.columns)),h.columns,rotation=45,fontsize=5);ax.set_xlabel('Inflow direction (degrees)');fig.colorbar(art,ax=ax,label=str(z.metric.iloc[0]))
  elif item=='fig_s7':
   zz=z[z.record_type=='plotted_histogram_bin']
   if len(zz):
    for j,(key,g) in enumerate(zz.groupby('group',sort=False)):ax.bar((g.bin_left+g.bin_right)/2,g.value,width=g.bin_right-g.bin_left,alpha=.5,label=str(key),color=PALETTE[j%len(PALETTE)])
    ax.legend(fontsize=6);ax.set(xlabel=str(z.metric.iloc[0]),ylabel='Count')
   else:ax.plot(np.arange(len(z)),z.value,'.',color=BLUE);ax.set_ylabel(str(z.metric.iloc[0]))
  elif item in ['fig_s3','fig_s4']:
   groups=['model','horizon'] if item=='fig_s3' else ['metric','horizon']
   for j,(key,g) in enumerate(z.groupby(groups,sort=False)):
    x=np.arange(len(g));ax.errorbar(x+j*.04,g.value,yerr=g.sd,fmt='.-',ms=3,lw=.7,label='/'.join(map(str,key)),color=PALETTE[j%len(PALETTE)])
   labs=list(dict.fromkeys(z.stratum));ax.set_xticks(range(len(labs)),labs,fontsize=6);ax.legend(fontsize=5,ncol=3);ax.set_ylabel('Error (%)' if item=='fig_s3' else 'Diagnostic value')
  else:
   y='value' if 'value' in z else 'mean';z=z[z[y].notna()]
   # Every stored plotted statistic is represented; long frozen value vectors
   # remain ordered by their public source rows rather than being sampled.
   x=np.arange(len(z));ax.plot(x,z[y],'.',ms=2,color=BLUE);ax.set(xlabel='Published source row',ylabel=str(z.metric.iloc[0]) if 'metric' in z and len(z) else y)
  ax.grid(axis='y',color='#E0E0E0',lw=.4)
 return finish(fig,item)

def render(item,paper=False):
 if item=='fig_6':return fig6(paper)
 if item=='fig_7':return timeline()
 if item in ['fig_15','fig_s1']:return attention(item)
 return source_redraw(item)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--item',required=True);p.add_argument('--paper',action='store_true');a=p.parse_args();print(json.dumps(render(a.item,a.paper)))
