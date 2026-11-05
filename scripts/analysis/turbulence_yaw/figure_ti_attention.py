"""Claim: show the actual entropy, maximum weight and DSI in each TI stratum.
Quantitative 1x3 grid; 183x69 mm; individual passed seeds, no substituted means.
Source k3_attention_by_ti.csv; each point averages finite query-turbine cells.
"""
from figure_style import *
def main():
    t=begin('python -B scripts/figure_ti_attention.py');d=pd.read_csv(O/'k3_attention_by_ti.csv');d=d[(d.row_type=='seed')&(d.site=='Kelmarsh')&(d.status=='PASS')]
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,69/25.4),layout='constrained');x=np.arange(5)
    for col,(metric,label) in enumerate([('normalized_entropy','Normalized entropy'),('maximum_weight','Maximum weight'),('dsi','DSI')]):
        ax=axes[col]
        for hi,h in enumerate(HORIZONS):
            q=d[(d.horizon==h)&(d.metric==metric)]
            for n,(s,g) in enumerate(q.groupby('seed',sort=False)):
                v=g.set_index('stratum').reindex(TI_LAYERS).attention_mean
                ax.plot(x+(hi-1)*.12+(n-1)*.035,v,marker=['o','s','^'][n],markersize=3,linewidth=.7,alpha=.85,color=['#0072B2','#009E73','#B5432F'][hi],label=f'{dict(zip(HORIZONS,["15 min","1 h","4 h"]))[h]}, s{int(s)}')
        ax.set_xticks(x,[LABELS[z] for z in TI_LAYERS],rotation=30,ha='right');ax.set_ylabel(label);panel(ax,chr(97+col));ax.grid(axis='y',alpha=.15)
        if metric=='dsi':ax.axhline(.5,color='#888888',linestyle=':',linewidth=.7)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('VF attention by target-time nacelle TI | individual passed seeds\nAll 3 seeds per horizon; *very high is a subset of high',fontsize=8)
    save(fig,'k3_ti_attention');end(t,'figure_ti_attention')
if __name__=='__main__':main()
