"""Claim: display every model's frozen TI-stratum error, retaining blocked VF gaps.
Quantitative 2x3 grid; 183x116 mm; baseline mean +/- sample SD (3 seeds),
VF individual passed seeds only. Source k1_ti_strata_errors.csv; no tests.
"""
from figure_style import *
def main():
    t=begin('python -B scripts/figure_ti_errors.py');d=pd.read_csv(O/'k1_ti_strata_errors.csv');fig,axes=plt.subplots(2,3,figsize=(183/25.4,116/25.4),sharex=True,layout='constrained');x=np.arange(5)
    for col,h in enumerate(HORIZONS):
        for row,metric in enumerate(['nrmse','nmae']):
            ax=axes[row,col]
            for mi,model in enumerate(COLORS):
                q=d[(d.row_type=='seed_summary')&(d.model==model)&(d.horizon==h)].set_index('stratum').reindex(TI_LAYERS)
                ax.errorbar(x+(mi-3)*.035,q[metric+'_pct_mean'],yerr=q[metric+'_pct_sd'],color=COLORS[model],linewidth=.85,marker='.',markersize=3,capsize=1.5,label=model)
            ax.set_title({1:'15 min',4:'1 h',16:'4 h'}[h]);ax.set_xticks(x,[LABELS[z] for z in TI_LAYERS],rotation=30,ha='right');ax.set_ylabel(metric.upper()+' (%)');panel(ax,chr(97+row*3+col));ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Target-time nacelle turbulence intensity | all models: 3 seeds\nMean +/- sample SD of 3 seeds; *very high is a subset of high',fontsize=8)
    save(fig,'k1_ti_errors');end(t,'figure_ti_errors')
if __name__=='__main__':main()
