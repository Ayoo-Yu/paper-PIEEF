"""Claim: show training empirical power curves in every frozen TI layer.
Quantitative grid, pooled plus all six turbines, 183x124 mm. No smoothing.
Source k2_ti_power_curves.csv; normal operation; counts remain in source CSV.
"""
from figure_style import *
def main():
    t=begin('python -B scripts/figure_ti_power_curves.py');d=pd.read_csv(O/'k2_ti_power_curves.csv');fig,axes=plt.subplots(2,4,figsize=(183/25.4,124/25.4),sharex=True,sharey=True,layout='constrained')
    for j,tid in enumerate(['pooled','1','2','3','4','5','6']):
        ax=axes.flat[j]
        for layer,color in zip(TI_LAYERS,['#0072B2','#009E73','#B5432F','#7B6DAA','#777777']):
            g=d[(d.turbine.astype(str)==tid)&(d.stratum==layer)]
            ax.plot(g.wind_center_ms,g.mean_power_mw,marker='.',linewidth=1,color=color,linestyle='--' if layer=='very_high' else '-',label=LABELS[layer])
        ax.set_title('Pooled' if tid=='pooled' else 'Turbine '+tid);ax.set_xlabel('Nacelle wind speed (m/s)');ax.set_ylabel('Mean power (MW)');panel(ax,chr(97+j));ax.grid(alpha=.15)
    axes.flat[-1].axis('off');h,l=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(h,l,loc='center')
    fig.suptitle('Training empirical power curves by nacelle TI | original normal-event filter\n1 m/s bins; no smoothing; *very high is a subset of high',fontsize=8)
    save(fig,'k2_ti_power_curves');end(t,'figure_ti_power_curves')
if __name__=='__main__':main()
