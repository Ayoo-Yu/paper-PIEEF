"""Claim: contrast yaw-power ratios and cos^p fits under both reference speeds.
Quantitative 1x2 grid; 183x76 mm; all 2-degree bins, day-block percentile 95% CI.
Source k4_yaw_power_ratio.csv and k4_cos_fit.json; no significance tests.
"""
from figure_style import *
def main():
    t=begin('python -B scripts/figure_yaw_power_ratio.py');d=pd.read_csv(O/'k4_yaw_power_ratio.csv');info=json.loads((O/'k4_cos_fit.json').read_text(encoding='utf8'))
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,76/25.4),layout='constrained')
    if info['status']=='completed':
        for j,f in enumerate(info['fits']):
            ax=axes[j];g=d[(d.reference_speed==f['reference_speed'])&(d.bin_type=='two_degree')]
            ax.errorbar(g.abs_phi_center_deg,g.mean_power_ratio,yerr=np.vstack([g.mean_power_ratio-g.ci95_lower,g.ci95_upper-g.mean_power_ratio]),fmt='o',color='#0072B2',markersize=3,capsize=2,linewidth=.8,label='Mean; day-block 95% CI')
            sparse=g.n_records<200;ax.scatter(g.loc[sparse,'abs_phi_center_deg'],g.loc[sparse,'mean_power_ratio'],s=14,facecolors='white',edgecolors='#0072B2',zorder=6,label='n < 200 (not fitted)')
            x=np.linspace(0,30,301);ax.plot(x,np.cos(np.deg2rad(x))**f['p'],color='#B5432F',linewidth=1.3,label='Least-squares cos^p fit')
            ax.axhline(1,color='#777777',linestyle=':',linewidth=.7);ax.set_xlabel('Absolute yaw misalignment (degrees)');ax.set_ylabel('Power / aligned reference power');ax.set_title(('Own nacelle speed' if j==0 else 'Median speed of other five turbines')+f'\np = {f["p"]:.2f} [{f["ci95_lower"]:.2f}, {f["ci95_upper"]:.2f}]');panel(ax,chr(97+j));ax.grid(axis='y',alpha=.15)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=1)
    else:
        for ax in axes:ax.text(.5,.5,'Blocked: raw alignment gate',ha='center',va='center',transform=ax.transAxes);ax.axis('off')
    fig.suptitle('Training normal operation; reference speed 4-11 m/s; no 3-sigma filter',fontsize=8)
    save(fig,'k4_yaw_power_ratio');end(t,'figure_yaw_power_ratio')
if __name__=='__main__':main()
