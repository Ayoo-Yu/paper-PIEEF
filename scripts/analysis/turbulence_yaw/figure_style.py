"""Python figure contract: quantitative grids, exact CSV values, no inference."""
from common import *
os.environ['MPLCONFIGDIR']=str(O/'scripts/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
COLORS={'VF':'#B5432F','DLinear':'#0072B2','Jensen-Net':'#7B6DAA','Wake-GNN':'#009E73','XGBoost':'#C28E00','Persistence':'#777777'}
LABELS={'low':'Low','medium':'Medium','high':'High','very_high':'Very high*','undefined':'Undefined'}
def save(fig,name):
    fig.savefig(O/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(O/'figures'/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
def panel(ax,label):ax.text(-.15,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
