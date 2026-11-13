"""Mechanism drawing helpers in page coordinates."""
from figure_common import *
from matplotlib.patches import FancyArrowPatch,Polygon
import importlib.util
plt.rcParams['mathtext.fontset']='stix'

BLUE,RED=COLORS[2],COLORS[3]
INK='#252525';MID='#777777';PALE='#E9EDF2';WARM='#FBE9E6'

def arrow(p,x0,y0,x1,y1,color=INK,lw=.7,dashed=False,style='-|>',connection='arc3'):
    a=FancyArrowPatch((x0/p.w,1-y0/p.h),(x1/p.w,1-y1/p.h),transform=p.fig.transFigure,
        arrowstyle=style,mutation_scale=8,linewidth=lw,color=color,
        linestyle=(0,(3,2)) if dashed else '-',connectionstyle=connection,
        shrinkA=2,shrinkB=2)
    p.fig.add_artist(a);return a

def rule(p,x0,y0,x1,y1,color=MID,lw=.6,ls='-'):
    p.fig.add_artist(Line2D([x0/p.w,x1/p.w],[1-y0/p.h,1-y1/p.h],
        transform=p.fig.transFigure,color=color,lw=lw,ls=ls))

def box(p,x,y,w,h,label='',color=INK,fill='white',size=9.5,weight='normal',ls='-'):
    p.fig.add_artist(Rectangle((x/p.w,1-(y+h)/p.h),w/p.w,h/p.h,
        transform=p.fig.transFigure,facecolor=fill,edgecolor=color,lw=.65,ls=ls,zorder=4))
    if label:p.text(x+w/2,y+h/2,label,size,ha='center',weight=weight,color=color,zorder=6)

def tree(p,x,y,w=12,h=9,color=INK):
    # Schematic tree topology.
    for a,b in [((.5,0),(.25,.5)),((.5,0),(.75,.5)),((.25,.5),(.1,1)),
                ((.25,.5),(.4,1)),((.75,.5),(.6,1)),((.75,.5),(.9,1))]:
        rule(p,x+a[0]*w,y+a[1]*h,x+b[0]*w,y+b[1]*h,color,.65)
    for xx,yy in [(.5,0),(.25,.5),(.75,.5),(.1,1),(.4,1),(.6,1),(.9,1)]:
        p.fig.add_artist(Line2D([ (x+xx*w)/p.w ],[1-(y+yy*h)/p.h],
            transform=p.fig.transFigure,ls='none',marker='s' if yy==1 else 'o',
            ms=2.8,mfc='white',mec=color,mew=.7))

def clean(ax):
    ax.spines[['top','right']].set_visible(True)
    ax.tick_params(top=True,right=True,direction='in')

def validate_source_data(root,n):
    name='validate_source_data'
    spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if n==2:
        f,ref,binary,h=mod.read_and_validate()
        return f,h,{'original_event_level_validation':True}
    f,h=read_data(root)
    return f,h,mod.validate(f)

def extra_checks(root):
    sums=json.loads((root/'source_checksums.json').read_text(encoding='utf-8'))
    for rel,sha in sums.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,rel
    return sums

def finish(page,root,n,caption,checks,hashes):
    export(page,root,n,caption,checks,hashes)
    return
