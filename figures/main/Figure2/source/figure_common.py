"""Shared final-size typography, method encodings and export validation."""
from pathlib import Path
import sys, json, hashlib, argparse, xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from PIL import Image

PROFILE=json.loads((Path(__file__).resolve().parent/'style/figure_style.json').read_text(encoding='utf-8'))
METHODS=['xgboost','elastic_net','single_window','multi_window']
NAMES=['XGBoost','Elastic Net','Single-window','Multi-window']
REGIONS=['china_2020_2022','china_2023','oceania','new_zealand']
REGION_NAMES=['China 2020–2022','China 2023','Oceania 2020–2022','New Zealand 2023–2024']
COLORS=[PROFILE['methods'][m]['colour'] for m in METHODS]
ENC={m:dict(color=PROFILE['methods'][m]['colour'],marker=PROFILE['methods'][m]['marker'],
            filled=PROFILE['methods'][m]['filled'],ls=ls)
     for m,ls in zip(METHODS,[(0,(5,2.2)),(0,(1,1.6)),'-','-'])}
SEQ=LinearSegmentedColormap.from_list('coverage',['#F7FAFC','#C1D4E8','#7499C5','#244F80'])
GAIN=LinearSegmentedColormap.from_list('gain',['#244F9E','#FFFFFF','#D73027'])
TRAIN=LinearSegmentedColormap.from_list('training',['#FFFFFF','#F5B3A6','#AF201A'])

def style():
    from unified_style import apply
    apply()
    plt.rcParams.update({'font.family':PROFILE['font_family'],'font.size':9.5,'axes.labelsize':10,
        'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9.5,
        'axes.linewidth':.7,'axes.edgecolor':'black','text.color':'black',
        'axes.spines.top':True,'axes.spines.right':True,
        'xtick.direction':'in','ytick.direction':'in','xtick.major.size':3.4,
        'ytick.major.size':3.4,'xtick.major.width':.7,'ytick.major.width':.7,
        'svg.fonttype':'none','pdf.fonttype':42,'hatch.linewidth':.45,
        'savefig.facecolor':'white','lines.solid_capstyle':'butt'})

class Page:
    def __init__(self,height=200):
        style();self.w=183;self.h=height
        self.fig=plt.figure(figsize=(self.w/25.4,height/25.4),dpi=180,facecolor='white')
    def text(self,x,y,s,size=9.5,ha='left',weight='normal',color='black',**kw):
        kw.setdefault('zorder',6)
        return self.fig.text(x/self.w,1-y/self.h,s,ha=ha,va='center',fontsize=size,
                             weight=weight,color=color,**kw)
    def axes(self,x,y,w,h):
        ax=self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h])
        ax.tick_params(which='both',top=True,right=True,pad=2.8)
        return ax
    def panel(self,x,y,letter,title):
        self.text(x,y,'('+letter+')',12,weight='bold')
    def method_legend(self,y=5.5):
        hs=[Line2D([],[],color=e['color'],marker=e['marker'],
               mfc=e['color'] if e['filled'] else 'none',mec=e['color'],mew=.85,
               ls=e['ls'],lw=1.3,ms=5.5 if m=='multi_window' else 4.2,label=n)
            for m,n in zip(METHODS,NAMES) for e in [ENC[m]]]
        self.fig.legend(handles=hs,loc='center',bbox_to_anchor=(.5,1-y/self.h),
                        ncol=4,frameon=False,handlelength=2.1,handletextpad=.6,
                        columnspacing=1.5,borderaxespad=0,fontsize=9.5)
    def scalarbar(self,x,y,w,cmap,vmin,vmax,label,ticks):
        ax=self.axes(x,y,w,2.4)
        cb=self.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin,vmax),cmap=cmap),
                            cax=ax,orientation='horizontal',ticks=ticks)
        cb.ax.tick_params(direction='out',length=2,pad=1.4,labelsize=9,top=False)
        cb.outline.set_linewidth(.6)
        self.text(x+w/2,y+10,label,9,ha='center')

def line(ax,x,y,method,markers=(5,10,20,30),lw=None):
    e=ENC[method];x=np.asarray(x);y=np.asarray(y)
    ax.plot(x,y,color=e['color'],ls=e['ls'],lw=lw or (1.5 if method=='multi_window' else 1.2),zorder=3)
    mask=np.isin(x,markers)
    ax.plot(x[mask],y[mask],ls='none',marker=e['marker'],ms=6 if method=='multi_window' else 4,
            mfc=e['color'] if e['filled'] else 'none',mec=e['color'],mew=.8,zorder=5,clip_on=False)

def read_data(root):
    d=json.loads((root/'data_dictionary.json').read_text(encoding='utf-8'))
    frames={};hashes={}
    for entry in d['data_files']:
        p=root/entry['file'];sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha==entry['sha256'],f'Checksum mismatch: {p.name}'
        df=pd.read_csv(p,dtype={'lake_id':str,'episode_id':str,'endpoint_id':str})
        assert len(df)==entry['rows'],p.name
        assert set(entry['columns']).issubset(df.columns),p.name
        assert not df.isna().any().any(),p.name
        frames[p.stem]=df;hashes[p.name]=sha
    return frames,hashes

def export(page,root,num,caption,checks,hashes):
    from unified_style import export_figure
    export_figure(page.fig,root,num,caption,checks,hashes)
    plt.close(page.fig)
