"""Plotting style and export checks for result figures."""
from pathlib import Path
import sys,json,hashlib,argparse,itertools,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.text import Text
from matplotlib.colors import LinearSegmentedColormap,Normalize
from PIL import Image
import numpy as np
import pandas as pd

NAVY='#1B3B8B';TEAL='#006D2C';GOLD='#A6761D';COPPER='#B2182B'
INK='#000000';MID='#4D4D4D';LIGHT='#E5E5E5'
METHODS=['xgboost','elastic_net','single_window','multi_window']
NAMES=['XGBoost','Elastic Net','Single-window','Multi-window']
REGIONS=['china_2020_2022','china_2023','oceania','new_zealand']
ENC={m:dict(color=c,marker=s,ls=l,filled=f) for m,c,s,l,f in zip(METHODS,
    ['#000000','#606060',NAVY,TEAL],['s','D','o','^'],[(0,(5,2.2)),(0,(1,1.6)),'-','-'],[True,False,True,False])}
ABS=LinearSegmentedColormap.from_list('coverage',['#FAFCFD','#C9D8E4','#7893B2','#233B60'])

class Page:
    def __init__(self,h):
        self.w=183;self.h=h
        plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
            'font.size':10,'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
            'legend.fontsize':9.5,'axes.linewidth':.7,'axes.edgecolor':INK,'text.color':INK,
            'axes.labelcolor':INK,'xtick.direction':'in','ytick.direction':'in',
            'xtick.major.size':3.4,'ytick.major.size':3.4,'xtick.major.width':.7,'ytick.major.width':.7,
            'xtick.minor.size':1.8,'ytick.minor.size':1.8,'xtick.minor.width':.55,'ytick.minor.width':.55,
            'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','lines.solid_capstyle':'butt'})
        self.fig=plt.figure(figsize=(183/25.4,h/25.4),dpi=180,facecolor='white')
    def text(self,x,y,s,size=9.5,ha='left',color=INK,weight='normal',**kwargs):
        return self.fig.text(x/self.w,1-y/self.h,s,va='center',ha=ha,fontsize=size,color=color,weight=weight,**kwargs)
    def axes(self,x,y,w,h):
        a=self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h])
        a.tick_params(which='both',top=True,right=True,pad=2)
        for spine in a.spines.values():spine.set_visible(True)
        return a
    def panel(self,x,y,letter,title):
        self.text(x,y,'('+letter+')',12,weight='bold')
    def method_legend(self,y):
        hs=[Line2D([],[],color=e['color'],ls=e['ls'],lw=1.2,marker=e['marker'],ms=4.5,
            mfc=e['color'] if e['filled'] else 'white',mec=e['color'],mew=.8,label=n)
            for m,n in zip(METHODS,NAMES) for e in [ENC[m]]]
        self.fig.legend(handles=hs,loc='center',bbox_to_anchor=(.5,1-y/self.h),ncol=4,
            frameon=False,handlelength=2,handletextpad=.5,columnspacing=1.5,borderaxespad=0,fontsize=9.5)

def read_data(root):
    sums=json.loads((root/'source_checksums.json').read_text(encoding='utf-8'));f={}
    for rel,digest in sums.items():
        path=root/rel;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
        if path.suffix=='.csv':f[path.stem]=pd.read_csv(path,dtype={'lake_id':str,'episode_id':str,'endpoint_id':str})
    return f,sums

def export(p,root,n,caption,checks,hashes):
    from unified_style import export_figure
    export_figure(p.fig,root,n,caption,checks,hashes)
    plt.close(p.fig)
