"""Draw the US main result panels."""
from pathlib import Path
import json,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap,Normalize,LogNorm
from matplotlib.ticker import AutoMinorLocator,NullLocator
import pandas as pd
import numpy as np
from unified_style import apply,export_figure
from connector_geometry import metric_arrow,component

ROOT=Path(__file__).resolve().parent
BLUE='#1B3B8B';GREEN='#006D2C';RED='#B2182B';GOLD='#A6761D';GREY='#606060'
METHODS=['xgboost','elastic_net','single_window','multi_window']
NAMES=['XGBoost','Elastic Net','Single-window','Multi-window']
COLORS=['#000000',GREY,BLUE,GREEN]
MARKERS=['s','D','o','^']
LINES=[(0,(5,2.2)),(0,(1,1.6)),'-','-']
ENC={m:dict(c=c,s=s,ls=ls,name=n) for m,c,s,ls,n in zip(METHODS,COLORS,MARKERS,LINES,NAMES)}
REGIONS=['china_2020_2022','china_2023','oceania','new_zealand']
RN=['China 2020–22','China 2023','Oceania 2020–22','NZ 2023–24']
RC=['China\n2020–22','China\n2023','Oceania\n2020–22','NZ\n2023–24']
SIGNED=LinearSegmentedColormap.from_list('signed',[RED,'white',GREEN])
COUNTS=LinearSegmentedColormap.from_list('counts',['#F7F9FC','#BDC9E0','#6A84BA',BLUE,'#071C4D'])
BANDS=['0','1–13','14–26','27–65','66–130']

def load(name):return pd.read_csv(ROOT/'data'/f'{name}.csv')
class Page:
    def __init__(self,h):
        apply();self.w=183;self.h=h;self.fig=plt.figure(figsize=(183/25.4,h/25.4),dpi=180,facecolor='white')
    def ax(self,x,y,w,h):
        a=self.fig.add_axes([x/183,1-(y+h)/self.h,w/183,h/self.h]);a.tick_params(which='both',top=True,right=True,pad=2)
        return a
    def text(self,x,y,s,size=10.5,**kwargs):return self.fig.text(x/183,1-y/self.h,s,fontsize=size,va='center',**kwargs)
    def panel(self,x,y,l):self.text(x,y,f'({l})',13,fontweight='bold')
    def legend(self,y=6):
        hs=[Line2D([],[],color=e['c'],ls=e['ls'],marker=e['s'],ms=4.5,mfc='white' if m in ['elastic_net','multi_window'] else e['c'],mew=.8,lw=1,label=e['name']) for m,e in ENC.items()]
        self.fig.legend(handles=hs,loc='center',bbox_to_anchor=(.5,1-y/self.h),ncol=4,frameon=False,fontsize=10.5,handlelength=1.8,columnspacing=1.3,handletextpad=.5)

def curve(ax,x,y,m,markers=False,**kwargs):
    e=ENC[m];return ax.plot(x,y,color=e['c'],ls=e['ls'],lw=1.1,marker=e['s'] if markers else None,ms=4.5,mfc='white' if m in ['elastic_net','multi_window'] else e['c'],mew=.8,**kwargs)

def f3():
    p=Page(190);p.legend();p.panel(5,12,'a')
    a=p.ax(25,18,145,34);a.set(xlim=(1,30),ylim=(0,100),xticks=[1,5,10,20,30],yticks=[0,50,100]);a.tick_params(labelbottom=False)
    a.set_ylabel('31–60 d coverage (%)',labelpad=3);a.yaxis.set_minor_locator(AutoMinorLocator(2))
    ref=463/593*100;a.axhline(ref,color=GREY,lw=.6,ls=(0,(3,3)));a.axvline(10,color=GOLD,lw=.65,ls='--')
    for m in METHODS:
        d=load('capacity_coverage').query('method_id==@m').sort_values('nominal_capacity_pct');curve(a,d.nominal_capacity_pct,d.coverage_pct,m)
        r=load('capacity_crossings').query('method_id==@m').iloc[0];e=ENC[m]
        a.plot(r.minimum_capacity_pct,r.actual_coverage_pct,marker=e['s'],ms=5,mfc='white' if m in ['elastic_net','multi_window'] else e['c'],mec=e['c'],mew=.9,ls='none')
    for m,xy in [('multi_window',(6.2,90)),('single_window',(12.6,90)),('elastic_net',(16,31)),('xgboost',(25,46))]:
        r=load('capacity_crossings').query('method_id==@m').iloc[0];e=ENC[m]
        a.annotate(f'{r.minimum_capacity_pct:g}%',(r.minimum_capacity_pct,r.actual_coverage_pct),xytext=xy,ha='center',va='center',color=e['c'],arrowprops=dict(arrowstyle='-',lw=.65,color=e['c'],shrinkB=3))
    h=p.ax(25,56,145,23);g=load('capacity_window_gain');positions=[0,1,2,3.35,4.35]
    for ep,y in zip(['1_14','15_30','31_60','1_30','1_60'],positions):
        for r in g.query('endpoint_id==@ep').itertuples():
            c=SIGNED((r.difference_pp+8)/16);h.add_patch(Rectangle((r.nominal_capacity_pct-.125,y-.45),.25,.9,fc=c,ec=c,lw=.2,antialiased=False))
    h.set(xlim=(1,30),ylim=(4.9,-.55),xticks=[1,5,10,20,30],yticks=positions,yticklabels=['1–14','15–30','31–60','1–30','1–60'])
    h.tick_params(axis='y',length=0);h.set_xlabel('Monitoring capacity (%)',labelpad=2);h.set_ylabel('Lead window (d)',labelpad=2)
    h.add_patch(Rectangle((9.875,-.5),.25,5.35,fill=False,ec=GOLD,lw=.65))
    cb=p.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-8,8),cmap=SIGNED),cax=p.ax(118,91,52,1.8),orientation='horizontal',ticks=[-8,0,8]);cb.ax.tick_params(direction='out',length=1.5,pad=.8,top=False)
    p.text(113,91.9,'Multi − Single (pp)',ha='right')
    p.panel(5,107,'b');p.panel(102,107,'c')
    b=p.ax(39,119,52,53);d=load('paired_differences');order=['elastic_net','xgboost','single_window'];ys=[2,1,0]
    for m,y in zip(order,ys):
        r=d.query('comparator_id==@m').iloc[0];e=ENC[m]
        b.errorbar(r.difference_pp,y,xerr=[[r.difference_pp-r.lower_pp],[r.upper_pp-r.difference_pp]],fmt=e['s'],color=e['c'],mfc='white' if m=='elastic_net' else e['c'],ms=5,capsize=3,lw=.85)
    b.axvline(0,color=GREY,ls='--',lw=.65);b.set(xlim=(-5,30),ylim=(-.6,2.6),xticks=[0,10,20,30],yticks=ys,yticklabels=[ENC[m]['name'] for m in order]);b.tick_params(axis='y',length=0,right=False)
    b.set_xlabel('Multi-window gain (pp)',labelpad=4)
    c=p.ax(118,119,59,53);r=load('rotation_event_lake_outcomes')
    for row in r.itertuples():
        e=ENC[row.method_id];x0,x1=row.selected_lakes_original,row.selected_lakes_rotation;y0,y1=row.original_coverage_pct,row.rotation_coverage_pct
        metric_arrow(c,(x0,y0),(x1,y1),key='scheduling-'+row.method_id,marker=e['s'],size_pt=5,color=e['c'],lw=.9)
        c.plot(x0,y0,marker=e['s'],ls='none',ms=5,mfc='white',mec=e['c'],mew=1)
        c.plot(x1,y1,marker=e['s'],ls='none',ms=5,mfc=e['c'],mec=e['c'],mew=1)
    c.set(xlim=(500,1550),ylim=(50,91),xticks=[600,1000,1400],yticks=[50,60,70,80,90]);c.set_xlabel('Distinct lakes selected',labelpad=4);c.set_ylabel('Event coverage (%)',labelpad=2)
    c.axhline(510/593*100,lw=.5,color=GREY,ls=':',zorder=0);c.text(520,87.5,'86.0%',color=GREEN,fontsize=10.5)
    hs=[Line2D([],[],marker='o',ls='none',color='black',mfc=f,ms=5,label=l) for f,l in [('white','Score-only'),('black','History-aware')]]
    c.legend(handles=hs,loc='lower right',frameon=False,ncol=1,fontsize=10.5,handletextpad=.4,labelspacing=.3)
    checks={'capacity_values':468,'window_cells':585,'primary_comparisons':3,'supplied_interval_level':.975,'same_slots_all_policies':bool(r.monitoring_slots.eq(36631).all()),'all_four_rotation_methods':len(r)==4}
    assert len(load('capacity_coverage'))==468 and len(g)==585 and len(d)==4
    assert np.allclose(r.original_events_covered/r.evaluable_events*100,r.original_coverage_pct)
    assert np.allclose(r.rotation_events_covered/r.evaluable_events*100,r.rotation_coverage_pct)
    return p,checks









def run(number):
    funcs={'3':f3}
    p,checks=funcs[number]()
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'data').glob('*') if f.is_file()}
    export_figure(p.fig,ROOT,number,(ROOT.parents[2]/'figures'/f'Figure{number}_caption.txt').read_text(encoding='utf-8'),checks,hashes)
    plt.close(p.fig)
