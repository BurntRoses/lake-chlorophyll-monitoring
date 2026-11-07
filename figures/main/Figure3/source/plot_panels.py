"""Plot main and supplementary result panels."""
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

def s1():
    p=Page(181);p.panel(5,7,'a')
    a=p.ax(39,18,132,30);a.set(xlim=(-24,8),ylim=(-.7,2.7),xticks=[-20,-10,0,7],yticks=[2,1,0],yticklabels=['Available','Pending','Recent high'])
    a.axvline(0,color='black',lw=.85);a.axvspan(0,8,color='#F1F1F1',zorder=0)
    for r,y in zip(load('information_timing_examples').itertuples(),[2,1,0]):
        c=BLUE if r.available_before_day0 else GREY
        a.plot([r.sample_day,r.available_day],[y,y],c=c,lw=.8);a.plot(r.sample_day,y,'o',mfc='white',mec=c,ms=5);a.plot(r.available_day,y,'s',mfc=c,mec=c,ms=5)
    a.tick_params(axis='y',length=0);a.set_xlabel('Days relative to decision',labelpad=3)
    a.legend(handles=[Line2D([],[],marker='o',mfc='white',color=BLUE,ls='none',label='Sample'),Line2D([],[],marker='s',color=BLUE,ls='none',label='Available')],loc='lower left',bbox_to_anchor=(0,1.05),ncol=2,frameon=False,handletextpad=.4)
    p.panel(5,63,'b');b=p.ax(39,70,132,29);b.set(xlim=(0,130),ylim=(-.6,1.5),xticks=[0,30,60,90,120],yticks=[]);b.set_xlabel('High-record day',labelpad=3)
    d=load('event_grouping_examples');b.plot([10,50],[0,0],c=RED,lw=1.1)
    for r in d.itertuples():b.plot(r.record_day,0,marker='^',ms=6,mfc=RED if r.first_record_of_event else 'white',mec=RED,mew=1,ls='none')
    metric_arrow(b,(50,.35),(120,.35),key='event-gap',marker='|',size_pt=0,color=GREY,lw=.7,double=True);b.text(85,.55,'70 d',ha='center');b.text(30,1,'Event 1',ha='center',color=RED);b.text(120,1,'Event 2',ha='center',color=RED)
    p.panel(5,114,'c');c=p.ax(39,124,132,39);c.set(xlim=(-25,65),ylim=(-.6,3.6),xticks=[-20,0,31,60],yticks=[3,2,1,0],yticklabels=['Positive','Negative','Unknown','Active'])
    c.axvspan(31,60,fc='#E5EBF5');c.axvline(0,c='black',lw=.8);c.tick_params(axis='y',length=0)
    c.plot(40,3,marker='^',ms=6,c=RED,ls='none');c.plot(45,2,marker='o',mfc='white',mec=BLUE,ms=6,ls='none');c.text(45,1,'?',ha='center',va='center',color=GREY,fontsize=12);c.plot(-20,0,marker='^',ms=6,c=RED,ls='none')
    for y in [3,2,1]:c.plot(0,y,marker='s',ms=4.5,color=GREEN,ls='none')
    c.plot(0,0,marker='x',ms=6,color=GREY,ls='none');c.set_xlabel('Days relative to decision',labelpad=3)
    p.text(120,117,'31–60 d outcome window',ha='center',color=BLUE)
    return p,{'schematic_dates_from_supplied_examples':True,'records':len(d),'positive_negative_unknown_active_distinguished':True}

def s2():
    p=Page(183);p.legend();p.panel(5,13,'a');a=p.ax(24,21,149,45)
    d=load('coverage_accumulation');assert len(d)==240 and d.cohort_events.eq(681).all()
    for m in METHODS:
        v=d.query('method_id==@m').sort_values('lead_days',ascending=False);e=ENC[m];a.step(v.lead_days,v.cumulative_coverage_pct,where='post',c=e['c'],ls=e['ls'],lw=1.1)
    a.set(xlim=(60,1),ylim=(0,100),xticks=[60,45,30,15,1],yticks=[0,25,50,75,100]);a.set_xlabel('Lead time (d)',labelpad=3);a.set_ylabel('Cumulative coverage (%)',labelpad=3)
    a.get_xticklabels()[0].set_ha('left')
    a.axvline(31,c=GOLD,lw=.65,ls='--');a.text(58,92,'n = 681',ha='left')
    p.panel(5,84,'b');p.panel(96,84,'c')
    b=p.ax(24,95,62,63);d=load('annual_coverage')
    for m in METHODS:
        v=d.query('method_id==@m').sort_values('event_year');curve(b,v.event_year,v.coverage_pct,m,True)
    b.set(xlim=(2020.3,2025.3),ylim=(40,100),xticks=range(2021,2026),yticks=[40,60,80,100]);b.set_ylabel('31–60 d coverage (%)',labelpad=3);b.set_xlabel('Evaluation year',labelpad=3)
    for t in b.get_xticklabels():t.set_rotation(45);t.set_ha('right')
    c=p.ax(118,95,55,63);cats=['31–60 d','15–30 d','1–14 d','Uncovered'];v=load('earliest_selection_transition');matrix=v.pivot(index='multi_category',columns='single_category',values='count').reindex(index=cats,columns=cats).to_numpy();assert matrix.sum()==681
    c.imshow(matrix,origin='upper',aspect='auto',cmap=COUNTS,norm=LogNorm(1,500));c.set(xticks=range(4),xticklabels=['31–60','15–30','1–14','None'],yticks=range(4),yticklabels=['31–60','15–30','1–14','None']);c.tick_params(length=0)
    for i in range(4):
        for j in range(4):c.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>35 else 'black')
    for edge in [.5,1.5,2.5]:c.axvline(edge,c='white',lw=.6);c.axhline(edge,c='white',lw=.6)
    for t in c.get_xticklabels():t.set_rotation(45);t.set_ha('right')
    c.set_xlabel('Single-window lead (d)',labelpad=3);c.set_ylabel('Multi-window lead (d)',labelpad=3)
    return p,{'fixed_timing_cohort':681,'all_240_step_values':True,'paired_transition_sum':int(matrix.sum()),'all_20_annual_estimates':len(d)==20}

def s3():
    p=Page(181);p.panel(5,8,'a');a=p.ax(67,19,106,47);d=load('component_captures')
    order=['single','three_long_primary_only','three_time_primary_only','three_long_joint','three_time_joint']
    labs=['Single-window','Same-window · primary','Multi-window · primary','Same-window · joint','Multi-window · joint']
    for ep,c,s,dy,lab in [('31_60','black','o',.18,'31–60 d'),('1_30',BLUE,'s',0,'1–30 d'),('1_60',GREEN,'^',-.18,'1–60 d')]:
        v=d.query('endpoint==@ep').set_index('method').loc[order];a.plot(v.capture_percent,np.arange(5)+dy,ls='none',marker=s,ms=4.5,c=c,mfc='white' if s=='^' else c,label=lab)
    a.set(xlim=(70,90),ylim=(4.7,-.7),xticks=[70,75,80,85,90],yticks=range(5),yticklabels=labs);a.tick_params(axis='y',length=0);a.set_xlabel('Event coverage (%)',labelpad=3)
    a.legend(loc='lower center',bbox_to_anchor=(.5,1.05),ncol=3,frameon=False,handletextpad=.3,columnspacing=.9)
    p.panel(5,91,'b');p.panel(96,91,'c');p.legend(85)
    b=p.ax(24,102,62,59);c=p.ax(115,102,58,59);rob=load('event_definition_robustness')
    for m in METHODS:
        v=rob.query('method_id==@m').set_index('threshold_id');curve(b,[25,50,75,100],v.loc[['absolute25','absolute50','absolute75','absolute100'],'coverage_pct'],m,True)
        curve(c,[30,60,90],v.loc[['absolute75_gap30','absolute75','absolute75_gap90'],'coverage_pct'],m,True)
    b.set(xlim=(20,105),ylim=(20,100),xticks=[25,50,75,100],yticks=[20,40,60,80,100]);b.set_xlabel('Chl-a threshold (µg/L)',labelpad=3);b.set_ylabel('31–60 d coverage (%)',labelpad=3)
    c.set(xlim=(25,95),ylim=(40,100),xticks=[30,60,90],yticks=[40,60,80,100]);c.set_xlabel('Grouping gap (d)',labelpad=3);c.set_ylabel('31–60 d coverage (%)',labelpad=3)
    assert len(rob)==24 and len(d.query("endpoint in ['31_60','1_30','1_60']"))==15
    return p,{'all_15_structure_estimates':True,'all_24_definition_estimates':True,'fixed_scores_no_refit':True}

def matrix(p,x,y,method):
    d=load('paired_visit_transitions').query('method_id==@method');mat=d.pivot(index='history_bin',columns='original_bin',values='lakes').reindex(index=range(5),columns=range(5)).to_numpy();assert mat.sum()==2844
    a=p.ax(x,y,57,48);a.imshow(np.ma.masked_equal(mat,0),origin='lower',aspect='auto',cmap=COUNTS,norm=LogNorm(1,2000));a.set(xticks=range(5),xticklabels=BANDS,yticks=range(5),yticklabels=BANDS);a.tick_params(length=0)
    for t in a.get_xticklabels():t.set_rotation(45);t.set_ha('right')
    for edge in np.arange(.5,4.5):a.axhline(edge,c='white',lw=.5);a.axvline(edge,c='white',lw=.5)
    for i in range(5):
        for j in range(5):
            if mat[i,j]:a.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',color='white' if mat[i,j]>=75 else 'black')
    a.add_patch(Rectangle((-.5,.5),1,3,fill=False,ec=GOLD,lw=.85));a.set_xlabel('Score-only selections',labelpad=3)
    return a

def s4():
    p=Page(193);p.panel(5,8,'a');p.panel(96,8,'b')
    a=matrix(p,27,20,'single_window');b=matrix(p,116,20,'multi_window');a.set_ylabel('History-aware selections',labelpad=3)
    p.text(56,12,'Single-window',color=BLUE,ha='center');p.text(145,12,'Multi-window',color=GREEN,ha='center')
    cb=p.fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1,2000),cmap=COUNTS),cax=p.ax(71,91,90,2.1),orientation='horizontal',ticks=[1,10,100,1000]);cb.ax.set_xticklabels(['1','10','100','1,000']);cb.ax.xaxis.set_minor_locator(NullLocator());cb.ax.tick_params(length=1.5,pad=1,top=False);p.text(65,92,'Lakes',ha='right')
    p.legend(108);p.panel(5,117,'c');p.panel(96,117,'d');d=load('slot_concentration_curves')
    for x,schedule,lab in [(24,'original','Score-only'),(115,'rotation','History-aware')]:
        a=p.ax(x,127,58,47)
        for m in METHODS:
            v=d.query('method_id==@m and schedule_id==@schedule');curve(a,v.lake_fraction*100,v.slot_fraction*100,m)
            assert len(v)==2845 and np.isclose(v.slot_fraction.iloc[-1],1)
        a.plot([0,100],[0,100],c=GREY,lw=.55,ls=':');a.set(xlim=(0,100),ylim=(0,100),xticks=[0,50,100],yticks=[0,50,100]);a.set_xlabel('Candidate lakes (%)',labelpad=3);a.set_ylabel('Monitoring slots (%)',labelpad=2);a.text(.05,.92,lab,transform=a.transAxes)
    return p,{'all_2844_lakes_in_each_matrix':True,'lorenz_points':len(d),'zero_selection_lakes_included':True,'same_36631_slots':True}

def s5():
    p=Page(219);p.legend();d=load('regional_capacity_response');v=load('regional_differences')
    for i,(x,y) in enumerate([(24,21),(115,21),(24,86),(115,86)]):
        region=REGIONS[i];p.panel(5 if i%2==0 else 96,y-9,'abcd'[i]);a=p.ax(x,y,58,43)
        for m in METHODS:
            q=d.query('region_id==@region and method_id==@m').sort_values('nominal_capacity_pct');curve(a,q.nominal_capacity_pct,q.coverage_pct,m)
        a.set(xlim=(1,30),ylim=(0,100),xticks=[1,10,20,30],yticks=[0,50,100]);a.axvline(10,c=GOLD,ls='--',lw=.6);a.text(.95,.08,RN[i],transform=a.transAxes,ha='right',va='bottom');a.set_xlabel('Monitoring capacity (%)',labelpad=2);a.set_ylabel('Event coverage (%)',labelpad=2)
    p.panel(5,148,'e');a=p.ax(48,157,125,43)
    for i,region in enumerate(REGIONS):
        for m,dy in [('xgboost',.14),('elastic_net',-.14)]:
            r=v.query('region_id==@region and comparator_id==@m').iloc[0];e=ENC[m];a.errorbar(r.difference_pp,3-i+dy,xerr=[[r.difference_pp-r.lower_pp],[r.upper_pp-r.difference_pp]],fmt=e['s'],color=e['c'],mfc='white' if m=='elastic_net' else e['c'],ms=4.5,capsize=2,lw=.8)
    a.axvline(0,c=GREY,lw=.6,ls='--');a.set(xlim=(-25,60),ylim=(-.55,3.55),xticks=[-20,0,20,40,60],yticks=[3,2,1,0],yticklabels=RN);a.tick_params(axis='y',length=0);a.set_xlabel('Multi-window − comparator (pp)',labelpad=3)
    assert v.lower_pp.min()>a.get_xlim()[0] and v.upper_pp.max()<a.get_xlim()[1]
    assert len(d)==1872 and len(v)==8 and v.confidence_level.eq(.9875).all()
    return p,{'regional_capacity_values':1872,'all_eight_paired_intervals':True,'interval_level':.9875,'separate_regional_cohorts':True}

def s6():
    p=Page(189);p.panel(5,8,'a');p.panel(96,8,'b')
    a=p.ax(23,20,62,53);q=load('pre2020_concentration_quantiles')
    for region,c,m,lab in [('source',GREY,'s','Source'),('china',GREEN,'o','China')]:
        d=q.query('region_id==@region').sort_values('percentile');a.plot(d.percentile,d.chl_a_ug_L,marker=m,c=c,lw=1,ms=4.5,label=lab)
    a.set(yscale='log',xlim=(20,101),xticks=[25,50,75,95],yticks=[1,10,100]);a.set_yticklabels(['1','10','100']);a.set_xlabel('Percentile',labelpad=3);a.set_ylabel('Pre-2020 Chl-a (µg/L)',labelpad=3);a.legend(frameon=False,loc='upper left',handlelength=1.5,handletextpad=.4)
    b=p.ax(119,20,54,53);d=load('historical_reference_support')
    for i,region in enumerate(REGIONS):
        r=d.query('region_id==@region').iloc[0];b.barh(3-i,r.candidate_lakes,color='#D9D9D9',height=.58);b.barh(3-i,r.supported_reference_lakes,color=GREEN,height=.58)
    b.set(xlim=(0,450),xticks=[0,200,400],yticks=[3,2,1,0],yticklabels=['China 20–22','China 2023','Oceania','NZ']);b.set_xlabel('Candidate lakes',labelpad=3);b.tick_params(axis='y',length=0)
    b.legend(handles=[Rectangle((0,0),1,1,fc=GREEN,label='Supported'),Rectangle((0,0),1,1,fc='#D9D9D9',label='Unsupported')],frameon=False,loc='lower left',bbox_to_anchor=(-.2,1.02),ncol=1,handlelength=1,labelspacing=.2,handletextpad=.4)
    p.panel(5,91,'c');p.panel(96,91,'d');c=p.ax(23,103,62,62);d=load('china_background_accumulation')
    for col,color,ls,lab in [('cumulative_evaluable_events',GREY,':','Evaluable'),('cumulative_absolute_covered',GOLD,'--','Absolute-trained'),('cumulative_relative_covered',GREEN,'-','Relative-trained')]:c.step(d.historical_q90_ug_L,d[col],where='post',c=color,ls=ls,lw=1.05,label=lab)
    c.set(xscale='log',xlim=(.25,200),ylim=(0,750),xticks=[1,5,25,75],yticks=[0,250,500,750]);c.set_xticklabels(['1','5','25','75']);c.set_xlabel('Historical Chl-a q90 (µg/L)',labelpad=3);c.set_ylabel('Cumulative events',labelpad=3)
    for xx in [25,75]:c.axvline(xx,c=GOLD,lw=.5,ls=':')
    c.legend(loc='upper left',frameon=False,fontsize=10.5,handlelength=1.4,handletextpad=.4,labelspacing=.3)
    e=p.ax(115,103,58,62);reach=load('regional_relative_reach')
    for i,region in enumerate(REGIONS):
        z=reach.query('region_id==@region').set_index('endpoint_id');short=z.loc['31_60'];wide=z.loc['1_60']
        metric_arrow(e,(short.coverage_pct,short.lake_coverage_pct),(wide.coverage_pct,wide.lake_coverage_pct),key='regional-reach-'+region,marker='o',size_pt=5,color=GREY,lw=.85)
        e.plot(short.coverage_pct,short.lake_coverage_pct,'o',ms=5,mfc='white',mec=GREEN);e.plot(wide.coverage_pct,wide.lake_coverage_pct,'o',ms=5,c=GREEN)
        # Label the filled endpoint from the open space above/right of its vector.
        placements={'china_2020_2022':(55,91,'China 20–22','right'),
                    'china_2023':(62,59,'China 2023','left'),
                    'oceania':(75,81,'Oceania','left'),
                    'new_zealand':(96,104,'NZ','right')}
        tx,ty,label,align=placements[region]
        e.annotate(label,(wide.coverage_pct,wide.lake_coverage_pct),xytext=(tx,ty),
            ha=align,va='center',fontsize=10.5,color='black',
            arrowprops=dict(arrowstyle='-',lw=.55,color=GREY,shrinkA=2,shrinkB=3.6))
    e.set(xlim=(25,110),ylim=(35,110),xticks=[25,50,75,100],yticks=[50,75,100]);e.set_xlabel('Events covered (%)',labelpad=3);e.set_ylabel('Event lakes covered (%)',labelpad=3)
    p.text(143,180,'○ 31–60 d    ● 1–60 d',ha='center',color=GREEN)
    assert int(d.iloc[-1].cumulative_evaluable_events)==718 and int(d.iloc[-1].cumulative_absolute_covered)==62 and int(d.iloc[-1].cumulative_relative_covered)==349
    return p,{'all_12_empirical_quantiles':len(q)==12,'all_four_reference_support_cohorts':True,'background_totals':[718,62,349],'eight_event_lake_pairs':len(reach)==8}

CAPTIONS = {'3': 'Figure 3. Capacity response, paired coverage differences and monitoring allocation.\n(a) Upper: coverage of 593 US primary-window events at capacities from 1% to 30% in 0.25-percentage-point increments. The horizontal reference marks Multi-window at 10% capacity (463/593, 78.08%); symbols mark the first evaluated capacity at which each method reaches that coverage. Lower: Multi-window minus Single-window coverage across five lead windows, whose event denominators are 679, 644, 593, 679 and 681 in the displayed order. Ochre marks 10% capacity. Both layers use score-only lists; pp denotes percentage points.\n(b) Multi-window minus Elastic Net, XGBoost and Single-window coverage at 10% capacity in the 31–60-day window. Bars show 97.5% paired whole-lake bootstrap intervals from 5,000 resamples, adjusted within the comparison families defined in Supplementary Methods S5.\n(c) Event coverage against the cumulative number of distinct lakes selected under score-only (open symbols) and history-aware (filled symbols) schedules. Arrows connect schedules for the same method. All schedules use 36,631 slots over 130 rounds among 2,844 lakes. Scheduling parameters were fixed during development; both capacity-based rankers reach 510/593 events (86.0%). Figure S4 shows allocation distributions and slot concentration. Results are retrospective for 2021–2025. Absolute events use Chl-a ≥75 μg/L and consecutive high-record gaps ≤60 days.', 'S1': 'Figure S1. Information availability, event grouping and label assignment.\n(a) Field sampling dates (open circles) and result-availability dates (filled squares), separated by a 7-day delay. The vertical line is the decision date. Available results enter the inputs; pending results become available later. Recent high records determine the active-state exclusion.\n(b) High records on days 10, 25 and 50 form one event because consecutive gaps are ≤60 days. The day-120 record follows a 70-day gap and starts a second event. Filled triangles mark first high records; open triangles mark subsequent high records within an event.\n(c) Positive, negative, unknown and active cases relative to a decision date. The shaded interval is the 31–60-day outcome window. Squares mark candidate decision opportunities, triangles high records, circles non-high observations and the cross an active-state exclusion. Label assignment uses observations in the outcome window; records with unknown outcomes remain eligible for ranking.\nThe rule schematics were drawn with Python/SVG code assisted by OpenAI Codex (OpenAI).', 'S2': 'Figure S2. Advance-selection timing, annual coverage and paired timing categories.\n(a) Cumulative earliest-selection curves for four methods on the same 681-event US 1–60-day cohort at 10% capacity. At each lead time, the curve gives the fraction of events whose lake has already been selected at least once. The ochre line marks 30 days.\n(b) Annual 31–60-day coverage in 2021–2025, with event denominators of 49, 39, 171, 225 and 109, respectively. Methods share eligibility and quotas.\n(c) Paired earliest-selection categories for Single-window (columns) and Multi-window (rows) on the 681-event cohort. Each event occupies one cell; None denotes no selection within 1–60 days. Cells give event counts on a logarithmic colour scale. Table S4 reports window-specific estimates and denominators.', 'S3': 'Figure S3. Ranking-structure comparisons and sensitivity to event definitions.\n(a) Five configurations evaluated in the 31–60, 1–30 and 1–60-day windows. Primary denotes training on the primary-window objective alone; joint includes auxiliary-output losses. The same-window full configuration uses depth 4 and auxiliary weights of 0.5; input-structure comparisons use depth 3. Tables S2 and S6 define all configurations.\n(b) Coverage at Chl-a thresholds of 25, 50, 75 and 100 μg/L, with scores fixed from training at 75 μg/L. (c) Coverage with consecutive high-record grouping gaps of 30, 60 and 90 days. Panels b and c use 10% capacity and the 31–60-day window. Tables S7 and S8 give the corresponding event denominators.', 'S4': 'Figure S4. Paired lake-selection distributions and monitoring-slot concentration.\n(a, b) Selection counts for Single-window and Multi-window across all 2,844 candidate lakes over 130 rounds. Columns show score-only bands and rows history-aware bands: 0, 1–13, 14–26, 27–65 and 66–130 selections. Cells show lake counts, zeros are blank, and both matrices use the same logarithmic colour scale. Ochre outlines the 281 and 278 additional lakes selected by history-aware scheduling. All originally selected lakes remain in the cumulative selected set, and the maximum count after scheduling is 65 for both methods.\n(c, d) Slot-concentration curves for score-only and history-aware schedules. Within each method, lakes are ordered by selection count and the curves show cumulative fractions of lakes and slots. The diagonal indicates equal allocation. All results use 10% capacity; coefficients were fixed during development (Supplementary Methods S3; Table S13).', 'S5': 'Figure S5. Regional capacity responses and paired absolute-event comparisons.\n(a–d) Absolute-event coverage in the 31–60-day window at capacities from 1% to 30%, in 0.25-percentage-point increments. Event cohorts are China 2020–2022, China 2023, Oceania 2020–2022 and New Zealand 2023–2024, with 337, 86, 29 and 77 events, respectively. Ochre lines mark 10% capacity. Model scores remain fixed as quotas change.\n(e) Multi-window minus XGBoost (black squares) and Elastic Net (grey open diamonds) coverage at 10% capacity. Bars show 98.75% paired whole-lake bootstrap intervals from 5,000 resamples, adjusted within the four-comparison regional families. The dashed line marks zero. These estimates use absolute events; Figure 4b evaluates refitting effects on relative events.', 'S6': 'Figure S6. Historical Chl-a, reference availability and relative-event coverage.\n(a) Pre-2020 Chl-a percentiles (25th–99th) pooled across 1,406,007 observations from 3,515 source lakes and 172,755 observations from 435 Chinese lakes, on a logarithmic concentration axis. Event labels use separate lake-specific pre-2011 references.\n(b) Relative-analysis candidates with and without ≥50 pre-2011 observations spanning ≥365 days. Candidates lacking these references remain eligible. In New Zealand, 19 of 166 candidates have references, and 13 contribute the 24 primary-window events.\n(c) Cumulative Chinese 2020–2022 relative events by lake-specific pre-2011 Chl-a q90. Endpoints are 718 evaluable events, 62 covered after absolute-target training and 349 after relative-target training. Equal q90 values are grouped. Guides at 25 and 75 μg/L delimit strata with 485, 215 and 18 events (Table S12).\n(d) Relative-trained Multi-window coverage of events versus the fraction of event-bearing lakes with at least one covered event. Open and filled symbols represent 31–60 and 1–60 days, with arrows connecting regional summaries. Window-specific event and lake denominators are in Table S15. The two New Zealand points coincide at 100%. All coverage results use 10% capacity.'}

def run(number):
    funcs={'3':f3,'S1':s1,'S2':s2,'S3':s3,'S4':s4,'S5':s5,'S6':s6}
    p,checks=funcs[number]()
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'data').glob('*') if f.is_file()}
    export_figure(p.fig,ROOT,number,(ROOT.parent/f'Figure{number}_caption.txt').read_text(encoding='utf-8'),checks,hashes)
    plt.close(p.fig)
