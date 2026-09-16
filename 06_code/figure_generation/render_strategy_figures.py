#!/usr/bin/env python3
"""Render capacity, list turnover and monitoring-support figures from native results."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

P=Path(__file__).resolve().parents[2];R=P/'04_analysis_data/strategy_evaluation';F=P/'02_figures';S=P/'03_figure_source_data';V=P/'07_metadata/provenance/figures'
Q='multi_horizon';D='direct_60_day';H='historical_risk'
C={Q:'#007C91',D:'#D55E00',H:'#68717C'}
LABEL={x['policy_id']:x['name_en'] for x in json.loads((P/'07_metadata/policy_registry.json').read_text())}
INK='#20242A'
plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7.5,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
for p in (F,S,V):p.mkdir(parents=True,exist_ok=True)
AUDIT={}

def axis(fig,rect):
 ax=fig.add_axes(rect)
 ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',pad=2)
 return ax

def title(fig,letter,text,x):
 fig.text(x,.965,letter,fontsize=9,weight='bold',va='top')
 fig.text(x+.025,.965,text,fontsize=8,va='top')

def key(fig,policies):
 handles=[Line2D([0],[0],color=C[p],lw=1,ls='--' if p==D else ':' if p==H else '-',marker='s' if p==D else 'o' if p==Q else None,markersize=3,markerfacecolor='white' if p==D else C[p],label=LABEL[p]) for p in policies]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.915),ncol=len(policies),frameon=False,handlelength=2,columnspacing=1.7)

def save(fig,name,source):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();b=fig.bbox
 outside=[];minfont=99
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text():continue
  # Do not count off-axis tick labels that the axis does not draw.
  if t in sum([a.get_xticklabels()+a.get_yticklabels() for a in fig.axes],[]):
   bb=t.get_window_extent(renderer)
  else:bb=t.get_window_extent(renderer)
  minfont=min(minfont,t.get_fontsize())
  if bb.x0<b.x0-1 or bb.y0<b.y0-1 or bb.x1>b.x1+1 or bb.y1>b.y1+1:outside.append(t.get_text())
 assert not outside,(name,outside)
 for suffix in ['pdf','svg','png']:
  fig.savefig(F/('supplementary' if name.startswith('Figure_S') else 'main')/f'{name}.{suffix}',dpi=450,facecolor='white')
 AUDIT[name]={'source_data':str(source.relative_to(P)),'size_mm':(fig.get_size_inches()*25.4).tolist(),'outside_canvas':outside,'minimum_font_pt':minfont,'matplotlib_version':matplotlib.__version__}
 plt.close(fig)

# Capacity: source data contain both panels; the multi_horizon-direct contrast repeats across the three policy rows.
cap=pd.read_csv(R/'capacity_coverage.csv');diff=pd.read_csv(R/'capacity_differences.csv')
cap=cap.loc[cap.endpoint.eq('primary_31_60')].copy();diff=diff.loc[diff.endpoint.eq('primary_31_60')].copy()
src=cap.merge(diff[['capacity_percent','difference_pp','ci_low_pp','ci_high_pp','ci_source']],on='capacity_percent',validate='many_to_one')
src['display_name']=src.policy.map(LABEL);sp=S/'Figure_5/source_data.csv';src.to_csv(sp,index=False)
# Plot only the exported source.
src=pd.read_csv(sp)
fig=plt.figure(figsize=(180/25.4,90/25.4),dpi=170)
a=axis(fig,[.080,.19,.405,.59]);b=axis(fig,[.590,.19,.385,.59])
title(fig,'a','Capacity and event coverage',.035);title(fig,'b','Multi-horizon versus Direct 60-day',.545);key(fig,[Q,D,H])
for p in (Q,D,H):
 z=src.loc[src.policy.eq(p)].sort_values('capacity_percent')
 a.plot(z.capacity_percent,z.coverage_percent,color=C[p],lw=1,ls='--' if p==D else ':' if p==H else '-')
 at=z.loc[z.capacity_percent.eq(10)].iloc[0]
 a.scatter([10],[at.coverage_percent],s=14,marker='s' if p==D else 'o' if p==Q else 'D',facecolors='white' if p==D else C[p],edgecolors=C[p],linewidths=.6,zorder=4)
z=src.loc[src.policy.eq(Q)].sort_values('capacity_percent')
b.plot(z.capacity_percent,z.difference_pp,color=C[Q],lw=1)
ci=z.loc[z.ci_low_pp.notna()]
b.errorbar(ci.capacity_percent,ci.difference_pp,yerr=[ci.difference_pp-ci.ci_low_pp,ci.ci_high_pp-ci.difference_pp],fmt='none',ecolor=C[Q],elinewidth=.55,capsize=2,zorder=2)
b.scatter(ci.capacity_percent,ci.difference_pp,s=10,color=C[Q],zorder=4)
b.annotate('+8.05 pp',(10,8.045977),(13.3,11.6),fontsize=7,arrowprops={'arrowstyle':'-','lw':.5,'color':C[Q]})
for ax in [a,b]:
 ax.axvline(10,color=INK,ls='--',lw=.55,zorder=0);ax.set_xlim(0,31);ax.set_xticks([1,5,10,15,20,25,30]);ax.set_xlabel('Monitoring capacity (%)')
 ax.text(10,.99,'10% primary',transform=ax.get_xaxis_transform(),va='top',ha='center',fontsize=6.8,bbox={'facecolor':'white','edgecolor':'none','pad':1})
a.set_ylim(0,80);a.set_yticks([0,20,40,60,80]);a.set_ylabel('Event coverage (%)')
b.axhline(0,color=INK,lw=.5);b.set_ylim(-.5,14);b.set_yticks([0,3,6,9,12]);b.set_ylabel('Event coverage difference (pp)')
fig.text(.08,.035,'31–60 days · 696 events · 10% prespecified; capacity curve descriptive',fontsize=7)
save(fig,'Figure_5',sp)

# List dynamics: ECDF Source Data preserve underlying observations, identity and summary annotations.
t=pd.read_csv(R/'turnover_adjacent_origins.csv');l=pd.read_csv(R/'turnover_lake_selection_counts.csv',dtype={'lake_id':str});sm=pd.read_csv(R/'turnover_summary.csv')
parts=[]
for p in [Q,D]:
 for panel,df,col,ident in [('a',t.loc[t.policy.eq(p)].copy(),'jaccard_overlap','current_origin'),('b',l.loc[l.policy.eq(p)&l.selection_n.gt(0)].copy(),'selection_n','lake_id')]:
  z=df.sort_values([col,ident],kind='mergesort');n=len(z)
  part=pd.DataFrame({'panel':panel,'policy':p,'display_name':LABEL[p],'observation_id':z[ident].to_numpy(),'value':z[col].to_numpy(),'ecdf':np.arange(1,n+1)/n,'denominator_n':n})
  part['metric']=col;parts.append(part)
src=pd.concat(parts,ignore_index=True).merge(sm,on='policy',validate='many_to_one');sp=S/'Figure_S9/source_data.csv';src.to_csv(sp,index=False);src=pd.read_csv(sp)
fig=plt.figure(figsize=(180/25.4,90/25.4),dpi=170)
a=axis(fig,[.080,.19,.405,.59]);b=axis(fig,[.590,.19,.385,.59])
title(fig,'a','Adjacent prediction-date list overlap',.035);title(fig,'b','Repeat selections and unique-lake reach',.545);key(fig,[Q,D])
for p in [Q,D]:
 for panel,ax in [('a',a),('b',b)]:
  z=src.loc[src.policy.eq(p)&src.panel.eq(panel)]
  ax.step(z.value*(100 if panel=='a' else 1),z.ecdf,where='post',color=C[p],lw=1,ls='--' if p==D else '-')
rowq=src.loc[src.policy.eq(Q)].iloc[0];rowd=src.loc[src.policy.eq(D)].iloc[0]
a.text(.025,.98,f'Mean Jaccard: {rowq.jaccard_mean:.1%} / {rowd.jaccard_mean:.1%}',transform=a.transAxes,va='top',fontsize=7)
b.text(.975,.26,f'Unique lakes: {int(rowq.unique_selected_lakes):,} / {int(rowd.unique_selected_lakes):,}\nMedian selections: {rowq.repeat_selection_median:.0f} / {rowd.repeat_selection_median:.0f}',transform=b.transAxes,ha='right',va='top',fontsize=7,linespacing=1.6)
a.set_xlim(0,100);a.set_xticks([0,20,40,60,80,100]);a.set_xlabel('Adjacent-list Jaccard overlap (%)')
b.set_xlim(0,130);b.set_xticks([0,25,50,75,100,130]);b.set_xlabel('Selections per ever-selected lake')
for ax in [a,b]:ax.set_ylim(0,1.05);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('ECDF')
fig.text(.08,.035,'10% capacity · 36,631 slots per strategy · 129 adjacent pairs · Descriptive operational analysis',fontsize=7)
save(fig,'Figure_S9',sp)

# Monitoring support: fixed global queues, descriptive support strata, prespecified overall reference.
src=pd.read_csv(R/'monitoring_support_strata_results.csv');src=src.loc[src.endpoint.eq('primary_31_60')].copy();sp=S/'Figure_S10/source_data.csv';src.to_csv(sp,index=False);src=pd.read_csv(sp)
fig=plt.figure(figsize=(180/25.4,85/25.4),dpi=170)
a=axis(fig,[.30,.20,.42,.62]);a.set_ylim(-.6,3.9)
fig.text(.04,.955,'Monitoring support and event coverage',fontsize=8,va='top')
fig.text(.04,.90,'Multi-horizon − Direct 60-day',fontsize=7,va='top')
labels=[];overall=src.loc[src.stratum.eq('overall')].iloc[0]
a.axvline(0,color=INK,lw=.6);a.axvline(overall.difference_pp,color=C[Q],ls=':',lw=.6)
for s,y in [('overall',3),('low',2),('medium',1),('high',0)]:
 z=src.loc[src.stratum.eq(s)].iloc[0];color=INK if s=='overall' else C[Q]
 a.errorbar(z.difference_pp,y,xerr=[[z.difference_pp-z.ci_low_pp],[z.ci_high_pp-z.difference_pp]],fmt='D' if s=='overall' else 'o',ms=3,elinewidth=.8,capsize=2,color=color)
 label='Overall' if s=='overall' else f'{s.title()} support ({z.observation_day_min}–{z.observation_day_max} days)'
 labels.append((y,label))
 # Place sample sizes and numerical intervals in a reserved table alongside the forest plot.
 fy=.20+.62*(y+.6)/4.5
 fig.text(.755,fy,f'{int(z.candidate_lake_n):,} / {int(z.event_n):,}',fontsize=7,va='center',ha='center')
 fig.text(.895,fy,f'{z.difference_pp:+.2f}\n[{z.ci_low_pp:.2f}, {z.ci_high_pp:.2f}]',fontsize=7,va='center',ha='center',linespacing=1.25)
a.set_yticks([y for y,l in labels],[l for y,l in labels]);a.tick_params(axis='y',length=0,pad=10)
a.set_xlim(-3,18);a.set_xticks([0,5,10,15]);a.set_xlabel('Event coverage difference (pp)')
fig.text(.755,.85,'Lakes / events',fontsize=7,ha='center');fig.text(.895,.85,'Difference [95% CI]',fontsize=7,ha='center')
fig.text(.04,.08,'31–60 days · 10% capacity · Post hoc strata; 5,000 lake bootstrap replicates with global queue rebuilding',fontsize=7)
zero=src.loc[src.stratum.eq('zero')].iloc[0]
fig.text(.04,.035,f'Zero observation support: {int(zero.candidate_lake_n):,} candidate lakes, 0 observed events; coverage not estimable.',fontsize=7)
save(fig,'Figure_S10',sp)
(V/'figure_render_audit.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
print('Rendered three figures with CSV Source Data; canvas checks PASS')
