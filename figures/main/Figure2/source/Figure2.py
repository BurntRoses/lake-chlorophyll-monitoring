"""Figure 2. Residual-tree learning and quota-constrained selection."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from mechanism_layout import *
from matplotlib.patches import Ellipse
from connector_geometry import Scene, component
ROOT=Path(__file__).resolve().parent

NAVY='#1B3B8B'; TEAL='#006D2C'; GOLD='#A6761D'; COPPER='#B2182B'
INK='#000000'; MID='#4D4D4D'; LIGHT='#E5E5E5'
COOL='#F0F2F8'; AQUA='#EDF5ED'; WHEAT='#F7F0DF'
COUNT=LinearSegmentedColormap.from_list('split_counts',['#FFFFFF','#CAD7E3',NAVY,'#23354F'])
DELTA=LinearSegmentedColormap.from_list('window_delta',[COPPER,'#FFFFFF',TEAL])
LEAF=LinearSegmentedColormap.from_list('leaf_direction',[COPPER,'#FFFFFF',TEAL])
GROUPS=['Algal observations','Other water quality','Weather','Fixed history','Seasonal context',
        'Lake / catchment','Interactions','Base logits','Window ID']
ALIASES={'wq_tp_lag60_mean':'TP, 60 d',
    'fixed_history_risk__x__origin_dayofyear_cos':'History × phase',
    'field_chla_lag60_source_record_days':'Chl-a support',
    'days_since_last_known_severe_sample':'Severe history',
    'window_head':'Window ID',
    'seasonal_31_60_high75_fraction':'Seasonal risk'}

def circle(p,x,y,label='',colour=INK,fill='white',diameter=4):
    p.fig.add_artist(Ellipse((x/p.w,1-y/p.h),diameter/p.w,diameter/p.h,
        transform=p.fig.transFigure,facecolor=fill,edgecolor=colour,lw=.75,zorder=5))
    if label:p.text(x,y,label,9,ha='center',color=colour,zorder=6)

def actual_tree(p,tree,x0,x1,y0,step,labelled=False):
    """Render the saved tree topology."""
    leaves=[]
    def gather(n):
        if 'leaf' in n:leaves.append(n['nodeid'])
        for c in n.get('children',[]):gather(c)
    gather(tree)
    leafx=dict(zip(leaves,np.linspace(x0,x1,len(leaves))))
    nodes={};edges=[]
    def layout(n,depth=0):
        child_positions=[layout(c,depth+1) for c in n.get('children',[])]
        x=float(np.mean(child_positions)) if child_positions else float(leafx[n['nodeid']])
        nodes[n['nodeid']]=(n,x,y0+depth*step)
        for c in n.get('children',[]):edges.append((n['nodeid'],c['nodeid']))
        return x
    layout(tree)
    scene=Scene(p.fig,'tree-'+str(x0)+'-'+str(y0))
    for nid,(n,x,y) in nodes.items():
        if labelled:
            label=ALIASES.get(n.get('feature_name'),'')
            width=10.4 if 'leaf' in n else 28 if 'History' in label else 25 if label not in ['TP, 60 d','Window ID'] else 19
            scene.node(str(nid),x-width/2,y-2.5,width,5)
        else:
            scene.node(str(nid),x-(.5 if 'leaf' in n else .55),y-(.45 if 'leaf' in n else .55),
                       1 if 'leaf' in n else 1.1,.9 if 'leaf' in n else 1.1,'rect' if 'leaf' in n else 'ellipse')
    for parent,child in edges:
        scene.connect(f'tree-{x0}-{y0}-{parent}-{child}',str(parent),str(child),
            s_side='bottom',t_side='top',head=False,gap=.08,
            color='#7B8A96',lw=.6 if labelled else .45,radius=0)
    for n,x,y in nodes.values():
        if 'leaf' in n:
            value=float(n['leaf'])
            if labelled:
                shade=LEAF((value+4.5)/9)
                box(p,x-5.2,y-2.5,10.4,5.0,color='white',fill=shade)
                p.text(x,y,f'{value:+.2f}',9,ha='center',
                    color='white' if abs(value)>3 else INK)
            else:
                p.fig.add_artist(Rectangle(((x-.5)/p.w,1-(y+.45)/p.h),1/p.w,.9/p.h,
                    transform=p.fig.transFigure,facecolor=TEAL if value>=0 else COPPER,
                    edgecolor='white',lw=.2,zorder=5))
        else:
            name=n['feature_name'];is_head=name=='window_head'
            if labelled:
                label=ALIASES[name]
                width=28 if 'History' in label else (25 if label not in ['TP, 60 d','Window ID'] else 19)
                box(p,x-width/2,y-2.5,width,5.0,label,color=GOLD if is_head else NAVY,
                    fill=WHEAT if is_head else 'white',size=10.5)
            else:
                circle(p,x,y,colour=GOLD if is_head else NAVY,fill=WHEAT if is_head else 'white',diameter=1.1)

def draw(f,components,model):
    p=Page(200)
    flow=Scene(p.fig,'learning-flow')
    flow.node('inputs',5,22,25,28)
    flow.node('pooled',75,22,25,27)
    flow.node('ensemble',108,22,43,16)
    flow.node('sum',152.5,32.5,5,5,'ellipse')
    flow.node('score',164,27,14,16)
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['svg.fonttype']='none'
    p.panel(5,8,'a','Multi-window residual learning')
    p.text(18,17,'Available inputs',9,ha='center',weight='bold')
    box(p,5,22,25,28,color='#88929B',fill='white')
    p.text(17.5,36,'Water quality\nWeather\nLake history',9,ha='center',color=INK)
    p.text(50,17,'Frozen predictors',9,ha='center',weight='bold')
    for j,(label,c) in enumerate([('31–60 d',TEAL),('1–14 d',NAVY),('15–30 d',GOLD)]):
        y=22+j*10.5
        key='predictor-'+str(j);flow.node(key,37,y,26,7)
        box(p,37,y,26,7,label,color=c,fill='white',size=9)
        flow.connect('input-'+str(j),'inputs',key,s_offset=y+3.5-36,color=MID,lw=.7)
        flow.connect('prediction-'+str(j),key,'pooled',t_offset=27.8+j*8-35.5,color=c,lw=.8)
    p.text(87.5,17,'Window inputs',9,ha='center',weight='bold')
    box(p,75,22,25,27,fill='white',color='#88929B')
    for j,c in enumerate([TEAL,NAVY,GOLD]):
        for k in range(5):
            box(p,78+k*3.8,26+j*8,3,3.6,color=c,
                fill=c if k==4 else ('#F3F6F8' if k<3 else 'white'))
    p.text(87.5,52,'[x, base logits, h]',9,ha='center',color=INK)
    flow.connect('shared-inputs','inputs','pooled',s_side='bottom',t_side='left',t_offset=10.5,
        via=[(17.5,55),(70,55),(70,46)],color=MID,lw=.7)
    p.text(45,58,'Shared inputs x',9,ha='center',color=MID)
    flow.connect('ensemble-input','pooled','ensemble',color=INK,lw=.85,
        via=[(104,35.5),(104,30)])
    p.text(131,17,'Residual tree ensemble',9,ha='center',weight='bold')
    for tree_index,x in [(0,110),(1,124),(59,138)]:
        actual_tree(p,model['trees'][tree_index],x,x+12,24,4)
    p.text(125,46,'60 trees · depth ≤3',9,ha='center',color=INK)
    # Sum tree outputs and the frozen predictor offset.
    for x in [116,130,144]:
        component(p.fig,[(x,36.7),(x,40.5)],key=f'ensemble-stem-{x}',scope='aggregation-stems',
            head=False,color=NAVY,lw=.65,radius=0)
    component(p.fig,[(116,40.5),(155,40.5),(155,37.73)],key='residual-aggregation',
        scope='aggregation-bus',color=NAVY,lw=.75,radius=.55)
    # Primary frozen offset bypasses the residual learner into the sum.
    flow.connect('frozen-offset','predictor-0','sum',s_side='top',t_side='top',
        via=[(50,20.5),(155,20.5)],color=TEAL,lw=.7)
    circle(p,155,35,'Σ',colour=TEAL,fill=AQUA,diameter=5)
    flow.connect('primary-score','sum','score',color=TEAL,lw=.85)
    box(p,164,27,14,16,'31–60 d\nscore',color=TEAL,fill=AQUA,size=9)
    # First tree: internal nodes and eight leaf values.
    p.text(5,68,'Tree 1',9.5,weight='bold',color=NAVY)
    p.text(5,87,r'$s_h=z_h+f_\theta([x,z,h])$',11.5)
    actual_tree(p,model['trees'][0],62,172,56,10,labelled=True)
    # Link the ensemble to the first-tree inset.
    component(p.fig,[(110,36.7),(104,48),(104,56),(107.27,56)],key='first-tree-detail',scope='detail-leader',
        color='#A3ADB6',lw=.65,head=False,dashed=True,radius=.7)

    p.panel(5,99,'b','Quota-constrained selection and event learning')
    p.text(46,106,'Top-k selection',9,ha='center',weight='bold')
    p.text(136,106,'Event pooling',9,ha='center',weight='bold')
    rd=pd.read_csv(ROOT/'data/selection_round.csv')
    meta=json.loads((ROOT/'data/selection_round_metadata.json').read_text())
    rank=rd.rank_pct.to_numpy();score=rd.multi_window.to_numpy();selected=rd.hard_selected.astype(bool).to_numpy()
    ax=p.axes(16,111,66,21)
    ax.axvspan(0,100*meta['actual_fraction'],color=WHEAT,zorder=0)
    ax.plot(rank,score,color=NAVY,lw=1.0)
    ax.plot(rank[selected],score[selected],color=GOLD,lw=1.2)
    ax.axvline(100*meta['actual_fraction'],color=GOLD,lw=.65,ls=(0,(3,2)))
    ax.plot(rank[meta['quota']-1],score[meta['quota']-1],marker='s',ms=3,mfc=GOLD,mec=GOLD)
    ax.set(xlim=(0,100),ylim=(-8.5,6.5),xticks=[],yticks=[-8,0,6])
    ax.set_ylabel('Score',fontsize=9,labelpad=3)
    ax.tick_params(labelsize=9,top=False,right=False,pad=1.6)
    clean(ax)
    ax.text(42,3.6,'k = 281',fontsize=9,color=GOLD,ha='left',va='center')
    # Binary top-k selections on the rank axis.
    stripe=p.axes(16,133,66,1.6)
    stripe.imshow(np.array([rd.hard_selected.to_numpy()]),aspect='auto',interpolation='nearest',
        cmap=LinearSegmentedColormap.from_list('hard_selection',['#E5E9ED',GOLD]),vmin=0,vmax=1,
        extent=[0,100,0,1])
    stripe.set(xlim=(0,100),xticks=[0,10,50,100],yticks=[])
    stripe.tick_params(top=False,right=False,left=False,length=1.5,pad=1,labelsize=9)
    for spine in stripe.spines.values():spine.set_visible(False)
    stripe.set_xlabel('Candidate rank (%)',fontsize=9,labelpad=1)
    # Event-pooling schematic.
    events=Scene(p.fig,'event-pooling')
    opportunity_ys=[113,117.5,121.5,126,130,134.5]
    codes=['A','A','B','B','C','C'];event_y={'A':115.25,'B':123.75,'C':132.25}
    events.node('utility',139,112,39,17)
    for c,y in event_y.items():
        events.node('event-'+c,122.6,y-2.4,4.8,4.8,'ellipse')
    for i,(c,y) in enumerate(zip(codes,opportunity_ys)):
        key='opportunity-'+str(i);events.node(key,95.85,y-2.15,4.3,4.3,'ellipse')
        events.connect('pool-'+str(i),key,'event-'+c,t_offset=-.6 if i%2==0 else .6,
            color=TEAL,lw=.75)
        circle(p,98,y,c,colour=TEAL,diameter=4.3)
    for i,(c,y) in enumerate(event_y.items()):
        circle(p,125,y,c,colour=TEAL,fill=AQUA,diameter=4.8)
        events.connect('event-utility-'+c,'event-'+c,'utility',
            t_offset=[-5.5,0,5.5][i],color=TEAL,lw=.75)
    box(p,139,112,39,17,'Event coverage\n+ log utility\n+ classification',color=TEAL,fill=AQUA,size=9)
    box(p,139,133,39,7,'Unknown-label anchor',color='#8B969F',fill='#F1F3F5',size=9)
    p.fig.add_artist(Ellipse((113/p.w,1-136.5/p.h),3.7/p.w,3.7/p.h,
        transform=p.fig.transFigure,facecolor='white',edgecolor=MID,lw=.7,ls=(0,(2,1.5)),zorder=5))
    p.text(113,136.5,'?',9,ha='center',color=MID,zorder=6)
    unknown=Scene(p.fig,'unknown-label')
    unknown.node('unknown',111.15,134.65,3.7,3.7,'ellipse')
    unknown.node('anchor',139,133,39,7)
    unknown.connect('unknown-anchor','unknown','anchor',color=MID,lw=.7,dashed=True)
    p.text(98,143.5,r'$\sum_{i\in t}p_i=k_t$',11)
    p.text(176,142.5,'Soft gates',10.5,ha='right',color=MID)
    # Training feedback path.
    component(p.fig,[(178.23,136.5),(180,136.5),(180,120.5)],
        key='anchor-gradient-branch',scope='feedback-branch',color=TEAL,lw=.75,
        dashed=True,head=False,radius=.6)
    component(p.fig,[(178.23,120.5),(180,120.5),(180,54),(150,54),(150,49),(145.5,49),(145.5,41)],
        key='joint-gradient',scope='feedback-trunk',color=TEAL,lw=.75,
        dashed=True,radius=.6)
    p.text(178,58,'Joint gradient',9,ha='right',color=TEAL)

    p.panel(5,149,'c','Saved model structure and window response')
    grouped=pd.read_csv(ROOT/'data/grouped_tree_splits.csv')
    matrices=[]
    for method in ['single_window','multi_window']:
        matrices.append(np.array([[int(grouped.loc[(grouped.method==method)&
            (grouped.feature_group==g)&(grouped.tree==t),'split_count'].iloc[0])
            for t in range(60)] for g in GROUPS]))
    top=161;height=34
    for x,matrix,name,c in zip([41,115],matrices,['Single-window','Multi-window'],[NAVY,TEAL]):
        p.text(x+30.5,151,name,11,ha='center',color=c)
        p.text(x+30.5,156,'Tree 1–60',10.5,ha='center',color=MID)
        aa=p.axes(x,top,61,height)
        aa.imshow(matrix,aspect='auto',interpolation='nearest',cmap=COUNT,vmin=0,vmax=7,extent=[.5,60.5,9,0])
        aa.set(xticks=[],yticks=[]);aa.tick_params(top=False,right=False,left=False,bottom=False)
        for y in range(1,9):aa.axhline(y,color='white',lw=.5)
        aa.axhline(7,color=TEAL,lw=.7)
        for s in aa.spines.values():s.set_linewidth(.5);s.set_edgecolor('#9AA6B0')
    for i,g in enumerate(GROUPS):p.text(5,top+(i+.5)*height/9,g,9,color=TEAL if i>=7 else INK)
    p.text(108.5,156,'Splits / tree',10.5,ha='center')
    cbax=p.axes(108,163,1.6,30)
    cb=p.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,7),cmap=COUNT),
        cax=cbax,orientation='vertical',ticks=[0,2,4,7])
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.tick_params(direction='out',length=1.5,pad=1.2,labelsize=9,top=False,right=False)
    cb.outline.set_linewidth(.5)
    return p

CAPTION = 'Figure 2. Shared residual-tree learning and quota-constrained lake selection.\n(a) Environmental inputs, fixed base-model logits and a window identifier enter a shared residual-tree ensemble. The three outputs correspond to 31–60, 1–14 and 15–30 days; the 31–60-day output determines the monitoring list. Miniatures show saved trees 1, 2 and 60, and the enlarged tree shows the full topology of tree 1 with raw residual leaf values. Split labels abbreviate transformed inputs: TP, total phosphorus; Chl-a, chlorophyll a; Window ID, lead window. Input blocks are schematic. The dashed pathway indicates training feedback, with auxiliary loss weights of 0.1 relative to the primary weight of 1. Training, leaf scaling and full split definitions are given in Supplementary Methods S2 and the source data.\n(b) Left: deployment scores for 2,805 eligible lakes on 25 June 2023; the top-k rule selects 281 lakes, marked in ochre. Right: eligible opportunities for events A, B and C are pooled so that selection on any one opportunity contributes event-level coverage. Smooth selection weights sum to each round’s integer quota. Known labels contribute classification loss, while candidates with unknown outcomes remain in the ranking and receive a base-prediction constraint (Supplementary Methods S2).\n(c) Splits by input family across the 60 trees of each fitted ranker. Columns represent trees and cells count splits, using a shared 0–7 colour scale. Multi-window trees use both base predictions and the window identifier as splitting variables. Source data give the family assignments and counts.\nThe learning schematic was drawn with Python/SVG code assisted by OpenAI Codex (OpenAI). Tree topologies and counts were read from the fitted models.'

if __name__=='__main__':
    f,h,checks=validate_source_data(ROOT,2);h.update(extra_checks(ROOT))
    comp=pd.read_csv(ROOT/'data/component_captures.csv')
    assert len(comp)==25 and np.allclose(comp.capture_percent,comp.captured/comp.events*100)
    model=json.loads((ROOT/'data/fitted_model_structure.json').read_text())
    assert model['tree_count']==len(model['trees'])==len(model['accepted_steps'])==60
    grouped=pd.read_csv(ROOT/'data/grouped_tree_splits.csv')
    raw=pd.read_csv(ROOT/'data/tree_feature_usage.csv')
    mapping=pd.read_csv(ROOT/'data/feature_family_mapping.csv')
    expanded=raw[raw.method.isin(['single_window','multi_window'])].merge(mapping,on='feature',validate='many_to_one')
    expected=expanded.groupby(['method','tree','feature_group']).split_count.sum()
    for r in grouped.itertuples():assert r.split_count==int(expected.get((r.method,r.tree,r.feature_group),0))
    assert grouped.groupby('method').split_count.sum().to_dict()=={'multi_window':405,'single_window':392}
    rd=pd.read_csv(ROOT/'data/selection_round.csv');meta=json.loads((ROOT/'data/selection_round_metadata.json').read_text())
    assert len(rd)==meta['candidates']==2805 and rd.hard_selected.sum()==meta['quota']==281
    assert rd.multi_window.is_monotonic_decreasing and not rd.candidate_row_key.duplicated().any()
    assert rd['rank'].tolist()==list(range(1,2806))
    assert np.allclose(rd.rank_pct,rd['rank']/2805*100)
    assert np.array_equal(rd.hard_selected,(rd['rank']<=281).astype(int))
    checks.update(actual_model_trees=60,all_1080_split_map_cells_reconciled=True,
        full_first_saved_tree_and_leaf_values=True,actual_2805_candidates_and_281_slots=True,
        five_window_differences_verified_as_audit_data=True,two_main_text_equations=True)
    finish(draw(f,comp,model),ROOT,2,CAPTION,checks,h)
