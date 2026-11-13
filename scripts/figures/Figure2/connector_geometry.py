"""Anchored, routed connector components for scientific vector figures.

All distances are final-page millimetres. A connector is one FancyArrowPatch
with a continuous route; its head is part of that artist, never a loose marker.
Source and target ports are derived from node geometry. Statistical markers
are clipped in display coordinates, so unequal axes and log axes are handled.
"""
from dataclasses import dataclass
from pathlib import Path as FilePath
import json, math
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.markers import MarkerStyle
from matplotlib.patheffects import AbstractPathEffect

MM_PER_PT=25.4/72


class _SolidArrowhead(AbstractPathEffect):
    def draw_path(self,renderer,gc,path,affine,rgbFace=None):
        if rgbFace is not None and path.codes is not None and Path.CLOSEPOLY in path.codes:
            head_gc=renderer.new_gc()
            head_gc.copy_properties(gc)
            head_gc.set_dashes(0,None)
            head_gc.set_capstyle('butt')
            head_gc.set_joinstyle('miter')
            renderer.draw_path(head_gc,path,affine,rgbFace)
            head_gc.restore()
        else:
            renderer.draw_path(gc,path,affine,rgbFace)


@dataclass(frozen=True)
class Node:
    key:str
    x:float
    y:float
    width:float
    height:float
    kind:str='rect'
    def port(self,side,offset=0):
        cx,cy=self.x+self.width/2,self.y+self.height/2
        if side in ['left','right']:
            dy=offset
            dx=self.width/2 if self.kind!='ellipse' else self.width/2*math.sqrt(max(0,1-(dy/(self.height/2))**2))
            return np.array([cx+(-dx if side=='left' else dx),cy+dy]),np.array([-1.,0.] if side=='left' else [1.,0.])
        dx=offset
        dy=self.height/2 if self.kind!='ellipse' else self.height/2*math.sqrt(max(0,1-(dx/(self.width/2))**2))
        return np.array([cx+dx,cy+(-dy if side=='top' else dy)]),np.array([0.,-1.] if side=='top' else [0.,1.])

def _rounded_path(points,radius=.65):
    points=np.asarray(points,dtype=float)
    if len(points)<2:raise ValueError('A connector requires two distinct points')
    pts=[points[0]]
    for p in points[1:]:
        if np.linalg.norm(p-pts[-1])>1e-8:pts.append(p)
    vertices=[pts[0]];codes=[Path.MOVETO]
    for i in range(1,len(pts)-1):
        prev,corner,nxt=pts[i-1:i+2]
        before=prev-corner;after=nxt-corner
        lb,la=np.linalg.norm(before),np.linalg.norm(after)
        d=min(radius,lb*.35,la*.35)
        if d<1e-6:
            vertices.append(corner);codes.append(Path.LINETO);continue
        vertices.extend([corner+before/lb*d,corner,corner+after/la*d])
        codes.extend([Path.LINETO,Path.CURVE3,Path.CURVE3])
    vertices.append(pts[-1]);codes.append(Path.LINETO)
    return Path(np.asarray(vertices),codes)

def _registry(fig):
    if not hasattr(fig,'_connector_components'):fig._connector_components=[]
    return fig._connector_components

def component(fig,points,*,key,scope='diagram',color='#4D4D4D',lw=.75,
              head=True,double=False,dashed=False,radius=.65,zorder=2,
              source=None,target=None):
    w,h=fig.get_size_inches()*25.4
    mm_path=_rounded_path(points,radius)
    verts=mm_path.vertices/np.array([w,-h])+np.array([0,1])
    route=Path(verts,mm_path.codes)
    style='<|-|>' if double else '-|>' if head else '-'
    artist=FancyArrowPatch(path=route,transform=fig.transFigure,
        arrowstyle=style,mutation_scale=8.5,linewidth=lw,color=color,
        linestyle=(0,(3.2,2.2)) if dashed else '-',
        capstyle='round',joinstyle='round',clip_on=False,zorder=zorder)
    if dashed and (head or double):
        artist.set_path_effects([_SolidArrowhead()])
    artist.set_gid('connector-'+key)
    fig.add_artist(artist)
    _registry(fig).append({'key':key,'scope':scope,'source':source,'target':target,
        'points_mm':np.asarray(points,dtype=float).tolist(),
        'rounding_mm':radius,'linewidth_pt':lw,'single_component':True,
        'arrowhead':style,'dashed':dashed})
    return artist

class Scene:
    def __init__(self,fig,scope):
        self.fig,self.scope=fig,scope
        self.nodes={}
    def node(self,key,x,y,w,h,kind='rect'):
        n=Node(key,x,y,w,h,kind);self.nodes[key]=n
        if not hasattr(self.fig,'_connector_nodes'):self.fig._connector_nodes=[]
        self.fig._connector_nodes.append((self.scope,n))
        return n
    def connect(self,key,source,target,*,s_side='right',t_side='left',
                s_offset=0,t_offset=0,via=(),gap=.23,**kwargs):
        a,an=self.nodes[source].port(s_side,s_offset)
        b,bn=self.nodes[target].port(t_side,t_offset)
        pts=[a+an*gap,*via,b+bn*gap]
        return component(self.fig,pts,key=key,scope=self.scope,
            source=source,target=target,**kwargs)

def _marker_radius(marker,size_pt,direction):
    m=MarkerStyle(marker)
    polygons=m.get_path().transformed(m.get_transform()).to_polygons()
    u=np.asarray(direction,dtype=float);u/=np.linalg.norm(u)
    candidates=[]
    for polygon in polygons:
        for a,b in zip(polygon,np.vstack([polygon[1:],polygon[:1]])):
            mat=np.column_stack([u,-(b-a)])
            if abs(np.linalg.det(mat))<1e-10:continue
            r,t=np.linalg.solve(mat,a)
            if r>=0 and 0<=t<=1:candidates.append(r*size_pt*MM_PER_PT)
    return min(candidates) if candidates else size_pt/2*MM_PER_PT

def metric_arrow(ax,start,end,*,key,marker='o',size_pt=5,color='#606060',lw=.85,double=False):
    # Resolve after final axis limits, aspect and fonts are established.
    fig=ax.figure
    if not hasattr(fig,'_pending_metric_connectors'):fig._pending_metric_connectors=[]
    fig._pending_metric_connectors.append((ax,start,end,dict(key=key,marker=marker,size_pt=size_pt,color=color,lw=lw,double=double)))

def _metric_arrow_now(ax,start,end,*,key,marker='o',size_pt=5,color='#606060',lw=.85,double=False):
    """Attach a directional connector to statistical-marker boundaries."""
    fig=ax.figure;w,h=fig.get_size_inches()*25.4
    def to_mm(xy):
        xy=fig.transFigure.inverted().transform(ax.transData.transform(xy))
        return np.array([xy[0]*w,(1-xy[1])*h])
    a,b=to_mm(start),to_mm(end)
    v=b-a;length=np.linalg.norm(v)
    if length<1e-7:return None  # coincident summaries have no direction
    u=v/length
    # MarkerStyle uses y-up; page coordinates use y-down.
    ra=_marker_radius(marker,size_pt,[u[0],-u[1]])+.18
    rb=_marker_radius(marker,size_pt,[-u[0],u[1]])+.18
    if length<=ra+rb+.8:return None
    return component(fig,[a+u*ra,b-u*rb],key=key,scope='data-vector',
        color=color,lw=lw,double=double,radius=0,zorder=2,
        source={'data':list(map(float,start)),'marker':marker},
        target={'data':list(map(float,end)),'marker':marker})

def _proper_cross(a,b,c,d):
    a,b,c,d=map(np.asarray,[a,b,c,d])
    def cross(x,y):return x[0]*y[1]-x[1]*y[0]
    ab,cd=b-a,d-c;den=cross(ab,cd)
    if abs(den)<1e-9:return False
    t,s=cross(c-a,cd)/den,cross(c-a,ab)/den
    return 1e-7<t<1-1e-7 and 1e-7<s<1-1e-7

def audit_connectors(fig,root):
    for ax,start,end,kwargs in getattr(fig,'_pending_metric_connectors',[]):
        _metric_arrow_now(ax,start,end,**kwargs)
    fig._pending_metric_connectors=[]
    rows=_registry(fig);crossings=[];intrusions=[]
    w,h=fig.get_size_inches()*25.4
    for row in rows:
        pts=np.asarray(row['points_mm'])
        assert np.all(pts[:,0]>=0) and np.all(pts[:,0]<=w) and np.all(pts[:,1]>=0) and np.all(pts[:,1]<=h),(row['key'],pts)
        for scope,node in getattr(fig,'_connector_nodes',[]):
            if scope!=row['scope'] or node.key in [row['source'],row['target']]:continue
            for a,b in zip(pts,pts[1:]):
                samples=np.linspace(a,b,max(3,int(np.linalg.norm(b-a)/.05)))
                x,y=samples[:,0],samples[:,1]
                inside=((x>node.x+.03)&(x<node.x+node.width-.03)&
                        (y>node.y+.03)&(y<node.y+node.height-.03))
                if inside.any():intrusions.append([row['key'],node.key])
    # Within each logical schematic, non-adjacent routes must not cross.
    # Cross-scope intersections include intentional tree/bus junctions and
    # quantitative vector crossings, and require the accompanying visual review.
    for i,a in enumerate(rows):
        if a['scope']=='data-vector':continue
        for b in rows[i+1:]:
            if a['scope']!=b['scope']:continue
            for p,q in zip(a['points_mm'],a['points_mm'][1:]):
                for r,s in zip(b['points_mm'],b['points_mm'][1:]):
                    if _proper_cross(p,q,r,s):crossings.append([a['key'],b['key']])
    result={'components':rows,'count':len(rows),'unrelated_crossings_within_schematic':crossings,
        'unrelated_node_intrusions':intrusions,
        'node_port_policy':'Geometry-derived edges; 0.23 mm diagram gap. Statistical markers clipped using display-space marker boundaries.',
        'construction':'One FancyArrowPatch per entire routed connector; no loose triangular arrowhead artists.'}
    FilePath(root,'connector_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    assert not crossings,crossings
    assert not intrusions,intrusions
    return result
