"""Technical graphical abstract: GIS context, model mechanisms and target refitting.

Install: python -m pip install pyproj resvg-py Pillow fonttools numpy scipy shapely
Run:     python scripts/render_figures.py GA
Outputs: GA.png, GA.svg and GA.tif; vector elements and text remain editable.

The source data include all 2,844 lake coordinates and geographic boundaries,
the GA endpoint tables, an elevation grid and contextual hydrography.
Map data and numerical results are empirical.
Candidate cells, model trees, event credit and threshold traces are schematics.
The threshold traces illustrate the event rules.
Source-domain refitting is distinct from the Chinese target-set evaluation.

Layout: 180 x 72 mm (2.5:1); raster: 6000 x 2400 px, 846.67 dpi.

Terrain: USGS elevation data via Mapzen/AWS Terrain Tiles, accessed 2026-11-03.
US terrain data courtesy of the U.S. Geological Survey. The US elevation grid is
reprojected from Web Mercator tiles at z=6 to EPSG:5070 with bilinear resampling,
rounded to integer metres and shaded from NW, W and NE with vertical exaggeration.
Colour encodes elevation, not event risk; hillshade only improves terrain reading.
USGS public-domain terrain material is not subject to US copyright protection.
Contextual coastlines and lakes: Natural Earth 1:50m, public domain.
https://registry.opendata.aws/terrain-tiles/
https://github.com/tilezen/joerd/blob/master/docs/attribution.md
https://www.naturalearthdata.com/about/terms-of-use/

SVG: the elevation base is an embedded raster. Lake points, outlines, graticules,
symbols, model geometry, soft shadow layers, legends and Times New Roman text
are editable vectors.
"""
from pathlib import Path
from io import BytesIO, StringIO
from html import escape
from math import sin, cos, pi
import os, csv, json, base64, zlib, re

import numpy as np
from scipy.ndimage import gaussian_filter
import pyproj
import resvg_py
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont
from fontTools import subset
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, substring, unary_union

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'data/figures/GA'
DATA = json.loads((ASSETS / 'study.json').read_text(encoding='utf-8'))
CAPTION_EN = 'Every fortnight, 10% of eligible lakes are selected by score. An event is covered if its lake is selected on at least one eligible occasion 31–60 days before the first high record. In the retrospective 2021–2025 evaluation of 2,844 US lakes, Single-window covered 461/593 events (77.7%) and Multi-window covered 463/593 (78.1%). Multi-window uses auxiliary lead windows during shared-tree training; the 31–60-day output determines the monitoring list. Relative-target refitting rebuilds labels and threshold-dependent inputs in the source region before refitting. On the same 718 Chinese relative events during 2020–2022, Multi-window coverage rises from 62 to 349 events. Lake A/B traces and conceptual trees are schematic. Map lines delineate study areas and do not necessarily depict accepted national boundaries.'
OUT = ROOT / 'figures'
FONT_DIR = Path(os.environ.get('BIOMASS_FONT_DIR', str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts')))
FONT_PATHS = [FONT_DIR/'times.ttf', FONT_DIR/'timesbd.ttf', FONT_DIR/'timesi.ttf']
for file in FONT_PATHS:
    if not file.exists(): raise RuntimeError('Times New Roman regular, bold and italic fonts are required.')

C = dict(ink='#111111', blue='#2166AC', orange='#D95F02', green='#008837',
         red='#B2182B', grey='#636363', mid='#969696', border='#777777',
         light='#E8E8E8', paper='#F4F4F4')
SHADOW_STRENGTH = 1.0  # 0: flat; 1: light presentation-style depth.
parts, labels = [], []


def emit(s): parts.append(s)


def group(attrs=''): emit('<g '+attrs+'>')


def end(): emit('</g>')


def path(d, fill='none', stroke='none', sw=1, **attrs):
    extra=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in attrs.items())
    emit(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
         f'stroke-linejoin="miter" stroke-linecap="butt" {extra}/>')


def line(x1,y1,x2,y2,color=None,sw=2,**attrs):
    path(f'M{x1},{y1} L{x2},{y2}',stroke=color or C['ink'],sw=sw,**attrs)


def rect(x,y,w,h,fill='white',stroke=None,sw=2,**attrs):
    extra=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in attrs.items())
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
         f'stroke="{stroke or C["ink"]}" stroke-width="{sw}" {extra}/>')


def card_shadow(x,y,w,h,strength=1,dx=4,dy=5):
    """Draw a lower-right shadow with nested vector rectangles.

    SHADOW_STRENGTH controls the shadow opacity; zero disables it.
    """
    if strength <= 0 or SHADOW_STRENGTH <= 0:
        return
    group('class="editable-shadow" aria-hidden="true"')
    for spread,alpha in [(3,.010),(2.2,.016),(1.3,.025),(.5,.034),(0,.060)]:
        rect(x+dx-spread,y+dy-spread,w+2*spread,h+2*spread,
             '#343A40','none',0,opacity=round(alpha*strength*SHADOW_STRENGTH,5))
    end()


def card(x,y,w,h,fill='white',stroke=None,sw=2,shadow=1,**attrs):
    """Draw a module with a vector shadow."""
    card_shadow(x,y,w,h,shadow)
    rect(x,y,w,h,fill,stroke,sw,**attrs)


def circle(x,y,r,fill,stroke='none',sw=1,**attrs):
    extra=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in attrs.items())
    emit(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def polygon(points,fill,stroke='none',sw=1):
    emit('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+
         f'" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x,y,s,size=44,weight='normal',color=None,anchor='middle',italic=False,rotate=0):
    labels.append((s,x,y,size,weight,anchor,italic,rotate))
    transform=f' transform="rotate({rotate} {x} {y})"' if rotate else ''
    emit(f'<text x="{x}" y="{y}" font-family="Times New Roman" font-size="{size}" '
         f'font-weight="{weight}" font-style="{"italic" if italic else "normal"}" '
         f'text-anchor="{anchor}" fill="{color or C["ink"]}"{transform}>{escape(s)}</text>')


def arrow(d,color=None,sw=3,dash=None,branches=()):
    """Draw a continuous arrow outline with a solid head."""
    route=LineString(np.asarray(re.findall(r'[-+]?(?:\d*\.\d+|\d+)',d),float).reshape(-1,2))
    points=np.asarray(route.coords)
    direction=points[-1]-points[-2]
    direction/=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    head_length=min(12.,route.length*.45,np.linalg.norm(points[-1]-points[-2])*.8)
    head_width=max(sw*2.8,min(12.,head_length))
    tip=points[-1]
    base=tip-head_length*direction
    shapes=[Polygon([tip,base+normal*head_width/2,base-normal*head_width/2])]
    neck=route.length-head_length
    if dash:
        on,off=map(float,dash.split())
        end=neck
        while end>0:
            segment=substring(route,max(0.,end-on),min(route.length,end+sw*.5))
            shapes.append(segment.buffer(sw/2,cap_style='flat',join_style='mitre'))
            end-=on+off
    else:
        body=substring(route,0.,neck+head_length*.5)
        network=unary_union([body,*[LineString(branch) for branch in branches]])
        if network.geom_type!='LineString':
            network=linemerge(network)
        shapes.append(network.buffer(sw/2,cap_style='flat',join_style='mitre'))
    outline=unary_union(shapes)
    polygons=[outline] if outline.geom_type=='Polygon' else list(outline.geoms)
    contours=[]
    for polygon in polygons:
        for ring in [polygon.exterior,*polygon.interiors]:
            contours.append('M'+' L'.join(f'{x:.4f},{y:.4f}' for x,y in ring.coords)+' Z')
    path(' '.join(contours),fill=color or C['ink'],sw=0,fill_rule='evenodd',
         data_connector='arrow',data_line_style='dashed' if dash else 'solid')


def mark(x,y,kind,color,filled=True,size=11):
    fill=color if filled else 'white'
    if kind=='square':rect(x-size,y-size,2*size,2*size,fill,color,2.5)
    elif kind=='circle':circle(x,y,size,fill,color,2.5)
    else:polygon([(x,y-size*1.12),(x-size*1.12,y+size),(x+size*1.12,y+size)],fill,color,2.5)


def tree(x,y,scale=1,color=None):
    color=color or C['grey']
    for xa,ya,xb,yb in [(0,-34,-34,0),(0,-34,34,0),(-34,0,-51,34),
                        (-34,0,-17,34),(34,0,17,34),(34,0,51,34)]:
        line(x+xa*scale,y+ya*scale,x+xb*scale,y+yb*scale,color,2.7)
    for dx,dy in [(0,-34),(-34,0),(34,0),(-51,34),(-17,34),(17,34),(51,34)]:
        if dy==34:
            rect(x+dx*scale-4.6*scale,y+dy*scale-3.5*scale,
                 9.2*scale,7*scale,'white',color,1.7)
        else:circle(x+dx*scale,y+dy*scale,4.2*scale,color)


def shortlist(x,y,color,kind):
    card(x,y,168,96,'white',color,2.4,shadow=.95)
    rect(x+1,y+1,166,27,color,color,0,opacity=.075)
    line(x+35,y+1,x+35,y+95,'#C7C7C7',1)
    line(x+36,y+28,x+166,y+28,'#BFBFBF',.9)
    for i in range(4):
        yy=y+17+i*21
        mark(x+20,yy,kind,color if i==0 else '#A7A7A7',i==0,5.5)
        line(x+43,yy,x+142-i*13,yy,color if i==0 else '#BDBDBD',3 if i==0 else 2)
    # Display the leading ranks.


def rows(name):return list(csv.DictReader(StringIO(DATA[name])))


us={r['method_id']:r for r in rows('us_primary.csv')}
training=rows('china_relative_training.csv')[0]
reach=rows('china_relative_lake_reach.csv')[0]
locations=rows('lake_coordinates.csv')
assert len(locations)==len({r['lake_id'] for r in locations})==2844
assert all(int(r['evaluable_events'])==593 for r in us.values())
assert int(training['evaluable_events'])==int(reach['evaluable_events'])==718
assert int(training['relative_events_covered'])==int(reach['events_covered'])==349
assert (int(reach['event_lakes_covered']),int(reach['evaluable_event_lakes']))==(196,258)
for row in us.values():
    assert abs(100*int(row['events_covered'])/593-float(row['coverage_pct']))<1e-8

emit('<?xml version="1.0" encoding="UTF-8"?>')
emit('<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="72mm" '
     'viewBox="0 0 2400 960" role="img" aria-labelledby="title desc">')
emit('<title id="title">Event-oriented monitoring under fixed capacity</title>')
emit('<desc id="desc">US lake network and fortnightly capacity allocation; record classification, '
     'Single-window and Multi-window ranking mechanisms; event grouping and credit; '
     'historical lake-relative references and source-domain refitting evaluated in China.</desc>')
emit('<metadata id="figure-caption">'+escape(CAPTION_EN)+'</metadata>')
emit('<metadata>Terrain: USGS elevation courtesy of the U.S. Geological Survey, via '
     'Mapzen/AWS Terrain Tiles; US public-domain material. Natural Earth 1:50m '
     'context geography, public domain. Terrain colour is elevation, not event risk. '
     'All 2,844 study-lake coordinates are retained; no schematic event locations are mapped.</metadata>')
emit('<defs>')
emit('<clipPath id="map-clip"><rect x="42" y="165" width="490" height="290"/></clipPath>')
emit('<pattern id="lead-hatch" width="10" height="10" patternUnits="userSpaceOnUse">'
     '<path d="M0,10 L10,0" stroke="#BCD0E5" stroke-width="1.3"/></pattern>')
emit('</defs>')
rect(0,0,2400,960,'white','none',0)

# Section headings.
text(285,80,'Limited capacity',56,'bold')
text(1190,80,'Event-oriented ranking',55,'bold')
text(2090,80,'Lake-relative targets',54,'bold')

# Study area and monitoring quota.
outline=json.loads(DATA['us_outline.geojson'])
states=json.loads(DATA['us_states.geojson'])
projection=pyproj.Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)

def rings(geom):
    if geom['type']=='Polygon':return geom['coordinates']
    if geom['type']=='MultiPolygon':return [r for p in geom['coordinates'] for r in p]
    raise ValueError(geom['type'])

projected=[projection.transform(lon,lat) for f in outline['features'] for r in rings(f['geometry']) for lon,lat in r]
xmin,xmax=min(x for x,y in projected),max(x for x,y in projected)
ymin,ymax=min(y for x,y in projected),max(y for x,y in projected)
map_scale=min(470/(xmax-xmin),270/(ymax-ymin))
map_cx,map_cy=287,310

def screen(lon,lat):
    x,y=projection.transform(lon,lat)
    return map_cx+(x-(xmin+xmax)/2)*map_scale, map_cy-(y-(ymin+ymax)/2)*map_scale

def geo_path(geometry):
    d=''
    for ring in rings(geometry):
        xy=[screen(lon,lat) for lon,lat in ring]
        d+='M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in xy)+' Z '
    return d

card(42,165,490,290,'#D7EAF0',C['ink'],2.2,shadow=.6)
group('clip-path="url(#map-clip)"')
context=json.loads((ASSETS / 'geography.json').read_bytes().decode('utf-8'))
for geom in context['land']:
    path(geo_path(geom),'#EAE8DD','#999B92',.65,fill_rule='evenodd')

# Raster elevation with vector overlays.
terrain=np.load(BytesIO((ASSETS / 'terrain.npz').read_bytes()))
elevation=terrain['elevation'].astype(np.float32)
left,bottom,right,top=terrain['bounds'].astype(float)
elevation=np.maximum(elevation,0)
breaks=np.array([0,200,500,1000,1600,2400,3200,4000,5000],np.float32)
terrain_colors=np.array([[166,194,117],[186,204,130],[210,214,142],
                        [226,205,126],[202,174,109],[170,133,85],
                        [177,163,143],[225,219,205],[247,244,233]],np.float32)
rgb=np.stack([np.interp(elevation,breaks,terrain_colors[:,i]) for i in range(3)],axis=-1)
dy=(top-bottom)/elevation.shape[0]
dx=(right-left)/elevation.shape[1]
smooth=gaussian_filter(elevation,.65)
gy,gx=np.gradient(smooth,dy,dx)
nx,ny,nz=-gx*6.5,-gy*6.5,np.ones_like(gx)
normal=np.sqrt(nx*nx+ny*ny+nz*nz)
shade=np.zeros_like(elevation)
for light,weight in [((-.58,-.58,.57),.60),((-.82,-.15,.55),.25),((.28,-.78,.56),.15)]:
    dot=(nx*light[0]+ny*light[1]+nz*light[2])/normal
    shade+=weight*np.clip(dot,0,1)
rgb=np.clip(rgb*(.74+.45*shade[...,None]),0,255).astype(np.uint8)
terrain_png=BytesIO();Image.fromarray(rgb).save(terrain_png,format='PNG',optimize=True)
terrain_encoded=base64.b64encode(terrain_png.getvalue()).decode('ascii')
usa_d=' '.join(geo_path(f['geometry']) for f in outline['features'])
emit('<defs><clipPath id="terrain-us-clip"><path d="'+usa_d+'"/></clipPath></defs>')
rx=map_cx+(left-(xmin+xmax)/2)*map_scale
ry=map_cy-(top-(ymin+ymax)/2)*map_scale
rw,rh=(right-left)*map_scale,(top-bottom)*map_scale
emit(f'<image x="{rx:.4f}" y="{ry:.4f}" width="{rw:.4f}" height="{rh:.4f}" '
     f'preserveAspectRatio="none" clip-path="url(#terrain-us-clip)" '
     f'href="data:image/png;base64,{terrain_encoded}"/>')

# Great Lakes and reference hydrography are drawn above the shaded terrain.
for geom in context['lakes']:
    path(geo_path(geom),'#D7EAF0','#6B9EB1',.65,fill_rule='evenodd')
for lon in (-120,-110,-100,-90,-80,-70):
    xy=[screen(lon,24+i*.4) for i in range(70)]
    path('M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in xy),stroke='#889792',sw=.75,stroke_dasharray='2.5 4',opacity=.72)
for lat in (25,30,35,40,45,50):
    xy=[screen(-126+i*.6,lat) for i in range(101)]
    path('M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in xy),stroke='#889792',sw=.75,stroke_dasharray='2.5 4',opacity=.72)
for f in states['features']:
    path(geo_path(f['geometry']),stroke='#68715E',sw=.75,fill_rule='evenodd')
for f in outline['features']:
    path(geo_path(f['geometry']),stroke='#4F574C',sw=1.7)
for row in locations:
    x,y=screen(float(row['longitude_deg']),float(row['latitude_deg']))
    circle(x,y,1.65,'none','#315988',.83,opacity=.83)
end()
# North arrow and metric scale bar.
text(76,204,'N',31,'bold')
polygon([(76,213),(68,244),(76,236),(84,244)],C['ink'])
scale_x,scale_y,scale_w=74,430,1_000_000*map_scale
rect(scale_x-8,scale_y-15,scale_w+20,29,'white','none',0,opacity=.9)
rect(scale_x,scale_y,scale_w/2,5,C['ink'],C['ink'],.8)
rect(scale_x+scale_w/2,scale_y,scale_w/2,5,'white',C['ink'],.8)
text(scale_x+scale_w/2,scale_y-4,'1000 km',25)
circle(92,537,5,'none','#315988',1.8)
text(116,550,'US · 2,844 lakes',45,anchor='start')

# Elevation legend.
emit('<defs><linearGradient id="elevation-key" x1="0" y1="0" x2="1" y2="0">')
for e,colour in zip(breaks,terrain_colors):
    if e>4000:break
    color='#'+''.join(f'{int(v):02x}' for v in colour)
    emit(f'<stop offset="{e/4000:.5f}" stop-color="{color}"/>')
emit('</linearGradient></defs>')
rect(64,465,222,10,'url(#elevation-key)','#555555',.8)
text(60,498,'0',25,anchor='end')
text(179,498,'2000',25)
text(295,498,'4000 m',25,anchor='start')
text(520,475,'Elevation (m)',26,anchor='end')
for tick in (0,2000,4000):line(64+222*tick/4000,475,64+222*tick/4000,480,'#555555',.8)

# Map -> eligible pool -> quota, with the cadence written on the return path.
# Ten schematic cells give one selected slot; actual k(t)=ceil[0.10*N(t)].
arrow('M285,575 L285,580 L178,580 L178,650',C['ink'],2.6)
for i in range(10):
    x,y=66+(i%5)*48,668+(i//5)*45
    rect(x,y,32,29,'#D9E5F0' if i==6 else 'white',C['blue'] if i==6 else '#777777',2)
    if i==6:line(x+6,y+15,x+26,y+15,C['blue'],4)
text(348,671,'10%',47,'bold',C['blue'])
arrow('M318,701 L374,701',C['blue'],3)
card(405,670,66,60,'white',C['blue'],3,shadow=.8)
path('M420,701 L433,713 L457,684',stroke=C['blue'],sw=4)
text(175,792,'Eligible N',43)
text(438,792,'Top k',43,color=C['blue'])
arrow('M438,817 L438,855 L175,855 L175,817',C['ink'],2.4)
rect(257,834,72,44,'white','none',0)
text(293,865,'14 d',43)

# Classification and event-coverage models.
# Model inputs use the available observations.
card(619,361,151,119,'white','#666666',2.2,shadow=.95)
rect(632,374,125,15,'#D9D9D9','none',0)
for row in range(4):
    for col in range(4):
        if (row,col) in [(1,2),(3,0)]:
            rect(633+col*31,400+row*17,23,10,'white','#A4A4A4',.8)
        else:rect(633+col*31,400+row*17,23,10,'#A8A8A8' if (row+col)%3==0 else '#E5E5E5','none',0)
circle(753,366,14,'white','#666666',1.8)
line(753,357,753,366,'#666666',1.7)
line(753,366,760,371,'#666666',1.7)
text(694,540,'Available',41)
text(694,581,'history',41)
arrow('M772,420 L815,420 L815,261 L861,261',C['grey'],3,
      branches=[[(815,420),(815,717.5)]])
arrow('M815,477 L861,477',C['blue'],3)
arrow('M815,716 L850,716',C['orange'],3)

# Conventional record-risk classification, followed by quota selection.
text(1040,185,'XGBoost / Elastic Net',48,color=C['grey'])
card(875,214,330,95,'#F3F3F3','#777777',1.8,shadow=.95)
tree(959,262,.91,C['grey'])
line(1090,228,1090,298,C['grey'],2.2)
for yy,x1,x2 in [(240,1090,1138),(262,1052,1090),(286,1090,1123)]:
    line(x1,yy,x2,yy,C['grey'],4)
arrow('M1207,261 L1246,261',C['grey'],3)
text(1360,275,'Record risk',44,color=C['grey'])
arrow('M1475,261 L1494,261',C['grey'],3,dash='9 6')
mark(1518,261,'square',C['grey'])
arrow('M1538,261 L1580,261',C['grey'],3,dash='9 6')
text(1674,185,'Top k lakes',45)
shortlist(1590,214,C['grey'],'square')

# One-window event/capacity ranking.
text(997,401,'Single-window',49,color=C['blue'])
card(875,430,235,95,'#E8EFF6',C['blue'],2.1,shadow=.9)
text(992,493,'31–60 d',45)
line(892,513,1094,513,'#B1C3D6',1)
line(993,513,1094,513,C['blue'],3)
for tick in (892,993,1094):line(tick,510,tick,517,C['blue'],1.3)
arrow('M1112,477 L1177,477',C['blue'],3.5)
text(1315,401,'Event ranking',44)
card(1190,430,250,95,'#F4F4F4','#666666',1.8,shadow=.95)
tree(1315,478,.94,C['grey'])
arrow('M1442,477 L1490,477',C['blue'],3.5)
mark(1518,477,'circle',C['blue'])
arrow('M1538,477 L1580,477',C['blue'],3.5)
shortlist(1590,430,C['blue'],'circle')

# Multi-window predictions, shared trees, two auxiliary heads and one output list.
text(997,607,'Multi-window',49,color=C['orange'])
for i,(yy,name) in enumerate([(626,'1–14 d'),(678,'15–30 d'),(730,'31–60 d')]):
    xx=905-20*i
    card_shadow(xx,yy,224,48,.6,dx=3,dy=3.5)
    rect(xx,yy,224,48,'white','#777777',1.8)
    line(xx,yy+1,xx+224,yy+1,C['orange'],3)
    text(xx+112,yy+35,name,40)
    line(xx+13,yy+42,xx+211,yy+42,'#C8C8C8',.9)
    lo,hi=[(0,14),(14,30),(30,60)][i]
    line(xx+13+198*lo/60,yy+42,xx+13+198*hi/60,yy+42,C['orange'],2.8)
arrow('M1154,716 L1180,716',C['orange'],2.2,branches=[
    [(1131,650),(1154,650),(1154,754)],
    [(1111,702),(1154,702)],[(1091,754),(1154,754)]])
card(1190,671,250,95,'#F4F4F4','#666666',1.8,shadow=.95)
tree(1294,718,.94,C['grey'])
line(1385,718,1475,650,'#969696',2.3,stroke_dasharray='4 5')
line(1385,718,1475,766,'#969696',2.3,stroke_dasharray='4 5')
circle(1475,650,7,'white','#8E8E8E',2)
circle(1475,766,6,'white','#8E8E8E',2)
text(1337,604,'Auxiliary',36,color=C['grey'])
text(1337,644,'training only',33,color=C['grey'])
arrow('M1442,716 L1490,716',C['orange'],3.6)
text(1523,688,'31–60 d',37,color=C['orange'])
mark(1518,716,'triangle',C['orange'])
arrow('M1538,716 L1580,716',C['orange'],3.6)
shortlist(1590,671,C['orange'],'triangle')
text(1315,812,'Shared trees',40)
text(997,812,'Forecasts',37,italic=True)
text(1657,564,'US · 77.7% (461/593)',35,'bold',C['blue'])
text(1657,803,f"US · {float(us['multi_window']['coverage_pct']):.1f}% (463/593)",35,'bold',C['orange'])

# A single integrated timeline defines selection, event grouping and event credit.
text(608,871,'Covered event',43,'bold',anchor='start')
text(608,925,'≥1 pre-event selection',37,anchor='start')
rect(1050,878,203,46,'url(#lead-hatch)',C['blue'],1.3)
text(1152,864,'31–60 d',38,color=C['blue'])
arrow('M1028,909 L1770,909',C['ink'],2.1)
for x in (1080,1185):
    circle(x,909,7,C['blue'])
    line(x,884,x,902,C['blue'],1.8)
text(1142,949,'selections',33,color=C['blue'])
for x in (1470,1570,1670):
    line(x,885,x,909,C['red'],1.8)
    mark(x,884,'triangle',C['red'],True,6.5)
path('M1470,874 L1470,861 L1670,861 L1670,874',stroke=C['red'],sw=2)
text(1570,850,'gaps ≤60 d',35,color=C['red'])
line(1470,909,1470,929,C['ink'],1.4)
text(1470,949,'0 = first high',33)
text(1772,949,'1 event',33,'bold',anchor='end')

# RIGHT: one coordinate system carries both lake backgrounds and the shared cutoff.
# Schematic threshold rules.
text(1843,300,'Chl-a (μg/L)',37,rotate=-90)
text(2090,132,'Lakes A, B · schematic',33,italic=True)
def yy(value):return 405-value*2.48
rect(2220,170,152,235,'#F3F3F3','none',0)
line(2220,170,2220,405,'#B2B2B2',1,stroke_dasharray='3 5')
line(1875,405,2372,405,C['ink'],1.8)
line(1875,170,1875,405,C['ink'],1.8)
for background,q,peak,label,peak_x in [(15,22,37,'q90,A',2288),(44,51,66,'q90,B',2334)]:
    pts=[]
    for i in range(28):
        x=1886+i*11.9
        val=background+2.4*sin(i*1.9)+1.1*cos(i*3.1)
        pts.append((x,yy(val)))
    pts.extend([(2220,yy(background+1)),(peak_x,yy(peak)),
                (min(2365,peak_x+24),yy(background+3)),(2370,yy(background))])
    path('M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in pts),stroke=C['ink'],sw=2.2)
    line(1875,yy(q),2372,yy(q),C['green'],2.6,stroke_dasharray='9 5')
    text(1883,yy(q)-10,label,38,color=C['green'],anchor='start',italic=True)
    mark(peak_x,yy(peak),'triangle',C['green'],True,6)
    line(peak_x,yy(peak)+10,peak_x,405,'#8EAF94',.8,stroke_dasharray='2 4')
line(1875,yy(75),2372,yy(75),C['red'],2.5,stroke_dasharray='12 6')
text(2355,yy(75)-14,'75 μg/L',38,color=C['red'],anchor='end')
text(2045,445,'pre-2011',36)
text(2300,445,'Evaluation',32)
arrow('M2045,460 L2045,485 L1940,485 L1940,558',C['green'],3)

# X and y show that threshold-dependent inputs and labels change before refitting.
card(1849,565,531,127,'white',C['green'],2.4,shadow=.9)
text(2114,552,'Source refit',47,'bold')
text(1900,608,'75',40,color=C['red'])
arrow('M1940,594 L1990,594',C['ink'],2.2)
text(2045,608,'q90',40,color=C['green'],italic=True)
for i in range(2):
    for j in range(3):rect(1870+j*27,626+i*16,21,9,C['green'] if j==2 else '#B9B9B9','none',0)
    rect(2007,626+i*16,24,9,C['green'],'none',0)
text(1930,682,'inputs',30)
text(2021,682,'labels',30)
arrow('M2050,644 L2165,644',C['green'],3)
tree(2252,635,.92,C['grey'])
arrow('M2114,695 L2114,718',C['green'],3)

# Shared brackets encode a fixed event cohort; the denominator is stated once.
text(2114,755,'China · 2020–2022',42)
text(1960,802,'Absolute training',34,color=C['grey'])
text(2255,802,'q90 training',34,color=C['green'])
text(1960,856,training['absolute_events_covered'],55,color=C['grey'])
text(2255,856,training['relative_events_covered'],55,'bold',C['green'])
arrow('M2030,835 L2166,835',C['green'],3.5)
path('M1888,868 L1888,880 L2342,880 L2342,868',stroke=C['grey'],sw=1.7)
text(2114,923,f"{training['evaluable_events']} relative events · same quota",35)

# Event-bearing lake coverage.
# Event-bearing lake reach is defined in the external caption.

# Embed editable Times New Roman subsets; only the elevation surface is raster.
font_css=[]
characters=''.join(item[0] for item in labels)
for (weight,style),file in zip([('normal','normal'),('bold','normal'),('normal','italic')],FONT_PATHS):
    font=TTFont(file,recalcTimestamp=False)
    opts=subset.Options();opts.recalc_timestamp=False;opts.drop_tables+=['meta']
    sub=subset.Subsetter(options=opts);sub.populate(text=characters);sub.subset(font)
    font.flavor='woff';stream=BytesIO();font.save(stream)
    encoded=base64.b64encode(stream.getvalue()).decode('ascii')
    font_css.append("@font-face{font-family:'Times New Roman';font-weight:"+weight+
                    ';font-style:'+style+';src:url(data:font/woff;base64,'+encoded+") format('woff');}")
emit('<defs><style type="text/css">'+''.join(font_css)+'</style></defs>')
emit('</svg>')
svg='\n'.join(parts)
(OUT/'GA.svg').write_text(svg,encoding='utf-8')
render_svg=svg.replace('width="180mm" height="72mm"','width="2400" height="960"',1)
png=resvg_py.svg_to_bytes(svg_string=render_svg,width=6000,height=2400,dpi=96,
                         background='white',font_files=[str(p) for p in FONT_PATHS],
                         font_family='Times New Roman',text_rendering='optimize_legibility')
with Image.open(BytesIO(png)) as im:im.save(OUT/'GA.png',dpi=(6000/180*25.4,6000/180*25.4),optimize=True)
print('Created GA.svg (180 x 72 mm) and GA.png (6000 x 2400 px).')
print('All figure text uses Times New Roman; all 2,844 coordinates retained.')
