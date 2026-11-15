"""Explicit semantic stroke topology and its annotated exports.

This module is a *post-processing and annotation layer* on the boundary-based
centerline reconstruction. It distinguishes initial skeleton segments from
continuous inferred pen strokes. Stroke continuity at a junction is defined by
reconstruction's boundary/tangent endpoint pairing, not by mere pixel contact.
It does not infer draw order, over/under, or actual pen-lift history.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np


PALETTE = ['#1769AA', '#D24A32', '#2A916B', '#8A50B7', '#CB801B',
           '#0E8B9A', '#AD4C82', '#606F19', '#A3484A', '#3F6FB9']


def _merge(parts):
    assembled = []
    for part in parts:
        p = np.asarray(part, dtype=float)
        if not len(p):
            continue
        if assembled and np.linalg.norm(assembled[-1][-1] - p[0]) < 1.5:
            p = p[1:]
        if len(p):
            assembled.append(p)
    return np.concatenate(assembled) if assembled else np.empty((0, 2), dtype=float)


def _part_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.


def assemble_strokes(pieces: dict, bridge_lines: list, bridge_records: list,
                     initial_segments: list, graph: dict, image_shape: tuple):
    """Stitch elementary geometries to inferred strokes and emit graph metadata.

    bridge_records entries: junction, kind, endpoints=((sid,side), ...) ;
    for extrapolate: endpoints contains only the originating terminal.

    An inferred stroke is a maximal path through *paired* skeleton endpoints;
    extrapolated fragments remain attached to their own stroke. All junctions
    are recorded separately even when a stroke passes smoothly through one.
    """
    by_sid = {int(s.sid): s for s in initial_segments}
    active_ids = sorted(pieces)
    matching = {}  # terminal -> (opposite terminal, bridge index)
    tails = {}
    for i, rec in enumerate(bridge_records):
        terminals = [tuple(a) for a in rec['endpoints']]
        if len(terminals) == 2:
            a, b = terminals
            if a not in matching and b not in matching:
                matching[a] = (b, i)
                matching[b] = (a, i)
        elif len(terminals) == 1:
            tails[terminals[0]] = i

    paths = []
    traces = []
    visited = set()

    def walk(start_sid, start_side):
        sid, side = start_sid, start_side
        components = []
        sid_path = []
        bridge_path = []
        is_cycle = False
        for _ in range(len(active_ids) + 2):
            if sid in visited:
                is_cycle = (sid == start_sid)
                break
            visited.add(sid)
            p = np.asarray(pieces[sid])
            if (sid, side) in tails:
                # An extrapolation points from the cutoff to the junction.
                b = tails[(sid, side)]
                components.append(bridge_lines[b][::-1])
                bridge_path.append(b)
            components.append(p if side == 0 else p[::-1])
            sid_path.append(sid)
            other = (sid, 1-side)
            if other in matching:
                (next_sid, next_side), b = matching[other]
                rec = bridge_records[b]
                oriented = np.asarray(bridge_lines[b])
                if tuple(rec['endpoints'][0]) != other:
                    oriented = oriented[::-1]
                components.append(oriented)
                bridge_path.append(b)
                if next_sid in visited:
                    is_cycle = (next_sid == start_sid)
                    break
                sid, side = next_sid, next_side
            else:
                if other in tails:
                    b = tails[other]
                    components.append(bridge_lines[b])
                    bridge_path.append(b)
                break
        return _merge(components), sid_path, sorted(set(bridge_path)), is_cycle

    # Begin at terminals not paired to another source segment, so orientation
    # and the chain ends are well defined. Remaining components are cycles.
    for sid in active_ids:
        if sid in visited:
            continue
        seg = by_sid[sid]
        if seg.closed:
            visited.add(sid)
            paths.append(np.asarray(pieces[sid]).copy())
            traces.append({'segment_ids':[sid], 'bridge_ids':[], 'closed':True})
            continue
        free_sides = [side for side in [0, 1] if (sid, side) not in matching]
        if free_sides:
            p, chain, bridges, closed = walk(sid, free_sides[0])
            if len(p) > 1:
                paths.append(p)
                traces.append({'segment_ids':chain, 'bridge_ids':bridges, 'closed':closed})
    for sid in active_ids:
        if sid not in visited:
            p, chain, bridges, closed = walk(sid, 0)
            if len(p) > 1:
                paths.append(p)
                traces.append({'segment_ids':chain,'bridge_ids':bridges, 'closed':closed})

    # Assemble physical junction markers from the segmentation graph, and then
    # attach inferred stroke IDs and continuity/pairing relations to each.
    sid_to_stroke = {}
    for i, tr in enumerate(traces):
        for sid in tr['segment_ids']:
            sid_to_stroke[sid] = f'S{i}'
    active_seg = [by_sid[sid] for sid in active_ids]
    node_meta = []
    junction_meta = []
    jnum = 0
    def outgoing_direction(segment_id, node_id):
        seg = by_sid[segment_id]
        p = np.asarray(seg.points, dtype=float)
        if len(p) < 2:
            return np.zeros(2)
        look = min(18, len(p)-1)
        if seg.start == node_id:
            delta = p[look]-p[0]
        else:
            delta = p[-1-look]-p[-1]
        norm = np.linalg.norm(delta)
        return delta / max(norm,1e-12)

    for nid in sorted(graph['nodes']):
        if not graph['junction'].get(nid, False):
            continue
        attached = sorted({s.sid for s in active_seg if nid in (s.start, s.end)})
        if not attached:
            continue
        actual = sorted({sid_to_stroke[s] for s in attached if s in sid_to_stroke}, key=lambda x:int(x[1:]))
        records = [r for r in bridge_records if r['junction'] == nid and len(r['endpoints']) == 2]
        through = []
        for r in records:
            a, b = r['endpoints']
            stroke = sid_to_stroke.get(a[0])
            if stroke and stroke == sid_to_stroke.get(b[0]):
                alignment = float(np.dot(outgoing_direction(a[0],nid), outgoing_direction(b[0],nid)))
                through.append({'stroke_id':stroke, 'from_segment':int(a[0]),
                                'to_segment':int(b[0]), 'method':r['kind'],
                                'outgoing_cosine':round(alignment,4)})
        deg = len(attached)
        if deg == 4 and len(through) == 2 and all(p['outgoing_cosine'] < -.9 for p in through):
            kind = 'X_crossing'
        elif deg == 3 and len(through) == 1 and through[0]['outgoing_cosine'] < -.9:
            kind = 'T_junction'
        elif deg == 3:
            kind = 'Y_or_three_way_junction'
        elif deg > 4:
            kind = 'multiway_junction'
        else:
            kind = 'junction'
        entry = {'id':f'J{jnum}', 'kind':kind, 'source_node':int(nid),
                 'xy':np.round(np.asarray(graph['nodes'][nid],float),3).tolist(),
                 'degree':deg, 'incident_strokes':actual,
                 'incident_segments':attached, 'continuations':through,
                 'ending_strokes':sorted(set(actual) - {p['stroke_id'] for p in through})}
        node_meta.append({'id':entry['id'], 'type':kind, 'xy':entry['xy']})
        junction_meta.append(entry)
        jnum += 1

    # Each semantic stroke is a single ordered centerline (except initial
    # closed loops). Each one has graph terminal references and optional via J's.
    end_nodes=[]
    end_counter=0
    def endpoint_node(pt, closed, tr):
        nonlocal end_counter
        if closed:
            return None
        # Exact semantic attachment wins over geometric snapping: an
        # extrapolation may end several pixels away from its source junction.
        for b in tr['bridge_ids']:
            if bridge_records[b]['kind'] != 'extrapolate':
                continue
            if np.linalg.norm(pt - np.asarray(bridge_lines[b][-1])) < 1.5:
                source_nid=bridge_records[b]['junction']
                return next((j['id'] for j in junction_meta if j['source_node'] == source_nid),None)
        candidate=None
        for j in junction_meta:
            d=float(np.linalg.norm(np.asarray(j['xy'])-pt))
            if d <= 5. and (candidate is None or d<candidate[0]):
                candidate=(d,j['id'])
        if candidate is not None:
            return candidate[1]
        name=f'E{end_counter}'
        end_counter += 1
        node_meta.append({'id':name, 'type':'endpoint', 'xy':np.round(pt,3).tolist()})
        end_nodes.append(name)
        return name

    info=[]
    for idx,(p,tr) in enumerate(zip(paths,traces)):
        name=f'S{idx}'
        start=endpoint_node(p[0],tr['closed'],tr)
        end=endpoint_node(p[-1],tr['closed'],tr)
        via=[j['id'] for j in junction_meta if name in j['incident_strokes']]
        stride=max(1,len(p)//250)
        reduced=np.round(p[::stride],2).tolist()
        if not np.allclose(reduced[-1],p[-1],atol=.01):
            reduced.append(np.round(p[-1],2).tolist())
        info.append({'id':name,'start_node':start,'end_node':end,
                     'via_junctions':via, 'closed':bool(tr['closed']),
                     'segment_ids':tr['segment_ids'], 'bridge_ids':tr['bridge_ids'],
                     'length_px':round(_part_length(p),2),
                     # Subpixel curve stored in abbreviated format; original
                     # arrays are in result.semantic_strokes if needed.
                     'polyline_xy':reduced})

    bridge_info=[]
    for i,rec in enumerate(bridge_records):
        owners=sorted({sid_to_stroke[sid] for sid,side in rec['endpoints'] if sid in sid_to_stroke})
        bridge_info.append({'id':f'B{i}', 'junction_id':next((j['id'] for j in junction_meta if j['source_node']==rec['junction']),None),
                            'kind':rec['kind'], 'endpoints':[{'segment_id':int(sid),'side':int(side)}
                                                               for sid,side in rec['endpoints']],
                            'stroke_ids':owners})
    topology={'schema_version':'1.0',
              'description':'Inferred continuous strokes and junction continuity; not actual drawing order or layering',
              'coordinate_system':'x-right, y-down, pixels',
              'image_size':[int(image_shape[1]),int(image_shape[0])],
              'nodes':node_meta,'junctions':junction_meta,
              'strokes':info,'bridges':bridge_info,
              'stroke_count':len(info),'endpoint_count':len(end_nodes),
              'junction_count':len(junction_meta)}
    return paths, topology


def _color(idx):
    return PALETTE[idx % len(PALETTE)]


def _label_position(p, junctions):
    """Place a stroke label along its body but away from crossing labels."""
    pts=np.asarray(p,float)
    if len(pts)==0:
        return np.array([0.,0.])
    spots=[pts[min(len(pts)-1,int(frac*(len(pts)-1)))] for frac in (.24,.35,.65,.76)]
    js=[np.asarray(j['xy'],float) for j in junctions]
    if not js:
        return spots[0]
    return max(spots,key=lambda pt: min(np.linalg.norm(pt-j) for j in js))


def _draw_annotated_svg(result, outpath, pathfn):
    import svgwrite  # explicit export only; skip in GNN CPU import path
    topology = result.topology
    h,w=result.binary.shape
    drawing=svgwrite.Drawing(str(outpath), size=(f'{w}px',f'{h}px'), viewBox=f'0 0 {w} {h}')
    drawing.add(drawing.rect(insert=(0,0),size=(w,h),fill='#FFFFFF'))
    for idx,(p,record) in enumerate(zip(result.semantic_strokes,topology['strokes'])):
        d=pathfn(p)
        if not d:
            continue
        g=drawing.g(id=f"stroke-{record['id']}")
        g.add(drawing.path(d=d,stroke=_color(idx),fill='none',stroke_width=2.8,
                           stroke_linecap='round', stroke_linejoin='round'))
        mid=_label_position(p,topology['junctions'])
        g.add(drawing.rect(insert=(float(mid[0]+3),float(mid[1]-13)),size=(32,15),
                              rx=3,ry=3,fill='white',fill_opacity=.92))
        g.add(drawing.text(record['id'],insert=(float(mid[0]+6),float(mid[1]-2)),
                           font_size=12,font_weight='bold',fill=_color(idx)))
        drawing.add(g)
    for j in topology['junctions']:
        x,y=j['xy']
        drawing.add(drawing.circle(center=(x,y),r=4,fill='white',stroke='#292929',stroke_width=1.8))
        drawing.add(drawing.text(j['id'],insert=(x+7,y-7),font_size=11,font_weight='bold',fill='#202020'))
    for node in topology['nodes']:
        if node['type']!='endpoint':
            continue
        x,y=node['xy']
        drawing.add(drawing.circle(center=(x,y),r=3,fill='#E6C45D',stroke='#424242',stroke_width=1))
        drawing.add(drawing.text(node['id'],insert=(x+4,y+13),font_size=10,fill='#353535'))
    drawing.save()


def _draw_annotated_png(result, outfile):
    h,w=result.binary.shape
    scale=2
    im=np.full((h*scale,w*scale,3),255,np.uint8)
    for i,(p,rec) in enumerate(zip(result.semantic_strokes,result.topology['strokes'])):
        col=_color(i)
        rgb=tuple(int(col[1+j:3+j],16) for j in [0,2,4])
        bgr=rgb[::-1]
        if len(p)>1:
            cv2.polylines(im,[np.rint(p*scale).astype(np.int32).reshape(-1,1,2)],
                          bool(rec['closed']),bgr,3,cv2.LINE_AA)
            mid=tuple(np.rint(_label_position(p,result.topology['junctions'])*scale).astype(int))
            cv2.putText(im,rec['id'],(mid[0]+8,mid[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.75,bgr,2,cv2.LINE_AA)
    for j in result.topology['junctions']:
        pt=tuple(np.rint(np.asarray(j['xy'])*scale).astype(int))
        cv2.circle(im,pt,5,(32,32,32),-1,cv2.LINE_AA)
        cv2.circle(im,pt,3,(255,255,255),-1,cv2.LINE_AA)
        cv2.putText(im,j['id'],(pt[0]+10,pt[1]-10),cv2.FONT_HERSHEY_SIMPLEX,.58,(45,45,45),2,cv2.LINE_AA)
    for n in result.topology['nodes']:
        if n['type']!='endpoint':
            continue
        pt=tuple(np.rint(np.asarray(n['xy'])*scale).astype(int))
        cv2.circle(im,pt,5,(70,180,230),-1,cv2.LINE_AA)
        cv2.circle(im,pt,5,(75,75,75),1,cv2.LINE_AA)
        cv2.putText(im,n['id'],(pt[0]+8,pt[1]+20),cv2.FONT_HERSHEY_SIMPLEX,.52,(55,55,55),1,cv2.LINE_AA)
    cv2.imwrite(str(outfile),im)


def save_topology(result, prefix: str | Path, pathfn):
    """Save _topology.json / _topology.svg / _topology.png at output prefix."""
    prefix=Path(prefix)
    (prefix.parent/(prefix.name+'_topology.json')).write_text(
        json.dumps(result.topology,ensure_ascii=False,indent=2),encoding='utf-8')
    _draw_annotated_svg(result,prefix.parent/(prefix.name+'_topology.svg'),pathfn)
    _draw_annotated_png(result,prefix.parent/(prefix.name+'_topology.png'))
