"""Boundary-guided line drawing vectorization (research-oriented reimplementation).

Reference: Zhang et al., Computer Graphics Forum 41(2), 2022,
DOI 10.1111/cgf.14485. Not official authors' implementation.

Coordinates are (x, y), origin at image top left. Binary mask True=ink.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional
import argparse
import json
import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from stroke_topology import assemble_strokes, save_topology
from width_profile import estimate_segment_width, SegmentMeanWidth
from boundary_rays import cast_exit_rays


@dataclass
class Config:
    threshold: int = -1               # -1 = Otsu
    invert: bool = False              # standard input: dark ink on light background
    smooth: float = 0.05              # spline smoothing strength (paper Sec. 4.1.1)
    sample_step: float = 0.5          # paper Sec. 4.2.1
    spike_cos: float = 0.99           # paper Eq. (2)
    connector_boundary_cos: float = 0.95  # paper Sec. 4.2.2
    collinear_cos: float = 0.99       # paper Sec. 4.3.2
    ray_limit: float = 90.0
    boundary_corner_deg: float = 43.0
    max_trim: float = 22.0
    max_bridge: float = 75.0
    min_component: int = 4
    svg_width: float = 1.0
    svg_mode: str = "centerline"      # centerline or original
    enable_completeness: bool = True
    diagonal_policy: str = 'topology_safe'  # redundant_only / legacy / all_8
    junction_merge_distance: float = 3.1
    # Only fuse nearby junction pixel clusters connected on the skeleton
    # through <=2 intermediate valence-two pixels. Euclidean-only fusing
    # joins unrelated crossings in tightly drawn art.
    junction_merge_mode: str = 'geodesic'  # geodesic / euclidean (legacy)
    junction_merge_max_hops: int = 3
    junction_blob_max_span: float = 5.0
    # A 2-5px "loop" returning to the SAME junction is often only a
    # pixel-graph shortcut around a multi-pixel intersection, not a real ring.
    junction_microcycle_max_length: float = 7.0
    junction_microcycle_max_radius: float = 3.0
    # Do not keep residual degree-two junctions caused by segment removal.
    collapse_degree2: bool = True
    demote_unresolved_pseudojunctions: bool = True
    degree2_outward_cos: float = 0.35
    # Cap cutting at each junction as a fraction of the local segment length.
    trim_fraction_limit: float = 0.24
    # RGBA transparent art has alpha ink; handle directly on CPU.
    alpha_threshold: int = 127
    pair_opposite_cos: float = .72
    boundary_query_k: int = 48
    boundary_normal_dot_limit: float = .34
    compute_widths: bool = True
    width_trim_fraction: float = .12  # ignore endpoints most affected by junctions
    width_max_samples: int = 96       # sampled pixels per segment, CPU bound


@dataclass
class Segment:
    sid: int
    start: int
    end: int
    points: np.ndarray
    closed: bool = False
    raw: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    # Dense local points, associated with the original segment orientation.
    dense: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    tangent: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    consistency: np.ndarray = field(default_factory=lambda: np.empty(0))
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)
    active: bool = True
    reason: str = ""


@dataclass(frozen=True)
class BoundaryHit:
    contour: int
    vertex: int
    group: int
    position: tuple[float, float]
    tangent: tuple[float, float]


@dataclass
class Endpoint:
    eid: int
    sid: int
    side: int                   # 0: beginning, 1: end
    junction: int
    position: np.ndarray
    toward: np.ndarray         # unit vector looking out from cutoff toward the junction
    left: Optional[BoundaryHit]
    right: Optional[BoundaryHit]
    original: np.ndarray


@dataclass
class Result:
    binary: np.ndarray
    raw_skeleton: np.ndarray
    boundaries: np.ndarray
    refined_skeleton: np.ndarray
    strokes: list[np.ndarray]
    segment_count: int
    removed_spikes: int
    removed_connectors: int
    junctions: int
    bridges: int
    meta: dict
    semantic_strokes: list[np.ndarray] = field(default_factory=list)
    topology: dict = field(default_factory=dict)
    # Geometric output primitives; no width curves in CPU inference.
    geometric_parts: list[np.ndarray] = field(default_factory=list)
    segment_widths: dict[str, SegmentMeanWidth] = field(default_factory=dict)
    part_widths: list[SegmentMeanWidth] = field(default_factory=list)  # unused for scalar widths
    source_graph: dict = field(default_factory=dict)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-10)


def poly_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def resample(p: np.ndarray, step: float, closed: bool = False) -> np.ndarray:
    p = np.asarray(p, float)
    if not len(p):
        return p.reshape((0, 2))
    if closed and len(p) > 2 and np.linalg.norm(p[0]-p[-1]) > 1e-7:
        p = np.vstack((p, p[0]))
    if len(p) < 2:
        return p.copy()
    dist = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(dist) > 1e-7]
    p, dist = p[keep], dist[keep]
    if len(p) < 2 or dist[-1] < 1e-6:
        return p[:1]
    n = max(2, int(np.ceil(dist[-1] / max(step, .05))) + 1)
    t = np.linspace(0, dist[-1], n)
    return np.c_[np.interp(t, dist, p[:, 0]), np.interp(t, dist, p[:, 1])]


def smooth_curve(p: np.ndarray, strength: float, step: float, closed: bool = False) -> np.ndarray:
    p = np.asarray(p, float)
    if len(p) < 4 or poly_length(p) < 3.:
        return resample(p, step, closed)
    if closed and np.linalg.norm(p[0]-p[-1]) < .5:
        p = p[:-1]
    arclen = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arclen) > 1e-8]
    p = p[keep]
    if len(p) < 4:
        return resample(p, step, closed)
    try:
        tck, _ = splprep([p[:, 0], p[:, 1]], s=max(0., strength)*len(p), k=min(3, len(p)-1), per=closed)
        n = max(2, int(np.ceil(poly_length(p) / step)))
        out = np.asarray(splev(np.linspace(0., 1., n, endpoint=not closed), tck)).T
        if not closed:
            out[0], out[-1] = p[0], p[-1]
        return out
    except (ValueError, TypeError, RuntimeError):
        return resample(p, step, closed)


def tangents(p: np.ndarray, look: int = 5, closed: bool = False) -> np.ndarray:
    if len(p) <= 1:
        return np.zeros((len(p), 2))
    look = min(look, max(1, (len(p)-1)//2))
    if closed:
        return unit(np.roll(p, -look, 0) - np.roll(p, look, 0))
    inds = np.arange(len(p))
    before = np.maximum(inds-look, 0)
    after = np.minimum(inds+look, len(p)-1)
    return unit(p[after] - p[before])


def preprocess(image: np.ndarray, cfg: Config) -> np.ndarray:
    """BGR/gray or BGRA with transparency -> bool ink mask (no visualization)."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4 and np.any(arr[..., 3] != 255):
        # Transparent PNG: RGB can be (0,0,0) even in transparent background.
        bi = (arr[..., 3] > cfg.alpha_threshold).astype(np.uint8)
    else:
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY if arr.shape[-1] == 4
                                    else cv2.COLOR_BGR2GRAY)
        else:
            gray = arr
        gray = np.asarray(gray, np.uint8)
        if cfg.threshold < 0:
            _, bi = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            bi = (gray < cfg.threshold).astype(np.uint8)
    if cfg.invert:
        bi = 1 - bi
    if cfg.min_component > 1 and np.any(bi):
        n, labels, sizes, _ = cv2.connectedComponentsWithStats(bi, 8)
        keep = sizes[:, cv2.CC_STAT_AREA] >= cfg.min_component
        keep[0] = False
        return keep[labels]
    return bi.astype(bool)


def _zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Vectorized two-subiteration Zhang--Suen thinning (1984).

    P2..P9 enumerate the 8-neighborhood clockwise, beginning at north.
    This is the actual algorithm used in the paper's initial step, with no
    requirement for OpenCV contrib or another thinning implementation.
    """
    im = np.pad(binary.astype(np.uint8), 1)
    while True:
        removed = 0
        for stage in (0, 1):
            core = im[1:-1, 1:-1]
            p2 = im[:-2, 1:-1]
            p3 = im[:-2, 2:]
            p4 = im[1:-1, 2:]
            p5 = im[2:, 2:]
            p6 = im[2:, 1:-1]
            p7 = im[2:, :-2]
            p8 = im[1:-1, :-2]
            p9 = im[:-2, :-2]
            ring = (p2,p3,p4,p5,p6,p7,p8,p9)
            n = sum(ring)
            crossings = sum((ring[i] == 0) & (ring[(i+1)%8] == 1)
                            for i in range(8))
            if stage == 0:
                condition = ((p2*p4*p6 == 0) & (p4*p6*p8 == 0))
            else:
                condition = ((p2*p4*p8 == 0) & (p2*p6*p8 == 0))
            erase = (core == 1) & (n >= 2) & (n <= 6) & (crossings == 1) & condition
            removed += int(np.count_nonzero(erase))
            core[erase] = 0
        if removed == 0:
            break
    return im[1:-1, 1:-1].astype(bool)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning on foreground bounding box (CPU-only).

    Trimming empty margins is pixel-equivalent to full-canvas thinning because
    all pixels outside the foreground bounding box are background; it often
    saves a large fraction of iterative-array work for sparse drawings.
    """
    binary=np.asarray(binary,dtype=bool)
    if not np.any(binary):return np.zeros_like(binary)
    ys,xs=np.nonzero(binary)
    y0,y1=int(ys.min()),int(ys.max())+1
    x0,x1=int(xs.min()),int(xs.max())+1
    region=binary[y0:y1,x0:x1]
    xp = getattr(cv2, 'ximgproc', None)
    if xp is not None and hasattr(xp, 'thinning'):
        thin=xp.thinning(np.uint8(region)*255,
                         thinningType=xp.THINNING_ZHANGSUEN)>0
    else:
        thin=_zhang_suen(region)
    out=np.zeros_like(binary)
    out[y0:y1,x0:x1]=thin
    return out


def _neighbors(p: tuple[int, int], present: set, policy: str = "topology_safe") -> list[tuple[int, int]]:
    x, y = p
    out = []
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
        q = (x+dx, y+dy)
        if q not in present:
            continue
        # Preserve staircase diagonal connections. A diagonal is redundant
        # only for a FULL 2x2 square (both orthogonal routes present). The
        # historical 'or' rule removed legitimate staircase edges.
        if dx and dy and policy != 'all_8':
            orth_a = (x+dx,y) in present
            orth_b = (x,y+dy) in present
            if policy in ('topology_safe','legacy'):
                # If an L-shaped two-edge path already connects these pixels,
                # its diagonal shortcut is topologically redundant. Retaining
                # it creates a 3-cycle and falsely marks the bend a junction.
                suppress = orth_a or orth_b
            elif policy == 'redundant_only':
                suppress = orth_a and orth_b  # opt-in, changes topology
            else:
                raise ValueError(f'unknown diagonal policy: {policy}')
            if suppress:
                continue
        out.append(q)
    return out


def skeleton_graph(img: np.ndarray, cfg: Config) -> tuple[list[Segment], dict]:
    yy, xx = np.nonzero(img)
    present = set(zip(xx.tolist(), yy.tolist()))
    adj = {p: _neighbors(p, present, cfg.diagonal_policy) for p in present}
    degree = {p: len(adj[p]) for p in present}
    junction_pix = {p for p in present if degree[p] >= 3}
    junction_blobs = []
    while junction_pix:
        start = junction_pix.pop()
        blob, stack = {start}, [start]
        while stack:
            for q in adj[stack.pop()]:
                if q in junction_pix:
                    junction_pix.remove(q)
                    blob.add(q)
                    stack.append(q)
        junction_blobs.append(blob)
    # A physical X-junction may contain multiple small valence>=3 clusters.
    # The legacy Euclidean-only merge silently fuses distinct nearby crossings.
    # Here the default requires a *short walk on the actual skeleton graph*,
    # with all intermediate pixels of degree 2. Merge-connectors become node
    # pixels, not spurious tiny self-loop segments.
    if cfg.junction_merge_mode not in ('geodesic','euclidean'):
        raise ValueError('junction_merge_mode must be geodesic or euclidean')
    if len(junction_blobs) > 1 and cfg.junction_merge_distance > 0:
        centers = np.asarray([np.mean(list(b),axis=0) for b in junction_blobs])
        near_pairs = sorted(cKDTree(centers).query_pairs(cfg.junction_merge_distance))
        parent=list(range(len(junction_blobs)))
        clusters=[set(x) for x in junction_blobs]
        merged_paths=[]
        original_members={p:i for i,blob in enumerate(junction_blobs) for p in blob}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def short_link(i,j):
            # Walk no more than max_hops edges and never traverse another
            # high-valence node. All intermediate pixels must be valence 2.
            target=junction_blobs[j]
            work=deque((p,(p,)) for p in sorted(junction_blobs[i]))
            seen=set(junction_blobs[i])
            while work:
                p,path=work.popleft()
                if len(path)-1>=cfg.junction_merge_max_hops:
                    continue
                for q in adj[p]:
                    if q in target:
                        return list(path[1:])  # only the interior pixels
                    if q in seen or degree[q]!=2 or q in original_members:
                        continue
                    seen.add(q)
                    work.append((q,path+(q,)))
            return None

        for i,j in near_pairs:
            a,b=find(i),find(j)
            if a==b:continue
            link=None
            if cfg.junction_merge_mode=='geodesic':
                link=short_link(i,j)
                if link is None:continue
            combined=clusters[a] | clusters[b]
            if link:combined.update(link)
            bounds=np.ptp(np.asarray(list(combined),dtype=float),axis=0)
            if cfg.junction_merge_mode=='geodesic' and np.max(bounds)>cfg.junction_blob_max_span:
                continue
            parent[b]=a
            clusters[a]=combined
            clusters[b]=set()
            if link:merged_paths.append(link)
        junction_blobs=[v for v in clusters if v]
    nodes = {}              # nodeid -> xy
    belonging = {}          # pixel -> nodeid
    is_junction = {}
    for blob in junction_blobs:
        nid = len(nodes)
        nodes[nid] = np.mean(np.asarray(list(blob), float), axis=0)
        is_junction[nid] = True
        for p in blob:
            belonging[p] = nid
    for p in present:
        if degree[p] <= 1 and p not in belonging:
            nid = len(nodes)
            nodes[nid] = np.asarray(p, float)
            belonging[p] = nid
            is_junction[nid] = False
    visited = set()
    segments = []

    def ek(a, b):
        return (a,b) if a < b else (b,a)

    def add_path(src, dest, path, closed=False):
        if len(path) < 2:
            return
        pix = np.asarray(path, float)
        if closed and src >= 0 and src == dest and len(pix)>1:
            # Never treat a 2x2 cycle inside a fat junction as an
            # independent closed stroke. Legitimate small loops not attached
            # to a junction (src == -1) are kept unconditionally.
            if (poly_length(pix) <= cfg.junction_microcycle_max_length and
                np.max(np.linalg.norm(pix-nodes[src],axis=1)) <= cfg.junction_microcycle_max_radius):
                return
        # Replace the very ends of junction blobs by the mean junction location.
        if src >= 0:
            pix[0] = nodes[src]
        if dest >= 0:
            pix[-1] = nodes[dest]
        pp = smooth_curve(pix, cfg.smooth, cfg.sample_step, closed)
        if len(pp) > 1:
            segments.append(Segment(len(segments), src, dest, pp, closed, pix, pp.copy()))

    for start, nid in sorted(belonging.items()):
        for q in adj[start]:
            if ek(start,q) in visited:
                continue
            if q in belonging and belonging[q] == nid:
                visited.add(ek(start,q))
                continue
            visited.add(ek(start,q))
            path = [start, q]
            prev, cur = start, q
            while cur not in belonging:
                nexts = [x for x in adj[cur] if x != prev and ek(cur,x) not in visited]
                if not nexts:
                    break
                nxt = nexts[0]
                visited.add(ek(cur,nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            dest = belonging.get(cur, -1)
            if dest >= 0:
                add_path(nid, dest, path, nid == dest)

    # Closed loops have only valence-2 pixels and no nodes.
    for p in sorted(present):
        for q in adj[p]:
            if ek(p,q) in visited:
                continue
            path = [p,q]
            visited.add(ek(p,q))
            prev, cur = p,q
            while cur != p:
                choices = [v for v in adj[cur] if v != prev and ek(cur,v) not in visited]
                if not choices:
                    break
                nxt = choices[0]
                visited.add(ek(cur,nxt))
                path.append(nxt)
                prev,cur = cur,nxt
            add_path(-1,-1,path,cur == p)
    return segments, {'nodes':nodes, 'junction':is_junction, 'adjacency':adj}


class Boundaries:
    """Dense continuous foreground boundaries, corner-based segment IDs and tangents."""
    def __init__(self, binary: np.ndarray, cfg: Config):
        self.cfg = cfg
        self.contours: list[np.ndarray] = []
        self.tvec: list[np.ndarray] = []
        self.groups: list[np.ndarray] = []
        self.positions = np.zeros((0,2))
        self.index = []
        self.tangent_lookup = np.zeros((0,2),dtype=float)
        cvcontours, _ = cv2.findContours(np.uint8(binary)*255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        gp = 0
        for contour in cvcontours:
            p = contour.reshape((-1,2)).astype(float)
            if len(p) < 4:
                continue
            cid = len(self.contours)
            self.contours.append(p)
            # Tangents: fit small-scale symmetric chord to reduce pixel staircases.
            t = tangents(p, 4, True)
            self.tvec.append(t)
            k = max(3, min(6, len(p)//12))
            incoming = unit(p-np.roll(p, k, axis=0))
            outgoing = unit(np.roll(p, -k, axis=0)-p)
            c = np.clip((incoming*outgoing).sum(axis=1), -1, 1)
            angles = np.degrees(np.arccos(c))
            # Equivalent non-maximum suppression, O(P*(k+2)) rather than
            # O(number_of_corner_candidates^2). Circular contours wrap.
            pivots=[]
            blocked=np.zeros(len(p),dtype=bool)
            window=np.arange(-(k+1),k+2)
            for pos in np.argsort(angles)[::-1]:
                if angles[pos] < cfg.boundary_corner_deg:
                    break
                if blocked[pos]:
                    continue
                pos=int(pos)
                pivots.append(pos)
                blocked[(pos+window)%len(p)]=True
            pivots.sort()
            group = np.empty(len(p), dtype=int)
            if not pivots:
                group.fill(gp)
                gp += 1
            else:
                for i, j in enumerate(pivots):
                    nxt = pivots[(i+1)%len(pivots)]
                    if nxt > j:
                        group[j:nxt] = gp
                    else:
                        group[j:] = gp
                        group[:nxt] = gp
                    gp += 1
            self.groups.append(group)
            self.index.extend((cid,j) for j in range(len(p)))
        if self.contours:
            self.positions = np.concatenate(self.contours)
            self.tangent_lookup = np.concatenate(self.tvec)
            self.tree = cKDTree(self.positions)
        else:
            self.tree = None

    def nearest(self, p: np.ndarray) -> Optional[BoundaryHit]:
        if self.tree is None:
            return None
        _, idx = self.tree.query(p)
        cid, j = self.index[int(idx)]
        v = self.contours[cid][j]
        t = self.tvec[cid][j]
        return BoundaryHit(cid,j,int(self.groups[cid][j]),tuple(v),tuple(t))

    def normal_hit(self, center: np.ndarray, direct: np.ndarray,
                   binary: np.ndarray, width_guess: float) -> Optional[BoundaryHit]:
        h, w = binary.shape
        if not (0<=center[0]<w and 0<=center[1]<h):
            return None
        cx=int(np.clip(round(center[0]),0,w-1))
        cy=int(np.clip(round(center[1]),0,h-1))
        if not binary[cy,cx]:
            return None
        limit = min(self.cfg.ray_limit,max(5.,2.8*width_guess+4.))
        step = self.cfg.sample_step
        last = center
        for i in range(1, int(limit/step)+1):
            p = center + direct*(i*step)
            x,y = int(round(p[0])), int(round(p[1]))
            if x < 0 or x >= w or y < 0 or y >= h or not binary[y,x]:
                return self.nearest(last)
            last = p
        return None

    def initiated(self, p: np.ndarray, t: np.ndarray, binary: np.ndarray,
                  edt: np.ndarray) -> tuple[float,Optional[BoundaryHit],Optional[BoundaryHit]]:
        h,w = binary.shape
        x,y = int(np.clip(round(p[0]),0,w-1)),int(np.clip(round(p[1]),0,h-1))
        if np.linalg.norm(t)<1e-5:
            return 0.,None,None
        normal = np.array([-t[1],t[0]])
        l = self.normal_hit(p,normal,binary,float(edt[y,x]))
        r = self.normal_hit(p,-normal,binary,float(edt[y,x]))
        vals = [abs(np.dot(t,hit.tangent)) for hit in (l,r) if hit is not None]
        return (float(max(vals)) if vals else 0.),l,r

    def initiated_batch(self, points: np.ndarray, vectors: np.ndarray,
                        binary: np.ndarray, edt: np.ndarray):
        """Bulk skeleton->boundary matches; 2 cv2.remap rays + bulk KDTree query.

        Compatibility: returns (score[N], left_hits[N], right_hits[N]).
        In contrast to independent per-sample loop, makes only two remaps
        per chunk and two KDTree queries for all successful intersections.
        """
        p=np.asarray(points,float).reshape(-1,2)
        t=unit(np.asarray(vectors,float).reshape(-1,2))
        n=len(p)
        if n == 0:
            return np.zeros(0),[],[]
        normal=np.column_stack([-t[:,1], t[:,0]])
        h,w=binary.shape
        x=np.clip(np.rint(p[:,0]).astype(int),0,w-1)
        y=np.clip(np.rint(p[:,1]).astype(int),0,h-1)
        safe = ((p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
                & binary[y,x] & (np.linalg.norm(t,axis=1)>.1))
        limits=np.minimum(self.cfg.ray_limit, np.maximum(5.,2.8*edt[y,x]+4.))
        left_d,left_ok=cast_exit_rays(binary,p,normal,limits,self.cfg.sample_step)
        right_d,right_ok=cast_exit_rays(binary,p,-normal,limits,self.cfg.sample_step)
        left_ok &= safe
        right_ok &= safe
        hits=[]
        scores=np.zeros((2,n),float)
        for k,(dist,ok,direction) in enumerate(((left_d,left_ok,normal),(right_d,right_ok,-normal))):
            cur=[None]*n
            ids=np.flatnonzero(ok)
            if len(ids) and self.tree is not None:
                endp=p[ids]+direction[ids]*(dist[ids,None]-.5*self.cfg.sample_step)
                # One bulk query for this entire side.
                _distance, near=self.tree.query(endp,k=1)
                for j,ki in enumerate(ids):
                    cid,vertex=self.index[int(near[j])]
                    tang=self.tvec[cid][vertex]
                    pt=self.contours[cid][vertex]
                    cur[ki]=BoundaryHit(cid,vertex,int(self.groups[cid][vertex]),tuple(pt),tuple(tang))
                    scores[k,ki]=abs(float(np.dot(t[ki],tang)))
            hits.append(cur)
        return np.max(scores,axis=0),hits[0],hits[1]

    def boundary_initiated_batch(self, points: np.ndarray, vectors: np.ndarray,
                                 radii: np.ndarray) -> np.ndarray:
        """Batch KDTree kNN with broadcasting; no Python per-sample query."""
        p=np.asarray(points,float).reshape(-1,2)
        t=unit(np.asarray(vectors,float).reshape(-1,2))
        r=np.broadcast_to(np.asarray(radii,float), (len(p),))
        out=np.zeros(len(p),float)
        if self.tree is None or not len(p):
            return out
        k=min(max(1,self.cfg.boundary_query_k),len(self.positions))
        for start in range(0,len(p),1024):
            stop=min(len(p),start+1024)
            distances,indices=self.tree.query(p[start:stop],k=k,distance_upper_bound=float(np.max(r[start:stop])))
            distances=np.asarray(distances).reshape(stop-start,k)
            indices=np.asarray(indices).reshape(stop-start,k)
            ok=(np.isfinite(distances)&(distances >= .2)
                &(distances <= r[start:stop,None])&(indices < len(self.positions)))
            safe_idx=np.minimum(indices,len(self.positions)-1)
            boundary_t=self.tangent_lookup[safe_idx]
            toward=(p[start:stop,None,:]-self.positions[safe_idx])/np.maximum(distances[:,:,None],1e-6)
            normal_ok=np.abs(np.sum(toward*boundary_t,axis=-1))<=self.cfg.boundary_normal_dot_limit
            angle=np.abs(np.sum(t[start:stop,None,:]*boundary_t,axis=-1))
            out[start:stop]=np.max(np.where(ok&normal_ok,angle,0.),axis=1)
        return out

    def boundary_initiated(self, p: np.ndarray, t: np.ndarray, radius: float) -> float:
        return float(self.boundary_initiated_batch(np.asarray(p)[None,:],
                        np.asarray(t)[None,:],np.asarray([radius]))[0])

    def reference_patch(self, ea: Endpoint, eb: Endpoint, max_length: float) -> Optional[np.ndarray]:
        candidates = []
        for ha in (ea.left,ea.right):
            for hb in (eb.left,eb.right):
                if ha is None or hb is None or ha.group != hb.group or ha.contour != hb.contour:
                    continue
                c = self.contours[ha.contour]
                n = len(c)
                i,j = ha.vertex,hb.vertex
                fwd = (j-i)%n
                bwd = (i-j)%n
                if min(fwd,bwd) < 1:
                    continue
                step = 1 if fwd <= bwd else -1
                count = min(fwd,bwd)
                if count > max_length*2.5:
                    continue
                arc = c[(i+step*np.arange(count+1))%n]
                if poly_length(arc)>max_length or poly_length(arc)>3.5*np.linalg.norm(ea.position-eb.position)+4:
                    continue
                # Copy reference boundary arc with a linearly changing offset.
                u = np.linspace(0.,1.,len(arc))[:,None]
                delta_a = ea.position-arc[0]
                delta_b = eb.position-arc[-1]
                patch = arc+(1-u)*delta_a+u*delta_b
                patch[0],patch[-1]=ea.position,eb.position
                candidates.append((poly_length(patch),patch))
        if not candidates:
            return None
        return smooth_curve(min(candidates,key=lambda k:k[0])[1],.03,.5)


def sample_segment(seg: Segment, binary: np.ndarray, edt: np.ndarray, bound: Boundaries):
    seg.dense = resample(seg.points,bound.cfg.sample_step,seg.closed)
    seg.tangent = tangents(seg.dense,5,seg.closed)
    if hasattr(bound, 'initiated_batch'):
        cs,left,right=bound.initiated_batch(seg.dense,seg.tangent,binary,edt)
    else:
        cs=[]; left=[]; right=[]
        for p,t in zip(seg.dense, seg.tangent):
            d,l,r=bound.initiated(p,t,binary,edt)
            cs.append(d);left.append(l);right.append(r)
    seg.consistency=np.asarray(cs)
    seg.left=left
    seg.right=right


def _is_junction(s: Segment, graph: dict) -> tuple[bool,bool]:
    return bool(graph['junction'].get(s.start,False)),bool(graph['junction'].get(s.end,False))


def classify_segments(segments: list[Segment], graph: dict, bound: Boundaries,
                      binary: np.ndarray, edt: np.ndarray, cfg: Config) -> dict:
    for s in segments:
        sample_segment(s,binary,edt,bound)
    # Boundary-to-skeleton segment correlation, using boundary group IDs.
    left_sets = {s.sid:{v.group for v in s.left if v is not None} for s in segments}
    right_sets = {s.sid:{v.group for v in s.right if v is not None} for s in segments}
    count_spike=count_connector=0
    incident=defaultdict(list)
    for s in segments:
        incident[s.start].append(s.sid)
        incident[s.end].append(s.sid)
    for s in segments:
        a,b = _is_junction(s,graph)
        is_spike = (a ^ b) and not s.closed
        is_connector = a and b and not s.closed
        if not (is_spike or is_connector):
            continue
        tangent_bad = len(s.consistency)>0 and bool(np.all(s.consistency < cfg.spike_cos))
        if is_spike:
            # Eq. (3) is a boundary-guided completeness criterion. Apply it
            # conservatively: other *incident* segments must use both sides.
            j = s.start if a else s.end
            siblings=[i for i in incident[j] if i != s.sid]
            sib_left = set().union(*(left_sets[k]|right_sets[k] for k in siblings)) if siblings else set()
            shared_left = left_sets[s.sid] & sib_left
            shared_right = right_sets[s.sid] & sib_left
            # Require distinct boundary segments on both sides, otherwise a
            # single unsplit closed contour would lead to false deletion.
            jpos=graph['nodes'][j]
            jy,jx=int(np.clip(round(jpos[1]),0,edt.shape[0]-1)),int(np.clip(round(jpos[0]),0,edt.shape[1]-1))
            local_width=2.*float(edt[jy,jx])
            short=poly_length(s.dense) < max(6., 2.0*local_width)
            mostly_inconsistent=float(np.mean(s.consistency < cfg.spike_cos)) > .25
            completeness_bad = (cfg.enable_completeness and short and mostly_inconsistent
                and bool(shared_left and shared_right)
                and len((left_sets[s.sid] | right_sets[s.sid])) >= 2)
            if tangent_bad or completeness_bad:
                s.active=False
                s.reason='spike_tangent' if tangent_bad else 'spike_completeness'
                count_spike+=1
        elif is_connector:
            # Paper Sec. 4.2.2: for every point both D_skel<.99 and D_bnd<.95.
            xy=np.clip(np.rint(s.dense).astype(int),[0,0],[binary.shape[1]-1,binary.shape[0]-1])
            radii=np.minimum(cfg.ray_limit,np.maximum(5.,edt[xy[:,1],xy[:,0]]*3.+3.))
            if hasattr(bound,'boundary_initiated_batch'):
                db=bound.boundary_initiated_batch(s.dense,s.tangent,radii)
            else:
                db=np.asarray([bound.boundary_initiated(p,t,r) for p,t,r in zip(s.dense,s.tangent,radii)])
            both=(s.consistency<cfg.spike_cos)&(db<cfg.connector_boundary_cos)
            if len(both)>0 and bool(np.all(both)):
                s.active=False;s.reason='connector_tangent';count_connector+=1
    return {'removed_spikes':count_spike, 'removed_connectors':count_connector}


def trim_to_reliable(s: Segment, side: int, edt: np.ndarray, cfg: Config) -> int:
    p=s.dense
    n=len(p)
    if n<4:
        return 0 if side==0 else n-1
    end = p[0] if side==0 else p[-1]
    x,y=np.clip(np.round(end).astype(int),[0,0],[edt.shape[1]-1,edt.shape[0]-1])
    maxtrim = min(cfg.max_trim, max(3., 2.0*float(edt[y,x])+3.),
                  max(0.75, cfg.trim_fraction_limit * poly_length(p)))
    if side==0:
        order=range(n)
    else:
        order=range(n-1,-1,-1)
    last=0 if side==0 else n-1
    # Trim only an unreliable prefix at a junction. Require 2 consecutive
    # reliable samples to avoid cutting far into a good line.
    good=0
    for idx in order:
        if np.linalg.norm(p[idx]-end)>maxtrim:
            break
        if s.consistency[idx]>=cfg.spike_cos and s.left[idx] is not None and s.right[idx] is not None:
            good+=1
            if good>=2:
                return last
        else:
            good=0
        last=idx
    return last


def _endpoint(s: Segment, side: int, junction: int, bound: Boundaries,
              edt: np.ndarray, cfg: Config, eid: int) -> Endpoint:
    arr=s.dense
    ix=trim_to_reliable(s,side,edt,cfg)
    if side==0:
        look=min(ix+7,len(arr)-1)
        # Toward junction is toward start of original curve.
        toward=unit(arr[ix]-arr[look])
        orig=arr[0]
    else:
        look=max(0,ix-7)
        toward=unit(arr[ix]-arr[look])
        orig=arr[-1]
    if np.linalg.norm(toward)<1e-8:
        toward=unit(orig-arr[ix])
    if np.linalg.norm(toward)<1e-8:
        toward=np.array([1.,0.])
    return Endpoint(eid,s.sid,side,junction,arr[ix].copy(),toward,
                    s.left[ix],s.right[ix],orig.copy())


def _quadratic_extension(s: Segment, ep: Endpoint, target: np.ndarray, cfg: Config) -> np.ndarray:
    # Paper fits a degree-2 polynomial to a broken line, then extrapolates.
    # Here we fit x(t), y(t) in arclength over an adjacent ~10px prefix;
    # a controlled Hermite correction makes the end land exactly at target.
    p=s.dense if ep.side==1 else s.dense[::-1]
    nearest=int(np.argmin(np.linalg.norm(p-ep.position,axis=1)))
    prefix=p[max(0,nearest-22):nearest+1]
    dist=np.linalg.norm(target-ep.position)
    if len(prefix)<3 or dist<.5:
        return np.vstack([ep.position,target])
    ar=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(prefix,axis=0),axis=1))]
    ar=ar-ar[-1]
    if len(np.unique(ar))<3:
        return np.vstack([ep.position,target])
    px=np.polyfit(ar,prefix[:,0],2)
    py=np.polyfit(ar,prefix[:,1],2)
    n=max(2,int(np.ceil(dist/cfg.sample_step)))
    t=np.linspace(0,1,n)
    step=min(dist,max(2.,dist*.9))
    extrap=np.c_[np.polyval(px,t*step),np.polyval(py,t*step)]
    # Keep polynomial's curvature near cutoff, and progressively steer
    # toward the detected junction to prevent run-away extrapolation.
    correction=(target-extrap[-1])[None,:]*(3*t**2-2*t**3)[:,None]
    curve=extrap+correction
    curve[0],curve[-1]=ep.position,target
    return curve


def collapse_degree2_junctions(segments: list[Segment], graph: dict,
                               cfg: Config) -> int:
    """Splice a pair of active segments at a residual 2-way pseudo-junction.

    Classification can delete a false spike, leaving a valence-3 source node
    with two physical arms. It is no longer a keypoint. Splice the *existing*
    sampled polylines rather than creating two extrapolated bridges. Only a
    sufficiently opposed pair is fused; ambiguous touching strokes are kept.
    No new boundary queries and no raster or visualization work.
    """
    if not cfg.collapse_degree2:
        return 0
    count=0
    while True:
        incid=defaultdict(list)
        for s in segments:
            if not s.active or s.closed:
                continue
            incid[s.start].append(s)
            incid[s.end].append(s)
        changed=False
        for nid in sorted(graph['nodes']):
            if not graph['junction'].get(nid,False):
                continue
            touching=incid[nid]
            if len(touching)!=2 or touching[0].sid==touching[1].sid:
                continue
            a,b=touching
            def outward(seg):
                pts=seg.dense
                if len(pts)<2:return np.zeros(2)
                k=min(len(pts)-1, max(2,int(np.ceil(5.0/max(cfg.sample_step,0.1)))))
                if seg.start==nid:return unit(pts[k]-pts[0])
                return unit(pts[-1-k]-pts[-1])
            cos=float(np.dot(outward(a),outward(b)))
            if cos>cfg.degree2_outward_cos:
                continue
            # Orient a toward the junction, b away from it. Reverse sampled
            # boundary sides when reversing the centerline orientation.
            pa=a.dense if a.end==nid else a.dense[::-1]
            pb=b.dense if b.start==nid else b.dense[::-1]
            if len(pa)<2 or len(pb)<2:continue
            if poly_length(pa)<1.0 or poly_length(pb)<1.0:continue
            a_outer=a.start if a.end==nid else a.end
            b_outer=b.end if b.start==nid else b.start
            if a_outer==nid or b_outer==nid:continue
            # Duplicate knot removal; no angle smoothing across a real corner.
            pts=np.vstack((pa,pb[1:]))
            if poly_length(pts)<1e-5:continue
            def orient_info(seg,reverse):
                vals=seg.consistency[::-1] if reverse else seg.consistency
                left=seg.right[::-1] if reverse else seg.left
                right=seg.left[::-1] if reverse else seg.right
                return vals,list(left),list(right)
            ca,la,ra=orient_info(a,a.end!=nid)
            cb,lb,rb=orient_info(b,b.start!=nid)
            # IDs remain stable: the lower-ID segment is the surviving unit.
            keep,drop=(a,b) if a.sid < b.sid else (b,a)
            keep.start,keep.end=a_outer,b_outer
            keep.closed=(a_outer==b_outer)
            keep.points=pts
            keep.dense=pts
            keep.raw=pts.copy()
            keep.tangent=tangents(pts,closed=keep.closed)
            keep.consistency=np.r_[ca,cb[1:]] if len(ca)==len(pa) and len(cb)==len(pb) else np.ones(len(pts))
            keep.left=la+lb[1:] if len(la)==len(pa) and len(lb)==len(pb) else [None]*len(pts)
            keep.right=ra+rb[1:] if len(ra)==len(pa) and len(rb)==len(pb) else [None]*len(pts)
            drop.active=False
            drop.reason='degree2_collapsed'
            graph['nodes'].pop(nid,None)
            graph['junction'].pop(nid,None)
            graph.setdefault('collapsed_junctions',[]).append(int(nid))
            count+=1
            changed=True
            break
        if not changed:break
    return count


def demote_pseudojunctions(segments: list[Segment], graph: dict,
                           cfg: Config) -> int:
    """Mark valence<3 residuals as endpoints/ambiguous touches, not junctions.

    A residual degree-two node with acute outward arms is *not* blindly
    joined: this is often two nearby lines that the original thinning
    accidentally squeezed together. Remove its false junction status but
    preserve two separate geometric segment terminals and their original IDs.
    """
    if not cfg.demote_unresolved_pseudojunctions:
        return 0
    incid=defaultdict(list)
    for s in segments:
        if not s.active or s.closed:continue
        incid[s.start].append(s.sid)
        incid[s.end].append(s.sid)
    demoted=[]
    for nid in sorted(graph['nodes']):
        if not graph['junction'].get(nid,False):continue
        attached=sorted(set(incid[nid]))
        if len(attached) >=3:continue
        graph['junction'][nid]=False
        demoted.append({'id':int(nid), 'remaining_valence':len(attached),
                        'segment_ids':attached})
    graph['demoted_pseudojunctions']=demoted
    return len(demoted)


def reconstruct(segments: list[Segment], graph: dict, bound: Boundaries,
                edt: np.ndarray, cfg: Config) -> tuple[list[np.ndarray],int]:
    active={s.sid:s for s in segments if s.active}
    # For each active segment retain a slice between its reliable cutoffs.
    epmap:dict[tuple[int,int],Endpoint]={}
    byjunction=defaultdict(list)
    counter=0
    for s in active.values():
        if s.closed:
            continue
        for side,j in ((0,s.start),(1,s.end)):
            if j >= 0 and graph['junction'].get(j,False):
                ep=_endpoint(s,side,j,bound,edt,cfg,counter)
                epmap[(s.sid,side)]=ep
                byjunction[j].append(ep)
                counter+=1
    pieces={}
    for s in active.values():
        start=epmap.get((s.sid,0))
        end=epmap.get((s.sid,1))
        a=0 if start is None else int(np.argmin(np.linalg.norm(s.dense-start.position,axis=1)))
        b=len(s.dense)-1 if end is None else int(np.argmin(np.linalg.norm(s.dense-end.position,axis=1)))
        if b<a+1:
            a,b=0,len(s.dense)-1
        pieces[s.sid]=s.dense[a:b+1].copy()

    # Reconstruct paths at each junction using boundary reference first;
    # next use collinear tangent pairs; leftover branches extrapolate.
    bridge_lines=[]
    bridge_records=[]
    for nid, endpoints in byjunction.items():
        if len(endpoints)<2:
            continue
        center=graph['nodes'][nid]
        unmatched={e.eid:e for e in endpoints}
        candidate=[]
        for ia,a in enumerate(endpoints):
            for b in endpoints[ia+1:]:
                if a.sid == b.sid:
                    continue
                direct=float(np.dot(a.toward,b.toward))
                # Opposite incoming directions means one continuous stroke.
                if direct > -cfg.pair_opposite_cos:
                    continue
                gap=float(np.linalg.norm(a.position-b.position))
                if gap>cfg.max_bridge:
                    continue
                patch=bound.reference_patch(a,b,cfg.max_bridge)
                if patch is not None:
                    # Boundary continuity takes priority, even when curvature is high.
                    candidate.append((2.+(-direct),a.eid,b.eid,patch,'boundary'))
                elif -direct>=cfg.collinear_cos:
                    candidate.append((1.+(-direct),a.eid,b.eid,
                                      resample(np.vstack([a.position,b.position]),cfg.sample_step),'tangent'))
        # Select disjoint endpoint pairings (greedy matching by confidence).
        for score,ia,ib,line,kind in sorted(candidate,key=lambda x:-x[0]):
            if ia not in unmatched or ib not in unmatched:
                continue
            a,b=unmatched.pop(ia),unmatched.pop(ib)
            bridge_lines.append(line)
            bridge_records.append({'junction':nid, 'kind':kind, 'endpoints':[(a.sid,a.side),(b.sid,b.side)]})
        for ep in unmatched.values():
            # At a T or Y junction, snap remaining branches to the closest
            # existing centerline near the junction, otherwise the junction.
            target=np.asarray(center,float)
            for line in bridge_lines:
                if np.min(np.linalg.norm(line-center,axis=1))>cfg.max_bridge/2:
                    continue
                j=int(np.argmin(np.linalg.norm(line-ep.position,axis=1)))
                if np.linalg.norm(line[j]-center) < max(4.,np.linalg.norm(ep.position-center)):
                    target=line[j]
            if np.linalg.norm(target-ep.position)>cfg.max_bridge:
                continue
            ext=_quadratic_extension(active[ep.sid],ep,target,cfg)
            bridge_lines.append(ext)
            bridge_records.append({'junction':nid, 'kind':'extrapolate', 'endpoints':[(ep.sid,ep.side)]})

    # Preserve consistent segment boundaries; bridges are separate SVG paths
    # with identical geometric endpoints, hence the output remains connected.
    strokes=[p for p in pieces.values() if len(p)>=2]
    strokes.extend(bridge_lines)
    return strokes,len(bridge_records),pieces,bridge_lines,bridge_records


def rasterize(strokes: list[np.ndarray], shape: tuple[int,int], thickness: int = 1) -> np.ndarray:
    # For downstream GNN, the refined skeleton MUST be a discrete 1px mask.
    # LINE_AA creates low-alpha neighbor pixels that become True after >0 and
    # can merge otherwise disjoint Region partitions. Use LINE_8 instead.
    image=np.zeros(shape,dtype=np.uint8)
    for p in strokes:
        if len(p)<2:
            continue
        pts=np.round(p).astype(np.int32).reshape((-1,1,2))
        cv2.polylines(image,[pts],False,255,thickness,cv2.LINE_8)
    return image


def _svg_path(p: np.ndarray, simplify: float=.65) -> str:
    p=np.asarray(p,float)
    if len(p)<2:
        return ''
    p=cv2.approxPolyDP(p.reshape((-1,1,2)).astype(np.float32),simplify,False).reshape((-1,2))
    if len(p)<2:
        return ''
    def xy(v):return f'{v[0]:.3f},{v[1]:.3f}'
    if len(p)==2:
        return 'M '+xy(p[0])+' L '+xy(p[1])
    d='M '+xy(p[0])
    for i in range(len(p)-1):
        p0=p[max(i-1,0)];p1=p[i];p2=p[i+1];p3=p[min(i+2,len(p)-1)]
        c1=p1+(p2-p0)/6.
        c2=p2-(p3-p1)/6.
        d+=' C '+xy(c1)+' '+xy(c2)+' '+xy(p2)
    return d


def save_svg(result: Result, filename: str | Path, cfg: Config):
    """EXPLICIT CLI/export only; never used by the callable CPU API."""
    import svgwrite
    h,w=result.binary.shape
    doc=svgwrite.Drawing(str(filename),size=(f'{w}px',f'{h}px'),viewBox=f'0 0 {w} {h}')
    doc.add(doc.rect(insert=(0,0),size=(w,h),fill='white'))
    piece_ids=result.meta.get('part_segment_ids', [])
    for i,path in enumerate(result.strokes):
        d=_svg_path(path)
        if not d:continue
        width=float(cfg.svg_width)
        if cfg.svg_mode=='original':
            sid=piece_ids[i] if i<len(piece_ids) else None
            prof=result.segment_widths.get(f'G{sid}') if sid is not None else None
            if prof is not None:
                width=prof.mean_width_px
        doc.add(doc.path(d=d,stroke='black',fill='none',stroke_width=max(.5,width),
                         stroke_linecap='round',stroke_linejoin='round'))
    doc.save()


def vectorize(image: np.ndarray, cfg: Config | None = None, *,
              output_mode: str = 'full') -> Result:
    """CPU-only extraction; no file I/O or visualization.

    output_mode='skeleton' only builds the refined skeleton; 'segments' also
    returns active keypoint-segment mean widths and the source graph without
    assembling semantic strokes, and 'full' adds full topology annotations.
    """
    if output_mode not in ('full','skeleton','segments'):
        raise ValueError('output_mode must be full, segments, or skeleton')
    cfg=cfg or Config()
    binary=preprocess(image,cfg)
    raw=skeletonize(binary)
    boundary_model=Boundaries(binary,cfg)
    segments,graph=skeleton_graph(raw,cfg)
    graph['segments']=segments
    edt=ndimage.distance_transform_edt(binary)
    removed=classify_segments(segments,graph,boundary_model,binary,edt,cfg)
    n_collapsed=collapse_degree2_junctions(segments,graph,cfg)
    n_demoted=demote_pseudojunctions(segments,graph,cfg)
    strokes,nbridges,pieces,bridge_lines,bridge_records=reconstruct(
        segments,graph,boundary_model,edt,cfg)
    out=rasterize(strokes,binary.shape)>0
    if output_mode=='skeleton':
        # No SVG, PNG, semantic stroke analysis, JSON or width sampling.
        return Result(binary,raw,np.zeros((0,0),bool),out,[],len(segments),
                      removed['removed_spikes'],removed['removed_connectors'],
                      sum(graph['junction'].values()),nbridges,
                      {'size':[int(binary.shape[1]),int(binary.shape[0])],
                       'active_segments':sum(s.active for s in segments),
                       'collapsed_degree2_junctions':n_collapsed,
                       'demoted_pseudojunctions':n_demoted,**removed},
                      source_graph=graph)

    boundary_mask=np.zeros_like(raw) if output_mode=='full' else np.zeros((0,0),bool)
    semantic_strokes=[]
    stroke_topology={}
    if output_mode=='full':
        if boundary_model.positions.size:
            coords=boundary_model.positions.astype(int)
            boundary_mask[coords[:,1],coords[:,0]]=True
        semantic_strokes,stroke_topology=assemble_strokes(
            pieces,bridge_lines,bridge_records,segments,graph,binary.shape)
    active={s.sid:s for s in segments if s.active}
    widths={}
    if cfg.compute_widths:
        # Reuse classification's ONE EDT array; one scalar per original
        # keypoint-to-keypoint skeleton graph edge. No ray casting/PCHIP.
        for sid,path in pieces.items():
            seg=active[sid]
            profile=estimate_segment_width(binary,seg.raw if len(seg.raw) else path,
                      distance_map=edt,seg_id=f'G{sid}', source_segment_id=sid,
                      start_node=seg.start,end_node=seg.end,closed=seg.closed,
                      trim_fraction=cfg.width_trim_fraction,
                      max_samples=cfg.width_max_samples)
            widths[profile.id]=profile
    if output_mode=='full':
        stroke_topology['geometric_segments']=[p.to_dict() for p in widths.values()]
    part_segment_ids=[int(sid) for sid,path in pieces.items() if len(path)>=2]
    part_segment_ids += [None]*len(bridge_lines)
    meta={'size':[int(binary.shape[1]),int(binary.shape[0])],
          'skeleton_segments':len(segments),'active_segments':len(active),
          'collapsed_degree2_junctions':n_collapsed,
          'demoted_pseudojunctions':n_demoted,
          'closed_segments':sum(x.active and x.closed for x in segments),
          'collapsed_segment_ids':[int(x.sid) for x in segments if x.reason=='degree2_collapsed'],
          'semantic_strokes':len(semantic_strokes),
          'junctions':sum(graph['junction'].values()),
          'bridges':nbridges,
          'part_segment_ids':part_segment_ids,
          'widths':{'schema_version':'2.0','units':'pixels',
                    'definition':'one scalar mean_width_px per active G segment',
                    'segment_count':len(widths)},
          **removed,
          'removed_ids':{str(s.sid):s.reason for s in segments
                         if not s.active and s.reason!='degree2_collapsed'},
          'config':vars(cfg)}
    stored_strokes=strokes if output_mode=='full' else []
    return Result(binary,raw,boundary_mask,out,stored_strokes,len(segments),
                  removed['removed_spikes'],removed['removed_connectors'],
                  meta['junctions'],nbridges,meta,semantic_strokes,stroke_topology,
                  list(stored_strokes),widths,[],graph)


def export_result(result: Result, cfg: Config, prefix: str | Path):
    prefix=Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    save_svg(result,prefix.with_suffix('.svg'),cfg)
    cv2.imwrite(str(prefix.parent/(prefix.name+'_binary.png')),np.uint8(~result.binary)*255)
    cv2.imwrite(str(prefix.parent/(prefix.name+'_skeleton.png')),np.uint8(~result.raw_skeleton)*255)
    cv2.imwrite(str(prefix.parent/(prefix.name+'_boundary.png')),np.uint8(~result.boundaries)*255)
    cv2.imwrite(str(prefix.parent/(prefix.name+'_refined.png')),np.uint8(~result.refined_skeleton)*255)
    save_topology(result,prefix,_svg_path)
    (prefix.parent/(prefix.name+'_widths.json')).write_text(
        json.dumps({'schema_version':'2.0','units':'pixels',
                    'description':'one mean_width_px per active keypoint segment',
                    'segments':[p.to_dict() for p in result.segment_widths.values()]},
                   ensure_ascii=False,indent=2),encoding='utf-8')
    (prefix.parent/(prefix.name+'_metadata.json')).write_text(json.dumps(result.meta,ensure_ascii=False,indent=2),encoding='utf-8')


def cli(argv=None):
    ap=argparse.ArgumentParser(description='Boundary-based topology reconstruction for line drawings')
    ap.add_argument('input',help='Raster input PNG/JPG')
    ap.add_argument('-o','--output',default='outputs/result',help='Output path prefix (without extension)')
    ap.add_argument('--threshold',type=int,default=-1,help='-1: Otsu; otherwise integer 0..255')
    ap.add_argument('--invert',action='store_true',help='White strokes on black background')
    ap.add_argument('--svg-mode',choices=['centerline','original'],default='centerline')
    ap.add_argument('--stroke-width',type=float,default=1.)
    ap.add_argument('--smooth',type=float,default=.05)
    ap.add_argument('--min-component',type=int,default=4)
    ap.add_argument('--spike-cos',type=float,default=.99)
    ap.add_argument('--sample-step',type=float,default=.5,help='Dense sampling step (pixels)')
    ap.add_argument('--ray-limit',type=float,default=90.,help='Max boundary-ray distance (pixels)')
    ap.add_argument('--max-bridge',type=float,default=75.,help='Max junction bridging distance')
    ap.add_argument('--no-completeness',action='store_true')
    ap.add_argument('--collinear-cos',type=float,default=.99)
    ap.add_argument('--connector-boundary-cos',type=float,default=.95)
    ap.add_argument('--max-trim',type=float,default=22.)
    ap.add_argument('--boundary-corner-deg',type=float,default=43.)
    ap.add_argument('--junction-merge-distance',type=float,default=3.1)
    ap.add_argument('--junction-merge-mode',choices=['geodesic','euclidean'],default='geodesic')
    ap.add_argument('--junction-merge-max-hops',type=int,default=3)
    ap.add_argument('--junction-blob-max-span',type=float,default=5.)
    ap.add_argument('--junction-microcycle-max-length',type=float,default=7.)
    ap.add_argument('--junction-microcycle-max-radius',type=float,default=3.)
    ap.add_argument('--no-degree2-collapse',action='store_true')
    ap.add_argument('--degree2-outward-cos',type=float,default=.35)
    ap.add_argument('--no-demote-pseudojunctions',action='store_true')
    ap.add_argument('--trim-fraction-limit',type=float,default=.24)
    ap.add_argument('--alpha-threshold',type=int,default=127)

    ap.add_argument('--boundary-query-k',type=int,default=48)
    ap.add_argument('--boundary-normal-dot-limit',type=float,default=.34)
    ap.add_argument('--pair-opposite-cos',type=float,default=.72)
    ap.add_argument('--diagonal-policy',choices=['topology_safe','redundant_only','legacy','all_8'],default='topology_safe')
    ap.add_argument('--no-widths',action='store_true',help='Skip width profiles (faster GNN skeleton-only preprocessing)')
    ap.add_argument('--width-trim-fraction',type=float,default=.12)
    ap.add_argument('--width-max-samples',type=int,default=96)
    args=ap.parse_args(argv)
    image=cv2.imread(args.input,cv2.IMREAD_UNCHANGED)
    if image is None:
        ap.error(f'Cannot read image: {args.input}')
    cfg=Config(threshold=args.threshold,invert=args.invert,svg_mode=args.svg_mode,
               svg_width=args.stroke_width,spike_cos=args.spike_cos,
               smooth=args.smooth,min_component=args.min_component,
               sample_step=args.sample_step,ray_limit=args.ray_limit,max_bridge=args.max_bridge,
               enable_completeness=not args.no_completeness,
               collinear_cos=args.collinear_cos,connector_boundary_cos=args.connector_boundary_cos,
               max_trim=args.max_trim,boundary_corner_deg=args.boundary_corner_deg,
               junction_merge_distance=args.junction_merge_distance,
               junction_merge_mode=args.junction_merge_mode,
               junction_merge_max_hops=args.junction_merge_max_hops,
               junction_blob_max_span=args.junction_blob_max_span,
               junction_microcycle_max_length=args.junction_microcycle_max_length,
               junction_microcycle_max_radius=args.junction_microcycle_max_radius,
               collapse_degree2=not args.no_degree2_collapse,
               degree2_outward_cos=args.degree2_outward_cos,
               demote_unresolved_pseudojunctions=not args.no_demote_pseudojunctions,
               trim_fraction_limit=args.trim_fraction_limit,
               alpha_threshold=args.alpha_threshold,
               boundary_query_k=args.boundary_query_k,
               boundary_normal_dot_limit=args.boundary_normal_dot_limit,
               pair_opposite_cos=args.pair_opposite_cos,diagonal_policy=args.diagonal_policy,
               compute_widths=not args.no_widths,
               width_trim_fraction=args.width_trim_fraction,
               width_max_samples=args.width_max_samples)
    result=vectorize(image,cfg)
    export_result(result,cfg,args.output)
    print(json.dumps({k:v for k,v in result.meta.items() if k not in ('removed_ids','config')},indent=2))
    print('Saved:',str(Path(args.output).with_suffix('.svg')))
    print('Stroke topology:',str(Path(args.output).parent/(Path(args.output).name+'_topology.json')))


if __name__=='__main__':
    cli()
