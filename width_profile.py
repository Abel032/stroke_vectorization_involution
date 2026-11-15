"""CPU-only ONE SCALAR width for each keypoint-to-keypoint segment.

Widths use a single OpenCV Euclidean distance transform of the original ink
mask, sampled away from junctions. No per-point ray marching, PCHIP, rendering,
SVG, image output, or torch. This is an approximation to the physical width,
not a boundary correspondence proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


def compute_distance_map(ink: np.ndarray) -> np.ndarray:
    """One reusable CV_32F Euclidean distance map for the entire foreground.

    At a skeleton pixel, 2*DT-0.5 approximates the raster stroke thickness.
    """
    ink = np.asarray(ink, dtype=bool)
    if ink.ndim != 2:
        raise ValueError('ink must be a 2-D mask')
    return cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


@dataclass(slots=True)
class SegmentMeanWidth:
    id: str
    source_segment_id: int | None
    start_node: int | None
    end_node: int | None
    mean_width_px: float
    quality: float
    samples_used: int
    length_px: float
    closed: bool = False
    source_kind: str = 'segment'
    method: str = 'trimmed_edt_mean'

    def width_at(self, s=None, normalized: bool = False):
        """Compatibility method: constant width everywhere on the segment."""
        q=np.asarray(0. if s is None else s)
        if q.ndim == 0:
            return float(self.mean_width_px)
        return np.full(q.shape, self.mean_width_px, dtype=float)

    def to_dict(self) -> dict:
        return {'id':self.id, 'source_segment_id':self.source_segment_id,
                'start_keypoint':self.start_node, 'end_keypoint':self.end_node,
                'mean_width_px':round(float(self.mean_width_px),3),
                'length_px':round(float(self.length_px),3),
                'quality':round(float(self.quality),3),
                'samples_used':int(self.samples_used),
                'closed':bool(self.closed), 'method':self.method,
                'source_kind':self.source_kind}


# Old symbol retained only to avoid import-time breakage; semantics are SCALAR.
WidthProfile=SegmentMeanWidth


def _simple_poly_length(xy: np.ndarray) -> float:
    if len(xy)<2: return 0.
    return float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())


def estimate_segment_width(ink: np.ndarray, path_xy: np.ndarray, *,
                           distance_map: np.ndarray | None = None,
                           seg_id: str = 'G0', source_segment_id=None,
                           start_node=None, end_node=None, closed=False,
                           trim_fraction: float = .12, max_samples: int = 96,
                           source_kind: str = 'segment', **_unused) -> SegmentMeanWidth:
    """Estimate a SINGLE scalar width over one segment, excluding end zones.

    Sample pixels along the supplied ordered skeleton path, not along its
    geometric spline; the EDT is computed ONCE per image when distance_map is
    supplied. Repeated traversal points are removed. The middle 76% of a
    segment reduces intersection-overlap bias; discard extreme 10% of width
    values for further robustness. Worst-case samples per segment <=96.
    """
    xy=np.asarray(path_xy, dtype=np.float32).reshape(-1,2)
    if distance_map is None:
        distance_map=compute_distance_map(ink)
    dist=np.asarray(distance_map)
    if dist.ndim!=2 or dist.shape!=np.shape(ink):
        raise ValueError('distance_map shape does not match ink')
    length=_simple_poly_length(xy)
    h,w=dist.shape
    if len(xy)==0:
        return SegmentMeanWidth(seg_id,source_segment_id,start_node,end_node,0.,0.,0,0.,closed,source_kind)
    # Avoid copies/linear-time arc resampling on the hot path; pixel skeleton
    # points are approximately equidistant. For subpixel splines, the samples
    # are still a stable discrete approximation to the segment mean.
    n=len(xy)
    trim=max(0,int(np.floor(n*max(0.,min(.4,trim_fraction))))) if not closed else 0
    view=xy[trim:n-trim] if n-2*trim>=2 else xy
    stride=max(1,int(np.ceil(len(view)/max(2,int(max_samples)))))
    view=view[::stride]
    xx=np.clip(np.rint(view[:,0]).astype(np.int32),0,w-1)
    yy=np.clip(np.rint(view[:,1]).astype(np.int32),0,h-1)
    flat=np.unique(yy.astype(np.int64)*w+xx)
    vals=2.*dist[(flat//w).astype(np.intp),(flat%w).astype(np.intp)]-.5
    vals=vals[np.isfinite(vals)&(vals>0.)]
    if not len(vals):
        return SegmentMeanWidth(seg_id,source_segment_id,start_node,end_node,0.,0.,0,length,closed,source_kind)
    if len(vals)>=10:
        lo,hi=np.percentile(vals,[10.,90.])
        valid=vals[(vals>=lo)&(vals<=hi)]
        if len(valid):vals=valid
    mean=max(1.,float(vals.mean()))
    quality=min(1.,len(vals)/8.)
    return SegmentMeanWidth(seg_id,source_segment_id,start_node,end_node,mean,
                            quality,int(len(vals)),length,closed,source_kind)


def means_from_line_label_map(ink: np.ndarray, labels: np.ndarray,
                              n_lines: int, *, distance_map: np.ndarray | None = None):
    """Very fast batch width aggregation via np.bincount, O(HW+N_line).

    labels: -1 for background / keypoint; 0..N-1 for GNN Line pixels.
    Returns (mean_width[N], sample_count[N]). No plotting or storage of curves.
    Pixels outside original foreground are not counted.
    """
    labels=np.asarray(labels)
    ink=np.asarray(ink,dtype=bool)
    if labels.shape!=ink.shape: raise ValueError('line label map / ink shape mismatch')
    if distance_map is None:distance_map=compute_distance_map(ink)
    keep=(labels>=0)&(labels<n_lines)&ink
    idx=labels[keep].astype(np.intp,copy=False)
    counts=np.bincount(idx,minlength=n_lines).astype(np.int32)
    values=2.*np.asarray(distance_map)[keep]-.5
    sums=np.bincount(idx,weights=values,minlength=n_lines)
    means=np.divide(sums,counts,out=np.zeros(n_lines,dtype=float),where=counts>0)
    means[counts>0]=np.maximum(1.,means[counts>0])
    return means.astype(np.float32),counts


__all__=['SegmentMeanWidth','WidthProfile','compute_distance_map',
         'estimate_segment_width','means_from_line_label_map']
